import torch
import numpy as np
import random
import os
from PIL import Image


from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler, DDIMScheduler
from huggingface_hub import hf_hub_download

from photomaker import PhotoMakerAnimateDiffXLPipline
from diffusers import MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_gif, load_image
from insightface.app import FaceAnalysis
from transformers import CLIPVisionModelWithProjection
import cv2
# gloal variable and function
import argparse
import torch_npu
from torch_npu.contrib import transfer_to_npu

def isimage(path):
    if 'png' in path.lower() or 'jpg' in path.lower() or 'jpeg' in path.lower():
        return True
    
def extract_face_features(image_lst: list, input_size=(640, 640)):
    # Extract Face features using insightface
    ref_images = []
    app = FaceAnalysis(name="buffalo_l",
                       root="./pretrain_model",
                       providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    app.prepare(ctx_id=0, det_size=input_size)
    for img in image_lst:
        image = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)
        faces = app.get(image)
        if len(faces)==0:
            continue
        image = torch.from_numpy(faces[0].normed_embedding)
        ref_images.append(image.unsqueeze(0))
    if len(ref_images)==0:
        return [None]
    else:
        ref_images = torch.stack(ref_images, dim=0).unsqueeze(0)

    return ref_images


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-s", "--seed", type=int, nargs='+',default=[42,128], help="seed for seed_everything")
    parser.add_argument("-p", "--prompt", type=str, default='emjoy.txt', help="prompt file path")
    parser.add_argument("-i", "--image", type=str, help="image")
    parser.add_argument("--unet_path", type=str, help="image", default='checkpoints/checkpoint-3000/pytorch_model_final.bin')
    parser.add_argument("-o","--output", type=str, default='outputs', help="output dir")
    parser.add_argument("--name", type=str, default='photomaker_mix', help="output name")
    parser.add_argument("-n", "--num_steps", type=int, default=50, help="number of steps")
    parser.add_argument("--size",type=int, default=512, help="size of image")
    return parser

def load_prompts(prompt_file):
    f = open(prompt_file, 'r')
    prompt_list = []
    for idx, line in enumerate(f.readlines()):
        l = line.strip()
        if len(l) != 0:
            prompt_list.append(l)
        f.close()
    return prompt_list

parser = get_parser()
args = parser.parse_args()
base_model_path = './pretrain_model/RealVisXL_V4.0'
device = "cuda"
adapter = MotionAdapter.from_pretrained("pretrain_model/animatediff-motion-adapter-sdxl-beta")


scheduler = DDIMScheduler.from_pretrained(
    base_model_path,
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)
image_encoder_path = "./pretrain_model/CLIP-ViT-H-14-laion2B-s32B-b79K"
image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path, torch_dtype=torch.float16)
pipe = PhotoMakerAnimateDiffXLPipline.from_pretrained(
    base_model_path,
    motion_adapter=adapter,
    scheduler=scheduler,
    torch_dtype=torch.float16,
    image_encoder=image_encoder,
    variant="fp16",
).to("cuda")
print("load unet from ",args.unet_path)
pipe.set_fusion_model(unet_path=args.unet_path)
# pipe.set_ip_adapter_scale([0.7,0.7])  
# pipe.enable_model_cpu_offload()
print("over")
# define and show the input ID images


latent = torch.load("/group/40034/jackeywu/code/PhotoMaker/datasets/CeleV-Text/processed_512/___5yD2BVx8_4_0.pt",map_location='cpu')
data = torch.load("/group/40034/jackeywu/code/PhotoMaker/datasets/CeleV-Text/processed/___5yD2BVx8_4_0.pt",map_location='cpu')
latent = latent['latent']

dir_name = "___5yD2BVx8_4_0"
## Note that the trigger word `img` must follow the class word for personalization
prompts = load_prompts(args.prompt)
negative_prompt = "(asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth"

face_id_embeds = data['face_ids'][0].to("cuda")

neg_face_id_embeds = torch.zeros_like(face_id_embeds)
id_embeds = torch.cat([neg_face_id_embeds, face_id_embeds], dim=0).to(dtype=torch.float16, device="cuda")
# id_embeds = face_id_embeds
clip_embeds = data["image_embeds"][0].to("cuda")
clip_embeds = pipe.prepare_ip_adapter_image_embeds(ip_adapter_image=None, ip_adapter_image_embeds=[clip_embeds,clip_embeds], device=torch.device("cuda"), num_images_per_prompt=1, do_classifier_free_guidance=True)[0]
    
## Parameter setting
num_steps = args.num_steps
style_strength_ratio = 20
start_merge_step = int(float(style_strength_ratio) / 100 * num_steps)
if start_merge_step > 30:
    start_merge_step = 30

pipe.enable_vae_slicing()
pipe.enable_vae_tiling()
# print(input_id_images[0] if args.ip_adapter else None)
seed_list = args.seed

size = (args.size, args.size)

for seed in seed_list:
    generator = torch.Generator(device=device).manual_seed(seed)
    frames = pipe(
        num_frames=16,
        height=size[0],
        width=size[1],
        guidance_scale=8,
        prompt_embeds=data["prompt_embeds_trigger"][0].unsqueeze(dim=0).to("cuda"),
        pooled_prompt_embeds=data["pooled_prompt_embeds_trigger"].unsqueeze(dim=0).to("cuda"),
        prompt_embeds_text_only=data["prompt_embeds"].unsqueeze(dim=0).to("cuda"),
        pooled_prompt_embeds_text_only=data["pooled_prompt_embeds"].unsqueeze(dim=0).to("cuda"),
        ip_adapter_image_embeds=[clip_embeds, id_embeds],
        num_videos_per_prompt=1,
        num_inference_steps=num_steps,
        start_merge_step=start_merge_step,
        generator=generator,
    ).frames[0]
    os.makedirs("{}/{}".format(args.output,dir_name), exist_ok=True)
    export_to_gif(frames, "{}/{}/{}_seed_{}.gif".format(args.output, dir_name, args.name, seed))
