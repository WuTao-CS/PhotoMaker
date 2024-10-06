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
from insightface.utils import face_align
# gloal variable and function
import argparse
def isimage(path):
    if 'png' in path.lower() or 'jpg' in path.lower() or 'jpeg' in path.lower():
        return True
    
def extract_face_features(image_lst: list, input_size=(640, 640)):
    # Extract Face features using insightface
    ref_images = []
    ref_images_emb = []
    app = FaceAnalysis(name="buffalo_l",
                       root="./pretrain_model",
                       providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    app.prepare(ctx_id=0, det_size=input_size)
    for img in image_lst:
        img = np.asarray(img)
        face_info = app.get(img)
        if len(face_info)==0:
                continue
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
        norm_face = face_align.norm_crop(img, landmark=face_info.kps, image_size=256)
        ref_images.append(norm_face)
        emb = torch.from_numpy(face_info.normed_embedding)
        ref_images_emb.append(emb)
    if len(ref_images)==0:
        print("no face detect")
        return [None, None]
    else:
        ref_images_emb = torch.stack(ref_images_emb, dim=0).unsqueeze(0)

    return ref_images, ref_images_emb


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-s", "--seed", type=int, nargs='+',default=[1024,128], help="seed for seed_everything")
    parser.add_argument("-p", "--prompt", type=str, default='emjoy.txt', help="prompt file path")
    parser.add_argument("-i", "--image", type=str, help="image")
    parser.add_argument("--pretrain_path", type=str, default="./pretrain_model/RealVisXL_V4.0")
    parser.add_argument("--unet_path", type=str, help="model path", default=None)
    parser.add_argument("--inject_block_txt", type=str, help="inject set", default="/group/40034/jackeywu/code/PhotoMaker/block.txt")
    parser.add_argument("-o","--output", type=str, default='outputs', help="output dir")
    parser.add_argument("--name", type=str, default='photomaker_mix', help="output name")
    parser.add_argument("-n", "--num_steps", type=int, default=50, help="number of steps")
    parser.add_argument("--size",type=int, default=512, help="size of video")
    parser.add_argument(
        "--enable_new_ip_adapter", action="store_true", help="Whether or not to use new ip-adapter."
    )
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
base_model_path = args.pretrain_path
device = "cuda"
adapter = MotionAdapter.from_pretrained("./pretrain_model/animatediff-motion-adapter-sdxl-beta")

print("load pretrain model from", base_model_path)
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
pipe.set_fusion_model(unet_path=args.unet_path,inject_block_txt=args.inject_block_txt,new_ip_adapter=args.enable_new_ip_adapter)
print(pipe.unet)
# pipe.set_ip_adapter_scale([0.7,0.7])  
# pipe.enable_model_cpu_offload()
print("over")
# define and show the input ID images

image_path_list=[args.image]

input_id_images=[]
for image_path in image_path_list:
    input_id_images.append(load_image(image_path))

for image_path,input_id_image in zip(image_path_list, input_id_images):
    input_id_images = [input_id_image]
    dir_name = os.path.basename(image_path).split('.')[0]
    ## Note that the trigger word `img` must follow the class word for personalization
    prompts = load_prompts(args.prompt)
    negative_prompt = "(asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth"

    face_id_image,face_id_embeds = extract_face_features(input_id_images)
    if face_id_embeds is None:
        continue
    neg_face_id_embeds = torch.zeros_like(face_id_embeds)
    id_embeds = torch.cat([neg_face_id_embeds, face_id_embeds], dim=0).to(dtype=torch.float16, device="cuda")
    # id_embeds = face_id_embeds
    clip_embeds = pipe.prepare_ip_adapter_image_embeds([face_id_image[0],face_id_image[0]], None, torch.device("cuda"), 1, True)[0]
    ## Parameter setting
    print("#######################")
    print(clip_embeds.shape)
    print(id_embeds.shape)
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
    for prompt in prompts:
        for seed in seed_list:
            generator = torch.Generator(device=device).manual_seed(seed)
            frames = pipe(
                prompt=prompt,
                num_frames=16,
                height=size[0],
                width=size[1],
                guidance_scale=7.5,
                input_id_images=input_id_images,
                negative_prompt=negative_prompt,
                ip_adapter_image_embeds=[clip_embeds, id_embeds],
                num_videos_per_prompt=1,
                num_inference_steps=num_steps,
                start_merge_step=start_merge_step,
                generator=generator,
            ).frames[0]
            os.makedirs("{}/{}".format(args.output,dir_name), exist_ok=True)
            export_to_gif(frames, "{}/{}/{}_{}_seed_{}.gif".format(args.output, dir_name, args.name, prompt.replace(' ','_'),seed))
