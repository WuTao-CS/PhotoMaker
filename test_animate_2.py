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
    ref_images = torch.stack(ref_images, dim=0).unsqueeze(0)

    return ref_images

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-s", "--seed", type=int, nargs='+',default=[42,128], help="seed for seed_everything")
    parser.add_argument("-p", "--prompt", type=str, default='lecun.txt', help="prompt file path")
    parser.add_argument("-i", "--image", type=str, default='./datasets/lecun', help="image prompt dir path")
    parser.add_argument("-o", "--output", type=str, default='photomakerwithadapter_clipl_animate', help="output name")
    parser.add_argument("--name", type=str, default='lecun', help="name")
    parser.add_argument("-n", "--num_steps", type=int, default=50, help="number of steps")
    parser.add_argument("--multi_ip_adapter", default=False, action='store_true')
    parser.add_argument("--index", type=int, default=0, help="number of steps")
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
base_model_path = './pretrain_model/RealVisXL_V3.0'
device = "cuda"
save_path = "./outputs"
adapter = MotionAdapter.from_pretrained("pretrain_model/animatediff-motion-adapter-sdxl-beta")
image_encoder_path = "./pretrain_model/CLIP-ViT-H-14-laion2B-s32B-b79K"
image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path, torch_dtype=torch.float16)

scheduler = DDIMScheduler.from_pretrained(
    base_model_path,
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)
pipe = PhotoMakerAnimateDiffXLPipline.from_pretrained(
    base_model_path,
    motion_adapter=adapter,
    scheduler=scheduler,
    torch_dtype=torch.float16,
    image_encoder=image_encoder,
    variant="fp16",
).to("cuda")


pipe.load_photomaker_adapter(
    "./pretrain_model/PhotoMaker",
    subfolder="",
    weight_name="photomaker-v1.bin",
    trigger_word="img"
)
pipe.id_encoder.to(device)
if args.multi_ip_adapter:
    pipe.load_ip_adapter("./pretrain_model/IP-Adapter/", subfolder="sdxl_models", weight_name="ip-adapter-plus-face_sdxl_vit-h.bin", image_encoder_folder=None)
pipe.load_ip_adapter("./pretrain_model/IP-Adapter-FaceID/", subfolder=None, weight_name="ip-adapter-faceid-portrait_sdxl.bin", image_encoder_folder=None)
# pipe.load_ip_adapter("./pretrain_model/IP-Adapter-FaceID/", subfolder=None, weight_name="ip-adapter-faceid_sdxl.bin", image_encoder_folder=None)
# pretrain_model/IP-Adapter-FaceID/ip-adapter-faceid-portrait_sdxl.bin
pipe.set_ip_adapter_scale(0.7)  
pipe.enable_model_cpu_offload()
print("over")
# define and show the input ID images
input_folder_name = args.image
image_basename_list =[base_name for base_name in os.listdir(args.image) if isimage(base_name)]
image_path_list = sorted([os.path.join(input_folder_name, basename) for basename in image_basename_list])

input_id_images = []
for image_path in image_path_list:
    input_id_images.append(load_image(image_path))

## Note that the trigger word `img` must follow the class word for personalization
prompts = load_prompts(args.prompt)
negative_prompt = "(asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch, multiple people, text on the screen, no people, unclear faces), open mouth"

face_id_embeds = extract_face_features(input_id_images)[0]
neg_face_id_embeds = torch.zeros_like(face_id_embeds)
id_embeds = torch.cat([neg_face_id_embeds, face_id_embeds], dim=0).to(dtype=torch.float16, device="cuda")
# id_embeds = face_id_embeds

if args.multi_ip_adapter:
    clip_embeds = pipe.prepare_ip_adapter_image_embeds([input_id_images], None, torch.device("cuda"), len(input_id_images), True)[0]
    pipe.unet.encoder_hid_proj.image_projection_layers[0].clip_embeds = clip_embeds.to(dtype=torch.float16)
    pipe.unet.encoder_hid_proj.image_projection_layers[0].shortcut = True # True if Plus v2

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

for prompt in prompts:
    for seed in seed_list:
        generator = torch.Generator(device=device).manual_seed(seed)
        frames = pipe(
            prompt=prompt,
            num_frames=16,
            guidance_scale=8,
            input_id_images=input_id_images,
            negative_prompt=negative_prompt,
            ip_adapter_image_embeds=[id_embeds],
            num_videos_per_prompt=1,
            num_inference_steps=num_steps,
            start_merge_step=start_merge_step,
            generator=generator,
        ).frames[0]
        os.makedirs("outputs/{}".format(args.name), exist_ok=True)
        export_to_gif(frames, "outputs/{}/{}_{}_seed_{}.gif".format(args.name, args.output, prompt.replace(' ','_'),seed))
