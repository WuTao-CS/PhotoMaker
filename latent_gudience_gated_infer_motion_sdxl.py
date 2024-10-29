import torch
import numpy as np
import random
import os
from PIL import Image


from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler, DDIMScheduler
from huggingface_hub import hf_hub_download

from model.pipline import VAEGatedAnimateDiffPipeline, VAEGatedAnimateDiffSDXLPipeline
from diffusers import MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_gif, load_image
from insightface.app import FaceAnalysis
from transformers import CLIPVisionModelWithProjection
import cv2
# gloal variable and function
import argparse
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu

    cann_available = True
except Exception:
    cann_available = False

def isimage(path):
    if 'png' in path.lower() or 'jpg' in path.lower() or 'jpeg' in path.lower():
        return True
    


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-s", "--seed", type=int, nargs='+',default=[42,128], help="seed for seed_everything")
    parser.add_argument("-p", "--prompt", type=str, default='emjoy2.txt', help="prompt file path")
    parser.add_argument("-i", "--image", type=str, help="image")
    parser.add_argument("--unet_path", type=str, help="model path", default=None)
    parser.add_argument("--inject_block_txt", type=str, help="inject set", default="/group/40034/jackeywu/code/PhotoMaker/block.txt")
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
adapter = MotionAdapter.from_pretrained("./pretrain_model/animatediff-motion-adapter-sdxl-beta")


scheduler = DDIMScheduler.from_pretrained(
    base_model_path,
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)

pipe = VAEGatedAnimateDiffSDXLPipeline.from_pretrained(
    base_model_path,
    motion_adapter=adapter,
    scheduler=scheduler,
)
print("load unet from ",args.unet_path)
pipe.set_fusion_model(unet_path=args.unet_path, inject_block_txt=args.inject_block_txt, with_motion=True)
pipe.to("cuda")
print("over")
# define and show the input ID images
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()
image_path_list=[args.image]

input_id_images=[]
for image_path in image_path_list:
    input_id_images.append(load_image(image_path).resize((args.size, args.size)))

for image_path,input_id_image in zip(image_path_list, input_id_images):
    dir_name = os.path.basename(image_path).split('.')[0]
    ## Note that the trigger word `img` must follow the class word for personalization
    prompts = load_prompts(args.prompt)
    negative_prompt = "(asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch, multiple people, text on the screen, no people, unclear faces)"
    # print(input_id_images[0] if args.ip_adapter else None)
    seed_list = args.seed
    input_id_image_emb = pipe.prepare_reference_image_embeds(input_id_image, None, torch.device("cuda"), 1)
    print(input_id_image_emb.shape)

    size = (args.size, args.size)
    for prompt in prompts:
        for seed in seed_list:
            generator = torch.Generator(device=device).manual_seed(seed)
            frames = pipe(
                prompt=prompt,
                num_frames=16,
                height=size[0],
                width=size[1],
                guidance_scale=7,
                reference_image_embeds = input_id_image_emb,
                negative_prompt=negative_prompt,
                num_videos_per_prompt=1,
                generator=generator,
            ).frames[0]
            os.makedirs("{}/{}".format(args.output,dir_name), exist_ok=True)
            export_to_gif(frames, "{}/{}/{}_{}_seed_{}.gif".format(args.output, dir_name, args.name, prompt.replace(' ','_'),seed))