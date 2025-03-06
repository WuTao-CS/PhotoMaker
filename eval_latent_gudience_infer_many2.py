import torch
import numpy as np
import random
import os
from PIL import Image
from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler, DDIMScheduler
from huggingface_hub import hf_hub_download
import sys
import torch
import clip
from decord import VideoReader, cpu
import json
from eval.eval_clip import ClipEval
from eval.eval_dino import DINOEvaluator
import pandas as pd
from model.pipline import VAEProjectAnimateDiffPipeline
from diffusers import MotionAdapter, DDIMScheduler, DPMSolverMultistepScheduler
from diffusers.utils import export_to_gif, load_image
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
    parser.add_argument("-s", "--seed", type=int, nargs='+',default=[42], help="seed for seed_everything")
    parser.add_argument("-p", "--prompt", type=str, default='benchmark.txt', help="prompt file path")
    parser.add_argument("--image_dir", type=str, default='datasets/Famous_people',help="image")
    parser.add_argument("--unet_path", type=str, help="image", default=None)
    parser.add_argument("-o","--output", type=str, default='outputs', help="output dir")
    parser.add_argument("--name", type=str, default='photomaker_mix', help="output name")
    parser.add_argument("-n", "--num_steps", type=int, default=50, help="number of steps")
    parser.add_argument("--cfg", type=int, default=8, help="number of steps")
    parser.add_argument("--size",type=int, default=512, help="size of image")
    parser.add_argument(
        "--enable_update_motion", action="store_true", help="Whether or not to update motion layer."
    )
    parser.add_argument(
        "--enable_crop_face", action="store_true", help="Whether or not to only face."
    )
    parser.add_argument(
        "--enable_euler", action="store_true", help="Whether or not to only face."
    )
    parser.add_argument(
        "--enable_dpm", action="store_true", help="Whether or not to only face."
    )
    parser.add_argument(
        "--enable_origin_cross_attn", action="store_true", help="Whether or not to use origin cross-attention."
    )
    parser.add_argument(
        "--enable_reference_image_noisy", action="store_true", help="Whether or not to use origin cross-attention."
    )
    parser.add_argument("--phase",type=int,default=0)
    parser.add_argument("--total",type=int,default=4)
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
base_model_path = './pretrain_model/Realistic_Vision_V5.1_noVAE'
device = "cuda"
adapter = MotionAdapter.from_pretrained("./pretrain_model/animatediff-motion-adapter-v1-5-3")

if args.enable_euler:
    scheduler = EulerDiscreteScheduler.from_pretrained(
        base_model_path,
        subfolder="scheduler",
        timestep_spacing='leading', 
        steps_offset=1,	
        beta_schedule="scaled_linear",
        beta_start=0.00085,
        beta_end=0.020
    )
elif args.enable_dpm:
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        base_model_path,
        subfolder="scheduler",
        beta_schedule="linear",
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True,
    )
else:
    scheduler = DDIMScheduler.from_pretrained(
        base_model_path,
        subfolder="scheduler",
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1,
    )
pipe = VAEProjectAnimateDiffPipeline.from_pretrained(
    base_model_path,
    motion_adapter=adapter,
    scheduler=scheduler,
).to("cuda")
print("load unet from ",args.unet_path)
pipe.set_fusion_model(unet_path=args.unet_path, enable_update_motion=args.enable_update_motion, enable_origin_cross_attn=args.enable_origin_cross_attn)
print("over")
# define and show the input ID images
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()


image_basename_list =[base_name for base_name in os.listdir(args.image_dir) if isimage(base_name)]
image_path_list = sorted([os.path.join(args.image_dir, basename) for basename in image_basename_list])

per_devie_num = len(image_path_list)/args.total
start = int(args.phase*per_devie_num)
end = int((args.phase+1)*per_devie_num)
image_path_list=image_path_list[start:end]
input_id_images=[]
for image_path in image_path_list:
    print(image_path)
    input_id_images.append(load_image(image_path).resize((512,512)))

for image_path,input_id_image in zip(image_path_list, input_id_images):
    dir_name = os.path.basename(image_path).split('.')[0]
    ## Note that the trigger word `img` must follow the class word for personalization
    prompts = load_prompts(args.prompt)
    negative_prompt = "semi-realistic, cgi, 3d, render, sketch, multiple people, cartoon, drawing, anime, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
    # print(input_id_images[0] if args.ip_adapter else None)
    seed_list = args.seed

    input_id_image_emb = pipe.prepare_reference_image_embeds(input_id_image, None, torch.device("cuda"), 1)
    

    size = (args.size, args.size)
    cnt=-1
    all_clip_t=[]
    all_frame_c=[]
    all_clip_i=[]
    all_dino_i=[]
    all_face_similarity=[]

    for prompt in prompts:
        cnt+=1
        for seed in seed_list:
            generator = torch.Generator(device=device).manual_seed(seed)
            frames = pipe(
                prompt=prompt,
                num_frames=16,
                guidance_scale=args.cfg,
                reference_image_embeds = input_id_image_emb,
                negative_prompt=negative_prompt,
                num_videos_per_prompt=1,
                generator=generator,
                num_inference_steps=args.num_steps,
                reference_image_noisy=args.enable_reference_image_noisy,
            ).frames[0]
            os.makedirs("{}/{}".format(args.output,dir_name), exist_ok=True)
            export_to_gif(frames, "{}/{}/{}_{}_seed_{}.gif".format(args.output, dir_name, cnt, prompt.replace(' ','_'),seed))