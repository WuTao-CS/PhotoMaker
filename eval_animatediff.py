import torch
import os


from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler, DDIMScheduler
import torch

import pandas as pd
import argparse
import json
from diffusers import DiffusionPipeline, AnimateDiffPipeline, MotionAdapter, DDIMScheduler,AnimateDiffSDXLPipeline
from diffusers.utils import export_to_gif, load_image
import argparse
# gloal variable and function

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-s", "--seed", type=int, nargs='+',default=[42], help="seed for seed_everything")
    parser.add_argument("-p", "--prompt", type=str, default='benchmark.txt', help="prompt file path")
    parser.add_argument("-o","--output", type=str, default='outputs', help="output dir")
    parser.add_argument("-n", "--num_steps", type=int, default=30, help="number of steps")
    parser.add_argument("--size",type=int, default=512, help="size of image")
    return parser


    
parser = get_parser()
args = parser.parse_args()
base_model_path = './pretrain_model/Realistic_Vision_V5.1_noVAE'
device = "cuda"
adapter = MotionAdapter.from_pretrained("./pretrain_model/animatediff-motion-adapter-v1-5-3/")
scheduler = DDIMScheduler.from_pretrained(
    base_model_path,
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)

pipe = AnimateDiffPipeline.from_pretrained(
    base_model_path,
    motion_adapter=adapter,
    scheduler=scheduler,
    torch_dtype=torch.float16,
).to("cuda")



pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

def load_prompts(prompt_file):
    f = open(prompt_file, 'r')
    prompt_list = []
    for idx, line in enumerate(f.readlines()):
        l = line.strip()
        if len(l) != 0:
            prompt_list.append(l)
        f.close()
    return prompt_list

dir_name = 'animatediff_V3'
## Note that the trigger word `img` must follow the class word for personalization
prompts = load_prompts(args.prompt)
negative_prompt = "asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch, multiple people, text on the screen, no people, unclear faces, awkward angles"
# print(input_id_images[0] if args.ip_adapter else None)
seed_list = args.seed

cnt = -1
for prompt in prompts:
    cnt+=1
    for seed in seed_list:
        generator = torch.Generator(device=device).manual_seed(seed)
        frames = pipe(
            prompt=prompt,
            num_frames=16,
            guidance_scale=8,
            negative_prompt=negative_prompt,
            num_videos_per_prompt=1,
            num_inference_steps=args.num_steps,
            generator=generator,
        ).frames[0]
        os.makedirs("{}/{}".format(args.output,dir_name), exist_ok=True)
        export_to_gif(frames, "{}/{}/{}_{}_seed_{}.gif".format(args.output, dir_name, cnt, prompt.replace(' ','_'),seed))

