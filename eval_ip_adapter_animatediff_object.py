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
import argparse
import json
from diffusers import DiffusionPipeline, AnimateDiffPipeline, MotionAdapter, DDIMScheduler,AnimateDiffSDXLPipeline
from diffusers.utils import export_to_gif, load_image
import argparse
from transformers import CLIPVisionModelWithProjection
import cv2
from insightface.app import FaceAnalysis
from insightface.utils import face_align
# gloal variable and function
def isimage(path):
    if 'png' in path.lower() or 'jpg' in path.lower() or 'jpeg' in path.lower():
        return True
    

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-s", "--seed", type=int, nargs='+',default=[42], help="seed for seed_everything")
    parser.add_argument("-i", "--image", type=str, default='dog.png', help="prompt file path")
    parser.add_argument("-p", "--prompt", type=str, default='benchmark.txt', help="prompt file path")
    parser.add_argument("-o","--output", type=str, default='outputs', help="output dir")
    parser.add_argument("-n", "--num_steps", type=int, default=50, help="number of steps")
    parser.add_argument("--size",type=int, default=512, help="size of image")
    parser.add_argument("--version", type=str, default="ip-adapter_sd15.bin")
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


pipe.load_ip_adapter("./pretrain_model/IP-Adapter", subfolder="models", weight_name=args.version)

pipe.enable_vae_slicing()
pipe.enable_vae_tiling()
clipeval = ClipEval()
Dinoeval = DINOEvaluator()

image_path_list = [args.image]


input_id_images=[]
for image_path in image_path_list:
    print(image_path)
    input_id_images.append(load_image(image_path).resize((640,640)))

score_vta_total, score_fc_total, count = 0, 0, 0
score_id_total, count_id = 0, 0
dino_id_total = 0

total_clip_t=[]
total_frame_c=[]
total_clip_i=[]
total_dino_i=[]

for image_path,input_id_image in zip(image_path_list, input_id_images):
    dir_name = os.path.basename(image_path).split('.')[0]
    ## Note that the trigger word `img` must follow the class word for personalization
    prompts = load_prompts(args.prompt)
    negative_prompt = "asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch"
    # print(input_id_images[0] if args.ip_adapter else None)
    seed_list = args.seed
    cnt=-1
    all_clip_t=[]
    all_frame_c=[]
    all_clip_i=[]
    all_dino_i=[]
    for prompt in prompts:
        cnt+=1
        for seed in seed_list:
            generator = torch.Generator(device=device).manual_seed(seed)
            frames = pipe(
                prompt=prompt,
                num_frames=16,
                guidance_scale=8,
                negative_prompt=negative_prompt,
                ip_adapter_image=input_id_image,
                num_videos_per_prompt=1,
                num_inference_steps=args.num_steps,
                generator=generator,
            ).frames[0]
            os.makedirs("{}/{}".format(args.output,dir_name), exist_ok=True)
            export_to_gif(frames, "{}/{}/{}_{}_seed_{}.gif".format(args.output, dir_name, cnt, prompt.replace(' ','_'),seed))

            clip_t = clipeval.cal_video_text_alignment(frames, prompt)
            frame_c = clipeval.cal_video_frame_consistency(frames)
            clip_i = clipeval.cal_video_id_consistency(frames, image_path)
            dino_i = Dinoeval.cal_video_id_consistency(frames, image_path).item()
            all_clip_t.append(clip_t)
            all_frame_c.append(frame_c)
            all_clip_i.append(clip_i)
            all_dino_i.append(dino_i)
    res = {'clip_t': np.mean(all_clip_t),'frame_c': np.mean(all_frame_c), 'clip_i': np.mean(all_clip_i), 'dino_i': np.mean(all_dino_i)}
    # write res to json
    with open("{}/{}/result.json".format(args.output,dir_name), "w") as f:
        json.dump(res, f, indent=4)
    table = {'clip_t': all_clip_t, 'frame_c': all_frame_c, 'clip_i': all_clip_i, 'dino_i': all_dino_i}
    # write table to csv
    df = pd.DataFrame(table)
    df.to_csv("{}/{}/table.csv".format(args.output,dir_name))
    total_clip_t.append(np.mean(all_clip_t))
    total_frame_c.append(np.mean(all_frame_c))
    total_clip_i.append(np.mean(all_clip_i))
    total_dino_i.append(np.mean(all_dino_i))

