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
from model.pipline import VAEProjectAnimateDiffPipeline, VAECannyProjectAnimateDiffPipeline
from diffusers import MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_gif, load_image
from insightface.app import FaceAnalysis
from transformers import CLIPVisionModelWithProjection
import cv2
from insightface.utils import face_align
# gloal variable and function
import argparse

def read_video_frames(video_path, stride=4, num_reference_frame=4, canny_threshold1=100, canny_threshold2=200):
    # 使用decord的VideoReader读取视频
    vr = VideoReader(video_path, ctx=cpu(0), width=512, height=512)  # ctx=cpu(0) 表示使用CPU

    original_frames = []
    canny_frames = []

    # 遍历视频帧
    for i in range(0, len(vr), stride):
        if len(original_frames) >= num_reference_frame:
            break  # 如果已经读取了足够的帧，则停止

        # 读取帧
        frame = vr[i].asnumpy()  # 将帧转换为numpy数组

        # 将帧从RGB转换为BGR（OpenCV默认使用BGR格式）
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 使用Canny算子进行边缘检测
        edges = cv2.Canny(frame_bgr, canny_threshold1, canny_threshold2)
        edges = cv2.merge([edges, edges, edges])

        # 将原始帧和Canny结果转换为PIL.Image
        original_pil = Image.fromarray(frame)  # 原始帧
        canny_pil = Image.fromarray(edges)     # Canny结果

        # 保存到列表
        original_frames.append(original_pil)
        canny_frames.append(canny_pil)

    return original_frames, canny_frames

def isimage(path):
    if 'png' in path.lower() or 'jpg' in path.lower() or 'jpeg' in path.lower():
        return True

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-s", "--seed", type=int, nargs='+',default=[42,1234], help="seed for seed_everything")
    parser.add_argument("-p", "--prompt", type=str, default='canny_prompt.txt', help="prompt file path")
    parser.add_argument("--video", type=str, default="datasets/test_video",help="image")
    parser.add_argument("--unet_path", type=str, help="image", default=None)
    parser.add_argument("-o","--output", type=str, default='outputs', help="output dir")
    parser.add_argument("--name", type=str, default='VideoMaker_mix', help="output name")
    parser.add_argument("-n", "--num_steps", type=int, default=30, help="number of steps")
    parser.add_argument("--cfg", type=int, default=8, help="number of steps")
    parser.add_argument("--size",type=int, default=512, help="size of image")
    parser.add_argument("--num_reference_frame",type=int, default=4, help="num_reference_frame")
    parser.add_argument(
        "--enable_euler", action="store_true", help="Whether or not to only face."
    )
    parser.add_argument(
        "--enable_reference_image_noisy", action="store_true", help="Whether or not to use origin cross-attention."
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
else:
    scheduler = DDIMScheduler.from_pretrained(
        base_model_path,
        subfolder="scheduler",
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1,
    )
pipe = VAECannyProjectAnimateDiffPipeline.from_pretrained(
    base_model_path,
    motion_adapter=adapter,
    scheduler=scheduler,
).to("cuda")
print("load unet from ",args.unet_path)
pipe.set_fusion_model(unet_path=args.unet_path, num_reference_frame=args.num_reference_frame)
print("over")
# define and show the input ID images
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

video_names = os.listdir(args.video)

for name in video_names:
    video_path = os.path.join(args.video,name)
    origin_frames, canny_frames = read_video_frames(video_path, stride=4, num_reference_frame=args.num_reference_frame)

    dir_name = os.path.basename(video_path).split('.')[0]
    ## Note that the trigger word `img` must follow the class word for personalization
    prompts = load_prompts(args.prompt)
    negative_prompt = "asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch"
    seed_list = args.seed

    input_id_image_emb = pipe.prepare_reference_image_embeds(canny_frames, None, torch.device("cuda"), 1)

    size = (args.size, args.size)
    cnt=-1
    os.makedirs("{}/{}".format(args.output,dir_name), exist_ok=True)
    export_to_gif(origin_frames,"{}/{}/origin.gif".format(args.output, dir_name))
    export_to_gif(canny_frames,"{}/{}/canny.gif".format(args.output, dir_name))

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
            export_to_gif(frames, "{}/{}/{}_{}_seed_{}.gif".format(args.output, dir_name, cnt, prompt.replace(' ','_'),seed))
            