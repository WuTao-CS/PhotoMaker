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

from model.pipline import VAEProjectAnimateDiffPipeline
from diffusers import MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_gif, load_image
from insightface.app import FaceAnalysis
from transformers import CLIPVisionModelWithProjection
import cv2
from insightface.utils import face_align
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
        norm_face = face_align.norm_crop(img, landmark=face_info.kps, image_size=512)
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
    parser.add_argument("-s", "--seed", type=int, nargs='+',default=[42,128], help="seed for seed_everything")
    parser.add_argument("-p", "--prompt", type=str, default='benchmark.txt', help="prompt file path")
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
        "--enable_origin_cross_attn", action="store_true", help="Whether or not to use origin cross-attention."
    )
    parser.add_argument(
        "--enable_reference_image_noisy", action="store_true", help="Whether or not to use origin cross-attention."
    )
    parser.add_argument("--phase",type=int,default=0)
    parser.add_argument("--total",type=int,default=8)
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
image_basename_list =[base_name for base_name in os.listdir("datasets/Famous_people/") if isimage(base_name)]
image_path_list = sorted([os.path.join("datasets/Famous_people/", basename) for basename in image_basename_list])

per_devie_num = len(image_path_list)/args.total
start = int(args.phase*per_devie_num)
end = int((args.phase+1)*per_devie_num)
image_path_list=image_path_list[start:end]
input_id_images=[]
for image_path in image_path_list:
    print(image_path)
    input_id_images.append(load_image(image_path).resize((640,640)))
    
for image_path,input_id_image in zip(image_path_list, input_id_images):
    
    dir_name = os.path.basename(image_path).split('.')[0]
    ## Note that the trigger word `img` must follow the class word for personalization
    prompts = load_prompts(args.prompt)
    negative_prompt = "asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch, multiple people, text on the screen, no people, unclear faces, awkward angles"
    # print(input_id_images[0] if args.ip_adapter else None)
    seed_list = args.seed
    if args.enable_crop_face:
        ref_image = extract_face_features([input_id_image])[0]
        if ref_image is not None:
            input_id_image_emb = pipe.prepare_reference_image_embeds(ref_image, None, torch.device("cuda"), 1)
    else:
        input_id_image_emb = pipe.prepare_reference_image_embeds(input_id_image, None, torch.device("cuda"), 1)
    

    size = (args.size, args.size)
    cnt=-1
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
                reference_image_noisy=args.enable_reference_image_noisy,
            ).frames[0]
            os.makedirs("{}/{}".format(args.output,dir_name), exist_ok=True)
            export_to_gif(frames, "{}/{}/{}_{}_seed_{}.gif".format(args.output, dir_name, cnt, prompt.replace(' ','_'),seed))
