import torch
import numpy as np
import random
import os
from PIL import Image

import cv2
from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler, DDIMScheduler
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, AnimateDiffPipeline, MotionAdapter, DDIMScheduler,AnimateDiffSDXLPipeline
from diffusers.utils import export_to_gif, load_image
import argparse
from insightface.app import FaceAnalysis
from transformers import CLIPVisionModelWithProjection

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
# gloal variable and function
def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-s", "--seed", type=int, nargs='+',default=[42,128], help="seed for seed_everything")
    parser.add_argument("-p", "--prompt", type=str, default="./person.txt", help="prompt")
    parser.add_argument("-i", "--image", type=str, default='./datasets/lecun', help="image prompt dir path")
    parser.add_argument("--name", type=str, default='lecun', help="name")
    parser.add_argument("-w", "--wight_adapter", type=str, default="ip-adapter-plus_sdxl_vit-h.bin", help="ip adapter weight name")
    parser.add_argument("--scale", type=float, default=0.6, help="scale")
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

def isimage(path):
    if 'png' in path.lower() or 'jpg' in path.lower() or 'jpeg' in path.lower():
        return True
    else:
        return False
    
parser = get_parser()
args = parser.parse_args()
base_model_path = './pretrain_model/RealVisXL_V3.0'
device = "cuda"
save_path = "./outputs"
image_encoder_path = "./pretrain_model/CLIP-ViT-H-14-laion2B-s32B-b79K"
image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path, torch_dtype=torch.float16)

adapter = MotionAdapter.from_pretrained("pretrain_model/animatediff-motion-adapter-sdxl-beta")
scheduler = DDIMScheduler.from_pretrained(
    base_model_path,
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)
pipe = AnimateDiffSDXLPipeline.from_pretrained(
    base_model_path,
    motion_adapter=adapter,
    scheduler=scheduler,
    image_encoder=image_encoder,
    torch_dtype=torch.float16,
).to("cuda")


# pipe.fuse_lora()
pipe.load_ip_adapter("pretrain_model/IP-Adapter", subfolder="sdxl_models", weight_name=args.wight_adapter,image_encoder_folder=None)
pipe.set_ip_adapter_scale(0.6)  
#pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
#pipe.fuse_lora()


# pipe.set_adapters(["photomaker"], adapter_weights=[1.0])
# define and show the input ID images
input_folder_name = args.image
image_basename_list =[base_name for base_name in os.listdir(args.image) if isimage(base_name)]
image_path_list = sorted([os.path.join(input_folder_name, basename) for basename in image_basename_list])
input_id_images = []
for image_path in image_path_list:
    input_id_images.append(load_image(image_path))

# face_id_embeds = extract_face_features(input_id_images)[0]
# neg_face_id_embeds = torch.zeros_like(face_id_embeds)
# id_embeds = torch.cat([neg_face_id_embeds, face_id_embeds], dim=0).to(dtype=torch.float16, device="cuda")
## Note that the trigger word `img` must follow the class word for personalization
prompts = load_prompts(args.prompt)
negative_prompt = "(asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth"

## Parameter setting
num_steps = 20
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()
seed_list = args.seed
for prompt in prompts:
    for seed in seed_list:
        generator = torch.Generator(device=device).manual_seed(seed)
        frames = pipe(
            prompt=prompt,
            num_frames=16,
            guidance_scale=8,
            ip_adapter_image=input_id_images[0],
            negative_prompt=negative_prompt,
            num_videos_per_prompt=1,
            num_inference_steps=num_steps,
            generator=generator,
        ).frames[0]
        os.makedirs("outputs/{}".format(args.name), exist_ok=True)
        export_to_gif(frames, "outputs/{}/{}_{}_seed_{}.gif".format(args.name, os.path.basename(args.wight_adapter) , prompt.replace(' ','_'), seed))
# export_to_gif(frames, "visual_animate_{}_{}.gif".format(text.replace(' ','_'), int(args.scale*10)))
