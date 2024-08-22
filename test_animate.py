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
# gloal variable and function
import argparse
def isimage(path):
    if 'png' in path.lower() or 'jpg' in path.lower() or 'jpeg' in path.lower():
        return True
    
def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-s", "--seed", type=int, nargs='+',default=[42,128], help="seed for seed_everything")
    parser.add_argument("-p", "--prompt", type=str, default='person.txt', help="prompt file path")
    parser.add_argument("-i", "--image", type=str, default='./datasets/lecun', help="image prompt dir path")
    parser.add_argument("-o", "--output", type=str, default='photomakerwithadapter_clipl_animate', help="output name")
    parser.add_argument("--name", type=str, default='lecun', help="name")
    parser.add_argument("-n", "--num_steps", type=int, default=50, help="number of steps")
    parser.add_argument("--ip_adapter", default=False, action='store_true')
    parser.add_argument("--use_clipl_embed", default=False, action='store_true')
    parser.add_argument("-r", "--ratio", type=int, default=20, help="style_strength_ratio")
    # style_strength_ratio
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
save_path = "./outputs"
adapter = MotionAdapter.from_pretrained("pretrain_model/animatediff-motion-adapter-sdxl-beta")
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
    variant="fp16",
).to("cuda")


pipe.load_photomaker_adapter(
    "./pretrain_model/PhotoMaker",
    subfolder="",
    weight_name="photomaker-v1.bin",
    trigger_word="img"
)
pipe.id_encoder.to(device)
# pipe.fuse_lora()
if args.ip_adapter:
    if args.use_clipl_embed:
        pipe.load_ip_adapter("./pretrain_model/IP-Adapter", subfolder="sdxl_models", weight_name='ip-adapter_sdxl.bin',image_encoder_folder=None)
    elif args.ip_adapter:
        pipe.load_ip_adapter("./pretrain_model/IP-Adapter", subfolder="sdxl_models", weight_name='ip-adapter_sdxl.bin')
    else:
        pipe.load_ip_adapter("./pretrain_model/IP-Adapter", subfolder="sdxl_models", weight_name='ip-adapter_sdxl.bin')
    pipe.set_ip_adapter_scale(0.7)  

# define and show the input ID images
if os.path.isdir(args.image):
    input_folder_name = args.image
    image_basename_list =[base_name for base_name in os.listdir(args.image) if isimage(base_name)]
    image_path_list = sorted([os.path.join(input_folder_name, basename) for basename in image_basename_list])
else:
    image_path_list = [args.image]
input_id_images = []
for image_path in image_path_list:
    input_id_images.append(load_image(image_path))

## Note that the trigger word `img` must follow the class word for personalization
prompts = load_prompts(args.prompt)
negative_prompt = "(asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth"


## Parameter setting
num_steps = args.num_steps
style_strength_ratio = args.ratio
start_merge_step = int(float(style_strength_ratio) / 100 * num_steps)
# if start_merge_step > 30:
#     start_merge_step = 30

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
            ip_adapter_image=input_id_images[0] if args.ip_adapter else None,
            num_videos_per_prompt=1,
            num_inference_steps=num_steps,
            start_merge_step=start_merge_step,
            use_clipl_embed=args.use_clipl_embed,
            generator=generator,
        ).frames[0]
        os.makedirs("outputs/{}".format(args.name), exist_ok=True)
        export_to_gif(frames, "outputs/{}/{}_ratio_{}_{}_seed_{}.gif".format(args.name, args.ratio, args.output, prompt.replace(' ','_'),seed))
