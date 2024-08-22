import torch
import numpy as np
import random
import os
from PIL import Image


from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler, DDIMScheduler
from huggingface_hub import hf_hub_download
import argparse
from photomaker import PhotoMakerStableDiffusionXLPipeline
# gloal variable and function
def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-s", "--seed", type=int, default=42, help="seed for seed_everything")
    parser.add_argument("-p", "--prompt", type=str, default="A man img running on the ground", help="prompt")
    parser.add_argument("--save", type=str, default="A teddybear running on the ground", help="prompt")
    parser.add_argument("-i", "--image", type=str, default="/group/40034/jerryxwli/code/VideoCrafter_Share/datasets/benchmark_dataset/plushie_teddybear/marina-shatskih-kBo2MFJz2QU-unsplash.jpg", help="prompt")
    return parser

def image_grid(imgs, rows, cols, size_after_resize):
    assert len(imgs) == rows*cols

    w, h = size_after_resize, size_after_resize

    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        img = img.resize((w,h))
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid
parser = get_parser()
args = parser.parse_args()

base_model_path = 'pretrain_model/RealVisXL_V3.0'
device = "cuda"
save_path = args.save

from huggingface_hub import hf_hub_download

# photomaker_ckpt = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")

pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
    variant="fp16",
).to(device)

pipe.load_photomaker_adapter(
    "./pretrain_model/PhotoMaker",
    subfolder="",
    weight_name="photomaker-v1.bin",
    trigger_word="img"
)
pipe.id_encoder.to(device)


#pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
#pipe.fuse_lora()
# pipe.set_adapters(["photomaker"], adapter_weights=[1.0])
# pipe.fuse_lora()

# define and show the input ID images
input_folder_name = args.image
image_basename_list = os.listdir(input_folder_name)
image_path_list = sorted([os.path.join(input_folder_name, basename) for basename in image_basename_list])

input_id_images = []
for image_path in image_path_list:
    input_id_images.append(load_image(image_path))

## Note that the trigger word `img` must follow the class word for personalization
prompt = args.prompt
negative_prompt = "(asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth"
generator = torch.Generator(device=device).manual_seed(args.seed)

## Parameter setting
num_steps = 50
style_strength_ratio = 20
start_merge_step = int(float(style_strength_ratio) / 100 * num_steps)
if start_merge_step > 30:
    start_merge_step = 30

images = pipe(
    prompt=prompt,
    input_id_images=input_id_images,
    negative_prompt=negative_prompt,
    num_images_per_prompt=4,
    num_inference_steps=num_steps,
    start_merge_step=start_merge_step,
    generator=generator,
).images

# Show and save the results
## Downsample for visualization
grid = image_grid(images, 1, 4, size_after_resize=512)

os.makedirs(save_path, exist_ok=True)
for idx, image in enumerate(images):
    image.save(os.path.join(save_path, "photomaker_{}.png".format(idx)))
