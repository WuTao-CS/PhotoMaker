import torch
import argparse
import os

parser = argparse.ArgumentParser(description='split the available weights from ckpts')

parser.add_argument("--ckpt-path",
    type=str,
    default="./projects/IDAdapter-diffusers/models/stable-diffusion-xl-base-1.0/ffhq/006-sdxl-lora-highres-loraproj1e-4-r64-finaldata-multiembed-append-1e-5-bs6-n8/checkpoint-150000/pytorch_model.bin"
)
parser.add_argument("--output-folder",
    type=str,
    default="./projects/IDAdapter-diffusers/release_models"
)
parser.add_argument("--output-type",
    type=str,
    # required=True, 
    choices=['safetensor', 'bin'],
    default="bin"
)
args = parser.parse_args()

ckpt_path = args.ckpt_path
output_folder = args.output_folder
output_type = args.output_type

if ckpt_path.endswith(".bin"):
    original_model_state_dict = torch.load(ckpt_path, map_location="cpu")
elif ckpt_path.endswith(".safetensors"):
    from safetensors import safe_open
    original_model_state_dict = {}
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            original_model_state_dict[k] = f.get_tensor(k)

    # from safetensors.torch import load_file
       

basename = os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))

image_encoder_state_dict = {}
lora_state_dict = {}

for k,v in original_model_state_dict.items():
    if 'image_encoder.' in k:
        new_k = k.replace('image_encoder.', '')
        image_encoder_state_dict[new_k] = v

    elif 'postfuse_module' in k:
        new_k = k.replace('postfuse_module.', 'fuse_module.')
        image_encoder_state_dict[new_k] = v
    
    elif 'lora.' in k:
        lora_state_dict[k] = v

save_state_dict = {
    "id_encoder": image_encoder_state_dict,
    "lora_weights": lora_state_dict,
}

if output_type == 'bin':
    save_tensor_name = torch.save(save_state_dict, os.path.join(output_folder, f'{basename}.bin'))
elif output_type == 'safetensor':
    pass