import os, sys
import numpy as np
import torch
from torchvision.transforms import Compose
import cv2
from PIL import Image
from glob import glob
from einops import rearrange
from tqdm import tqdm
from torch import nn
from typing import Optional
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection,CLIPImageProcessor
from model.datasets.ffhq import FFHQProcessDataset
import argparse

import json
# import torch_npu
# from torch_npu.contrib import transfer_to_npu

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--root", type=str, default='./datasets/ffhq/', help="data path")
    parser.add_argument("--save_path", type=str, default='/group/40007/public_datasets/ffhq', help="data path")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default='./pretrain_model/stable-diffusion-v1-5', help="pretrained model path")
    parser.add_argument("--phase",type=int,default=0)
    parser.add_argument("--total",type=int,default=2)
    return parser

def rectangle_to_square_and_resize(arr, target_size=512):
    # 获取数组的形状
    height, width = arr.shape[:2]
    
    # 确定长边和短边的长度
    max_side = max(height, width)
    
    # 创建一个新的正方形数组，边长为长边的长度
    square_arr = np.full((max_side, max_side) + arr.shape[2:], 255, dtype=arr.dtype)
    
    # 计算原数组在新数组中的起始位置
    start_y = (max_side - height) // 2
    start_x = (max_side - width) // 2
    
    # 将原数组的内容复制到新数组的中心位置
    square_arr[start_y:start_y+height, start_x:start_x+width] = arr
    
    # 将正方形数组 resize 到目标大小
    try:
        resized_arr = cv2.resize(square_arr, (target_size, target_size), interpolation=cv2.INTER_AREA)
    except:
        print("Error in cv2.resize")
        resized_arr = None

    return resized_arr

class Annotator(nn.Module):
    def __init__(self, pretrained_model_name_or_path="./pretrain_model/stable-diffusion-v1-5"):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae",
            use_safetensors=True, 
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            use_safetensors=True, 
        )
        self.vae.enable_tiling()
        self.vae.enable_slicing()
        self.device = "cuda"


    def encode_video_with_vae(self, pixel_values):
        latents = self.vae.encode(pixel_values).latent_dist
        latents = latents.sample()
        return latents

    def encode_prompt(
        self,
        prompt,
        clip_skip: Optional[int] = None,
    ):
        device = self.device
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            )
            print(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        if clip_skip is None:
            prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
            prompt_embeds = prompt_embeds[0]
        else:
            prompt_embeds = self.text_encoder(
                text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
            )
            # Access the `hidden_states` first, that contains a tuple of
            # all the hidden states from the encoder layers. Then index into
            # the tuple to access the hidden states from the desired layer.
            prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
            # We also need to apply the final LayerNorm here to not mess with the
            # representations. The `last_hidden_states` that we typically use for
            # obtaining the final prompt representations passes through the LayerNorm
            # layer.
            prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)


        prompt_embeds_dtype = self.text_encoder.dtype
        
        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        return prompt_embeds

    def forward(self, batch):
        image = batch["pixel_values"].to(self.device)
        ref_image = batch["ref_pixel_values"].to(self.device)

        prompt = batch["prompt"][0]

        prompt_embeds = self.encode_prompt(prompt)
        latent = self.encode_video_with_vae(image)
        ref_images_latent = self.encode_video_with_vae(ref_image)
        return {"latent": latent, "prompt_embeds": prompt_embeds.squeeze(dim=0), "ref_images_latent": ref_images_latent}

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    annotator = Annotator()
    annotator.eval()
    annotator.to("cuda")
    print(args.phase)
    tokenizer = CLIPTokenizer.from_pretrained("./pretrain_model/stable-diffusion-v1-5", subfolder="tokenizer")
    data = FFHQProcessDataset(root=args.root,tokenizer=tokenizer,phase=args.phase,total=args.total)
    print(len(data))
    data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    all_data = []
    for batch in tqdm(data_loader):
        with torch.no_grad():
            output = annotator(batch)
        output['prompt'] = batch['prompt'][0]
        save_path = f"{args.root}/processed_sd15_blank/{batch['name'][0]}.pt"
        all_data.append({"path":save_path, "prompt":batch['prompt'][0]})
        torch.save(output, f"{args.save_path}/processed_sd15_blank/{batch['name'][0]}.pt")

    json_file_name = os.path.join(args.save_path, "processed_sd15_blank_image_{}.json".format(args.phase))
    with open(json_file_name, 'w') as  f:
        json.dump(all_data, f, indent=4)
    print("ok")

