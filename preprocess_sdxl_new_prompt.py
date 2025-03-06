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
import argparse
from photomaker.model import PhotoMakerIDEncoder
from insightface.app import FaceAnaly
from insightface.utils import face_align
from photomaker.datasets.celebv_text import CelebVTextProcessDataset
import json

# import torch_npu
# from torch_npu.contrib import transfer_to_npu

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--root", type=str, default='/group/40033/public_datasets/CeleV-Text/', help="data path")
    parser.add_argument("--save_path", type=str, default='/group/40033/public_datasets/CeleV-Text', help="data path")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default='./pretrain_model/RealVisXL_V4.0', help="pretrained model path")
    parser.add_argument("--phase",type=int,default=0)
    parser.add_argument("--total",type=int,default=4)
    return parser



class Annotator(nn.Module):
    def __init__(self, pretrained_model_name_or_path="./pretrain_model/RealVisXL_V4.0"):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            use_safetensors=True, 
        )
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            use_safetensors=True, 
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_2")
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae",
            use_safetensors=True, 
        )
        self.feature_extractor = CLIPImageProcessor()
        self.vae.enable_tiling()
        self.vae.enable_slicing()
        self.device = "cuda"

    def encode_video_with_vae(self, video):
        video_length = video.shape[1]
        pixel_values = rearrange(video, "b f c h w -> (b f) c h w")
        latents = self.vae.encode(pixel_values).latent_dist
        latents = latents.sample()
        latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
        return latents

    def encode_prompt(self, prompt, device, prompt_2=None, clip_skip=None):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )
        prompt_2 = prompt_2 or prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

        # textual inversion: process multi-vector tokens if necessary
        prompt_embeds_list = []
        prompts = [prompt, prompt_2]
        for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            if clip_skip is None:
                prompt_embeds = prompt_embeds.hidden_states[-2]
            else:
                # "2" because SDXL always indexes from the penultimate layer.
                prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        return prompt_embeds, pooled_prompt_embeds


    def forward(self, prompt):
        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(prompt, self.device)
        return prompt_embeds.squeeze(dim=0), pooled_prompt_embeds.squeeze(dim=0)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    annotator = Annotator()
    annotator.eval()
    annotator.to("cuda")
    print(args.phase)
    with open("datasets/sh_CeleV-Text/qwen7b_caption.json", 'r') as file:
        all_data = json.load(file)
    # 计算每个阶段的处理数量
    per_device_num = len(all_data) / args.total
    start = int(args.phase * per_device_num)
    end = int((args.phase + 1) * per_device_num)

    # 如果 end 超出范围，则处理剩余的所有数据
    if end >= len(all_data):
        all_data = all_data[start:]
    else:
        all_data = all_data[start:end]
    
    for data in tqdm(all_data):
        path = data['path']
        prompt = data['prompt']
        base_name = os.path.basename(path).split(".")[0]
        process_data_path = os.path.join(
            "datasets/sh_CeleV-Text",
            "processed_sdxl_512",
            f"{base_name}.pt",
        )
        try:
            process_data = torch.load(process_data_path, map_location='cpu')
        except:
            print(process_data_path)
            continue
        if 'new_prompt_embeds' not in process_data.keys() or 'new_pooled_prompt_embeds' not in process_data.keys():
            with torch.no_grad():
                process_data['new_prompt_embeds'],process_data['new_pooled_prompt_embeds'] = annotator(prompt)
            torch.save(process_data, process_data_path)
        



