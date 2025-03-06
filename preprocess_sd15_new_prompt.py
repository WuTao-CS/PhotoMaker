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
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from photomaker.datasets.celebv_text import CelebVTextProcessDataset
import json
# import torch_npu
# from torch_npu.contrib import transfer_to_npu

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--root", type=str, default='/group/40033/public_datasets/CeleV-Text/', help="data path")
    parser.add_argument("--save_path", type=str, default='/group/40033/public_datasets/CeleV-Text', help="data path")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default='./pretrain_model/stable-diffusion-v1-5', help="pretrained model path")
    parser.add_argument("--phase",type=int,default=0)
    parser.add_argument("--total",type=int,default=4)
    return parser



class Annotator(nn.Module):
    def __init__(self, pretrained_model_name_or_path="./pretrain_model/stable-diffusion-v1-5"):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            use_safetensors=True, 
        )
        self.device = "cuda"

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


    def forward(self, prompt):
        prompt_embeds = self.encode_prompt(prompt)
        return prompt_embeds.squeeze(dim=0)


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
            "processed_sd15",
            f"{base_name}.pt",
        )
        try:
            process_data = torch.load(process_data_path, map_location='cpu')
        except:
            print(process_data_path)
            continue
        with torch.no_grad():
            new_prompt_embeds=annotator(prompt)
        process_data['new_prompt_embeds'] = annotator(prompt)
        torch.save(process_data, process_data_path)
        



