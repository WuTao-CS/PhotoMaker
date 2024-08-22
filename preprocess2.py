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
from photomaker.datasets.celebv_text import CelebVTextProcessDataset
import json

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--root", type=str, default='./datasets/CeleV-Text', help="data path")
    parser.add_argument("--save_path", type=str, default='/data1/wutao', help="data path")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default='./pretrain_model/RealVisXL_V4.0', help="pretrained model path")
    parser.add_argument("--phase",type=int,default=0)
    parser.add_argument("--total",type=int,default=8)
    return parser

class PhotoMakerIDEncoderForlabel(PhotoMakerIDEncoder):
    def __init__(self):
        super().__init__()
    def forward(self, id_pixel_values, prompt_embeds, class_tokens_mask):
        b, num_inputs, c, h, w = id_pixel_values.shape
        id_pixel_values = id_pixel_values.view(b * num_inputs, c, h, w)

        vision_outputs= self.vision_model(id_pixel_values)
        shared_id_embeds = vision_outputs[1] # 768
        id_embeds = self.visual_projection(shared_id_embeds)
        id_embeds_2 = self.visual_projection_2(shared_id_embeds)

        id_embeds = id_embeds.view(b, num_inputs, 1, -1)
        id_embeds_2 = id_embeds_2.view(b, num_inputs, 1, -1)    

        id_embeds = torch.cat((id_embeds, id_embeds_2), dim=-1) #[b, num_inputs, 1, 2048]
        updated_prompt_embeds=[]
        for i in num_inputs:
            updated_prompt_embed = self.fuse_module(prompt_embeds, id_embeds[:,i,:,:], class_tokens_mask)
            updated_prompt_embeds.append(updated_prompt_embed)
        updated_prompt_embeds = torch.stack(updated_prompt_embeds, dim=0)
        return updated_prompt_embeds, id_embeds_2

class Annotator(nn.Module):
    def __init__(self, pretrained_model_name_or_path="./pretrain_model/RealVisXL_V4.0"):
        super().__init__()
        self.id_encoder = PhotoMakerIDEncoderForlabel()
        self.id_encoder.load_from_pretrained('./pretrain_model/PhotoMaker/photomaker-v1.bin')
        self.trigger_word = "img"
        self.feature_extractor = CLIPImageProcessor()
        self.device = "cuda"
    def encode_photomaker_prompt(self, video, prompt_embeds, class_tokens_mask, device):
        # 5. Prepare the input ID images
        dtype = next(self.id_encoder.parameters()).dtype
        input_id_images = video
        id_pixel_values = self.id_image_processor(input_id_images, return_tensors="pt").pixel_values

        id_pixel_values = id_pixel_values.unsqueeze(0).to(device=device, dtype=dtype) # TODO: multiple prompts

        # 6. Get the update text embedding with the stacked ID embedding
        prompt_embeds, _ip_adapter_image_embeds = self.id_encoder(id_pixel_values, prompt_embeds, class_tokens_mask)

        return prompt_embeds, _ip_adapter_image_embeds

    def forward(self, batch, flag=True):
        video = batch["video"].to(self.device)
        ref_frames = batch["ref_frames"][0]
        prompt = batch["prompt"][0]
        prompt_trigger = batch["prompt_trigger"][0]
        prompt_embeds_trigger, _ip_adapter_image_embeds = self.encode_photomaker_prompt(image_embeds, prompt_embeds_trigger, class_tokens_mask, self.device)
        return {"prompt_embeds_trigger":prompt_embeds_trigger.squeeze(dim=0)}


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    annotator = Annotator()
    annotator.eval()
    annotator.to("cuda")
    print(args.phase)
    data = CelebVTextProcessDataset(root=args.root,load_all_frames=False,phase=args.phase,total=args.total)
    data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False,num_workers=4)
    all_data = []
    for batch in tqdm(data_loader):
        if batch == {}:
            continue
        
        if os.path.exists(f"{args.save_path}/processed/{batch['name'][0]}.pt"):
            output = torch.load(f"{args.save_path}/processed/{batch['name'][0]}.pt")
            with torch.no_grad():
                out = annotator(batch,flag=False)
                if out is None:
                    continue
                output["prompt_embeds_trigger"] = out(batch,flag=False)["prompt_embeds_trigger"]
        else:
            with torch.no_grad():
                output = annotator(batch)
        if output is None:
            continue
        output['prompt'] = batch['prompt'][0]
        output['prompt_trigger'] = batch['prompt_trigger'][0]
        save_path = f"{args.root}/processed/{batch['name'][0]}.pt"
        all_data.append({"path":save_path, "prompt":batch['prompt'][0], "prompt_trigger":batch['prompt_trigger'][0]})
        torch.save(output, f"{args.save_path}/processed/{batch['name'][0]}.pt")

    json_file_name = os.path.join(args.save_path, "processed_{}.json".format(args.phase))
    with open(json_file_name, 'w') as  f:
        json.dump(all_data, f, indent=4)
    print("ok")

    