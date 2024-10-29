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
    parser.add_argument("--save_path", type=str, default='/group/40007/public_datasets/CeleV-Text', help="data path")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default='./pretrain_model/stable-diffusion-v1-5', help="pretrained model path")
    parser.add_argument("--phase",type=int,default=0)
    parser.add_argument("--total",type=int,default=8)
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
        self.app = FaceAnalysis(name="buffalo_l",
                        root="./pretrain_model",
                        providers=['CUDAExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.vae.enable_tiling()
        self.vae.enable_slicing()
        self.device = "cuda"

    def get_face_image(self, video):
        # Extract Face features using insightface
        ref_images = []
        for i in range(video.shape[0]):
            img = video[i]
            img = np.array(img)
            face_info = self.app.get(img)
            
            if len(face_info)==0:
                continue
            face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
            bbox = face_info.bbox.astype(int)
            x1, y1, x2, y2 = bbox[:4]
            face_image = rectangle_to_square_and_resize(img[y1:y2, x1:x2,:])
            if face_image is None:
                continue
            ref_images.append(torch.tensor(face_image))
        return ref_images

    def encode_video_with_vae(self, video):
        video_length = video.shape[1]
        pixel_values = rearrange(video, "b f c h w -> (b f) c h w")
        latents = self.vae.encode(pixel_values).latent_dist
        latents = latents.sample()
        latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
        return latents



    def forward(self, batch, ref_data):
        video = batch["video"].to(self.device)
        ref_frames = batch["ref_frames"][0]
        prompt = batch["prompt"][0]
        # Extract Face features using insightface
        ref_images = self.get_face_image(ref_frames)
        if len(ref_images) == 0:
            return {}
        ref_images = torch.stack(ref_images, dim=0)
        ref_images = ref_images.permute(0, 3, 1, 2).unsqueeze(dim=0).to(device=self.device, dtype=video.dtype)
        ref_images = (ref_images / 255. - 0.5) * 2 
        ref_images_latent = self.encode_video_with_vae(ref_images)
        ref_data['new_ref_images_latent']=ref_images_latent.squeeze(dim=0)
        return ref_data

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    annotator = Annotator()
    annotator.eval()
    annotator.to("cuda")
    print(args.phase)
    data = CelebVTextProcessDataset(root=args.root,resolution=[512,512],load_all_frames=False,phase=args.phase,total=args.total)
    print(len(data))
    data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    all_data = []
    for batch in tqdm(data_loader):
        if batch == {}:
            continue
        ref_data = None
        if os.path.exists(f"{args.save_path}/processed_sd15/{batch['name'][0]}.pt"):
            save_path = f"{args.save_path}/processed_sd15/{batch['name'][0]}.pt"
            ref_data = torch.load(save_path)
            if 'new_ref_images_latent' in ref_data.keys():
                output = ref_data
            else:
                with torch.no_grad():
                    output = annotator(batch,ref_data)
        else:
            output == {}
        if output == {}:
            continue
        output['prompt'] = batch['prompt'][0]
        save_path = f"{args.root}/processed_sd15/{batch['name'][0]}.pt"
        all_data.append({"path":save_path, "prompt":batch['prompt'][0]})
        torch.save(output, f"{args.save_path}/processed_sd15/{batch['name'][0]}.pt")

    json_file_name = os.path.join(args.save_path, "processed_sd15_new_face_{}.json".format(args.phase))
    with open(json_file_name, 'w') as  f:
        json.dump(all_data, f, indent=4)
    print("ok")

