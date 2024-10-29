import os, sys
import numpy as np
import torch
import torch.nn as nn
from glob import glob
import torch.utils
import torch.utils.data
from tqdm import tqdm
import logging
from decord import VideoReader, cpu
from typing import List, Optional, Tuple, Union, Any
from time import time
from glob import glob
import random
import json
import cv2
from torchvision import transforms
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from PIL import Image

def exists(val: Any) -> bool:
    return val is not None


def rectangle_to_square_and_resize(arr, target_size=512):
    # 获取数组的形状
    height, width = arr.shape[:2]
    
    # 确定长边和短边的长度
    max_side = max(height, width)
    
    # 创建一个新的正方形数组，边长为长边的长度
    square_arr = np.full((max_side, max_side) + arr.shape[2:], arr.max(), dtype=arr.dtype)
    
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


def make_spatial_transformations(resolution, type, ori_resolution=None):
    """ 
    resolution: target resolution, a list of int, [h, w]
    """
    if type == "random_crop":
        transformations = transforms.RandomCrop(resolution)
    elif type == "resize_center_crop":
        is_square = (resolution[0] == resolution[1])
        if is_square:
            transformations = transforms.Compose([
                transforms.Resize(resolution[0]),
                transforms.CenterCrop(resolution[0]),
                ])
        else:
            if ori_resolution is not None:
                # resize while keeping original aspect ratio,
                # then centercrop to target resolution
                resize_ratio = max(resolution[0] / ori_resolution[0], resolution[1] / ori_resolution[1])
                resolution_after_resize = [int(ori_resolution[0] * resize_ratio), int(ori_resolution[1] * resize_ratio)]
                transformations = transforms.Compose([
                    transforms.Resize(resolution_after_resize),
                    transforms.CenterCrop(resolution),
                    ])
            else:
                # directly resize to target resolution
                transformations = transforms.Compose([
                    transforms.Resize(resolution),
                    ])
    elif type == "resize":
        transformations = transforms.Compose([
            transforms.Resize(resolution),
            ])
    else:
        raise NotImplementedError
    return transformations


    
class CelebVTextSD15Dataset(torch.utils.data.Dataset):
    def __init__(self, root: str, video_length = 16, resolution = [512,512], frame_stride=8, spatial_transform_type="resize_center_crop", get_latent_from_video=False, fixed_fps=None, with_scaling_factor=True, text_drop_ratio=0.05, image_drop_ratio=0.05, image_text_drop_ratio=0.05):
        self.root = root
        self.video_length = video_length
        self.resolution = resolution
        if self.resolution[0]==512:
            with open(os.path.join(self.root, "processed_sd15.json"), 'r') as file:
                self.all_data = json.load(file)
        else:
            Exception("Only support 512 resolution")
        self.spatial_transform_type = spatial_transform_type
        self.spatial_transform = make_spatial_transformations(self.resolution, type=self.spatial_transform_type) \
            if self.spatial_transform_type is not None else None
        self.fixed_fps = fixed_fps
        self.get_latent_from_video = get_latent_from_video
        self.frame_stride = frame_stride
        self.with_scaling_factor =with_scaling_factor
        self.text_drop_ratio = text_drop_ratio
        self.image_drop_ratio = image_drop_ratio
        self.image_text_drop_ratio = image_text_drop_ratio
        

    def __len__(self) -> int:
        return len(self.all_data)

    def __getitem__(self, index):
        index = index % len(self.all_data)
        data = self.all_data[index]
        while True:
            prompt = data["prompt"]
            process_data_path = data["path"]
            base_name = os.path.basename(process_data_path).split(".")[0]
            process_data_path = os.path.join(
                self.root,
                "processed_sd15",
                f"{base_name}.pt",
            )
            try:
                process_data = torch.load(process_data_path, map_location='cpu')
                break
            except:
                index = (index+1) % len(self.all_data)
        
        if self.get_latent_from_video:
            base_name = os.path.basename(process_data_path).split(".")[0]
            video_path = os.path.join(
                self.root,
                "celebvtext_6",
                f"{base_name}.mp4",
            )
            video_reader = VideoReader(video_path, ctx=cpu(0), width=self.resolution[1], height=self.resolution[0])
            fps_ori = video_reader.get_avg_fps()
            if self.fixed_fps is not None:
                frame_stride = int(self.frame_stride * (1.0 * fps_ori / self.fixed_fps))
            else:
                frame_stride = self.frame_stride
            frame_stride = max(frame_stride, 1)
            required_frame_num = frame_stride * (self.video_length-1) + 1
            frame_num = len(video_reader)
            if frame_num < required_frame_num:
                frame_stride = frame_num // self.video_length
                required_frame_num = frame_stride * (self.video_length-1) + 1
            random_range = frame_num - required_frame_num
            start_idx = random.randint(0, random_range) if random_range > 0 else 0
            frame_indices = [start_idx + frame_stride*i for i in range(self.video_length)]
            frames = video_reader.get_batch(frame_indices)
            assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
            frames = torch.tensor(frames.asnumpy()).permute(0,3,1,2).float() # [t,h,w,c] -> [t,c,h,w]
            frames = (frames / 255 - 0.5) * 2
        else:
            frames = process_data["latent"][:,:self.video_length,:,:]
            if self.with_scaling_factor:
                frames = frames * 0.18215
        prompt_embeds = process_data["prompt_embeds"]
        ref_latent_id = random.randint(0, process_data["ref_images_latent"].shape[1]-1)
        ref_images_latent = process_data["ref_images_latent"][:,ref_latent_id,:,:].unsqueeze(1)
        if self.with_scaling_factor:
            ref_images_latent = ref_images_latent * 0.18215
        

        # random drop
        rand_num = random.random()
        if rand_num < self.text_drop_ratio:
            prompt_embeds = torch.zeros_like(prompt_embeds)
        elif rand_num < (self.image_drop_ratio + self.text_drop_ratio):
            ref_images_latent = torch.zeros_like(ref_images_latent)
        elif rand_num < (self.image_drop_ratio + self.text_drop_ratio + self.image_text_drop_ratio):
            prompt_embeds = torch.zeros_like(prompt_embeds)
            ref_images_latent = torch.zeros_like(ref_images_latent)
        
        return {"video":frames, "prompt_embeds":prompt_embeds, "prompt":prompt, "ref_images_latent":ref_images_latent}
        
            
class CelebVTextSDXLDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, video_length = 16, resolution = [512,512], frame_stride=8, spatial_transform_type="resize_center_crop", get_latent_from_video=False, fixed_fps=None, with_scaling_factor=True, text_drop_ratio=0.05, image_drop_ratio=0.05, image_text_drop_ratio=0.05):
        self.root = root
        self.video_length = video_length
        self.resolution = resolution
        if self.resolution[0]==512:
            with open(os.path.join(self.root, "processed_sdxl_512_final.json"), 'r') as file:
                self.all_data = json.load(file)
        else:
            Exception("Only support 512 resolution")
        self.spatial_transform_type = spatial_transform_type
        self.spatial_transform = make_spatial_transformations(self.resolution, type=self.spatial_transform_type) \
            if self.spatial_transform_type is not None else None
        self.fixed_fps = fixed_fps
        self.get_latent_from_video = get_latent_from_video
        self.frame_stride = frame_stride
        self.with_scaling_factor =with_scaling_factor
        self.text_drop_ratio = text_drop_ratio
        self.image_drop_ratio = image_drop_ratio
        self.image_text_drop_ratio = image_text_drop_ratio
        

    def __len__(self) -> int:
        return len(self.all_data)
    
    def __getitem__(self, index):
        index = index % len(self.all_data)
        data = self.all_data[index]
        prompt = data["prompt"]
        process_data_path = data["path"]
        base_name = os.path.basename(process_data_path).split(".")[0]
        process_data_path = os.path.join(
            self.root,
            "processed_sdxl_512",
            f"{base_name}.pt",
        )
        process_data = torch.load(process_data_path, map_location='cpu')
        
        if process_data["latent"].shape[0]==1:
            video_ref = torch.cat([process_data["latent"].unsqueeze(dim=1),process_data["ref_images_latent"]],dim=1)
            process_data["latent"]=video_ref[:,:,:self.video_length,:,:]
            process_data["ref_images_latent"]=video_ref[:,:,self.video_length:,:,:]

        frames = process_data["latent"][:,:self.video_length,:,:]
        
        if self.with_scaling_factor:
            frames = frames * 0.13025
        prompt_embeds = process_data["prompt_embeds"]
        pooled_prompt_emb = process_data["pooled_prompt_embeds"]
        ref_latent_id = random.randint(0, process_data["ref_images_latent"].shape[2]-1)
        ref_images_latent = process_data["ref_images_latent"][:,:,ref_latent_id,:,:].unsqueeze(2)
        if self.with_scaling_factor:
            ref_images_latent = ref_images_latent * 0.13025
        

        # random drop
        rand_num = random.random()
        if rand_num < self.text_drop_ratio:
            prompt_embeds = torch.zeros_like(prompt_embeds)
            pooled_prompt_emb = torch.zeros_like(pooled_prompt_emb)
        elif rand_num < (self.image_drop_ratio + self.text_drop_ratio):
            ref_images_latent = torch.zeros_like(ref_images_latent)
        elif rand_num < (self.image_drop_ratio + self.text_drop_ratio + self.image_text_drop_ratio):
            prompt_embeds = torch.zeros_like(prompt_embeds)
            pooled_prompt_emb = torch.zeros_like(pooled_prompt_emb)
            ref_images_latent = torch.zeros_like(ref_images_latent)
        
        return {"video":frames.squeeze(dim=0), "prompt_emb":prompt_embeds, "prompt":prompt, "ref_images_latent":ref_images_latent.squeeze(dim=0), "pooled_prompt_emb":pooled_prompt_emb}


class CelebVTextSD15OnlineDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, video_length = 16, resolution = [512,512], frame_stride=8, spatial_transform_type="resize_center_crop", get_latent_from_video=False, fixed_fps=None, with_scaling_factor=True, text_drop_ratio=0.05, image_drop_ratio=0.05, image_text_drop_ratio=0.05):
        self.root = root
        self.video_length = video_length
        self.resolution = resolution
        if self.resolution[0]==512:
            with open(os.path.join(self.root, "processed_sd15.json"), 'r') as file:
                self.all_data = json.load(file)
        else:
            Exception("Only support 512 resolution")
        self.spatial_transform_type = spatial_transform_type
        self.spatial_transform = make_spatial_transformations(self.resolution, type=self.spatial_transform_type) \
            if self.spatial_transform_type is not None else None
        self.fixed_fps = fixed_fps
        self.get_latent_from_video = get_latent_from_video
        self.frame_stride = frame_stride
        self.with_scaling_factor =with_scaling_factor
        self.text_drop_ratio = text_drop_ratio
        self.image_drop_ratio = image_drop_ratio
        self.image_text_drop_ratio = image_text_drop_ratio
        self.app = FaceAnalysis(name="buffalo_l",
                        root="./pretrain_model",
                        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
    
    def get_face_image(self, video):
        # Extract Face features using insightface
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

            return face_image
        return None

    def __len__(self) -> int:
        return len(self.all_data)
    
    def __getitem__(self, index):
        index = index % len(self.all_data)
        data = self.all_data[index]
        prompt = data["prompt"]
        process_data_path = data["path"]
        base_name = os.path.basename(process_data_path).split(".")[0]
        process_data_path = os.path.join(
            self.root,
            "processed_sd15",
            f"{base_name}.pt",
        )
        process_data = torch.load(process_data_path, map_location='cpu')
        
        base_name = os.path.basename(process_data_path).split(".")[0]
        video_path = os.path.join(
            self.root,
            "celebvtext_6",
            f"{base_name}.mp4",
        )
        video_reader = VideoReader(video_path, ctx=cpu(0), width=self.resolution[1], height=self.resolution[0])
        fps_ori = video_reader.get_avg_fps()
        if self.fixed_fps is not None:
            frame_stride = int(self.frame_stride * (1.0 * fps_ori / self.fixed_fps))
        else:
            frame_stride = self.frame_stride
        frame_stride = max(frame_stride, 1)
        required_frame_num = frame_stride * (self.video_length-1) + 1
        frame_num = len(video_reader)
        if frame_num < required_frame_num:
            frame_stride = frame_num // self.video_length
            required_frame_num = frame_stride * (self.video_length-1) + 1
        random_range = frame_num - required_frame_num
        start_idx = random.randint(0, random_range) if random_range > 0 else 0
        frame_indices = [start_idx + frame_stride*i for i in range(self.video_length)]
        frames = video_reader.get_batch(frame_indices)
        assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
        frames = frames.asnumpy()
        ref_image = self.get_face_image(frames)
        if ref_image is None:
            ref_image = frames[0]
        ref_image = torch.tensor(ref_image).permute(2,0,1).float().unsqueeze(0) # [h,w,c] -> [1,c,h,w]
        frames = torch.tensor(frames).permute(0,3,1,2).float() # [t,h,w,c] -> [t,c,h,w]
        frames = torch.cat([frames, ref_image], dim=0)
        frames = (frames / 255 - 0.5) * 2
        prompt_embeds = process_data["prompt_embeds"]
        # random drop
        rand_num = random.random()
        drop_image = False
        if rand_num < self.text_drop_ratio:
            prompt_embeds = torch.zeros_like(prompt_embeds)
            drop_image = False
        elif rand_num < (self.image_drop_ratio + self.text_drop_ratio):
            ref_image = torch.zeros_like(ref_image)
            drop_image = True
        elif rand_num < (self.image_drop_ratio + self.text_drop_ratio + self.image_text_drop_ratio):
            prompt_embeds = torch.zeros_like(prompt_embeds)
            ref_image = torch.zeros_like(ref_image)
            drop_image = True

        return {"video":frames, "prompt_embeds":prompt_embeds, "prompt":prompt, "ref_images":ref_image, 'drop_image':drop_image}
    
    
class CelebVTextSD15NewFaceDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, video_length = 16, resolution = [512,512], frame_stride=8, spatial_transform_type="resize_center_crop", get_latent_from_video=False, fixed_fps=None, with_scaling_factor=True, text_drop_ratio=0.05, image_drop_ratio=0.05, image_text_drop_ratio=0.05):
        self.root = root
        self.video_length = video_length
        self.resolution = resolution
        if self.resolution[0]==512:
            with open(os.path.join(self.root, "processed_sd15.json"), 'r') as file:
                self.all_data = json.load(file)
        else:
            Exception("Only support 512 resolution")
        self.spatial_transform_type = spatial_transform_type
        self.spatial_transform = make_spatial_transformations(self.resolution, type=self.spatial_transform_type) \
            if self.spatial_transform_type is not None else None
        self.fixed_fps = fixed_fps
        self.get_latent_from_video = get_latent_from_video
        self.frame_stride = frame_stride
        self.with_scaling_factor =with_scaling_factor
        self.text_drop_ratio = text_drop_ratio
        self.image_drop_ratio = image_drop_ratio
        self.image_text_drop_ratio = image_text_drop_ratio
        

    def __len__(self) -> int:
        return len(self.all_data)

    def __getitem__(self, index):
        index = index % len(self.all_data)
        data = self.all_data[index]
        while True:
            prompt = data["prompt"]
            process_data_path = data["path"]
            base_name = os.path.basename(process_data_path).split(".")[0]
            process_data_path = os.path.join(
                self.root,
                "processed_sd15",
                f"{base_name}.pt",
            )
            try:
                process_data = torch.load(process_data_path, map_location='cpu')
                break
            except:
                index = (index+1) % len(self.all_data)
        
        if self.get_latent_from_video:
            base_name = os.path.basename(process_data_path).split(".")[0]
            video_path = os.path.join(
                self.root,
                "celebvtext_6",
                f"{base_name}.mp4",
            )
            video_reader = VideoReader(video_path, ctx=cpu(0), width=self.resolution[1], height=self.resolution[0])
            fps_ori = video_reader.get_avg_fps()
            if self.fixed_fps is not None:
                frame_stride = int(self.frame_stride * (1.0 * fps_ori / self.fixed_fps))
            else:
                frame_stride = self.frame_stride
            frame_stride = max(frame_stride, 1)
            required_frame_num = frame_stride * (self.video_length-1) + 1
            frame_num = len(video_reader)
            if frame_num < required_frame_num:
                frame_stride = frame_num // self.video_length
                required_frame_num = frame_stride * (self.video_length-1) + 1
            random_range = frame_num - required_frame_num
            start_idx = random.randint(0, random_range) if random_range > 0 else 0
            frame_indices = [start_idx + frame_stride*i for i in range(self.video_length)]
            frames = video_reader.get_batch(frame_indices)
            assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
            frames = torch.tensor(frames.asnumpy()).permute(0,3,1,2).float() # [t,h,w,c] -> [t,c,h,w]
            frames = (frames / 255 - 0.5) * 2
        else:
            frames = process_data["latent"][:,:self.video_length,:,:]
            if self.with_scaling_factor:
                frames = frames * 0.18215
        prompt_embeds = process_data["prompt_embeds"]
        if "new_ref_images_latent" in process_data.keys():
            ref_latent_id = random.randint(0, process_data["new_ref_images_latent"].shape[1]-1)
            ref_images_latent = process_data["new_ref_images_latent"][:,ref_latent_id,:,:].unsqueeze(1)
        else:
            ref_latent_id = random.randint(0, process_data["ref_images_latent"].shape[1]-1)
            ref_images_latent = process_data["ref_images_latent"][:,ref_latent_id,:,:].unsqueeze(1)
        if self.with_scaling_factor:
            ref_images_latent = ref_images_latent * 0.18215
        

        # random drop
        rand_num = random.random()
        if rand_num < self.text_drop_ratio:
            prompt_embeds = torch.zeros_like(prompt_embeds)
        elif rand_num < (self.image_drop_ratio + self.text_drop_ratio):
            ref_images_latent = torch.zeros_like(ref_images_latent)
        elif rand_num < (self.image_drop_ratio + self.text_drop_ratio + self.image_text_drop_ratio):
            prompt_embeds = torch.zeros_like(prompt_embeds)
            ref_images_latent = torch.zeros_like(ref_images_latent)
        
        return {"video":frames, "prompt_embeds":prompt_embeds, "prompt":prompt, "ref_images_latent":ref_images_latent}
 