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
from torchvision import transforms


def exists(val: Any) -> bool:
    return val is not None

def make_spatial_transformations(resolution, type, ori_resolution=None):
    """ 
    resolution: target resolution, a list of int, [h, w]
    """
    if type == "random_crop":
        transformations = transforms.RandomCropss(resolution)
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
        
            
