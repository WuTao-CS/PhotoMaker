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
from concurrent.futures import ThreadPoolExecutor

def exists(val: Any) -> bool:
    return val is not None


# 定义处理单帧的函数
def canny_process_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    canny_frame = cv2.Canny(gray_frame, threshold1=100, threshold2=200)
    canny_frame_3channel = cv2.merge([canny_frame, canny_frame, canny_frame])
    return canny_frame_3channel

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

class CelebVCannySD15LatentDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, ref_root='./datasets/sh_CeleV-Text/segment_results_head', video_length = 16, resolution = [512,512], frame_stride=8, fixed_fps=None, with_scaling_factor=True, text_drop_ratio=0.05, image_drop_ratio=0.05, image_text_drop_ratio=0.05,new_prompt=False):
        self.root = root
        self.ref_root = ref_root
        self.video_length = video_length
        self.resolution = resolution
        if self.resolution[0]==512:
            with open("datasets/sh_CeleV-Text/all_segment_results_head_path.json", 'r') as file:
                self.all_data = json.load(file)
        else:
            Exception("Only support 512 resolution")


        self.ref_img_random_trans = transforms.Compose([
            transforms.Resize(self.resolution),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=30,
                translate=(0.1, 0.1),
                scale=(0.6, 1.2),
                fill=255)
        ])
        self.fixed_fps = fixed_fps
        self.frame_stride = frame_stride
        self.with_scaling_factor =with_scaling_factor
        self.text_drop_ratio = text_drop_ratio
        self.image_drop_ratio = image_drop_ratio
        self.image_text_drop_ratio = image_text_drop_ratio
        self.new_prompt=new_prompt
    

    def __len__(self) -> int:
        return len(self.all_data)
    
    def __getitem__(self, index):
        index = index % len(self.all_data)
        data = self.all_data[index]
        prompt=''
        while True:
            prompt = ""
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
        
        base_name = os.path.basename(process_data_path).split(".")[0]
        # video_path = os.path.join(
        #     self.root,
        #     "celebvtext_6",
        #     f"{base_name}.mp4",
        # )
        # ref_video_path = os.path.join(
        #     self.root,
        #     "celebvtext_6_canny",
        #     f"{base_name}.mp4",
        # )
        # video_reader = VideoReader(video_path, ctx=cpu(), width=self.resolution[1], height=self.resolution[0])
        # ref_video_reader = VideoReader(ref_video_path, ctx=cpu(), width=self.resolution[1], height=self.resolution[0])
        # fps_ori = video_reader.get_avg_fps()
        # if self.fixed_fps is not None:
        #     frame_stride = int(self.frame_stride * (1.0 * fps_ori / self.fixed_fps))
        # else:
        #     frame_stride = self.frame_stride
        # frame_stride = max(frame_stride, 1)
        # required_frame_num = frame_stride * (self.video_length-1) + 1
        # frame_num = len(video_reader)
        # if frame_num < required_frame_num:
        #     frame_stride = frame_num // self.video_length
        #     required_frame_num = frame_stride * (self.video_length-1) + 1
        # random_range = frame_num - required_frame_num
        # start_idx = random.randint(0, random_range) if random_range > 0 else 0
        # frame_indices = [start_idx + frame_stride*i for i in range(self.video_length)]
        # frames = video_reader.get_batch(frame_indices)
        # frames = frames.asnumpy()
        # assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
        # ref_frames = ref_video_reader.get_batch(frame_indices)
        # ref_frames = ref_frames.asnumpy()
    
        
        # frames = torch.tensor(frames).permute(0,3,1,2).float() # [t,h,w,c] -> [t,c,h,w]
        # ref_frames = torch.tensor(ref_frames).permute(0,3,1,2).float() # [t,h,w,c] -> [t,c,h,w]
        frames = process_data["latent"][:,:self.video_length,:,:]
        ref_frames = frames
        if self.with_scaling_factor:
            frames = frames * 0.18215
            ref_frames = ref_frames * 0.18215
        if self.new_prompt:
            try:
                prompt_embeds = process_data["new_prompt_embeds"].detach()
            except:
                prompt_embeds = process_data["prompt_embeds"]
        else:
            prompt_embeds = process_data["prompt_embeds"]
        # random drop
        rand_num = random.random()
        drop_image = False
        if rand_num < self.text_drop_ratio:
            prompt_embeds = torch.zeros_like(prompt_embeds)
        elif rand_num < (self.image_drop_ratio + self.text_drop_ratio):
            ref_frames = torch.zeros_like(ref_frames)
        elif rand_num < (self.image_drop_ratio + self.text_drop_ratio + self.image_text_drop_ratio):
            prompt_embeds = torch.zeros_like(prompt_embeds)
            ref_frames = torch.zeros_like(ref_frames)
        # frames = torch.cat([frames, ref_frames], dim=0)
        frames = torch.cat([frames, ref_frames], dim=1)
        return {"video":frames, "prompt_embeds":prompt_embeds, "prompt":prompt}

class CelebVTextCannySD15Dataset(torch.utils.data.Dataset):
    def __init__(self, root: str, ref_root='./datasets/sh_CeleV-Text/segment_results_head', video_length = 16, resolution = [512,512], frame_stride=8, fixed_fps=None, with_scaling_factor=True, text_drop_ratio=0.05, image_drop_ratio=0.05, image_text_drop_ratio=0.05,new_prompt=False):
        self.root = root
        self.ref_root = ref_root
        self.video_length = video_length
        self.resolution = resolution
        if self.resolution[0]==512:
            with open("datasets/sh_CeleV-Text/all_segment_results_head_path.json", 'r') as file:
                self.all_data = json.load(file)
        else:
            Exception("Only support 512 resolution")


        self.ref_img_random_trans = transforms.Compose([
            transforms.Resize(self.resolution),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=30,
                translate=(0.1, 0.1),
                scale=(0.6, 1.2),
                fill=255)
        ])
        self.fixed_fps = fixed_fps
        self.frame_stride = frame_stride
        self.with_scaling_factor =with_scaling_factor
        self.text_drop_ratio = text_drop_ratio
        self.image_drop_ratio = image_drop_ratio
        self.image_text_drop_ratio = image_text_drop_ratio
        self.new_prompt=new_prompt
    

    def __len__(self) -> int:
        return len(self.all_data)
    
    def __getitem__(self, index):
        index = index % len(self.all_data)
        prompt=''
        while True:
            data = self.all_data[index]
            process_data_path = data["path"]
            # print(process_data_path)
            base_name = os.path.basename(process_data_path).split(".")[0]
            process_data_path = os.path.join(
                self.root,
                "processed_sd15",
                f"{base_name}.pt",
            )
            try:
                process_data = torch.load(process_data_path, map_location='cpu')
                base_name = os.path.basename(process_data_path).split(".")[0]
                video_path = os.path.join(
                    self.root,
                    "celebvtext_6",
                    f"{base_name}.mp4",
                )
                # ref_video_path = os.path.join(
                #     self.root,
                #     "celebvtext_6_canny",
                #     f"{base_name}.mp4",
                # )
                video_reader = VideoReader(video_path, ctx=cpu(), width=self.resolution[1], height=self.resolution[0])
                # ref_video_reader = VideoReader(ref_video_path, ctx=cpu(), width=self.resolution[1], height=self.resolution[0])
                # print(len(video_reader)==len(ref_video_reader))
                break
            except:
                index = (index+1) % len(self.all_data)
        video_reader = VideoReader(video_path, ctx=cpu(), width=self.resolution[1], height=self.resolution[0])
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
        frames = frames.asnumpy()
        assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
        # ref_frames = ref_video_reader.get_batch(frame_indices)
        # ref_frames = ref_frames.asnumpy()
        # 使用Canny算子处理每一帧
        ref_frames = []
        ref_frames = []
        with ThreadPoolExecutor() as executor:
            ref_frames = list(executor.map(canny_process_frame, frames))
        # for frame in frames:
        #     # 将帧从BGR转换为灰度图像
        #     gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        #     # 应用Canny边缘检测
        #     canny_frame = cv2.Canny(gray_frame, threshold1=100, threshold2=200)
        #     # 将处理后的帧添加到ref_frames列表中
        #     # 将单通道的Canny图像复制为三通道
        #     canny_frame_3channel = cv2.merge([canny_frame, canny_frame, canny_frame])
        #     # 将处理后的帧添加到ref_frames列表中
        #     ref_frames.append(canny_frame_3channel)
        ref_frames = np.array(ref_frames)
        # print(ref_frames.shape)
        
        frames = torch.tensor(frames).permute(0,3,1,2).float() # [t,h,w,c] -> [t,c,h,w]
        ref_frames = torch.tensor(ref_frames).permute(0,3,1,2).float() # [t,h,w,c] -> [t,c,h,w]
        
        if self.new_prompt:
            try:
                prompt_embeds = process_data["new_prompt_embeds"].detach()
            except:
                prompt_embeds = process_data["prompt_embeds"]
        else:
            prompt_embeds = process_data["prompt_embeds"]
        # random drop
        rand_num = random.random()
        drop_image = False
        if rand_num < self.text_drop_ratio:
            prompt_embeds = torch.zeros_like(prompt_embeds)
        elif rand_num < (self.image_drop_ratio + self.text_drop_ratio):
            ref_frames = torch.zeros_like(ref_frames)
        elif rand_num < (self.image_drop_ratio + self.text_drop_ratio + self.image_text_drop_ratio):
            prompt_embeds = torch.zeros_like(prompt_embeds)
            ref_frames = torch.zeros_like(ref_frames)
        frames = torch.cat([frames, ref_frames], dim=0)
        frames = (frames / 255 - 0.5) * 2
        return {"video":frames, "prompt_embeds":prompt_embeds, "prompt":prompt}
    
    
class CelebVTextSelectCannySD15Dataset(torch.utils.data.Dataset):
    def __init__(self, root: str, ref_root='./datasets/sh_CeleV-Text/segment_results_head', video_length = 16, resolution = [512,512], frame_stride=8, fixed_fps=None, with_scaling_factor=True, text_drop_ratio=0.05, image_drop_ratio=0.05, image_text_drop_ratio=0.05,new_prompt=False, num_reference_frame=16):
        self.root = root
        self.ref_root = ref_root
        self.video_length = video_length
        self.resolution = resolution
        if self.resolution[0]==512:
            with open("datasets/sh_CeleV-Text/all_segment_results_head_path.json", 'r') as file:
                self.all_data = json.load(file)
        else:
            Exception("Only support 512 resolution")


        self.ref_img_random_trans = transforms.Compose([
            transforms.Resize(self.resolution),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=30,
                translate=(0.1, 0.1),
                scale=(0.6, 1.2),
                fill=255)
        ])
        self.fixed_fps = fixed_fps
        self.frame_stride = frame_stride
        self.with_scaling_factor =with_scaling_factor
        self.text_drop_ratio = text_drop_ratio
        self.image_drop_ratio = image_drop_ratio
        self.image_text_drop_ratio = image_text_drop_ratio
        self.new_prompt=new_prompt
        self.num_reference_frame = num_reference_frame
    

    def __len__(self) -> int:
        return len(self.all_data)
    
    def __getitem__(self, index):
        index = index % len(self.all_data)
        prompt=''
        while True:
            data = self.all_data[index]
            process_data_path = data["path"]
            # print(process_data_path)
            base_name = os.path.basename(process_data_path).split(".")[0]
            process_data_path = os.path.join(
                self.root,
                "processed_sd15",
                f"{base_name}.pt",
            )
            try:
                process_data = torch.load(process_data_path, map_location='cpu')
                base_name = os.path.basename(process_data_path).split(".")[0]
                video_path = os.path.join(
                    self.root,
                    "celebvtext_6",
                    f"{base_name}.mp4",
                )
                # ref_video_path = os.path.join(
                #     self.root,
                #     "celebvtext_6_canny",
                #     f"{base_name}.mp4",
                # )
                video_reader = VideoReader(video_path, ctx=cpu(), width=self.resolution[1], height=self.resolution[0])
                # ref_video_reader = VideoReader(ref_video_path, ctx=cpu(), width=self.resolution[1], height=self.resolution[0])
                # print(len(video_reader)==len(ref_video_reader))
                break
            except:
                index = (index+1) % len(self.all_data)
        video_reader = VideoReader(video_path, ctx=cpu(), width=self.resolution[1], height=self.resolution[0])
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
        frames = frames.asnumpy()
        assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
        # ref_frames = ref_video_reader.get_batch(frame_indices)
        # ref_frames = ref_frames.asnumpy()
        # 使用Canny算子处
        frame_indices = [i * (len(frames) // self.num_reference_frame) for i in range(self.num_reference_frame)]
        selected_frames = frames[frame_indices]
        ref_frames = []
        with ThreadPoolExecutor() as executor:
            ref_frames = list(executor.map(canny_process_frame, selected_frames))
        ref_frames = np.array(ref_frames)
        # print(ref_frames.shape)
        
        frames = torch.tensor(frames).permute(0,3,1,2).float() # [t,h,w,c] -> [t,c,h,w]
        ref_frames = torch.tensor(ref_frames).permute(0,3,1,2).float() # [t,h,w,c] -> [t,c,h,w]
        
        if self.new_prompt:
            try:
                prompt_embeds = process_data["new_prompt_embeds"].detach()
            except:
                prompt_embeds = process_data["prompt_embeds"]
        else:
            prompt_embeds = process_data["prompt_embeds"]
        # random drop
        rand_num = random.random()
        drop_image = False
        if rand_num < self.text_drop_ratio:
            prompt_embeds = torch.zeros_like(prompt_embeds)
        elif rand_num < (self.image_drop_ratio + self.text_drop_ratio):
            ref_frames = torch.zeros_like(ref_frames)
        elif rand_num < (self.image_drop_ratio + self.text_drop_ratio + self.image_text_drop_ratio):
            prompt_embeds = torch.zeros_like(prompt_embeds)
            ref_frames = torch.zeros_like(ref_frames)
        frames = torch.cat([frames, ref_frames], dim=0)
        frames = (frames / 255 - 0.5) * 2
        # ref_frames = (ref_frames / 255 - 0.5) * 2
        return {"video":frames, "prompt_embeds":prompt_embeds, "prompt":prompt}