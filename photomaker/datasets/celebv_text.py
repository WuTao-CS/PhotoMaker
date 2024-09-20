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

MATCHED_WORDS = ["person", "male", "female", "man", "woman", "men", "women", "girl", "boy", "girls", "boys", "lady", "ladies", "teen", "teens", "student", "students"]


def insert_img_after_keyword(text):
    for word in MATCHED_WORDS:
        index = text.find(word)
        if index != -1:
            # Find the end index of the matched word
            end_index = index + len(word)
            # Insert "img" after the matched word
            new_text = text[:end_index] + " img" + text[end_index:]
            return new_text, True
    return text, False

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

class CelebVTextProcessDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, video_length = 16, resolution = [1024,1024], frame_stride=8, spatial_transform_type="resize_center_crop",fixed_fps=None,load_all_frames=False, num_ref_frames=16, phase=1, total=4):
        self.root = root
        self.video_length = video_length
        self.resolution = resolution
        self.frame_stride = frame_stride
        if self.resolution[0]==512:
            with open(os.path.join(self.root, "processed_512.json"), 'r') as file:
                self.all_data = json.load(file)
        else:
            with open(os.path.join(self.root, "processed.json"), 'r') as file:
                self.all_data = json.load(file)
        per_devie_num = len(self.all_data)/total
        start = int(phase*per_devie_num)
        end = int((phase+1)*per_devie_num)
        if end >= len(self.all_data):
            self.all_data = self.all_data[start:]
        else:
            self.all_data = self.all_data[start:end]
        self.spatial_transform_type = spatial_transform_type
        self.spatial_transform = make_spatial_transformations(self.resolution, type=self.spatial_transform_type) \
            if self.spatial_transform_type is not None else None
        self.faild_case = []
        self.fixed_fps = fixed_fps
        self.load_all_frames = load_all_frames
        self.num_ref_frames = num_ref_frames

    def __len__(self) -> int:
        return len(self.all_data)

    def __getitem__(self, index):
        index = index % len(self.all_data)
        data = self.all_data[index]
        text = data["prompt"]
        name = os.path.basename(data["path"]).split(".")[0]
        video_path = os.path.join(
            self.root,
            "celebvtext_6",
            f"{name}.mp4",
        )
        video_reader = VideoReader(video_path, ctx=cpu(0), width=self.resolution[1], height=self.resolution[0])
        if len(video_reader) < self.video_length:
            self.faild_case.append(video_path)
            return {}
        
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
        if self.load_all_frames:
            frames = video_reader.get_batch(range(frame_num))
            
        else:
            random_range = frame_num - required_frame_num
            start_idx = random.randint(0, random_range) if random_range > 0 else 0
            frame_indices = [start_idx + frame_stride*i for i in range(self.video_length)]
            try:
                frames = video_reader.get_batch(frame_indices)
            except:
                print(f"Get frames failed! path = {video_path}; [max_ind vs frame_total:{max(frame_indices)} / {frame_num}]")
                return {}
            assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
        ref_frames = video_reader.get_batch(np.linspace(0, frame_num-1, num=self.num_ref_frames, dtype=int)).asnumpy()

        frames = torch.tensor(frames.asnumpy()).permute(0,3,1,2).float() # [t,h,w,c] -> [t,c,h,w]
        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
        frames = (frames / 255 - 0.5) * 2
        prompt_trigger,_ = insert_img_after_keyword(text)
        data={"video":frames, "prompt":text, "video_path":video_path, "name":name,"ref_frames":ref_frames, "prompt_trigger":prompt_trigger}
        return data

    
class CelebVTextDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, video_length = 16, resolution = [512,512], frame_stride=8, spatial_transform_type="resize_center_crop", num_ref_frames=16, get_latent_from_video=False, fixed_fps=None, prompt_trigger=True, with_scaling_factor=True,image_drop_ratio=0.05, text_drop_ratio=0.05, image_text_drop_ratio=0.05):
        self.root = root
        self.video_length = video_length
        self.resolution = resolution
        with open(os.path.join(self.root, "processed_sdxl_512_final.json"), 'r') as file:
            self.all_data = json.load(file)
        
        self.spatial_transform_type = spatial_transform_type
        self.spatial_transform = make_spatial_transformations(self.resolution, type=self.spatial_transform_type) \
            if self.spatial_transform_type is not None else None
        self.num_ref_frames = num_ref_frames
        self.fixed_fps = fixed_fps
        self.get_latent_from_video = get_latent_from_video
        self.frame_stride = frame_stride
        self.prompt_trigger = prompt_trigger
        self.with_scaling_factor =with_scaling_factor
        self.image_drop_ratio = image_drop_ratio
        self.text_drop_ratio = text_drop_ratio
        self.image_text_drop_ratio = image_text_drop_ratio
        

    def __len__(self) -> int:
        return len(self.all_data)
    
    def __getitem__(self, index):
        index = index % len(self.all_data)
        data = self.all_data[index]
        prompt = data["prompt"]
        prompt_trigger = data["prompt_trigger"]
        process_data_path = data["path"]
        process_data = torch.load(process_data_path, map_location='cpu')
        
        if process_data["latent"].shape[0]==1:
            video_ref = torch.cat([process_data["latent"].unsqueeze(dim=1),process_data["ref_images_latent"]],dim=1)
            process_data["latent"]=video_ref[:,:,:self.video_length,:,:].squeeze(0)
            process_data["ref_images_latent"]=video_ref[:,:,self.video_length:,:,:]

        
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
                frames = frames * 0.13025
        idx = random.randint(0, self.num_ref_frames-1)
        faces_id_idx = random.randint(0, process_data["face_ids"].shape[0]-1)
        faces_id = process_data["face_ids"][faces_id_idx].unsqueeze(0)
        clip_emb = process_data["image_embeds"][faces_id_idx].unsqueeze(0)
        if self.prompt_trigger:
            if len(process_data["prompt_embeds_trigger"].shape)==2:
                process_data["prompt_embeds_trigger"]=process_data["prompt_embeds_trigger"].unsqueeze(0)
            prompt_emb = process_data["prompt_embeds_trigger"][faces_id_idx]
            pooled_prompt_emb = process_data["pooled_prompt_embeds_trigger"]
        else:
            prompt_emb = process_data["prompt_emb"]
            pooled_prompt_emb = process_data["pooled_prompt_emb"]

        # random drop
        rand_num = random.random()
        if rand_num < self.image_drop_ratio:
            clip_emb = torch.zeros_like(clip_emb)
            faces_id = torch.zeros_like(faces_id)
        elif rand_num < (self.image_drop_ratio + self.text_drop_ratio):
            prompt_emb = torch.zeros_like(prompt_emb)
            pooled_prompt_emb = torch.zeros_like(pooled_prompt_emb)
        elif rand_num < (self.image_drop_ratio + self.text_drop_ratio + self.image_text_drop_ratio):
            prompt_emb = torch.zeros_like(prompt_emb)
            pooled_prompt_emb = torch.zeros_like(pooled_prompt_emb)
            clip_emb = torch.zeros_like(clip_emb)
            faces_id = torch.zeros_like(faces_id)
        
        return {"video":frames, "clip_emb":clip_emb, "prompt_emb":prompt_emb, "pooled_prompt_emb":pooled_prompt_emb, "faces_id":faces_id, "prompt":prompt, "prompt_trigger":prompt_trigger,"process_data_path":process_data_path}
        
            
