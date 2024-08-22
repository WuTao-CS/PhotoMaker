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
        self.details_path = glob(os.path.join(self.root,"texts/face40_details_new", "*.txt"))
        self.details_path.sort()
        per_devie_num = len(self.details_path)/total
        start = int(phase*per_devie_num)
        end = int((phase+1)*per_devie_num)
        if end >= len(self.details_path):
            self.details_path = self.details_path[start:]
        else:
            self.details_path = self.details_path[start:end]
        self.spatial_transform_type = spatial_transform_type
        self.spatial_transform = make_spatial_transformations(self.resolution, type=self.spatial_transform_type) \
            if self.spatial_transform_type is not None else None
        self.faild_case = []
        self.fixed_fps = fixed_fps
        self.load_all_frames = load_all_frames
        self.num_ref_frames = num_ref_frames

    def __len__(self) -> int:
        return len(self.details_path)

    def _load_text(self, file_path: str) -> str:
        # 打开文件并逐行读取内容
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # 去掉每行末尾的换行符
        lines = [line.strip() for line in lines]
        final_new_text=""
        final_text = ""
        for text in lines:
            new_text, is_inserted = insert_img_after_keyword(text)
            if len(new_text)>len(final_new_text):
                final_new_text=new_text
                final_text=text
            
        if len(final_new_text)>0:
            additionals = ["action_dur", "emotion", "light_dir", "light_color_temp", "light_intensity"]
            for additional in additionals:
                additional_path = os.path.join(
                    self.root, 'texts', additional, os.path.basename(file_path)
                )
                try:
                    with open(additional_path, "r") as f:
                        add_text = f.readline().strip()
                        final_new_text+=add_text
                        final_text+=add_text
                except FileNotFoundError:
                    print(f"File {additional_path} not found.", "yellow")    
            return final_text, final_new_text, True
        return text, None, False
    
    def __getitem__(self, index):
        index = index % len(self.details_path)
        text_file = self.details_path[index]
        text, new_text, is_inserted = self._load_text(text_file)
        video_path = os.path.join(
            self.root,
            "celebvtext_6",
            f"{os.path.splitext(os.path.basename(text_file))[0]}.mp4",
        )
        name = os.path.splitext(os.path.basename(text_file))[0]
        video_reader = VideoReader(video_path, ctx=cpu(0), width=self.resolution[1], height=self.resolution[0])
        if len(video_reader) < self.video_length or is_inserted is False:
            self.faild_case.append(text_file)
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
        data={"video":frames, "prompt":text, "prompt_trigger":new_text, "video_path":video_path, "name":name,"ref_frames":ref_frames}
        return data

    
class CelebVTextDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, video_length = 16, resolution = [512,512], frame_stride=8, spatial_transform_type="resize_center_crop", num_ref_frames=16, get_latent_from_video=False, fixed_fps=None, prompt_trigger=True, with_scaling_factor=True):
        self.root = root
        self.video_length = video_length
        self.resolution = resolution
        if self.resolution[0]==512:
            with open(os.path.join(self.root, "processed_512.json"), 'r') as file:
                self.all_data = json.load(file)
        else:
            with open(os.path.join(self.root, "processed.json"), 'r') as file:
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
        

    def __len__(self) -> int:
        return len(self.all_data)
    
    def __getitem__(self, index):
        index = index % len(self.all_data)
        data = self.all_data[index]
        prompt = data["prompt"]
        prompt_trigger = data["prompt_trigger"]
        if self.resolution[0]==512:
            process_data_path = data["path"].replace("processed_512","processed")
        else:
            process_data_path = data["path"]
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
            if self.resolution[0]==512:
                frames = torch.load(data["path"], map_location='cpu')["latent"][:,:self.video_length,:,:]
            else:
                frames = process_data["latent"][:,:self.video_length,:,:]
        idx = random.randint(0, self.num_ref_frames-1)
        faces_id_idx = random.randint(0, process_data["face_ids"].shape[0]-1)
        faces_id = process_data["face_ids"][faces_id_idx]
        clip_emb = process_data["image_embeds"][idx].unsqueeze(0)
        if self.prompt_trigger:
            prompt_emb = process_data["prompt_embeds_trigger"][idx]
            pooled_prompt_emb = process_data["pooled_prompt_embeds_trigger"]
        else:
            prompt_emb = process_data["prompt_emb"]
            pooled_prompt_emb = process_data["pooled_prompt_emb"]
        class_token_mask = process_data["class_tokens_mask"]
        if self.with_scaling_factor:
            frames = frames * 0.13025

        return {"video":frames, "clip_emb":clip_emb, "prompt_emb":prompt_emb, "pooled_prompt_emb":pooled_prompt_emb, "faces_id":faces_id, "class_token_mask":class_token_mask, "prompt":prompt, "prompt_trigger":prompt_trigger}
        
            
