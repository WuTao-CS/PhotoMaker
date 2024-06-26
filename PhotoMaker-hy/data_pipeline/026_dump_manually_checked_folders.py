"""
Compare cropping face with expanding facial region

TODO: 
1. image resolution & face resolution filtering
2. no face filtering
3. if only detect only one face, using `bbox_meta` to replace `bbox`

"""
from torchvision.transforms.functional import to_tensor
import os
import argparse
import json
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import torch
from copy import deepcopy
# import pandas as pd
import math
from tqdm import tqdm
import glob
import shutil
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.utils import make_grid
import cv2
import concurrent.futures
import time


# parser = argparse.ArgumentParser(description='pexels get meta information')
# parser.add_argument('--file_path', type=str, help='Saved information for each txt file')
# parser.add_argument('--save_meta_name', type=str, help='The save meta name')
# opt = parser.parse_args()

poco_file_path = "./projects/IDAdapter-diffusers/data-process/poco/manually_list_poco.txt"
src_path = "./projects/IDAdapter-diffusers/data/poco_celeb_images_cropped_1024"
tgt_path = "./projects/IDAdapter-diffusers/data/poco_celeb_images_cropped_1024_dumped_026"

# 检查目标文件夹是否存在，如果不存在则创建
if not os.path.exists(tgt_path):
    os.makedirs(tgt_path, exist_ok=True)

print(f"Remove meta file: {poco_file_path}")
with open(poco_file_path, 'r') as f:
    file_list = f.readlines()

folder_name_list = []
for folder_name in file_list:
    folder_name = folder_name.strip()
    folder_name = os.path.splitext(folder_name)[0]
    folder_name_list.append(folder_name)

print(folder_name_list)

# single_face_dirname_list = []
for folder_name in tqdm(folder_name_list):
    full_src_path = os.path.join(src_path, folder_name)
    full_tgt_path = os.path.join(tgt_path, folder_name)

    # 移动文件夹
    try:
        shutil.move(full_src_path, full_tgt_path)
        print(f"文件夹已成功移动从 {full_src_path} 到 {full_tgt_path}")
    except Exception as e:
        print(f"移动文件夹时出错: {e}")