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



if __name__ == '__main__':
    root_path = './projects/IDAdapter-diffusers/data/poco_celeb_images'
    folder_list = os.listdir(root_path)

    start_idx = 0
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']  # 支持的图像文件扩展名

    # with open(, 'w') as face_f:
    text_logger = "data-process/poco/010_find_single_face.txt"
    if os.path.exists(text_logger):
        os.remove(text_logger)

    for folder_name in tqdm(sorted(folder_list)[start_idx:]):
        folder = os.path.join(root_path, folder_name)
        files = os.listdir(folder)
        file_list = []
        for filename in files:
            img_ext = os.path.splitext(filename)[1].lower()
            # 如果是图像文件，则将文件路径添加到列表中
            if img_ext in image_extensions:
                file_list.append(os.path.join(folder, filename))

        file_list = sorted(file_list) # Important!!!!!!
        if len(file_list) == 0:
            print(f"No image found. Ignoring {folder} ")
            continue
        
        num_face_indicator = []
        for idx, img_path in enumerate(file_list):
            extension = os.path.splitext(img_path)[-1]
            # file_path = './projects/IDAdapter-diffusers/data/imdb_celeb_images/0003460/MV5BY2EzNDFiOGUtOTY4My00MDVjLThjYTctYWRkYjlhMGU3ZTNlXkEyXkFqcGdeQXVyMjQwMDg0Ng@@._V1_.jpg'
            json_path = img_path.replace(extension, '.json')
            if os.path.exists(json_path):
                with open(json_path) as f:
                    meta_dict = json.load(f)

            bbox_meta = meta_dict['bbox_meta']
            num_face_indicator.append(len(bbox_meta))

        if all(list(map(lambda x: x<2, num_face_indicator))):
            with open(text_logger, 'a+') as f:
                f.write(f"{folder}\n")