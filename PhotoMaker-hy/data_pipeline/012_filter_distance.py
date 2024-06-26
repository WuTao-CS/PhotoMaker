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

with open(os.path.join('data-process/poco', '010_find_single_face.txt'), 'r') as f:
    file_list = f.readlines()
print(len(file_list))

single_face_dirname_list = []
for filename in file_list:
    single_face_dirname_list.append(os.path.basename(filename.strip()))

text_logger = open(os.path.join('data-process/poco', '012_failed_mv.txt'), 'a+')

# 遍历文件夹中的所有文件和子文件夹
data_root= 'data/poco_celeb_images'

image_paths = []
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp'] 
num_total_images = 0
for root, dirs, files in tqdm(os.walk(data_root)):
    for filename in files:
        # 获取文件扩展名
        extension = os.path.splitext(filename)[1].lower()
        # 如果是图像文件，则将文件路径添加到列表中
        if extension in image_extensions:
            num_total_images += 1
            file_path = os.path.join(root, filename)
            if os.path.exists(file_path.replace(extension, '.json')):
                image_paths.append(file_path)

print(f"数据集图像数量: {num_total_images}, 包含json文件数量: {len(image_paths)}")
image_paths = sorted(image_paths)

# dump_data_root = data_root + '_dumped'
dump_data_root = 'data/poco_celeb_images_dumped_012'
os.makedirs(dump_data_root, exist_ok=True)

THRESHOLD = 8.0

print(f"开始处理数据集...")
start=time.time()
# 下载图片的函数
def remove_lowdistance_face(img_path):
    extension = os.path.splitext(img_path)[-1]

    with open(img_path.replace(extension, '.json')) as f:
        meta_dict = json.load(f)
    
    id_dirname = os.path.basename(os.path.dirname(img_path))
    dump_dirname = os.path.join(dump_data_root, id_dirname)

    img_name = os.path.basename(img_path)
    dump_path = os.path.join(dump_dirname, img_name)

    if 'distance_norm' in meta_dict.keys():
        distance_norm = meta_dict['distance_norm']
        # if len(bbox_list) == 0:
        #     shutil.move(img_path, dump_path)
        #     shutil.move(img_path.replace('.jpg', '.json'), dump_path.replace('.jpg', '.json'))
        # else:
        # if not any(indicator):
        if distance_norm > THRESHOLD and (id_dirname not in single_face_dirname_list):
            os.makedirs(dump_dirname, exist_ok=True)
            shutil.move(img_path, dump_path)
            shutil.move(img_path.replace(extension, '.json'), dump_path.replace(extension, '.json'))
        

# # 使用线程池并行下载图片
with tqdm(total=len(image_paths)) as pbar:
    with concurrent.futures.ThreadPoolExecutor() as executor:
    #     # 将下载图片的函数提交给线程池，返回一个future对象列表
        future_to_url = {executor.submit(remove_lowdistance_face, img_path): img_path for img_path in image_paths}
        for future in concurrent.futures.as_completed(future_to_url):
            # 获取已完成的future对象并输出结果
            img_path = future_to_url[future]
            pbar.update()
            try:
                future.result()
            except Exception as exc:
                print(f'{img_path} 移除失败，原因为：{exc}')
                text_logger.write(f'{img_path}\n')

text_logger.close()
end = time.time()

print(f"处理时间为: {end - start}")