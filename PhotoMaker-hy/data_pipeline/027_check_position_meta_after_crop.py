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

text_logger = open(os.path.join('data-process/poco/027_failed_valid_meta.txt'), 'a+')

# 遍历文件夹中的所有文件和子文件夹
data_root='data/poco_celeb_images_cropped_1024'

image_paths = []
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp'] 
num_total_images = 0
for root, dirs, files in os.walk(data_root):
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

print(f"开始处理数据集...")
start=time.time()
# 下载图片的函数
def count_valid_meta(img_path):

    with open(img_path.replace('.png', '.json')) as f:
        meta_dict = json.load(f)
    
    valid_pos_count = 0
    if 'end_pos' in meta_dict.keys():
        valid_pos_count += 1

    valid_caption_count = 0
    if 'caption' in meta_dict.keys():
        valid_caption_count += 1

    valid_mask_count = 0
    if os.path.exists(img_path.replace('.png', '.mask.png')):
        valid_mask_count += 1

    return valid_pos_count, valid_caption_count, valid_mask_count
        

total_pos_count = 0
total_caption_count = 0
total_mask_count = 0

# # 使用线程池并行下载图片
with tqdm(total=len(image_paths)) as pbar:
    with concurrent.futures.ThreadPoolExecutor() as executor:
    #     # 将下载图片的函数提交给线程池，返回一个future对象列表
        future_to_url = {executor.submit(count_valid_meta, img_path): img_path for img_path in image_paths}
        for future in concurrent.futures.as_completed(future_to_url):
            # 获取已完成的future对象并输出结果
            img_path = future_to_url[future]
            pbar.update()
            try:
                valid_pos_count, valid_caption_count, valid_mask_count = future.result()
                if valid_pos_count == 0:
                    with open("data-process/poco/027_no_trigger_word.txt", 'a+') as f:
                        f.write(f"{img_path}\n")
                total_pos_count += valid_pos_count
                total_caption_count += valid_caption_count
                total_mask_count += valid_mask_count
            except Exception as exc:
                print(f'{img_path} json失败，原因为：{exc}')
                text_logger.write(f'{img_path}\n')


print(total_pos_count, total_caption_count, total_mask_count)

text_logger.close()
end = time.time()

print(f"处理时间为: {end - start}")