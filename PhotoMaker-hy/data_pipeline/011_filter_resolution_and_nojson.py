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

text_logger = open(os.path.join('data-process/poco', '011_failed_mv.txt'), 'a+')

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
dump_data_root = 'data/poco_celeb_images_dumped_011'
os.makedirs(dump_data_root, exist_ok=True)

print(f"开始处理数据集...")
start=time.time()
# 下载图片的函数
def remove_unsatisfied_image(img_path):
    need_to_move = False

    extension = os.path.splitext(img_path)[-1]
    
    id_dirname = os.path.basename(os.path.dirname(img_path))
    dump_dirname = os.path.join(dump_data_root, id_dirname)

    img_name = os.path.basename(img_path)

    json_path = img_path.replace(extension, '.json')
    if os.path.exists(json_path):
        with open(json_path) as f:
            meta_dict = json.load(f)

        bbox_list = meta_dict['bbox_meta']

        # 未检测出来人脸的需要筛选出来
        if len(bbox_list) == 0:
            need_to_move = True
            bbox = None
        elif len(bbox_list) == 1:
            bbox = bbox_list[0]
        else:
            bbox = meta_dict['bbox']

        # bbox最长需要大于256
        if bbox:
            w_box = bbox[2] - bbox[0]
            h_box = bbox[3] - bbox[1]
            if max(w_box, h_box) < 256:
                need_to_move = True   

        # 图像分辨率最长边小于等于600的筛选出来
        original_resolution = meta_dict['original_resolution']
        if max(original_resolution) < 601:
            need_to_move = True
        if need_to_move:
            os.makedirs(dump_dirname, exist_ok=True)
            dump_path = os.path.join(dump_dirname, img_name)
            # move image
            shutil.move(img_path, dump_path)
            # move json
            shutil.move(json_path, dump_path.replace(extension, '.json'))
        # else:
        #     meta_dict['bbox'] = bbox
        #     # 将列表转换为json字符串
        #     json_str = json.dumps(meta_dict)

        #     # # 将json字符串保存到文件中
        #     with open(img_path.replace('.jpg', '.json'), 'w') as f:
        #         f.write(json_str)
    else:
        os.makedirs(dump_dirname, exist_ok=True)
        dump_path = os.path.join(dump_dirname, img_name)
        shutil.move(img_path, dump_path)

    # else:
    #     meta_dict['bbox'] = bbox
    #     # 将列表转换为json字符串
    #     json_str = json.dumps(meta_dict)

    #     # # 将json字符串保存到文件中
    #     with open(img_path.replace(extension, '.json'), 'w') as f:
    #         f.write(json_str)
        

# # 使用线程池并行下载图片
with tqdm(total=len(image_paths)) as pbar:
    with concurrent.futures.ThreadPoolExecutor() as executor:
    #     # 将下载图片的函数提交给线程池，返回一个future对象列表
        future_to_url = {executor.submit(remove_unsatisfied_image, img_path): img_path for img_path in image_paths}
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