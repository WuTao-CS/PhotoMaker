from torchvision.transforms.functional import to_tensor
import os
import argparse
import json
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import torch
from copy import deepcopy
import math
from tqdm import tqdm
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.utils import make_grid
import cv2
import concurrent.futures

parser = argparse.ArgumentParser(description='download images of imdb celebrities')

parser.add_argument("--data-root",
    type=str,
    default="./dataset/imdb_celeb_images_cropped_1024_bak"
)
parser.add_argument("--output-name",
    type=str,
    default="./dataset/imdb_celeb_filelist.txt"
)
args = parser.parse_args()

# 遍历文件夹中的所有文件和子文件夹
data_root= args.data_root

image_paths = []
# image_extensions = ['.jpg', '.jpeg', '.png', '.bmp'] 
image_extensions = ['.png'] 
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
                image_paths.append(file_path.replace('.jpg', '.png')+'\n')

print(f"数据集图像数量: {num_total_images}, 包含json文件数量: {len(image_paths)}")

with open(args.output_name, 'w') as f:
    f.writelines(sorted(image_paths))
# image_paths = sorted(image_paths)