import requests
import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import os
from idadapter.transforms import tensor_to_image
from tqdm import tqdm
import glob
import json

"""
TODO: count IOU part for imdb faces 
"""

with open('data-process/poco/014_failed_segment.txt', 'r') as f:
    file_paths = f.readlines()


for img_path in file_paths:
    print(img_path)
    img_path = img_path.strip()
    json_path = img_path.replace('.png', '.json')
    with open(json_path) as f:
        meta_dict = json.load(f)
    face_bbox = meta_dict["bbox_after_cropped"]
    img_size = meta_dict["size_after_crop"]
    print(face_bbox, img_size)
    # 创建一个黑色背景的图像
    img = Image.new('RGB', img_size, color='black')

    # 创建一个白色的画笔
    draw = ImageDraw.Draw(img)

    # 在bbox内部填充白色
    draw.rectangle(face_bbox, fill='white')

    # 保存图像
    filename = img_path.replace(".png", ".mask.png")
    print(filename)
    img.save(filename)
    # import pdb; pdb.set_trace()
    # we refer to the demo notebooks for visualization (see "Resources" section in the Mask2Former docs)
