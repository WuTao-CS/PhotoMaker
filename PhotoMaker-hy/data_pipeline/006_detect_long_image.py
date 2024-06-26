import os
import argparse
import json
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from copy import deepcopy
# import pandas as pd
import math
from tqdm import tqdm
import glob
import shutil
import cv2
import concurrent.futures
import time

data_root = "./projects/IDAdapter-diffusers/data/poco_celeb_images"

parent_path = "./projects/IDAdapter-diffusers/data-process/poco"
print(parent_path)
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
            # if os.path.exists(file_path.replace(extension, '.json')):
            image_paths.append(file_path)

print(f"数据集图像数量: {num_total_images}")
image_paths = sorted(image_paths)

text_logger = open(os.path.join(parent_path, '006_detected_long_images.txt'), 'a+')
bug_logger = open(os.path.join(parent_path, '006_failed_images.txt'), 'a+')
# 遍历文件夹中的所有文件和子文件夹


print(f"开始处理数据集...")
start=time.time()
# 下载图片的函数
def detect_long_image(img_path):
    image_name = os.path.splitext(os.path.basename(img_path))[0]
    image = cv2.imread(img_path)
    width = image.shape[1]
    height = image.shape[0]
    # 使用Canny边缘检测
    edges = cv2.Canny(image, 100, 180)
    
    # cv2.imwrite(f'edge.png', edges)
    # 使用Hough变换找到图像中的直线
    # lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    # lines = cv2.HoughLines(edges, 1, np.pi/180, int(width*0.5))
    lines = cv2.HoughLines(edges, 1, np.pi/180, int(width*0.5))

    if lines is not None:
        epsilon = 1e-3
        horizontal_lines = []
        vertical_lines = []
        for rho, theta in lines[:, 0]:
            # print(round(theta, 3), round(np.pi/2, 3))
            # round_theta = round(theta, 3)
            # print(round_theta, round_horizontal_theta, round_theta == round_horizontal_theta)
            if  np.abs(theta - np.pi/2) < epsilon:
                horizontal_lines.append((rho, theta))
            if np.abs(theta) < epsilon:
                vertical_lines.append((rho, theta))

        # print(len(horizontal_lines), len(vertical_lines))
        if (len(horizontal_lines) > 0) or (len(vertical_lines) > 0):
            # print(f"检测到可能为长图")
            # 找到长图分割线
            ratio = max(height, width) / float(min(height, width))
            if ratio > 1.7:
                return img_path, ratio
            else:
                return None
        else:
            return None
        
    else:
        return None
        
        
# # 使用线程池并行下载图片
with tqdm(total=len(image_paths)) as pbar:
    with concurrent.futures.ThreadPoolExecutor() as executor:
    #     # 将下载图片的函数提交给线程池，返回一个future对象列表
        future_to_url = {executor.submit(detect_long_image, img_path): img_path for img_path in image_paths}
        for future in concurrent.futures.as_completed(future_to_url):
            # 获取已完成的future对象并输出结果
            img_path = future_to_url[future]
            pbar.update()
            try:
                results = future.result()
                if results is not None:
                    text_logger.write(f'{results[0]} {results[1]}\n')
            except Exception as exc:
                print(f'{img_path} 检测失败，原因为：{exc}')
                bug_logger.write(f'{img_path}\n')

text_logger.close()
bug_logger.close()
end = time.time()

print(f"处理时间为: {end - start}")