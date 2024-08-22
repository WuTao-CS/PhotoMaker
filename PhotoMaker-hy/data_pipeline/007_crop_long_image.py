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

detected_long_images_file_list = "./projects/IDAdapter-diffusers/data-process/poco/006_detected_long_images.txt"
with open(detected_long_images_file_list, 'r') as f:
    detected_info = f.readlines()

filename_list = []
for info in detected_info:
    filename_list.append(info.split()[0])

dump_data_root = f'data/poco_celeb_images_toy_dumped_007'

# text_logger = open(os.path.join(parent_path, 'detected_images.txt'), 'a+')
bug_logger = open(os.path.join('data-process/poco', '007_failed_images.txt'), 'a+')
# 遍历文件夹中的所有文件和子文件夹


print(f"开始处理数据集...")
start=time.time()
# 下载图片的函数
def detect_long_image_and_split(img_path):
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
        id_dirname = os.path.basename(os.path.dirname(img_path))
        dump_dirname = os.path.join(dump_data_root, id_dirname)
        img_name = os.path.basename(img_path)
        os.makedirs(dump_dirname, exist_ok=True)
        dump_path = os.path.join(dump_dirname, img_name)
        # 找到水平线
        # round_horizontal_theta = round(np.pi/2, 3)
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

        if len(horizontal_lines) > 0:
            horizontal_lines.sort(key=lambda x: x[0])

            # # 切割图像
            sub_images = []
            start_row = 0
            for rho, _ in horizontal_lines:
                end_row = int(rho)
                sub_image = image[start_row:end_row, :]
                # 只有有效切割时才保存图像
                if end_row - start_row > 100:
                    sub_images.append(sub_image)
                start_row = end_row

            #最后一个子图(水印顺便也去除了)
            sub_image = image[start_row:, :]
            if height - start_row > 100:
                sub_images.append(sub_image)

            # os.remove(img_path)
            # # 保存子图
            for i, sub_image in enumerate(sub_images):
                sub_save_path = img_path.replace(image_name, image_name+f'_{i}')
                extension = os.path.splitext(sub_save_path)[-1]
                sub_save_path = sub_save_path.replace(extension, '.png')
                cv2.imwrite(sub_save_path, sub_image)
                
            shutil.move(img_path, dump_path)
            return img_path
        
        if len(vertical_lines) > 0:
            vertical_lines.sort(key=lambda x: x[0])

            # # 切割图像
            sub_images = []
            start_column = 0
            for rho, _ in vertical_lines:
                end_column = int(rho)
                sub_image = image[:, start_column:end_column]
                # 只有有效切割时才保存图像
                if end_column - start_column > 100:
                    sub_images.append(sub_image)
                start_column = end_column

            #最后一个子图(水印顺便也去除了)
            sub_image = image[:, start_column:]
            if width - start_column > 100:
                sub_images.append(sub_image)

            # os.remove(img_path)
            # # 保存子图
            for i, sub_image in enumerate(sub_images):
                sub_save_path = img_path.replace(image_name, image_name+f'_{i}')
                extension = os.path.splitext(sub_save_path)[-1]
                sub_save_path = sub_save_path.replace(extension, '.png')
                cv2.imwrite(sub_save_path, sub_image)
            
            shutil.move(img_path, dump_path)
            return img_path            
    else:
        return None
        
        

# # 使用线程池并行下载图片
with tqdm(total=len(filename_list)) as pbar:
    with concurrent.futures.ThreadPoolExecutor() as executor:
    #     # 将下载图片的函数提交给线程池，返回一个future对象列表
        future_to_url = {executor.submit(detect_long_image_and_split, img_path): img_path for img_path in filename_list}
        for future in concurrent.futures.as_completed(future_to_url):
            # 获取已完成的future对象并输出结果
            img_path = future_to_url[future]
            pbar.update()
            try:
                status = future.result()
                if status:
                    print("切分移动成功")
                # if status is not None:
                #     text_logger.write(f'{img_path}\n')
            except Exception as exc:
                print(f'{img_path} 移动失败，原因为：{exc}')
                bug_logger.write(f'{img_path}\n')

bug_logger.close()
end = time.time()

print(f"处理时间为: {end - start}")