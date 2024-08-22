"""
Compare cropping face with expanding facial region
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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.utils import make_grid
import cv2

# TODO: update with alignment
Image.MAX_IMAGE_PIXELS = 1000000000


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    root_path = './projects/IDAdapter-diffusers/data/poco_celeb_images_cropped_1024'
    folder_list = os.listdir(root_path)

    # with open('./projects/IDAdapter-diffusers/dump_data/manually_redetect_data_ordered_imdb.txt', 'r') as f:
    #     imdb_dir_list = f.readlines()

    # folder_list = [item.strip() for item in imdb_dir_list]
    # print(folder_list)
    debug_count = 0

    start_idx = 0

    save_path = f'dump_data/verify_face_identity_manually_poco'
    os.makedirs(save_path, exist_ok=True)
    for folder_name in tqdm(sorted(folder_list)[start_idx:]):
        folder = os.path.join(root_path, folder_name)
        id_name = os.path.basename(folder)
        mask_list = sorted(glob.glob(os.path.join(folder, '*.mask.png')))
        img_list = [mask_path.replace('.mask.png', '.png') for mask_path in mask_list]
        save_image_list = [] 

        for name in img_list:
            instance_image = Image.open(name)
            if not instance_image.mode == "RGB":
                instance_image = instance_image.convert("RGB") 

            ### json path
            json_path = name.replace('.png', '.json')
            try:
                with open(json_path, 'r') as f:
                    meta_dict = json.load(f)
            except:
                with open('data-process/poco/024_no_json_file.txt', 'a+') as f:
                    f.write(f'{name}\n')

            bbox = meta_dict['bbox']

            instance_image = instance_image.crop(bbox).resize((112,112))   
            save_image_list.append(to_tensor(instance_image))
        if len(save_image_list) == 0:
            print(f"empty for {folder_name}")
            continue
        save_image_grid = torch.stack(save_image_list, dim=0)
        save_image_grid = make_grid(save_image_grid, nrow=5)
        save_image_grid = F.to_pil_image(save_image_grid)
        same_image_name = os.path.join(save_path, f'{id_name}.jpg')
        save_image_grid.save(same_image_name)