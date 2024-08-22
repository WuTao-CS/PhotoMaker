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


parser = argparse.ArgumentParser(description='pexels get meta information')
parser.add_argument('--file_path', type=str, help='Saved information for each txt file')
# parser.add_argument('--save_meta_name', type=str, help='The save meta name')
opt = parser.parse_args()

print(f"Remove meta file: {opt.file_path}")
with open(opt.file_path, 'r') as f:
    file_list = f.readlines()

single_face_dirname_list = []
for filename in tqdm(file_list):
    filename = filename.strip()
    if os.path.exists(filename):
        os.remove(filename)
    extension = os.path.splitext(filename)[-1]

    json_path = filename.replace(extension, '.json')
    if os.path.exists(json_path):
        os.remove(json_path)

    mask_path = filename.replace(extension, '.mask.png')
    if os.path.exists(mask_path):
        os.remove(mask_path)