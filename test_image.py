import torch
import numpy as np
import random
import os
from PIL import Image

from shutil import copyfile
from diffusers.utils import load_image


from insightface.app import FaceAnalysis
from transformers import CLIPVisionModelWithProjection
import cv2
from insightface.utils import face_align
# gloal variable and function
import argparse
def isimage(path):
    if 'png' in path.lower() or 'jpg' in path.lower() or 'jpeg' in path.lower():
        return True
def extract_face_features(image_lst: list, input_size=(640, 640)):
    # Extract Face features using insightface
    ref_images = []
    ref_images_emb = []
    app = FaceAnalysis(name="buffalo_l",
                       root="./pretrain_model",
                       providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    app.prepare(ctx_id=0, det_size=input_size)
    for img in image_lst:
        img = np.asarray(img)
        face_info = app.get(img)
        if len(face_info)==0:
                continue
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
        norm_face = face_align.norm_crop(img, landmark=face_info.kps, image_size=512)
        ref_images.append(norm_face)
        emb = torch.from_numpy(face_info.normed_embedding)
        ref_images_emb.append(emb)
    if len(ref_images)==0:
        print("no face detect")
        return [None, None]
    else:
        ref_images_emb = torch.stack(ref_images_emb, dim=0).unsqueeze(0)

    return ref_images, ref_images_emb
image_basename_list =[base_name for base_name in os.listdir("datasets/Famous_people") if isimage(base_name)]
image_path_list = sorted([os.path.join("datasets/Famous_people", basename) for basename in image_basename_list])
input_id_images=[]
target_dir='./datasets/common_people/'
for image_path in image_path_list:
    input_id_image=load_image(image_path).resize((640,640))
    ref_image = extract_face_features([input_id_image])[0]
    if ref_image is None:
        print(image_path)
    else:
        target_path=os.path.join(target_dir,os.path.basename(image_path))
        copyfile(image_path,target_path)
        input_id_images.append(image_path)

print(len(input_id_images))
print(input_id_images)