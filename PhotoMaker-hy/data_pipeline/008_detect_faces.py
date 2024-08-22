import os
import argparse
import json
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import facexlib
import torch
from copy import deepcopy
import pandas as pd
from tqdm import tqdm
from facexlib.visualization import visualize_detection
from facexlib.detection import RetinaFace
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

Image.MAX_IMAGE_PIXELS = 1000000000
"""
测试用的file:
./projects/weibo-crawler/weibo/1998858463/img/原创微博图片/20190903T_4412347655084206/20190903T_4412347655084206_1_0.png
"""

def init_detection_model(model_name='retinaface_resnet50', half=False, device='cuda', model_rootpath=None):
    if model_name == 'retinaface_resnet50':
        model = RetinaFace(network_name='resnet50', half=half)
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    model_path = './projects/diffusion-exp/detection_Resnet50_Final.pth'

    # TODO: clean pretrained model
    load_net = torch.load(model_path, map_location=lambda storage, loc: storage)
    # remove unnecessary 'module.'
    for k, v in deepcopy(load_net).items():
        if k.startswith('module.'):
            load_net[k[7:]] = v
            load_net.pop(k)
    model.load_state_dict(load_net, strict=True)
    model.eval()
    model = model.to(device)
    return model


def visualize_detection(img, bboxes, save_path=None, to_bgr=False):
    """Visualize detection results.

    Args:
        img (Numpy array): Input image. CHW, BGR, [0, 255], uint8.
    """
    img = np.copy(img)
    if to_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    height, width, _ = img.shape
    # font_scale = width * height / 8000000
    font_scale = 2
    cv2.putText(img, f'{width}x{height}', (255, 255), cv2.FONT_HERSHEY_DUPLEX,  font_scale, (255, 255, 255))
    for b in bboxes:
        cv2.putText(img, f'{b[2]-b[0]}x{b[3]-b[1]}', (int(b[0]), int(b[1] + 12)), cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255))
        # bounding boxes
        b = list(map(int, b))
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)

    # save img
    if save_path is not None:
        cv2.imwrite(save_path, img)



class IMDBDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
        self,
        data_root='./projects/IDAdapter-diffusers/data/poco_celeb_images',
        size=512
    ):
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']  # 支持的图像文件扩展名
        json_extensions = ['.json']  # 支持的json文件扩展名
        image_paths = []  # 存储图像文件路径的列表
        detected_image_paths = []  # 存储已检测好的json文件路径的列表

        # 遍历文件夹中的所有文件和子文件夹
        num_total_images = 0
        for root, dirs, files in os.walk(data_root):
            for filename in files:
                # 获取文件扩展名
                extension = os.path.splitext(filename)[1].lower()
                # 如果是图像文件，则将文件路径添加到列表中
                if extension in image_extensions:
                    num_total_images += 1
                    file_path = os.path.join(root, filename)
                    # TODO: 如果要重新处理，就需要删除这一行
                    # if not os.path.exists(file_path.replace(extension, '.json')):
                    image_paths.append(file_path)

        print(f"数据集图像数量: {num_total_images}, 未处理数量: {len(image_paths)}")
        self.image_paths = sorted(image_paths)
        self.image_transforms = transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR)
        self.size = size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        example = {}
        file_path = self.image_paths[index]
        try:
            instance_image = Image.open(file_path)
            # instance_image = exif_transpose(instance_image)

            if not instance_image.mode == "RGB":
                instance_image = instance_image.convert("RGB")

            ori_width, ori_height = instance_image.size
            example["instance_image"] = self.image_transforms(instance_image)
        except Exception as exc:
            with open('data-process/poco/008_failed_to_detect.txt', 'a+') as f:
                f.write(f"{file_path}\n")
            print(f"Warninig: {file_path} 错误信息为: {exc}")
            instance_image = Image.fromarray(np.zeros((self.size, self.size, 3)).astype(np.uint8)).convert('RGB')
            ori_width, ori_height = instance_image.size
            example["instance_image"] = instance_image
            
        example['w_scale'] = ori_width / float(self.size)
        example['h_scale'] = ori_height / float(self.size)

        example["file_path"] = file_path
        example["original_resolution"] = (ori_width, ori_height)

        return example


def collate_fn(examples):
    pil_images = [example["instance_image"] for example in examples]
    file_paths = [example["file_path"] for example in examples]
    scales = [(example["w_scale"], example["h_scale"], example["w_scale"], example["h_scale"]) for example in examples]
    original_resolution = [example["original_resolution"] for example in examples]

    batch = {"pil_images": pil_images, 'scales': scales, 'file_paths': file_paths, "original_resolution": original_resolution}
    return batch

# save the json file 
# number of faces
# each bounding box
det_net = init_detection_model()
det_dataset = IMDBDataset(data_root='./projects/IDAdapter-diffusers/data/poco_celeb_images')
det_dataloader = DataLoader(det_dataset, batch_size=8, num_workers=4, collate_fn=collate_fn)

# batch = next(det_dataloader)
with torch.no_grad():
    for step, batch in enumerate(tqdm(det_dataloader)):
        pil_images = batch['pil_images']
        scales = batch['scales']
        file_paths = batch['file_paths']
        original_resolution = batch['original_resolution']
        bboxes, _ = det_net.batched_detect_faces(pil_images, 0.80)
        # print(f"Tot")
        # count faces
        for box_per_face, filename, upscale, ori_res in zip(bboxes, file_paths, scales, original_resolution):
            det_meta = {}
            det_meta['number'] = len(box_per_face)
            det_meta['original_resolution'] = ori_res
            bbox_meta = []
            for box in box_per_face:
                upscaled_bbox = (box[:4] * np.array(upscale)).tolist()
                upscaled_bbox = [int(ub) for ub in upscaled_bbox]
                bbox_meta.append(upscaled_bbox)

            det_meta['bbox_meta'] = bbox_meta
            json_str = json.dumps(det_meta)

            # print(len(b), fn)
        # print(len(bboxes), [len(b) for b in bboxes])
            extension = os.path.splitext(filename)[-1]
            with open(filename.replace(extension, '.json'), 'w') as f:
                # print(filename.replace('jpg', 'json'))
                f.write(json_str)
# print(batch)

# 打印所有图像文件路径
# print(len(image_paths))