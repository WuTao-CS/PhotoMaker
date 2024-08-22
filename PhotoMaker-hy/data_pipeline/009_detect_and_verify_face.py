"""
Compare cropping face with expanding facial region

TODO: 
If only multiple faces in, start filtering
"""
from torchvision.transforms.functional import to_tensor
import os
import argparse
import json
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import facexlib
import torch
from copy import deepcopy
import pandas as pd
import math
from tqdm import tqdm
from facexlib.visualization import visualize_detection
from facexlib.recognition.arcface_arch import Backbone
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.utils import make_grid
import cv2
from facexlib.detection import RetinaFace
import shutil

# TODO: update with alignment
Image.MAX_IMAGE_PIXELS = 1000000000

def visualize_detection(img, bbox, save_path=None, to_bgr=False):
    """Visualize detection results.

    Args:
        img (Numpy array): Input image. CHW, BGR, [0, 255], uint8.
    """
    img = np.copy(img)
    if to_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    height, width, _ = img.shape
    # bounding boxes
    bbox = list(map(int, bbox))
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

    # save img
    if save_path is not None:
        cv2.imwrite(save_path, img)

def init_recognition_model(model_name, half=False, device='cuda', model_rootpath=None):
    if model_name == 'arcface':
        model = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se').to(device).eval() 
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    model_path = './projects/IDAdapter-diffusers/recognition_arcface_ir_se50.pth'

    # TODO: clean pretrained model
    load_net = torch.load(model_path, map_location=lambda storage, loc: storage)
    # remove unnecessary 'module.'
    # for k, v in deepcopy(load_net).items():
    #     if k.startswith('module.'):
    #         load_net[k[7:]] = v
    #         load_net.pop(k)
    model.load_state_dict(load_net, strict=True)
    model.eval()
    model = model.to(device)
    return model


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

def find_min_index(lst):
    min_value = min(lst)
    min_index = lst.index(min_value)
    return min_index

def find_max_index(lst):
    max_value = max(lst)
    max_index = lst.index(max_value)
    return max_index


def find_median(lst):
    sorted_lst = sorted(lst)
    lst_len = len(lst)
    mid = lst_len // 2
    if lst_len % 2 == 0:
        return (sorted_lst[mid - 1] + sorted_lst[mid]) / 2
    else:
        return sorted_lst[mid]


def load_image(file_path, det_net, crop_before_resize=False):
    extension = os.path.splitext(file_path)[-1]
    # file_path = './projects/IDAdapter-diffusers/data/imdb_celeb_images/0003460/MV5BY2EzNDFiOGUtOTY4My00MDVjLThjYTctYWRkYjlhMGU3ZTNlXkEyXkFqcGdeQXVyMjQwMDg0Ng@@._V1_.jpg'
    json_path = file_path.replace(extension, '.json')
    if os.path.exists(json_path):
        try:
            with open(json_path) as f:
                meta_dict = json.load(f)
        except:
            return [], [], []
        ## TODO resume
        # if 'distance_norm' in meta_dict.keys():
        #     return [], [], []

    instance_image_list = []
    try:
        instance_image = Image.open(file_path)
    except:
        with open('data-process/poco/009_failed_images.txt', 'a+') as f:
            f.write(f'{file_path}\n')
        return [], [], [] 
    # print(min(instance_image.size))
    ########## TODO
    # if max(instance_image.size) > 10000:
    #     target_size = (instance_image.width // 3, instance_image.height // 3)
    #     instance_image = instance_image.resize(target_size)
    #     instance_image.save(file_path)
    ##############
    width, height = instance_image.size

    # 确定 resize 后的大小
    if width < height:
        new_width = 1024
        new_height = int(height * new_width / width)
        w_scale = width / float(new_width)
        h_scale = height / float(new_height)
    else:
        new_height = 1024
        new_width = int(width * new_height / height)
        w_scale = width / float(new_width)
        h_scale = height / float(new_height)
    try:
        instance_image = instance_image.resize((new_width, new_height))
    except:
        with open('data-process/poco/009_failed_images.txt', 'a+') as f:
            f.write(f'{file_path}\n')
        return [], [], [] 
        # bbox_meta = []
        # for bbox in meta_dict['bbox_meta']:
        #     bbox_meta.append(list(map(lambda x: x/3, bbox)))

        # filtered_bbox_meta = []
        # for bbox in meta_dict['filtered_bbox_meta']:
        #     filtered_bbox_meta.append(list(map(lambda x: x/3, bbox)))
        # if 'bbox' in meta_dict.keys():
        #     bbox_meta()
    # print(min(instance_image.size))
    # exit()
    scales = [w_scale, h_scale, w_scale, h_scale]
    if not instance_image.mode == "RGB":
        instance_image = instance_image.convert("RGB")
    with torch.no_grad():
        # TODO: resolution lose heavily
        try:
            det_info, faces = det_net.align_multi(instance_image, conf_threshold=0.80)
        except:
            return [], [], []
    print(file_path, len(faces))
    if len(faces) == 0:
        return [], [], []
    bboxes = det_info[:, :4].tolist()
    selected_faces = []
    selected_bboxes = []
    for bbox, face in zip(bboxes, faces):
        bbox = (bbox[:4] * np.array(scales)).tolist()
        bbox = [int(b) for b in bbox]

        #### height selection
        # height = bbox[3] - bbox[1]
        # if height >= 255:

        selected_faces.append(face[:,:,::-1])
        selected_bboxes.append(bbox)

    if crop_before_resize:
        # image_tensor = to_tensor(instance_image.crop(bbox).resize((112,112))) 
        for face in selected_faces:
            image_tensor = to_tensor(face.copy()) 
            instance_image_list.append(F.normalize(image_tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True))

    name_list = [file_path for _ in instance_image_list]
    return instance_image_list, name_list, selected_bboxes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    root_path = './projects/IDAdapter-diffusers/data/poco_celeb_images'
    folder_list = os.listdir(root_path)
    # print(folder_list)
    # exit()
    debug_count = 0
    det_net = init_detection_model()
    recog_net = init_recognition_model(model_name='arcface')

    start_idx = 4745
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']  # 支持的图像文件扩展名

    for folder_name in tqdm(sorted(folder_list)[start_idx:]):
        folder = os.path.join(root_path, folder_name)
        files = os.listdir(folder)
        file_list = []
        for filename in files:
            img_ext = os.path.splitext(filename)[1].lower()
            # 如果是图像文件，则将文件路径添加到列表中
            if img_ext in image_extensions:
                file_list.append(os.path.join(folder, filename))
        file_list = sorted(file_list) # Important!!!!!!
        # img_list = sorted(glob.glob(os.path.join(folder, '*.jpg')))
        if len(file_list) == 0:
            print(f"No image found. Ignoring {folder} ")
            continue

        dist_list = []
        identical_count = 0

        face_tensor_list = []
        full_name_list = []
        full_bbox_list = []
        for idx, img_path in enumerate(file_list):
            instance_image_list, name_list, bbox_list = load_image(img_path, det_net, crop_before_resize=True)
            face_tensor_list.extend(instance_image_list)
            full_bbox_list.extend(bbox_list)
            full_name_list.extend(name_list)
        if len(face_tensor_list) == 0:
            print(f"Verify meta exists. Ignoring {folder} ")
            continue
        # update image list to avoid detecting/filtering zero face 
        img_list = sorted(list(set(full_name_list)))
        score_dict = {}
        for name in list(set(full_name_list)):
            score_dict[name] = []
                    
        face_tensors = torch.stack(face_tensor_list, dim=0)
        print("Face count: ", face_tensors.shape, len(face_tensor_list))
        with torch.no_grad():
            face_tensors = face_tensors.to(torch.device('cuda'))
            output = recog_net(face_tensors)

            # face_grid = make_grid(face_tensors, nrow=5)
            # face_grid = F.to_pil_image(face_grid)
            # face_grid.save('face_grid.jpg')

        score_list = []
        # vote_list = [0 for _ in range(output.shape[0])]
        for i in range(output.shape[0]):
            # TODO: list 
            # 2. align and warp 
            # a = torch.linalg.norm((output[i] - output), ord=2, dim=1, keepdim=False)
            # print(torch.linalg.norm((output[i] - output), ord=2, dim=1, keepdim=False))
            distance = torch.linalg.norm((output[i] - output), ord=2, dim=1, keepdim=False)

            #### Rule 1: sum
            score = torch.sum(distance)
            score_list.append(score.item() / len(face_tensor_list))

            #### Rule 2 : rank 
            # print(distance)
            # indices = torch.argsort(distance)[1:int(len(img_list) / 2)]

            # for vote_idx in indices.tolist():
            #     vote_list[vote_idx] += 1 
            # print(indices.tolist())
            # exit()
            # score = torch.linalg.norm((output[i] - output), ord=2, dim=1, keepdim=False)
            # print(full_name_list)
            # exit()

        for name, score in zip(full_name_list, score_list):
            score_dict[name].append(score)

        # construct filtering mask (mask out unrelated person)
        select_dict = {}
        for key, value in score_dict.items():
            min_value = min(value)
            min_index = value.index(min_value)
            indicator = [False for _ in range(len(value))]
            indicator[min_index] = True
            select_dict[key] = indicator

        indicator_list = []
        for name in img_list:
            indicator_list.extend(select_dict[name])
        
        indicator_mask = torch.tensor(indicator_list, device='cuda')
        # print(output.shape, indicator_mask[:, None])
        selected_output = output.masked_select(indicator_mask[:, None])
        selected_output = selected_output.view(-1, output.shape[1])
        # print(selected_output.view(-1, output.shape[1]).shape)
        
        full_bbox_list = [full_bbox_list[i] for i in range(len(indicator_list)) if indicator_list[i]]
        assert len(full_bbox_list) == selected_output.shape[0]
        assert len(img_list) == selected_output.shape[0]
        print("Filtered Face count: ", selected_output.shape, len(full_bbox_list))
        score_list = []
        for i in range(selected_output.shape[0]):
            # TODO: list 
            # 2. align and warp 
            # a = torch.linalg.norm((selected_output[i] - selected_output), ord=2, dim=1, keepdim=False)
            # print(torch.linalg.norm((selected_output[i] - selected_output), ord=2, dim=1, keepdim=False))
            distance = torch.linalg.norm((selected_output[i] - selected_output), ord=2, dim=1, keepdim=False)

            #### Rule 1: sum
            score = torch.sum(distance)
            score_list.append(score.item() / selected_output.shape[0])
        
        ######################
        ## visualize bounding bbox
        epsilon = 1e-8
        threshold = 8
        folder_name = os.path.basename(folder)
        # src_path = f'data/imdb_celeb_images_toy/{folder_name}'
        save_path = f'dump_data/det_ori_aligned_sorted_faces_poco/{folder_name}'
        os.makedirs(save_path, exist_ok=True)

        min_score = min(score_list)
        variance = sum([((x - min_score) ** 2) for x in score_list]) / len(score_list)
        std_dev = math.sqrt(variance)
        dis_score_list = list(map(lambda x: (x-min_score), score_list))
        print(std_dev)
        same_image_list = []
        diff_image_list = []

        same_score_list = []
        diff_score_list = []
        for name, score, bbox in zip(img_list, dis_score_list, full_bbox_list):
            instance_image = Image.open(name)
            if not instance_image.mode == "RGB":
                instance_image = instance_image.convert("RGB") 

            extension = os.path.splitext(name)[-1]
            ### json path
            json_path = name.replace(extension, '.json')
            # meta_dict = {}
            if not os.path.exists(json_path):
                old_json_path = name.replace(extension, '.json')
                shutil.move(old_json_path, json_path) 

            with open(json_path, 'r') as f:
                meta_dict = json.load(f)

            # if "verify_meta" in meta_dict.keys():
            #     meta_dict.pop("verify_meta")
            meta_dict['distance'] = int(score *1000) / 1000.
            meta_dict['distance_norm'] = int(score/(std_dev**2 + epsilon) * 1000) / 1000.
            meta_dict['bbox'] = list(map(int, bbox))
            # 将列表转换为json字符串
            json_str = json.dumps(meta_dict)

            # # 将json字符串保存到文件中
            with open(json_path, 'w') as f:
                f.write(json_str)
            ###

            instance_image = instance_image.crop(bbox).resize((112,112))    
            position_1 = (10, 10)
            position_2 = (10, 100)
            
            draw = ImageDraw.Draw(instance_image)
            font = ImageFont.load_default()
            color = (255, 255, 255)
            draw.text(position_1, f"{score/(std_dev**2 + epsilon):.2f}", fill=color, font=font)
            draw.text(position_2, f"{score*10:.2f}", fill=color, font=font)
            if score / (std_dev + epsilon) < threshold * std_dev:
                same_image_list.append(to_tensor(instance_image))
                same_score_list.append(score)
            else:
                diff_image_list.append(to_tensor(instance_image))
                diff_score_list.append(score)

        if len(same_image_list) > 0:
            zipped = list(zip(same_image_list, same_score_list))
            # 根据list1的排序方式对元组列表进行排序
            zipped.sort(key=lambda x: x[1])
            # 将排序后的元组列表解包成两个列表
            same_image_list, same_score_list = zip(*zipped)
            same_image_grid = torch.stack(same_image_list, dim=0)
            same_image_grid = make_grid(same_image_grid, nrow=5)
            same_image_grid = F.to_pil_image(same_image_grid)
            same_image_name = os.path.join(save_path, f'face_same_grid.jpg')
            same_image_grid.save(same_image_name)

        if len(diff_image_list) > 0:
            zipped = list(zip(diff_image_list, diff_score_list))
            # 根据list1的排序方式对元组列表进行排序
            zipped.sort(key=lambda x: x[1])
            # 将排序后的元组列表解包成两个列表
            diff_image_list, diff_score_list = zip(*zipped)
            diff_image_grid = torch.stack(diff_image_list, dim=0)
            diff_image_grid = make_grid(diff_image_grid, nrow=5)
            diff_image_grid = F.to_pil_image(diff_image_grid)
            diff_image_name = os.path.join(save_path, f'face_diff_grid.jpg')
            diff_image_grid.save(diff_image_name)

        # exit()
    # torch.
        
        # median_dict = {}
        # for idx, (key, img_score_list) in enumerate(sorted(score_dict.items())):
        #     json_path = key.replace('.jpg', '.json')
        #     with open(json_path, 'r') as f:
        #         meta_dict = json.load(f)
            
        #     verify_meta = {}
        #     verify_meta['per_head_score'] = img_score_list
        #     verify_meta['min_score_position'] = find_min_index(img_score_list)
        #     verify_meta['per_head_vote'] = vote_dict[key]
        #     verify_meta['max_vote_position'] = find_max_index(vote_dict[key])

        #     meta_dict['verify_meta'] = verify_meta
            # print(key, f"{len(img_score_list)} score: {find_min_index(img_score_list) + 1} vote: {meta_dict['max_vote_position'] + 1}")

            # img = cv2.imread(key)
            # vote_bbox = meta_dict['bbox_meta'][meta_dict['max_vote_position']]
            # score_bbox = meta_dict['bbox_meta'][meta_dict['min_score_position']]

            # save_vote_name = os.path.join(save_path, f'det_vote_{idx:03d}.jpg')
            # save_score_name = os.path.join(save_path, f'det_score_{idx:03d}.jpg')
            # visualize_detection(img, vote_bbox, save_path = save_vote_name)
            # visualize_detection(img, score_bbox, save_path = save_score_name)

            # 将列表转换为json字符串
            # json_str = json.dumps(meta_dict)

            # # 将json字符串保存到文件中
            # with open(json_path, 'w') as f:
            #     f.write(json_str)

    # print(score_dict)
    # print(output.shape)
    
        # print(output.size())
#         output = output.data.cpu().numpy()
#         dist = cosin_metric(output[0], output[1])
#         dist = np.arccos(dist) / math.pi * 180
#         print(f'{idx} - {dist} o : {basename}')
#         if dist < 1:
#             print(f'{basename} is almost identical to original.')
#             identical_count += 1
#         else:
#             dist_list.append(dist)

#     print(f'Result dist: {sum(dist_list) / len(dist_list):.6f}')
#     print(f'identical count: {identical_count}')



# model = init_recognition_model(model_name='arcface')
# print(model)

# similarity_score = (src_img_features @ gen_img_features.T).mean()