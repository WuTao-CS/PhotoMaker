import torch
import numpy as np
import json
import io
import os
import cv2
import math
import struct
from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw
from copy import deepcopy
# from openpose import OpenposeDetector
import glob
"""
    pip install scikit-image
    pip install -U openmim
    mim install mmengine
    mim install mmdet
"""
# apply_openpose = OpenposeDetector()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def visualize_detection(img, bboxes_and_landmarks, save_path=None, to_bgr=False):
    """Visualize detection results.

    Args:
        img (Numpy array): Input image. CHW, BGR, [0, 255], uint8.
    """
    img = np.copy(img)
    if to_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for idx, b in enumerate(bboxes_and_landmarks):
        if idx == len(bboxes_and_landmarks) - 1:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)
        # confidence
        # bounding boxes
        b = list(map(int, b))
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), color, 2)
    # save img
    if save_path is not None:
        cv2.imwrite(save_path, img)

# draw the body keypoint and lims
def draw_bodypose(canvas, candidate, subset):
    stickwidth = 4
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    # plt.imsave("preview.jpg", canvas[:, :, [2, 1, 0]])
    # plt.imshow(canvas[:, :, [2, 1, 0]])
    return canvas

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img


def process(input_image):
    with torch.no_grad():
        input_image = HWC3(input_image)
        ori_height, ori_width = input_image.shape[0:2]
        img = resize_image(input_image, 512)
        now_h, now_w = img.shape[0:2]
        w_scale = ori_width / now_w
        h_scale = ori_height / now_h

        detected_map, coordinate = apply_openpose(img)
        res = []
        for coor in detected_map:
            # print(coor[0])
            coor[0] = coor[0] * w_scale
            coor[1] = coor[1] * h_scale
            res.append(coor)

    return np.array(res), coordinate


def process_with_dect(input_image):
    with torch.no_grad():
        input_image = HWC3(input_image)
        ori_height, ori_width = input_image.shape[0:2]
        img = resize_image(input_image, 512)
        now_h, now_w = img.shape[0:2]
        w_scale = ori_width / now_w
        h_scale = ori_height / now_h
        human_bboxs = apply_openpose.get_bbox(img)
        res = []
        for pose_result in human_bboxs:
            x1, y1, x2, y2 = pose_result[0], pose_result[1], pose_result[2], pose_result[3]
            x1 = x1 * w_scale
            x2 = x2 * w_scale
            y1 = y1 * h_scale
            y2 = y2 * h_scale
            # x1 = max(0, x1 - (x2 - x1) * padding_scale)
            # x2 = min(ori_width, x2 + (x2 - x1) * padding_scale)
            # y1 = max(0, y1 - (y2 - y1) * padding_scale)
            # y2 = min(ori_height, y2 + (y2 - y1) * padding_scale)
            # if (x2 - x1) > 400 and (y2 - y1) > 400:
            res.append([x1, y1, x2, y2])

    return res


def process_keypoint(candidate, subset, img):
    n = len(subset)
    h, w = img.shape[0:2]
    bboxs = []
    for i in range(n):
        keypoints = subset[i]
        lh = h
        lw = w
        rh = 0
        rw = 0
        left = int(keypoints[8])
        right = int(keypoints[11])
        if int(keypoints[0]) == -1 or (int(keypoints[14]) == -1
                                       and int(keypoints[15] == -1)) or (int(keypoints[16]) == -1
                                                                         and int(keypoints[17] == -1)):
            continue
        max_h = 0
        if left >= 0:
            max_h = max(max_h, candidate[left][1])
        if right >= 0:
            max_h = max(max_h, candidate[right][1])
        if max_h == 0:
            max_h = h
        h = max_h
        for point in keypoints[0:18]:
            if point == -1:
                continue
            point = int(point)
            x, y = candidate[point][0:2]
            lh = min(lh, y)
            lw = min(lw, x)
            rh = max(rh, y)
            rw = max(rw, x)
        len_w = (rw - lw)
        len_h = (rh - lh)
        if len_w < 400 or len_h < 400:
            continue
        if (lh - len_h * 0.5 < 0):
            lh = 0
        else:
            lh = lh - len_h * 0.5
        if (rh + len_h * 0.2 > h):
            rh = h
        else:
            rh = rh + len_h * 0.2

        if (lw - len_w * 0.2 < 0):
            lw = 0
        else:
            lw = lw - len_w * 0.2
        if (rw + len_w * 0.2 > w):
            rw = w
        else:
            rw = rw + len_w * 0.2
        bboxs.append([lw, lh, rw, rh])
    return bboxs


def threshold_keypoint(subset):
    n = len(subset)
    if n == 1:
        return subset
    res = []
    for i in range(n):
        keypoints = subset[i]
        if int(keypoints[0]) == -1 or (int(keypoints[14]) == -1
                                       and int(keypoints[15] == -1)) or (int(keypoints[16]) == -1
                                                                         and int(keypoints[17] == -1)):
            continue
        if keypoints[-1] < 5:
            continue
        res.append(subset[i])
    return res

def computer_bbox_iou(human_bbox, face_bbox):
    """
    计算两个bbox的IoU，并判断face_bbox是否被完全包括住
    :param bbox1: 表示bbox1的列表，格式为 [x1, y1, x2, y2]
    :param bbox2: 表示bbox2的列表，格式为 [x1, y1, x2, y2]
    :return: 两个bbox的IoU
    """
    # 计算交集部分的面积
    x1 = max(human_bbox[0], face_bbox[0])
    y1 = max(human_bbox[1], face_bbox[1])
    x2 = min(human_bbox[2], face_bbox[2])
    y2 = min(human_bbox[3], face_bbox[3])
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    human_bbox_area = (human_bbox[2] - human_bbox[0]) * (human_bbox[3] - human_bbox[1])
    face_bbox_area = (face_bbox[2] - face_bbox[0]) * (face_bbox[3] - face_bbox[1])
    union_area = human_bbox_area + face_bbox_area - intersection_area
    
    # 计算IoU
    iou = intersection_area / union_area

    # 判断是否被完全覆盖
    if intersection_area < face_bbox_area:
        is_covered = False
    else:
        is_covered = True
    # 计算并集部分的面积
    return iou, is_covered
   
def filter_human_bboxes(human_bboxes, face_bbox):
    # if len(human_bboxes > 1):
    
    max_iou = 0
    best_bbox = []
    for human_bbox in (human_bboxes):
        hx1, hy1, hx2, hy2 = human_bbox
        cur_iou, is_covered = computer_bbox_iou(human_bbox, face_bbox)
        if not is_covered:
            continue
        if cur_iou > max_iou:
            best_bbox = [human_bbox]
            max_iou = cur_iou

    filtered_bbox = best_bbox
    return filtered_bbox


def is_point_in_bbox(point, bbox):
    """
    判断一个点是否在bbox内

    :param point: 点的坐标，格式为 (x, y)
    :param bbox: bbox的坐标，格式为 (x1, y1, x2, y2)
    :return: True表示点在bbox内，False表示点不在bbox内
    """
    x, y = point
    x1, y1, x2, y2 = bbox
    if x >= x1 and x <= x2 and y >= y1 and y <= y2:
        return True
    else:
        return False

def crop_based_on_face_bbox(ori_resolution, face_bbox):
    """
    计算bbox尺寸，长宽各扩大三倍
    """
    ori_height, ori_width = ori_resolution
    x1, y1, x2, y2 = face_bbox

    # 计算中心点坐标
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # 计算原始长和宽
    w = x2 - x1
    h = y2 - y1

    # 计算最长边, 将最长边视为新的bbox长度
    square_bbox_width = max(w, h) * 4

    # 更新crop的边长
    if square_bbox_width > min(ori_height, ori_width):
        square_bbox_width = min(ori_height, ori_width)

    # 计算新的左上角和右下角坐标
    x1_new = cx - square_bbox_width / 2
    y1_new = cy - square_bbox_width / 2
    x2_new = cx + square_bbox_width / 2
    y2_new = cy + square_bbox_width / 2

    # 如果bbox超出了图像边界，则将超出部分的坐标修改为图像边界的坐标
    if x1_new < 0:
        x2_new += abs(x1_new)
        x1_new = 0
    if y1_new < 0:
        y2_new += abs(y1_new)
        y1_new = 0
    if x2_new > ori_width:
        x1_new -= (x2_new - ori_width)
        x2_new = ori_width
    if y2_new > ori_height:
        y1_new -= (y2_new - ori_height)
        y2_new = ori_height

    return int(x1_new), int(y1_new), int(x2_new), int(y2_new)


def update_face_bbox(crop_bbox, face_bbox):
    c_x1, c_y1, _, _ = crop_bbox
    f_x1, f_y1, f_x2, f_y2 = face_bbox
    return f_x1 - c_x1, f_y1 - c_y1, f_x2 - c_x1, f_y2 - c_y1

if __name__ == '__main__':
    # os.makedirs(web_dst_tar_path, exist_ok=True)
    # root_path = 'weibo/1998858463/img/原创微博图片'
    root_path = './projects/IDAdapter-diffusers/data/poco_celeb_images'
    root_name = os.path.basename(root_path)
    folder_list = os.listdir(root_path)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp'] 

    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-6.7b-coco")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-6.7b-coco", torch_dtype=torch.float16
    )
    model.to(device)

    # folder_list = ["0003807"] # TODO
    for folder_name in tqdm(sorted(folder_list)):
        folder = os.path.join(root_path, folder_name)   
        files = os.listdir(folder)
        img_list = []
        for filename in files:
            img_ext = os.path.splitext(filename)[1].lower()
            # 如果是图像文件，则将文件路径添加到列表中
            if img_ext in image_extensions:
                img_list.append(os.path.join(folder, filename))

        for idx, img_path in enumerate(img_list):
            extension = os.path.splitext(img_path)[-1]

            # img_path = "./projects/IDAdapter-diffusers/data/imdb_celeb_images/0000001/MV5BMDNmMTRmOGEtNjM3NC00MGRlLTg3ZjctODg1Yjc1ZDhmOWZlXkEyXkFqcGdeQXVyMDM2NDM2MQ@@._V1_.jpg"
            json_path = img_path.replace(extension, '.json')
            print(img_path)
            try:
                with open(json_path) as f:
                    meta_dict = json.load(f)
            except:
                with open(os.path.join('data-process/poco', '013_failed_json.txt'), 'a+') as f:
                    f.write(f'{img_path}\n') 
                continue

            #####
            save_img_path = img_path.replace(extension, '.png').replace(root_name, root_name + '_cropped')
            save_json_path = save_img_path.replace('.png', '.json')
            # if os.path.exists(save_json_path):
            #     continue

            if 'bbox' not in meta_dict:
                with open(os.path.join('data-process/poco', '013_failed_keys.txt'), 'a+') as f:
                    f.write(f'{img_path}\n') 
                continue
            face_bbox = meta_dict["bbox"]
            # load image and detect person bbox
            instance_image = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
            ori_height, ori_width = instance_image.shape[0:2]
            crop_bbox = crop_based_on_face_bbox([ori_height, ori_width], face_bbox)
            face_bbox = update_face_bbox(crop_bbox, face_bbox)

            # crop bbox
            # size before crop
            meta_dict['bbox_after_cropped'] = face_bbox
            meta_dict['crop_bbox'] = crop_bbox
            meta_dict['size_before_crop'] = [ori_height, ori_width]
            # 裁切图像
            crop_img_array = instance_image[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2]]
            meta_dict['size_after_crop'] = crop_img_array.shape[0:2]

            if not os.path.exists(os.path.dirname(save_img_path)):
                os.makedirs(os.path.dirname(save_img_path))
            cv2.imwrite(save_img_path, crop_img_array[:,:,::-1])

            # inference caption
            inputs = processor(images=Image.fromarray(crop_img_array), return_tensors="pt").to(device, torch.float16)

            generated_ids = model.generate(**inputs)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            print(generated_text)
            meta_dict["caption"] = generated_text

            json_str = json.dumps(meta_dict)

            # # 将json字符串保存到文件中
            with open(save_json_path, 'w') as f:
                f.write(json_str)
