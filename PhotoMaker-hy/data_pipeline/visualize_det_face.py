import cv2
import numpy as np
import os
import json


"""
获得脸部分辨率
获得占比
"""

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


def parse_path(path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']  # 支持的图像文件扩展名
    json_extensions = ['.json']  # 支持的json文件扩展名
    meta_group_paths = []  # 存储图像文件路径的列表

    # 遍历文件夹中的所有文件和子文件夹
    for root, dirs, files in os.walk(path):
        for filename in files:
            # 获取文件扩展名
            extension = os.path.splitext(filename)[1].lower()
            # 如果是图像文件，则将文件路径添加到列表中
            if extension in image_extensions:
                file_path = os.path.join(root, filename)
                meta_path = {}
                meta_path['img_path'] = file_path
                meta_path['json_path'] = file_path.replace(extension, '.json')
                meta_group_paths.append(meta_path)

    return meta_group_paths


src_path = 'data/poco_celeb_images/id20536947'

id_name = os.path.basename(src_path)

save_path = f'dump_data/det_poco_faces/{id_name}'
os.makedirs(save_path, exist_ok=True)
meta_group_paths = parse_path(src_path)
print(len(meta_group_paths))


for idx, meta_path in enumerate(meta_group_paths):
    img_path = meta_path['img_path']
    json_path = meta_path['json_path']
    img = cv2.imread(img_path)
    # print(img.shape)
    print(json_path)
    with open(json_path) as f:
        meta = json.load(f)
    
    bboxes = meta['bbox_meta']
    # bboxes = [meta['bbox']]

    visualize_detection(img, bboxes, save_path=os.path.join(save_path, f'det_{idx:03d}.jpg'))
    # exit()
