import os
from tqdm import tqdm
import concurrent.futures
import time
import numpy as np
import json
# image_ids_path = os.path.join("./projects/IDAdapter-diffusers/data/ffhq_wild_files", "image_ids_train.txt")

source_path = "./projects/IDAdapter-diffusers/data/poco_celeb_images_cropped"
target_path = "./projects/IDAdapter-diffusers/data/poco_celeb_images_cropped_1024"


from PIL import Image

from PIL import PngImagePlugin
PngImagePlugin.MAX_TEXT_CHUNK = 1048576  # this is the current value
Image.MAX_IMAGE_PIXELS = 1000000000

image_paths = []
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp'] 
num_total_images = 0
for root, dirs, files in os.walk(source_path):
    for filename in files:
        # 获取文件扩展名
        extension = os.path.splitext(filename)[1].lower()
        # 如果是图像文件，则将文件路径添加到列表中
        if extension in image_extensions:
            num_total_images += 1
            file_path = os.path.join(root, filename)
            if os.path.exists(file_path.replace(extension, '.json')):
                image_paths.append(file_path)

print(f"数据集图像数量: {num_total_images}, 包含json文件数量: {len(image_paths)}")
image_paths = sorted(image_paths)

def get_mask_bbox(mask):
    mask_array = np.array(mask)

    # 计算非零像素的x和y坐标
    non_zero_pixels = np.nonzero(mask_array)

    y_coords = non_zero_pixels[0]
    x_coords = non_zero_pixels[1]

    # 计算边界框 (left, upper, right, lower)
    bbox = (x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max())
    return bbox

def parallel():
    text_logger = open(os.path.join('data-process/poco/022_failed_resize_imdb.txt'), 'a+')

    # os.path.join()
    print(f"开始下载")
    start=time.time()
    # 下载图片的函数
    def resize_and_save_img(image_path):
        dirname = os.path.basename(os.path.dirname(image_path))
        folder_name = os.path.join(target_path, dirname)
        os.makedirs(folder_name, exist_ok=True)

        save_img_path = os.path.join(folder_name, os.path.basename(image_path))
        mask_path = image_path.replace('.png', '.mask.png')
        json_path = image_path.replace('.png', '.json')
        save_mask_path = save_img_path.replace('.png', '.mask.png')
        save_json_path = save_img_path.replace('.png', '.json')

        if (os.path.exists(save_img_path) and Image.open(save_img_path)) and (os.path.exists(save_mask_path) and Image.open(save_mask_path)) and (os.path.exists(save_json_path)):
            print(f"{save_img_path}已经存在, 所以跳过...")
        else:
            with Image.open(image_path) as image:
                # 获取图像的宽度和高度
                width, height = image.size

                if max([width, height]) < 601:
                    with open('data-process/poco/022_undesired_resolution.txt', 'a+') as f:
                        f.write(f"{image_path}\n")
                    return

                # 计算缩放比例
                scale = 1024. / min(width, height)

                # 计算缩放后的宽度和高度
                new_width = int(width * scale)
                new_height = int(height * scale)

                # 缩放图像并保存
                resized_image = image.resize((new_width, new_height), resample=Image.BICUBIC)
                print(f"存储至: {save_img_path}")
                resized_image.save(save_img_path)
            
            with Image.open(mask_path) as mask:
                mask = mask.convert('L')
                resized_mask = mask.resize((new_width, new_height), resample=Image.NEAREST)
                resized_mask.save(save_mask_path)
                mask_bbox = get_mask_bbox(resized_mask)

            save_dict = {}
            with open(json_path, "r") as f:
                info_dict = json.load(f)
            save_dict['caption'] = info_dict['caption_coco_singular']
            save_dict['end_pos'] = info_dict['end_pos']
            save_dict['original_size'] =  info_dict['size_after_crop']
            save_dict['bbox'] = [int(bb * scale) for bb in info_dict['bbox_after_cropped']]
            save_dict['mask_bbox'] =  [int(bb) for bb in mask_bbox]

            json_str = json.dumps(save_dict, indent=2)

            # # 将json字符串保存到文件中
            with open(save_json_path, 'w') as f:
                f.write(json_str)

    # # 使用线程池并行下载图片
    with tqdm(total=len(image_paths)) as pbar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
        #     # 将下载图片的函数提交给线程池，返回一个future对象列表
            future_to_url = {executor.submit(resize_and_save_img, image_path): image_path for image_path in image_paths}
            for future in concurrent.futures.as_completed(future_to_url):
                # 获取已完成的future对象并输出结果
                image_path = future_to_url[future]
                pbar.update()
                try:
                    future.result()
                except Exception as exc:
                    print(f'{image_path} resize失败，原因为：{exc}')
                    text_logger.write(f'{image_path}\n')

    text_logger.close()
    end = time.time()

    print(f"处理时间为: {end - start}")


def quene():
    for image_path in tqdm(image_paths):
        dirname = os.path.basename(os.path.dirname(image_path))
        folder_name = os.path.join(target_path, dirname)
        os.makedirs(folder_name, exist_ok=True)

        save_img_path = os.path.join(folder_name, os.path.basename(image_path))
        mask_path = image_path.replace('.png', '.mask.png')
        json_path = image_path.replace('.png', '.json')
        save_mask_path = save_img_path.replace('.png', '.mask.png')
        save_json_path = save_img_path.replace('.png', '.json')

        if (os.path.exists(save_img_path) and Image.open(save_img_path)) and (os.path.exists(save_mask_path) and Image.open(save_mask_path)) and (os.path.exists(save_json_path)):
            print(f"{save_img_path}已经存在, 所以跳过...")
        else:
            with Image.open(image_path) as image:
                # 获取图像的宽度和高度
                width, height = image.size

                # 计算缩放比例
                scale = 1024. / min(width, height)

                # 计算缩放后的宽度和高度
                new_width = int(width * scale)
                new_height = int(height * scale)

                # 缩放图像并保存
                resized_image = image.resize((new_width, new_height), resample=Image.BICUBIC)
                print(f"存储至: {save_img_path}")
                resized_image.save(save_img_path)
            
            with Image.open(mask_path) as mask:
                mask = mask.convert('L')
                resized_mask = mask.resize((new_width, new_height), resample=Image.NEAREST)
                resized_mask.save(save_mask_path)
                mask_bbox = get_mask_bbox(resized_mask)

            save_dict = {}
            with open(json_path, "r") as f:
                info_dict = json.load(f)
            save_dict['caption'] = info_dict['caption_coco_singular']
            save_dict['end_pos'] = info_dict['end_pos']
            save_dict['original_size'] =  info_dict['size_after_crop']
            save_dict['bbox'] = [int(bb * scale) for bb in info_dict['bbox_after_cropped']]
            save_dict['mask_bbox'] = [int(bb) for bb in mask_bbox]

            json_str = json.dumps(save_dict, indent=2)

            # # 将json字符串保存到文件中
            with open(save_json_path, 'w') as f:
                f.write(json_str)   


if __name__ == '__main__':
    parallel()