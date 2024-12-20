import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from tqdm import tqdm
import locale
import numpy as np
import argparse
import requests
import concurrent.futures
import time
import glob
from PIL import Image, ImageFont, ImageDraw
from PIL import ImageFile

Image.MAX_IMAGE_PIXELS = 1000000000


def is_image_truncated(file_path):
    try:
        parser = ImageFile.Parser()
        with open(file_path, 'rb') as f:
            while True:
                data = f.read(1024)
                if not data:
                    break
                parser.feed(data)
            parser.close()
        return False  # 如果没有抛出异常，那么图像没有被截断
    except IOError:
        return True  # 如果抛出异常，那么图像被截断
        
# 创建一个 ArgumentParser 对象
parser = argparse.ArgumentParser(description='download images of imdb celebrities')

# 添加一个参数

parser.add_argument("--link-path",
    type=str,
    default="./projects/IDAdapter-diffusers/data-process/poco/truncated_images_poco_1.txt"
)
parser.add_argument("--save-path",
    type=str,
    default="./projects/IDAdapter-diffusers/data/poco_celeb_images"
)

# 解析命令行参数
args = parser.parse_args()

# 写入header
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
BASE_URL = 'https://www.imdb.com'

# if not os.path.exists(args.save_path):
os.makedirs(args.save_path, exist_ok=True)

df_list = []
meta_path = open(args.link_path, 'r')
failed_links_lines = meta_path.readlines()
for link in failed_links_lines:
    meta_dict = {}
    split_link = link.split()
    meta_dict['id'] = split_link[0]
    meta_dict['image_url'] = split_link[1].strip()
    df_list.append(meta_dict)

print(df_list)
# exit()
num_files = len(sorted(glob.glob(os.path.join('data-process/poco', 'failed_links*'))))
# It always have one base file
text_logger = open(os.path.join('data-process/poco', f'failed_links_{num_files-1}.txt'), 'a+')

print(f"开始下载")
start=time.time()
# 下载图片的函数
def download_image(meta_item):
    id_save_path = os.path.join(args.save_path, meta_item["id"])
    os.makedirs(id_save_path, exist_ok=True)
    completed_images = os.listdir(id_save_path)
    file_name = os.path.basename(meta_item['image_url'])
    # if file_name in completed_images:
    #     print(f'{file_name} 已经下载，所以跳过')
    # else:
    file_path = os.path.join(id_save_path, file_name)
    if is_image_truncated(file_path):
        print(f'{file_path} 图像截断，重新处理')
        response = requests.get('https:' + meta_item['image_url'], headers=headers)
        # filename = os.path.join(id_save_path, file_path)
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f'{file_path} 下载完成')
        return True
    else:
        print(f'{file_path} 图像未截断，所以跳过')
        return None


num_download = 0
# # 使用线程池并行下载图片
with tqdm(total=len(df_list)) as pbar:
    with concurrent.futures.ThreadPoolExecutor() as executor:
    #     # 将下载图片的函数提交给线程池，返回一个future对象列表
        future_to_url = {executor.submit(download_image, meta_item): meta_item for meta_item in df_list}
        for future in concurrent.futures.as_completed(future_to_url):
            # 获取已完成的future对象并输出结果
            meta_item = future_to_url[future]
            pbar.update()
            try:
                status = future.result()
                if status:
                    num_download += 1
                    with open(os.path.join('data-process/poco', 'truncated_images_poco_2.txt'), 'a+') as f:
                        f.write(f'{meta_item["id"]} {meta_item["image_url"]}\n')
            except Exception as exc:
                print(f'{meta_item["id"]} | {meta_item["image_url"]} 下载失败，原因为：{exc}')
                text_logger.write(f'{meta_item["id"]} {meta_item["image_url"]}\n')

text_logger.close()
end = time.time()

print(f"总下载数据量: {num_download}")
print(f"处理时间为: {end - start}")