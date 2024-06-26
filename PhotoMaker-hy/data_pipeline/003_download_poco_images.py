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
from PIL import Image, ImageFont, ImageDraw

Image.MAX_IMAGE_PIXELS = 1000000000

# 创建一个 ArgumentParser 对象
parser = argparse.ArgumentParser(description='download images of imdb celebrities')

# 添加一个参数

parser.add_argument("--meta-path",
    type=str,
    default="./projects/IDAdapter-diffusers/data-process/poco/poco_celebs_meta_0.csv"
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

df = pd.read_csv(args.meta_path)
print(f"包含ID数量：{df['id'].nunique()}")

df = df.sort_values(by='id')
print(f"原始url数量: {len(df)}")
# 删除包含缺失值的行
df = df.dropna(subset=['image_url'])
# 重置索引
df = df.reset_index(drop=True)
print(f"有效url数量: {len(df)}")
df_list = df.to_dict('records')

text_logger = open(os.path.join('data-process/poco', 'failed_links_poco.txt'), 'a+')

print(f"开始下载")
start=time.time()
# 下载图片的函数
def download_image(meta_item):
    id_save_path = os.path.join(args.save_path, f'{meta_item["id"]}')
    os.makedirs(id_save_path, exist_ok=True)
    completed_images = os.listdir(id_save_path)
    file_name = os.path.basename(meta_item['image_url'])
    if (file_name in completed_images):
        try:
            raw_image = Image.open(os.path.join(id_save_path, file_name))
            raw_image.verify()  # 验证图像，如果损坏，会抛出异常
        except:
            print(f'{file_name} 未下载完整，重新处理')
        else:
            print(f'{file_name} 已经下载，所以跳过')
            return None

    response = requests.get('https:' + meta_item['image_url'], headers=headers)
    filename = os.path.join(id_save_path, file_name)
    with open(filename, 'wb') as f:
        f.write(response.content)
    print(f'{filename} 下载完成')
    return True

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
                    with open(os.path.join('data-process/poco', 'corrupted_images_poco.txt'), 'a+') as f:
                        f.write(f'{meta_item["id"]} {meta_item["image_url"]}\n')
            except Exception as exc:
                print(f'{meta_item["id"]} | {meta_item["image_url"]} 下载失败，原因为：{exc}')
                text_logger.write(f'{meta_item["id"]} {meta_item["image_url"]}\n')

text_logger.close()
end = time.time()

print(f"总下载数据量: {num_download}")
print(f"处理时间为: {end - start}")