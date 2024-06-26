import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from tqdm import tqdm
import locale
import numpy as np
import argparse
import time
import json
import concurrent.futures

import spacy
from spacy.matcher import Matcher

# 使用inflection库将复数名词转换为单数形式
# 安装 inflection 库：pip install spacy inflection inflect
# python -m spacy download en_core_web_sm
import inflection
import inflect

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("merge_noun_chunks")

MATCHED_WORDS = ["man", "woman", "men", "women", "girl", "boy", "girls", "boys", "person", "lady", "ladies", "teen", "teens", "student", "students"]
FILTERED_WORDS = []

# 遍历文件夹中的所有文件和子文件夹
data_root='data/poco_celeb_images_cropped'

# img_path = 'data/imdb_celeb_images_cropped/0000000/MV5BMDc2ZDkzNmMtYzQ4ZS00NTAyLTkzNzAtNzQ5NDQ3ZGFjZDFmXkEyXkFqcGdeQXVyNjUwNzk3NDc@._V1_.png'
# json_path = img_path.replace('.png', '.json')
# with open(json_path) as f:
#     meta_dict = json.load(f)

# caption = meta_dict['caption_coco_singular']
# doc = nlp(caption)
# noun_tokens = [token for token in doc if token.pos_ == "NOUN"]
# noun_tokens_idx = [token.i for token in noun_tokens]
# matched_noun_tokens = [token for token in noun_tokens if any([word in token.text.lower().split() for word in MATCHED_WORDS]) ]
# if len(matched_noun_tokens) == 1:
#     token = matched_noun_tokens[0]
#     start_pos = token.idx
#     end_pos = start_pos + len(token.text)
# print(caption, end_pos)

# exit()
image_paths = []
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp'] 
num_total_images = 0
for root, dirs, files in os.walk(data_root):
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

# text_logger = open(os.path.join('data-process', 'check_caption.txt'), 'a+')


print(f"开始处理数据集...")
start=time.time()
# 下载图片的函数
def update_end_pos(img_path):
    json_path = img_path.replace('.png', '.json')
    with open(json_path) as f:
        meta_dict = json.load(f)

    caption = meta_dict['caption_coco_singular']
    doc = nlp(caption)
    noun_tokens = [token for token in doc if token.pos_ == "NOUN"]
    noun_tokens_idx = [token.i for token in noun_tokens]
    matched_noun_tokens = [token for token in noun_tokens if any([word in token.text.lower().split() for word in MATCHED_WORDS]) ]
    if len(matched_noun_tokens) == 1:
        token = matched_noun_tokens[0]
        start_pos = token.idx
        end_pos = start_pos + len(token.text)
        meta_dict['end_pos'] = end_pos
        json_str = json.dumps(meta_dict, indent=2)

        # # 将json字符串保存到文件中
        with open(json_path, 'w') as f:
            f.write(json_str)

# # 使用线程池并行下载图片
with tqdm(total=len(image_paths)) as pbar:
    with concurrent.futures.ThreadPoolExecutor() as executor:
    #     # 将下载图片的函数提交给线程池，返回一个future对象列表
        future_to_url = {executor.submit(update_end_pos, img_path): img_path for img_path in image_paths}
        for future in concurrent.futures.as_completed(future_to_url):
            # 获取已完成的future对象并输出结果
            img_path = future_to_url[future]
            pbar.update()
            try:
                future.result()
            except Exception as exc:
                print(f'{img_path} caption更新失败，原因为：{exc}')
                # text_logger.write(f'{img_path}\n')

text_logger.close()
end = time.time()

print(f"处理时间为: {end - start}")