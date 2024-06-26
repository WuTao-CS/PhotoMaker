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
import re
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

with open('data-process/poco/018_log_clip_score.txt', 'r') as f:
    clip_score_lines = f.readlines()

clip_score_dict = {}
for line in clip_score_lines:
    img_path = line.split()[0]

    # 使用正则表达式匹配方括号中的内容
    list_pattern = re.compile(r'\[(.*?)\]')

    # 使用正则表达式提取list值
    list_values = list_pattern.findall(line)

    if list_values:
        # 将找到的list值转换为Python列表
        list_values = list_values[0].split(',')
        list_values = [float(value.strip()) for value in list_values]

    clip_score_dict[img_path] = list_values

# print(clip_score_dict)

with open("data-process/poco/018_best_matched_word_per_folder.txt", "r") as f:
    folder_info = f.readlines()

logger_path = 'data-process/poco/019_bug_caption_position.txt'
if os.path.exists(os.path.join(logger_path)):
    os.remove(os.path.join(logger_path))
    print(f"File '{os.path.join(logger_path)}' has been deleted.")
else:
    print(f"File '{os.path.join(logger_path)}' does not exist.")

folder_word_dict = {}
for line in folder_info:
    line = line.strip()
    folder_base_name = line.split()[0].strip()
    folder_word = line.split()[1].strip()
    folder_word_dict[folder_base_name] = folder_word

print(folder_word_dict)
# text_logger = open(os.path.join('data-process', 'check_caption.txt'), 'a+')

print(f"开始处理数据集...")
start=time.time()

for img_path, clip_score in tqdm(clip_score_dict.items()):
    json_path = img_path.replace('.png', '.json')
    print(f"{img_path}")
    with open(json_path) as f:
        meta_dict = json.load(f)
    folder_base_name = os.path.basename(os.path.dirname(img_path))
    folder_word = folder_word_dict[folder_base_name]
    clip_score_list = clip_score_dict[img_path]
    print("***********************")
    print(f"{folder_base_name} {folder_word}")
    # exit()
    caption = meta_dict['caption_coco_singular'].strip()

    doc = nlp(caption)
    noun_tokens = [token for token in doc if token.pos_ == "NOUN"]
    matched_noun_tokens = [token for token in noun_tokens if any([word in token.text.lower().split() for word in MATCHED_WORDS]) ]
    if not (len(clip_score_list) == len(matched_noun_tokens)):
        with open(os.path.join('data-process/poco/019_bug_caption_position.txt'), 'a+') as f:
            f.write(f"{img_path} | {len(clip_score_list)} does not equal to {len(matched_noun_tokens)}\n")

    matched_noun_tokens_idx = [token.idx for token in matched_noun_tokens]
    matched_words = list()
    for token in matched_noun_tokens:
        for word in MATCHED_WORDS:
            if word in token.text.lower().split():
                matched_words.append(word)

    matched_word_idx = [index for index, token in enumerate(matched_noun_tokens) if folder_word in token.text]
    if len(matched_word_idx) == 1:
        matched_word_idx = matched_word_idx[0]
        token = matched_noun_tokens[matched_word_idx]
        start_pos = token.idx
        end_pos = start_pos + len(token.text)
        meta_dict['end_pos'] = end_pos
        meta_dict['start_pos'] = start_pos
        json_str = json.dumps(meta_dict, indent=2)

        # # 将json字符串保存到文件中
        with open(json_path, 'w') as f:
            f.write(json_str)

    elif len(matched_word_idx) == 0:
        with open(os.path.join('data-process/poco/019_bug_caption_position.txt'), 'a+') as f:
            f.write(f"{img_path} | No trigger word found\n")
    else:
        best_score = 0
        for cur_idx in matched_word_idx:
            cur_clip_score = clip_score_list[cur_idx]
            if cur_clip_score > best_score:
                best_score = cur_clip_score
                best_idx = cur_idx

        token = matched_noun_tokens[best_idx]
        start_pos = token.idx
        end_pos = start_pos + len(token.text)
        meta_dict['end_pos'] = end_pos
        meta_dict['start_pos'] = start_pos
        json_str = json.dumps(meta_dict, indent=2)

        # # 将json字符串保存到文件中
        with open(json_path, 'w') as f:
            f.write(json_str)        
        
        # 优先匹配与matched word相符的
        # if folder_word in matched_words:
        #     find_idx = matched_words.index(folder_word)
        # # 如果不相符则测算sentenceformer score


end = time.time()

print(f"处理时间为: {end - start}")