
import os
import cv2
import argparse
import json
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import torch
from copy import deepcopy
import pandas as pd
from tqdm import tqdm
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import spacy
from spacy.matcher import Matcher
import os 
import json
# 使用inflection库将复数名词转换为单数形式
# 安装 inflection 库：pip install spacy inflection inflect
# python -m spacy download en_core_web_sm
import inflection
import inflect
import time
from tqdm import tqdm
import concurrent.futures
import torch
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("merge_noun_chunks")
MATCHED_WORDS = ["man", "woman", "men", "women", "girl", "boy", "person"]
UNDESIRED_WORDS = ["actor", "actress", "json_path:"]

PLURAL_DICT = {
    "men": "man",
    "women": "woman",
    "girls": "girl",
    "boys": "boy",
    "ladies": "lady",
    "teens": "teen",
    "students": "student"
}

MATCHED_PLURAL_WORDS = list(PLURAL_DICT.keys())

def read_image_and_zero_background(img_path, return_pil=False):
    mask_path = img_path.replace('.png', '.mask.png')
    mask_image = np.array(Image.open(mask_path).convert("L"))
    ori_image = Image.open(img_path).convert("RGB")

    object_image = (mask_image > 0)[..., None] * ori_image
    if return_pil:
        object_image = Image.fromarray(object_image)
    # object_image.save('test.jpg')
    # exit()
    return object_image


def get_mask_bbox(mask):
    mask_array = np.array(mask)

    # 计算非零像素的x和y坐标
    non_zero_pixels = np.nonzero(mask_array)

    y_coords = non_zero_pixels[0]
    x_coords = non_zero_pixels[1]

    # 计算边界框 (left, upper, right, lower)
    bbox = (x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max())
    return bbox

def update_caption_with_bugs():

    with open("data-process/poco/016_bug_caption.txt", "r") as f:
        meta_info = f.readlines()

    meta_info = [item.split('|')[0].strip() for item in meta_info]

    logger_path = 'data-process/poco/016_update_caption_with_bugs.txt'

    if os.path.exists(os.path.join(logger_path)):
        os.remove(os.path.join(logger_path))
        print(f"File '{os.path.join(logger_path)}' has been deleted.")
    else:
        print(f"File '{os.path.join(logger_path)}' does not exist.")

    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-6.7b-coco")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-6.7b-coco", torch_dtype=torch.float16
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for img_path in tqdm(meta_info):
        json_path = img_path.replace(".png", ".json")
        mask_path  = img_path.replace(".png", ".mask.png")
        # with open(json_path) as f:
        #     meta_dict = json.load(f)

        object_image = read_image_and_zero_background(img_path, return_pil=True)
        # original_image = Image.open(img_path).convert("RGB")
        # mask_image = Image.open(mask_path).convert("L")
        # bbox = get_mask_bbox(mask_image)
        # bbox_image = original_image.crop(bbox)
        unfind_matched_caption = True
        while unfind_matched_caption:
            inputs = processor(images=object_image, return_tensors="pt").to(device, torch.float16)

            generated_ids = model.generate(**inputs, do_sample=True, num_return_sequences=5, max_length=70)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

            best_caption = []

            for caption in generated_text:
                caption = caption.lower().strip()
                doc = nlp(caption)
                noun_tokens = [token for token in doc if token.pos_ == "NOUN"]
                matched_noun_tokens = [token for token in noun_tokens if any([word in token.text.lower().split() for word in MATCHED_WORDS]) ]
                undesired_noun_tokens = [token for token in noun_tokens if any([word in token.text.lower().split() for word in UNDESIRED_WORDS]) ]
                # if_word_exists = any([word in caption.split() for word in MATCHED_WORDS])
                # if if_word_exists:
                #     best_caption.append(caption)
                if (len(matched_noun_tokens) > 0) and (len(undesired_noun_tokens) == 0):
                    best_caption.append(caption)
                    unfind_matched_caption = False

            print(best_caption)

        # if len(best_caption) > 0:
        with open(os.path.join(logger_path), 'a+') as f:
            f.write(f"{img_path} | caption: {best_caption[0]}\n")
        # else:
            # with open(os.path.join('data-process', logger_path), 'a+') as f:
                # f.write(f"json_path: {img_path.replace('.png', '.json')} | caption: No caption found\n")        


def record_json_without_words():
    text_logger = open(os.path.join('data-process/poco/016_bug_caption.txt'), 'a+')

    # 遍历文件夹中的所有文件和子文件夹
    data_root='data/poco_celeb_images_cropped'
    # img_path = './projects/IDAdapter-diffusers/data/imdb_celeb_images_cropped/0000000/MV5BMDc2ZDkzNmMtYzQ4ZS00NTAyLTkzNzAtNzQ5NDQ3ZGFjZDFmXkEyXkFqcGdeQXVyNjUwNzk3NDc@._V1_.png'
    # with open(img_path.replace('.png', '.json')) as f:
    #     meta_dict = json.load(f)
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

    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("merge_noun_chunks")

    print(f"开始处理数据集...")
    start=time.time()

    # 下载图片的函数
    def find_bug_caption(img_path):
        with open(img_path.replace('.png', '.json')) as f:
            meta_dict = json.load(f)
        
        caption = meta_dict['caption_coco_singular']
        # stat_key = {}
        # for idx, (json_path, caption) in enumerate(meta_dict.items()): 
        #     for idx, (json_path, caption) in enumerate(meta_dict.items()):   
    
        doc = nlp(caption)
        noun_tokens = [token for token in doc if token.pos_ == "NOUN"]
        noun_tokens_idx = [token.i for token in noun_tokens]
        matched_noun_tokens = [token for token in noun_tokens if any([word in token.text.lower().split() for word in MATCHED_WORDS]) ]
        undesired_noun_tokens = [token for token in noun_tokens if any([word in token.text.lower().split() for word in UNDESIRED_WORDS]) ]
        write_line = None
        # print(len(matched_noun_tokens))
        if len(matched_noun_tokens) == 0:
            write_line = f'{img_path} | caption: {caption.strip()}\n'
        elif len(undesired_noun_tokens) > 0:
            write_line = f'{img_path} | caption: {caption.strip()}\n'
        return write_line
    
    # # 使用线程池并行下载图片
    with tqdm(total=len(image_paths)) as pbar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
        #     # 将下载图片的函数提交给线程池，返回一个future对象列表
            future_to_url = {executor.submit(find_bug_caption, img_path): img_path for img_path in image_paths}
            for future in concurrent.futures.as_completed(future_to_url):
                # 获取已完成的future对象并输出结果
                img_path = future_to_url[future]
                pbar.update()
                try:
                    write_line = future.result()
                    if write_line is not None:
                        text_logger.write(write_line)
                except Exception as exc:
                    print(f'{img_path} caption处理失败，原因为：{exc}')
    

def update_bug_caption_with_singular():
    with open('data-process/poco/016_update_caption_with_bugs.txt', 'r') as f:
        caption_info = f.readlines()

    caption_dict = {}
    for item in caption_info:
        img_path = item.split('|')[0].strip()
        caption = item.split('|')[1][10:].strip()
        caption_dict[img_path] = caption

    for img_path, caption in tqdm(caption_dict.items()):
        json_path = img_path.replace('.png', '.json')
        print(json_path)
        with open(json_path) as f:
            meta_dict = json.load(f)
        
        doc = nlp(caption)
        # for ent in doc.ents:
        #     if ent.label_ == "PERSON":
        #         caption = caption.replace(ent.text, "a person")
        #         print(caption)
        # 找出名词及其在文本中的位置
        # doc[0]
        noun_tokens = [token for token in doc if token.pos_ == "NOUN"]
        noun_tokens_idx = [token.i for token in noun_tokens]
        plural_noun_tokens = [token for token in noun_tokens if (token.text.lower() in MATCHED_PLURAL_WORDS) ]
        # quantifiers = {}
        for token in plural_noun_tokens:
            if noun_tokens_idx.index(token.i) == 0:
                previous_noun_idx = -1
            else:
                previous_noun_idx = noun_tokens_idx[noun_tokens_idx.index(token.i) - 1]

            count_word = None 
            for idx_doc in range(token.i-1, previous_noun_idx, -1):
                if doc[idx_doc].pos_ == "NUM":
                    count_word = doc[idx_doc]
        
            if count_word is not None:
                replaced_caption = caption.replace(count_word.text, "a")
            else:
                replaced_caption = caption

            replaced_caption = replaced_caption.replace(token.text, PLURAL_DICT[token.text])            
            print(f"Before: {caption.strip()}")
            print(f"After: {replaced_caption.strip()}")
            caption = replaced_caption
        
        meta_dict['caption_coco_singular'] = caption
        json_str = json.dumps(meta_dict, indent=2)
        # print(json_path)
        # # 将json字符串保存到文件中
        with open(json_path, 'w') as f:
            f.write(json_str)

if __name__ == "__main__":
    # stage 1
    # record_json_without_words()
    # stage 2
    # update_caption_with_bugs()
    # stage 3
    update_bug_caption_with_singular()
