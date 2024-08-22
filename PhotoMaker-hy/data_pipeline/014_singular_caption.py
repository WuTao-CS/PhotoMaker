
import spacy
from spacy.matcher import Matcher

# 使用inflection库将复数名词转换为单数形式
# 安装 inflection 库：pip install spacy inflection inflect
# python -m spacy download en_core_web_sm
import inflection
import inflect
import json
from tqdm import tqdm
import os

# MATCHED_WORDS = ["man", "woman", "men", "women", "girl", "boy", "girls", "boys", "person", "lady", "ladies", "teen", "teens", "student", "students"]

MATCHED_PLURAL_WORDS = ["man", "woman", "girl", "boy", "lady", "teen", "student"]

PLURAL_DICT = {
    "men": "man",
    "women": "woman",
    "girls": "girl",
    "boys": "boy",
    "ladies": "lady",
    "teens": "teen",
    "students": "student"
}

MATCHED_PLURAL_WORDS = PLURAL_DICT.keys()
MATCHED_SINGULAR_WORDS = PLURAL_DICT.keys()
# # 加载spacy英语模型
nlp = spacy.load("en_core_web_sm")
# # # import pdb; pdb.set_trace()
matcher = Matcher(nlp.vocab)
# matcher.add("WOMEN", [[{"LOWER": "women"}]])
# matcher.add("MEN", [[{"LOWER": "men"}]])
# matcher.add("GIRLS", [[{"LOWER": "women"}]])
# matcher.add("WOMEN", [[{"LOWER": "women"}]])
def replace_singular():
    with open("data-process/poco/014_update_without_caption.txt", "r") as f:
        meta_info = f.readlines()

    meta_dict = {}

    for item in meta_info:
        json_path = item.split('|')[0].lstrip('json_path: ').strip()
        caption = item.split('|')[1][10:]
        meta_dict[json_path] = caption


    # ### 
    # 1. 先找到演员名字，都统一替换为person
    # 2. 然后找复数词和对应的前面的量词，根据量词将复数词转化为单数形式，比如如果是two women，则转换为a woman and a woman
    # 3. 找到所有句子中，matched words对应的位置，如果一句话中存在多个matched words，则重新计算数字
    for json_path, caption in meta_dict.items():
        print("****************")
        print(caption)
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
        print(noun_tokens, noun_tokens_idx, plural_noun_tokens)
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



def main():
    data_root='data/poco_celeb_images_cropped'
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


    # ### 
    # 1. 先找到演员名字，都统一替换为person
    # 2. 然后找复数词和对应的前面的量词，根据量词将复数词转化为单数形式，比如如果是two women，则转换为a woman and a woman
    # 3. 找到所有句子中，matched words对应的位置，如果一句话中存在多个matched words，则重新计算数字
    for img_path in tqdm(image_paths):
        json_path = img_path.replace('.png', '.json')
        with open(json_path) as f:
            meta_dict = json.load(f)
        
        # if 'caption_coco_updated' in meta_dict.keys():
        #     print("using caption_coco_updated")
        #     caption = meta_dict['caption_coco_updated']
        # else:
        caption = meta_dict['caption']
        
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

        # # 将json字符串保存到文件中
        with open(json_path, 'w') as f:
            f.write(json_str)
    # dump_data_root = data_root + '_dumped_lowres_second'
    # os.makedirs(dump_data_root, exist_ok=True)

if __name__ == "__main__":
    # replace_singular()
    main()
