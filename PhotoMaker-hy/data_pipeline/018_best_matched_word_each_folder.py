
import spacy
from spacy.matcher import Matcher

# 使用inflection库将复数名词转换为单数形式
# 安装 inflection 库：pip install spacy inflection inflect
# python -m spacy download en_core_web_sm
import inflection
import inflect
import json
from tqdm import tqdm
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import transformers
from torchmetrics.multimodal.clip_score import CLIPScore
from functools import partial
import torch
import os
import glob
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("merge_noun_chunks")
metric = CLIPScore( model_name_or_path="openai/clip-vit-large-patch14")

MATCHED_WORDS = ["man", "woman", "men", "women", "girl", "boy", "girls", "boys", "person", "lady", "ladies", "teen", "teens", "student", "students"]
# MATCHED_PLURAL_WORDS = ["man", "woman", "girl", "boy", "lady", "teen", "student"]


# PLURAL_DICT = {
#     "men": "man",
#     "women": "woman",
#     "girls": "girl",
#     "boys": "boy",
#     "ladies": "lady",
#     "teens": "teen",
#     "students": "student"
# }

# MATCHED_PLURAL_WORDS = list(PLURAL_DICT.keys())
# MATCHED_SINGULAR_WORDS = [PLURAL_DICT[key] for key in PLURAL_DICT.keys()]

# MATCHED_WORDS = MATCHED_PLURAL_WORDS + MATCHED_SINGULAR_WORDS

# # 加载spacy英语模型
# nlp = spacy.load("en_core_web_sm")
# # # import pdb; pdb.set_trace()
# matcher = Matcher(nlp.vocab)
# matcher.add("WOMEN", [[{"LOWER": "women"}]])
# matcher.add("MEN", [[{"LOWER": "men"}]])
# matcher.add("GIRLS", [[{"LOWER": "women"}]])
# matcher.add("WOMEN", [[{"LOWER": "women"}]])
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


def dependency_parsing():
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("merge_noun_chunks")

    with open("data-process/poco/018_update_without_caption.txt", "r") as f:
        meta_info = f.readlines()

    meta_dict = {}

    for item in meta_info:
        json_path = item.split('|')[0].lstrip('json_path: ').strip()
        caption = item.split('|')[1][10:]
        meta_dict[json_path] = caption

    stat_key = {}
    for idx, (json_path, caption) in enumerate(meta_dict.items()):   
        doc = nlp(caption)
        noun_tokens = [token for token in doc if token.pos_ == "NOUN"]
        noun_tokens_idx = [token.i for token in noun_tokens]
        matched_noun_tokens = [token for token in noun_tokens if any([word in token.text.lower().split() for word in MATCHED_WORDS]) ]
        # print(len(matched_noun_tokens))
        if len(matched_noun_tokens) > 1:
            print("****************")
            print(f"{idx}: {caption}")
            print(matched_noun_tokens)
            # 统计依存关系
            for token in matched_noun_tokens:
                print(token.text, token.pos_, token.dep_, [t for t in token.ancestors], [t for t in token.children])
                dependency = token.dep_
                if dependency in stat_key:
                    stat_key[dependency] += 1
                else:
                    stat_key[dependency] = 1
            
            # for token in doc:
            #     print(token.text, token.pos_, token.dep_, token.head.text)

    # print(stat_key)
    # 如果人是pobj和dobj的情况下 则往children节点寻找，children如果没有，就只截取一个单词，如果有找到有下一个人为止
    # 如果ROOT和nsubj是人，则找到下一个人出现为止
    # 如果conj是人，则找到下一个人出现为止

        # 遍历分析结果，输出每个单词的词性、依存关系和父节点
        # for token in matched_noun_tokens:
        #     print(token.text, token.pos_, token.dep_, token.head.text)

def find_dependency_for_matched_words():
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("merge_noun_chunks")

    with open("data-process/poco/018_duplicate_matched_words.txt", "r") as f:
        meta_info = f.readlines()

    caption_dict = {}

    # TODO
    for item in tqdm(meta_info[:10]):
        img_path = item.split('|')[0].strip()
        json_path = img_path.replace('.png', '.json')
        with open(json_path) as f:
            meta_dict = json.load(f)
        
        caption = meta_dict['caption_coco_singular'].strip()
        caption_dict[json_path] = caption

    # print()
    for idx, (json_path, caption) in enumerate(caption_dict.items()):   
        img_path = json_path.replace('.json', '.png')
        object_image = read_image_and_zero_background(img_path)
        object_image_tensor = torch.from_numpy(object_image)[None].permute(0, 3, 1, 2)

        doc = nlp(caption)
        len_caption = len(caption)
        noun_tokens = [token for token in doc if token.pos_ == "NOUN"]
        # noun_tokens_idx = [token.i for token in noun_tokens]
        matched_noun_tokens = [token for token in noun_tokens if any([word in token.text.lower().split() for word in MATCHED_WORDS]) ]
        matched_noun_tokens_idx = [token.idx for token in matched_noun_tokens]
        write_line = None
        # print(len(matched_noun_tokens))
        # get matched words list
        matched_words = list()
        for token in matched_noun_tokens:
            for word in MATCHED_WORDS:
                if word in token.text.lower().split():
                    matched_words.append(word)
                    
        if len(matched_noun_tokens) > 1:
            print("******************")
            print(img_path)
            print(caption)
            sub_caption_candidate = []
            caption_dependency_list = []
            for idx_tk, token in enumerate(matched_noun_tokens):
                caption_dependency_list.append(token.dep_)
                if (token.dep_ == 'pobj') or (token.dep_ == 'dobj'):
                    if len([t for t in token.children]) == 0:
                        sub_caption_candidate.append(token.text)
                # print(caption[start_idx: end_idx])
                    else:
                        start_idx = matched_noun_tokens_idx[idx_tk]
                        end_idx = len_caption if idx_tk == (len(matched_noun_tokens) -1) else matched_noun_tokens_idx[idx_tk+1]
                        sub_caption_candidate.append(caption[start_idx: end_idx])
                else:
                    start_idx = matched_noun_tokens_idx[idx_tk]
                    end_idx = len_caption if idx_tk == (len(matched_noun_tokens) -1) else matched_noun_tokens_idx[idx_tk+1]
                    sub_caption_candidate.append(caption[start_idx: end_idx].strip())                    
            
            # print(sub_caption_candidate)
            # print(caption_dependency_list)
            assert len(sub_caption_candidate) == len(matched_noun_tokens), f"{len(sub_caption_candidate)} != {len(matched_noun_tokens)}"
                # dependency = token.dep_
            
            clip_score_list = []
            for prompt in sub_caption_candidate:
                clip_score_list.append(metric(object_image_tensor, prompt).detach().item())
            clip_score_list.append(metric(object_image_tensor.repeat(len(sub_caption_candidate), 1, 1, 1), sub_caption_candidate).detach().item())
            print(sub_caption_candidate, clip_score_list)
            # if matched words are duplicated
            # if len(matched_words) > len(list(set(matched_words))):
            #     write_line = f'{img_path} | caption: {caption.strip()} | matched_noun_tokens: {matched_noun_tokens} | token pos & dep: {[(token.pos_, token.dep_) for token in matched_noun_tokens]} \n'
            #     print(write_line)
            #     if idx > 10:
            #         break

if __name__ == "__main__":
    # dependency_parsing()
    root_path='data/poco_celeb_images_cropped'
    folder_list = os.listdir(root_path)

    if os.path.exists('data-process/poco/018_best_matched_word_per_folder.txt'):
        with open('data-process/poco/018_best_matched_word_per_folder.txt', 'r') as f:
            cur_matched_results = f.readlines()

        completed_folder = []
        for result in cur_matched_results:
            completed_folder.append(result.split()[0])
        
        folder_list = list(set(folder_list) - set(completed_folder))

    for folder_name in tqdm(sorted(folder_list)):
        folder = os.path.join(root_path, folder_name)

        img_list = sorted(glob.glob(os.path.join(folder, '*.png')))
        mask_list = sorted(glob.glob(os.path.join(folder, '*.mask.png')))
        img_list = sorted(list(set(img_list) - set(mask_list)))

        stat_dict_per_folder = {}
        for word in MATCHED_WORDS:
            stat_dict_per_folder[word] = 0

        for idx, img_path in enumerate(img_list):
            json_path = img_path.replace('.png', '.json')
            with open(json_path) as f:
                meta_dict = json.load(f)
        
            caption = meta_dict['caption_coco_singular'].strip()
            doc = nlp(caption)
            len_caption = len(caption)
            noun_tokens = [token for token in doc if token.pos_ == "NOUN"]
            # noun_tokens_idx = [token.i for token in noun_tokens]
            matched_noun_tokens = [token for token in noun_tokens if any([word in token.text.lower().split() for word in MATCHED_WORDS]) ]
            matched_noun_tokens_idx = [token.idx for token in matched_noun_tokens]

            write_line = None
            # print(len(matched_noun_tokens))
            # get matched words list
            matched_words = list()
            for token in matched_noun_tokens:
                for word in MATCHED_WORDS:
                    if word in token.text.lower().split():
                        matched_words.append(word)
            
            if len(matched_words) == 1:
                stat_dict_per_folder[matched_words[0]] += 1
            elif len(matched_words) == 0:
                with open('data-process/poco/018_bug_no_any_matched_word.txt', 'a+') as f:
                    f.write(f'{img_path} {caption}\n')                
            else:
                sub_caption_candidate = []
                caption_dependency_list = []
                for idx_tk, token in enumerate(matched_noun_tokens):
                    caption_dependency_list.append(token.dep_)
                    if (token.dep_ == 'pobj') or (token.dep_ == 'dobj'):
                        if len([t for t in token.children]) == 0:
                            sub_caption_candidate.append(token.text)
                    # print(caption[start_idx: end_idx])
                        else:
                            start_idx = matched_noun_tokens_idx[idx_tk]
                            end_idx = len_caption if idx_tk == (len(matched_noun_tokens) -1) else matched_noun_tokens_idx[idx_tk+1]
                            sub_caption_candidate.append(caption[start_idx: end_idx])
                    else:
                        start_idx = matched_noun_tokens_idx[idx_tk]
                        end_idx = len_caption if idx_tk == (len(matched_noun_tokens) -1) else matched_noun_tokens_idx[idx_tk+1]
                        sub_caption_candidate.append(caption[start_idx: end_idx].strip())                    
                
                # print(sub_caption_candidate)
                # print(caption_dependency_list)
                assert len(sub_caption_candidate) == len(matched_noun_tokens), f"{len(sub_caption_candidate)} != {len(matched_noun_tokens)}"
                    # dependency = token.dep_
                
                object_image = read_image_and_zero_background(img_path)
                object_image_tensor = torch.from_numpy(object_image)[None].permute(0, 3, 1, 2)
                clip_score_list = []
                for prompt in sub_caption_candidate:
                    clip_score_list.append(metric(object_image_tensor, prompt).detach().item())
                # clip_score_list.append(metric(object_image_tensor.repeat(len(sub_caption_candidate), 1, 1, 1), sub_caption_candidate).detach().item())
                max_index = clip_score_list.index(max(clip_score_list))
                stat_dict_per_folder[matched_words[max_index]] += 1
                print(caption)                
                print(sub_caption_candidate, clip_score_list)
                with open('data-process/poco/018_log_clip_score.txt', 'a+') as f:  
                    f.write(f'{img_path} {clip_score_list}\n')    

        # predict the best word
        best_word = max(stat_dict_per_folder.items(), key=lambda x: x[1])[0]
        print(stat_dict_per_folder)
        print(f'{folder_name} {best_word}')
        with open('data-process/poco/018_best_matched_word_per_folder.txt', 'a+') as f:
            f.write(f'{folder_name} {best_word}\n')

    # find_dependency_for_matched_words()
    # import spacy

    # nlp = spacy.load("en_core_web_sm")
    # doc = nlp("Credit and mortgage account holders must submit their requests")

    # root = [token for token in doc if token.head == token][0]
    # print([token for token in doc if token.head == token], root, list(root.lefts))
    # subject = list(root.lefts)[0]
    # for descendant in subject.subtree:
    #     assert subject is descendant or subject.is_ancestor(descendant)
    #     print(descendant.text, descendant.dep_, descendant.n_lefts,
    #             descendant.n_rights,
    #             [ancestor.text for ancestor in descendant.ancestors])
    
    # nlp = spacy.load("en_core_web_sm")
    # nlp.add_pipe("merge_noun_chunks")

    # doc = nlp("Credit and mortgage account holders must submit their requests")
    # # span = doc[doc[4].left_edge.i : doc[4].right_edge.i+1]
    # # with doc.retokenize() as retokenizer:
    # #     retokenizer.merge(span)
    # for token in doc:
    #     print(token.text, token.pos_, token.dep_, token.head.text)
    # main()
