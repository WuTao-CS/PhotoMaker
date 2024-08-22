import os
import torch
import torchvision
from torchvision.io import read_image, ImageReadMode
from idadapter.utils import save_image
import glob
import json
import numpy as np
import random
from copy import deepcopy

def expand_bbox(bbox, img_width, img_height, n=1.5):
    x0, y0, x1, y1 = bbox
    center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
    new_w, new_h = (x1 - x0) * n, (y1 - y0) * n
    new_x0, new_y0 = center_x - new_w / 2, center_y - new_h / 2
    new_x1, new_y1 = center_x + new_w / 2, center_y + new_h / 2

    # check if the new bbox exceeds the image boundary
    if new_x0 < 0:
        new_x0 = 0
    if new_y0 < 0:
        new_y0 = 0
    if new_x1 > img_width:
        new_x1 = img_width
    if new_y1 > img_height:
        new_y1 = img_height

    return [int(new_x0), int(new_y0), int(new_x1), int(new_y1)]


def prepare_image_token_idx(image_token_mask, max_num_objects):
    image_token_idx = torch.nonzero(image_token_mask, as_tuple=True)[1]
    image_token_idx_mask = torch.ones_like(image_token_idx, dtype=torch.bool)
    if len(image_token_idx) < max_num_objects:
        image_token_idx = torch.cat(
            [
                image_token_idx,
                torch.zeros(max_num_objects - len(image_token_idx), dtype=torch.long),
            ]
        )
        image_token_idx_mask = torch.cat(
            [
                image_token_idx_mask,
                torch.zeros(
                    max_num_objects - len(image_token_idx_mask),
                    dtype=torch.bool,
                ),
            ]
        )

    image_token_idx = image_token_idx.unsqueeze(0)
    image_token_idx_mask = image_token_idx_mask.unsqueeze(0)
    return image_token_idx, image_token_idx_mask


class IDAdapterFileListDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        tokenizer,
        train_transforms,
        object_transforms,
        object_processor,
        device=None,
        max_num_objects=4,
        image_token="<|image|>",
        object_appear_prob=1,
        uncondition_prob=0,
        text_only_prob=0,
        object_resolution=224,
        use_multi_embeds=False,
        use_face_mask=False,
        enlarge_ratio=1,
        repeat_token=False,
    ):
        self.root = root
        self.tokenizer = tokenizer
        self.train_transforms = train_transforms
        self.object_transforms = object_transforms
        self.object_processor = object_processor
        self.object_resolution = object_resolution
        self.max_num_objects = max_num_objects
        self.image_token = image_token
        self.object_appear_prob = object_appear_prob
        self.device = device
        self.uncondition_prob = uncondition_prob
        self.text_only_prob = text_only_prob
        self.use_multi_embeds = use_multi_embeds
        self.use_face_mask = use_face_mask
        self.enlarge_ratio = enlarge_ratio
        self.repeat_token = repeat_token

        wallhaven_file_path = os.path.join(root, "wallhaven_celeb_filelist.txt")
        imdb_file_path = os.path.join(root, "imdb_celeb_filelist.txt")

        with open(wallhaven_file_path, 'r') as f:
            wallhaven_file_list = f.readlines()
            
        wallhaven_file_list = [item.strip() for item in wallhaven_file_list]

        with open(imdb_file_path, 'r') as f:
            imdb_file_list = f.readlines()  

        imdb_file_list = [item.strip() for item in imdb_file_list]

        self.file_list = sorted(imdb_file_list + wallhaven_file_list)
        tokenizer.add_tokens([image_token], special_tokens=True)
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)

        print(f"Using multi-embed: {use_multi_embeds} | Using face mask: {use_face_mask}")

    def __len__(self):
        return len(self.file_list)

    def _tokenize_and_mask_noun_phrases_ends(self, caption, end_pos, replace_token):
        """
            The two tokenizers share a single mask
        """
        if replace_token:
            caption = caption[:end_pos] + self.image_token + caption[end_pos:]

        input_ids = self.tokenizer.encode(caption)

        noun_phrase_end_mask = [False for _ in input_ids]
        clean_input_ids = []
        clean_index = 0

        for i, id in enumerate(input_ids):
            if id == self.image_token_id:
                noun_phrase_end_mask[clean_index - 1] = True
            else:
                clean_input_ids.append(id)
                clean_index += 1

        max_len = self.tokenizer.model_max_length


        if len(clean_input_ids) > max_len:
            clean_input_ids = clean_input_ids[:max_len]
        else:
            clean_input_ids = clean_input_ids + [self.tokenizer.pad_token_id] * (
                max_len - len(clean_input_ids)
            )

        if len(noun_phrase_end_mask) > max_len:
            noun_phrase_end_mask = noun_phrase_end_mask[:max_len]
        else:
            noun_phrase_end_mask = noun_phrase_end_mask + [False] * (
                max_len - len(noun_phrase_end_mask)
            )

        clean_input_ids = torch.tensor(clean_input_ids, dtype=torch.long)
        noun_phrase_end_mask = torch.tensor(noun_phrase_end_mask, dtype=torch.bool)
        return clean_input_ids.unsqueeze(0), noun_phrase_end_mask.unsqueeze(0)

    @torch.no_grad()
    def preprocess(self, target_image, target_mask, object_images, object_masks, object_bboxes, segment, id_name):
        caption = segment["caption"]
        end_pos = int(segment["end_pos"])
        num_objects = len(object_images)
        pixel_values, transformed_segmap = self.train_transforms(target_image, target_mask)
        object_pixel_values = []
        object_segmaps = []

        prob = random.random()
        if prob < self.uncondition_prob:
            caption = ""
            replace_token = False
            num_objects = 0
        elif prob < self.uncondition_prob + self.text_only_prob:
            replace_token = False
            num_objects = 0
        else:
            replace_token = True

        # TODO: compare pixel_values and target_image
        for oid, (object_image, object_mask) in enumerate(zip(object_images, object_masks)):
            background = self.object_processor.get_background(object_image)
            # object_mask = T.functional.resize(segmap, (self.object_resolution, self.object_resolution), interpolation=T.InterpolationMode.NEAREST)
            h, w = object_mask.shape
            cls_id = 255
            bbox = expand_bbox(object_bboxes[oid], w, h) if len(object_bboxes) > 0 else [0,0,w,h]
            # TODO: edit bbox width and length
            object_image = self.object_processor(
                object_image, background, object_mask, id=cls_id, bbox=bbox
            )
            object_transformed_pixel_values = self.object_transforms(object_image)
            object_pixel_values.append(object_transformed_pixel_values)
            # object_segmaps.append(object_mask == cls_id)

        input_ids, image_token_mask = self._tokenize_and_mask_noun_phrases_ends(
            caption, end_pos, replace_token
        )

        if replace_token and self.repeat_token:
            nonzero_idx = torch.nonzero(image_token_mask.flatten()).item()
            true_elements = torch.tensor([image_token_mask[:,nonzero_idx].item()] * (num_objects-1), dtype=torch.bool)
            image_token_mask = torch.cat((image_token_mask[:,:(nonzero_idx+1)], true_elements[None], image_token_mask[:,(nonzero_idx+1):]), dim=1)
            trigger_token = torch.tensor([input_ids[:,nonzero_idx].item()] * (num_objects-1), dtype=torch.long)
            input_ids = torch.cat((input_ids[:,:(nonzero_idx+1)], trigger_token[None], input_ids[:,(nonzero_idx+1):]), dim=1)

            max_len = self.tokenizer.model_max_length
            image_token_mask = image_token_mask[:, :max_len]
            input_ids = input_ids[:, :max_len]

        # num_objects = image_token_idx_mask.sum().item()
        object_pixel_values = object_pixel_values[:num_objects]
        # object_segmaps = object_segmaps[:num_objects]

        if num_objects > 0:
            padding_object_pixel_values = torch.zeros_like(object_pixel_values[0])
        else:
            padding_object_pixel_values = self.object_transforms(background)
            padding_object_pixel_values[:] = 0

        if num_objects < self.max_num_objects:
            object_pixel_values += [
                torch.zeros_like(padding_object_pixel_values)
                for _ in range(self.max_num_objects - num_objects)
            ]

        object_pixel_values = torch.stack(
            object_pixel_values
        )  # [max_num_objects, 3, 256, 256]
        object_pixel_values = object_pixel_values.to(
            memory_format=torch.contiguous_format
        ).float()

        object_segmaps = torch.stack(
            [transformed_segmap == cls_id]
        ).float()  # [max_num_objects, 256, 256]

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "image_token_mask": image_token_mask,
            "object_pixel_values": object_pixel_values,
            "object_segmaps": object_segmaps,
            "num_objects": torch.tensor(num_objects),
            "image_ids": torch.tensor(id_name),
        }

    def __getitem__(self, idx):
        image_path = self.file_list[idx]
        id_dir_path = os.path.dirname(image_path)
        id_name = os.path.basename(id_dir_path)
        full_mask_paths = sorted(list(glob.glob(os.path.join(id_dir_path, '*.mask.png'))))
        if self.use_multi_embeds:
            num_objects = random.randint(1, min(self.max_num_objects, len(full_mask_paths)))  # 生成1到列表长度之间的随机整数
        else:
            num_objects = 1  # 生成1到列表长度之间的随机整
        target_mask_path = image_path.replace('.png', '.mask.png')
        object_mask_paths = random.sample(full_mask_paths, num_objects)

        # get path
        target_image_path = image_path
        target_json_path = image_path.replace(".png", ".json")
        object_image_paths = list(map(lambda x: x.replace(".mask.png", ".png"), object_mask_paths))
        object_json_paths = list(map(lambda x: x.replace(".mask.png", ".json"), object_mask_paths))
        
        # read mask and images
        target_image = read_image(target_image_path, mode=ImageReadMode.RGB)
        target_mask = read_image(target_mask_path, mode=ImageReadMode.UNCHANGED).squeeze()
        object_images = [read_image(oimg_path, mode=ImageReadMode.RGB) for oimg_path in object_image_paths]
        object_masks = [read_image(omask_path, mode=ImageReadMode.UNCHANGED).squeeze() for omask_path in object_mask_paths]

        object_bboxes = []
        if self.use_face_mask:
            for ojson_path in object_json_paths:
                with open(ojson_path, "r") as f:
                    info_dict = json.load(f) 

                x0, y0, x1, y1 = info_dict['bbox']
                x0 = max(0, x0)
                y0 = max(0, y0)
                object_bboxes.append([x0, y0, x1, y1])          

        segment = {}
        # read key information
        with open(target_json_path, "r") as f:
            info_dict = json.load(f)
            segment['end_pos'] = info_dict['end_pos']
            # caption = info_dict['caption']
            # if 'caption_coco_singular' in info_dict.keys():
                # segment['caption'] = info_dict['caption_coco_singular']
            # else:
            segment['caption'] = info_dict['caption']
        
        # if self.device is not None:
        #     target_image = target_image.to(self.device)
        #     target_mask = target_mask.to(self.device)
        #     object_images = [img.to(self.device) for img in object_images]
        #     object_masks = [mask.to(self.device) for mask in object_masks]

        return self.preprocess(target_image, target_mask, object_images, object_masks, object_bboxes, segment, int(id_name))


class IDAdapterChineseFileListDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        tokenizer,
        train_transforms,
        object_transforms,
        object_processor,
        device=None,
        max_num_objects=4,
        image_token="<|image|>",
        object_appear_prob=1,
        uncondition_prob=0,
        text_only_prob=0,
        object_resolution=224,
        use_multi_embeds=False,
        use_face_mask=False,
        enlarge_ratio=1,
        repeat_token=False,
    ):
        self.root = root
        self.tokenizer = tokenizer
        self.train_transforms = train_transforms
        self.object_transforms = object_transforms
        self.object_processor = object_processor
        self.object_resolution = object_resolution
        self.max_num_objects = max_num_objects
        self.image_token = image_token
        self.object_appear_prob = object_appear_prob
        self.device = device
        self.uncondition_prob = uncondition_prob
        self.text_only_prob = text_only_prob
        self.use_multi_embeds = use_multi_embeds
        self.use_face_mask = use_face_mask
        self.enlarge_ratio = enlarge_ratio
        self.repeat_token = repeat_token

        poco_file_path = os.path.join(root, "poco_celeb_filelist.txt")
        weibo_file_path = os.path.join(root, "weibo_celeb_filelist.txt")

        with open(poco_file_path, 'r') as f:
            poco_file_list = f.readlines()
            
        poco_file_list = [item.strip() for item in poco_file_list]

        with open(weibo_file_path, 'r') as f:
            weibo_file_list = f.readlines()  

        weibo_file_list = [item.strip() for item in weibo_file_list]

        self.file_list = sorted(weibo_file_list + poco_file_list)
        tokenizer.add_tokens([image_token], special_tokens=True)
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)

        print(f"Using multi-embed: {use_multi_embeds} | Using face mask: {use_face_mask}")

    def __len__(self):
        return len(self.file_list)

    def _tokenize_and_mask_noun_phrases_ends(self, caption, end_pos, replace_token):
        """
            The two tokenizers share a single mask
        """
        if replace_token:
            caption = caption[:end_pos] + self.image_token + caption[end_pos:]

        input_ids = self.tokenizer.encode(caption)

        noun_phrase_end_mask = [False for _ in input_ids]
        clean_input_ids = []
        clean_index = 0

        for i, id in enumerate(input_ids):
            if id == self.image_token_id:
                noun_phrase_end_mask[clean_index - 1] = True
            else:
                clean_input_ids.append(id)
                clean_index += 1

        max_len = self.tokenizer.model_max_length


        if len(clean_input_ids) > max_len:
            clean_input_ids = clean_input_ids[:max_len]
        else:
            clean_input_ids = clean_input_ids + [self.tokenizer.pad_token_id] * (
                max_len - len(clean_input_ids)
            )

        if len(noun_phrase_end_mask) > max_len:
            noun_phrase_end_mask = noun_phrase_end_mask[:max_len]
        else:
            noun_phrase_end_mask = noun_phrase_end_mask + [False] * (
                max_len - len(noun_phrase_end_mask)
            )

        clean_input_ids = torch.tensor(clean_input_ids, dtype=torch.long)
        noun_phrase_end_mask = torch.tensor(noun_phrase_end_mask, dtype=torch.bool)
        return clean_input_ids.unsqueeze(0), noun_phrase_end_mask.unsqueeze(0)

    @torch.no_grad()
    def preprocess(self, target_image, target_mask, object_images, object_masks, object_bboxes, segment, id_name):
        caption = segment["caption"]
        end_pos = int(segment["end_pos"])
        num_objects = len(object_images)
        pixel_values, transformed_segmap = self.train_transforms(target_image, target_mask)
        object_pixel_values = []
        object_segmaps = []

        prob = random.random()
        if prob < self.uncondition_prob:
            caption = ""
            replace_token = False
            num_objects = 0
        elif prob < self.uncondition_prob + self.text_only_prob:
            replace_token = False
            num_objects = 0
        else:
            replace_token = True

        # TODO: compare pixel_values and target_image
        for oid, (object_image, object_mask) in enumerate(zip(object_images, object_masks)):
            background = self.object_processor.get_background(object_image)
            # object_mask = T.functional.resize(segmap, (self.object_resolution, self.object_resolution), interpolation=T.InterpolationMode.NEAREST)
            h, w = object_mask.shape
            cls_id = 255
            bbox = expand_bbox(object_bboxes[oid], w, h) if len(object_bboxes) > 0 else [0,0,w,h]
            # TODO: edit bbox width and length
            object_image = self.object_processor(
                object_image, background, object_mask, id=cls_id, bbox=bbox
            )
            object_transformed_pixel_values = self.object_transforms(object_image)
            object_pixel_values.append(object_transformed_pixel_values)
            # object_segmaps.append(object_mask == cls_id)

        input_ids, image_token_mask = self._tokenize_and_mask_noun_phrases_ends(
            caption, end_pos, replace_token
        )

        if replace_token and self.repeat_token:
            nonzero_idx = torch.nonzero(image_token_mask.flatten()).item()
            true_elements = torch.tensor([image_token_mask[:,nonzero_idx].item()] * (num_objects-1), dtype=torch.bool)
            image_token_mask = torch.cat((image_token_mask[:,:(nonzero_idx+1)], true_elements[None], image_token_mask[:,(nonzero_idx+1):]), dim=1)
            trigger_token = torch.tensor([input_ids[:,nonzero_idx].item()] * (num_objects-1), dtype=torch.long)
            input_ids = torch.cat((input_ids[:,:(nonzero_idx+1)], trigger_token[None], input_ids[:,(nonzero_idx+1):]), dim=1)

            max_len = self.tokenizer.model_max_length
            image_token_mask = image_token_mask[:, :max_len]
            input_ids = input_ids[:, :max_len]

        # num_objects = image_token_idx_mask.sum().item()
        object_pixel_values = object_pixel_values[:num_objects]
        # object_segmaps = object_segmaps[:num_objects]

        if num_objects > 0:
            padding_object_pixel_values = torch.zeros_like(object_pixel_values[0])
        else:
            padding_object_pixel_values = self.object_transforms(background)
            padding_object_pixel_values[:] = 0

        if num_objects < self.max_num_objects:
            object_pixel_values += [
                torch.zeros_like(padding_object_pixel_values)
                for _ in range(self.max_num_objects - num_objects)
            ]

        object_pixel_values = torch.stack(
            object_pixel_values
        )  # [max_num_objects, 3, 256, 256]
        object_pixel_values = object_pixel_values.to(
            memory_format=torch.contiguous_format
        ).float()

        object_segmaps = torch.stack(
            [transformed_segmap == cls_id]
        ).float()  # [max_num_objects, 256, 256]

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "image_token_mask": image_token_mask,
            "object_pixel_values": object_pixel_values,
            "object_segmaps": object_segmaps,
            "num_objects": torch.tensor(num_objects),
            "image_ids": id_name,
        }

    def __getitem__(self, idx):
        image_path = self.file_list[idx]
        id_dir_path = os.path.dirname(image_path)
        id_name = os.path.basename(id_dir_path)
        full_mask_paths = sorted(list(glob.glob(os.path.join(id_dir_path, '*.mask.png'))))
        if self.use_multi_embeds:
            # print(id_name)
            num_objects = random.randint(1, min(self.max_num_objects, len(full_mask_paths)))  # 生成1到列表长度之间的随机整数
        else:
        # TODO edit
            num_objects = 1  # 生成1到列表长度之间的随机整
        target_mask_path = image_path.replace('.png', '.mask.png')
        object_mask_paths = random.sample(full_mask_paths, num_objects)

        # get path
        target_image_path = image_path
        target_json_path = image_path.replace(".png", ".json")
        object_image_paths = list(map(lambda x: x.replace(".mask.png", ".png"), object_mask_paths))
        object_json_paths = list(map(lambda x: x.replace(".mask.png", ".json"), object_mask_paths))
        
        # read mask and images
        target_image = read_image(target_image_path, mode=ImageReadMode.RGB)
        target_mask = read_image(target_mask_path, mode=ImageReadMode.UNCHANGED).squeeze()
        object_images = [read_image(oimg_path, mode=ImageReadMode.RGB) for oimg_path in object_image_paths]
        object_masks = [read_image(omask_path, mode=ImageReadMode.UNCHANGED).squeeze() for omask_path in object_mask_paths]

        object_bboxes = []
        if self.use_face_mask:
            for ojson_path in object_json_paths:
                with open(ojson_path, "r") as f:
                    info_dict = json.load(f) 

                x0, y0, x1, y1 = info_dict['bbox']
                x0 = max(0, x0)
                y0 = max(0, y0)
                object_bboxes.append([x0, y0, x1, y1])          

        segment = {}
        # read key information
        with open(target_json_path, "r") as f:
            info_dict = json.load(f)
            segment['end_pos'] = info_dict['end_pos']
            # caption = info_dict['caption']
            # if 'caption_coco_singular' in info_dict.keys():
                # segment['caption'] = info_dict['caption_coco_singular']
            # else:
            segment['caption'] = info_dict['caption']
        
        # if self.device is not None:
        #     target_image = target_image.to(self.device)
        #     target_mask = target_mask.to(self.device)
        #     object_images = [img.to(self.device) for img in object_images]
        #     object_masks = [mask.to(self.device) for mask in object_masks]

        return self.preprocess(target_image, target_mask, object_images, object_masks, object_bboxes, segment, id_name)


class IDAdapterFFHQDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        tokenizer,
        train_transforms,
        object_transforms,
        object_processor,
        device=None,
        max_num_objects=4,
        image_token="<|image|>",
        object_appear_prob=1,
        uncondition_prob=0,
        text_only_prob=0,
        object_resolution=224,
        use_multi_embeds=False,
    ):
        self.root = root
        self.tokenizer = tokenizer
        self.train_transforms = train_transforms
        self.object_transforms = object_transforms
        self.object_processor = object_processor
        self.object_resolution = object_resolution
        self.max_num_objects = max_num_objects
        self.image_token = image_token
        self.object_appear_prob = object_appear_prob
        self.device = device
        self.uncondition_prob = uncondition_prob
        self.text_only_prob = text_only_prob
        self.use_multi_embeds = use_multi_embeds

        image_ids_path = os.path.join(root, "image_ids_highres.txt")
        with open(image_ids_path, "r") as f:
            self.image_ids = f.read().splitlines()

        tokenizer.add_tokens([image_token], special_tokens=True)
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)

        print(f"Using multi-embed: {use_multi_embeds}")

    def __len__(self):
        return len(self.image_ids)

    def _tokenize_and_mask_noun_phrases_ends(self, caption, end_pos, replace_token):
        """
            The two tokenizers share a single mask
        """
        if replace_token:
            caption = caption[:end_pos] + self.image_token + caption[end_pos:]

        input_ids = self.tokenizer.encode(caption)

        noun_phrase_end_mask = [False for _ in input_ids]
        clean_input_ids = []
        clean_index = 0

        for i, id in enumerate(input_ids):
            if id == self.image_token_id:
                noun_phrase_end_mask[clean_index - 1] = True
            else:
                clean_input_ids.append(id)
                clean_index += 1

        max_len = self.tokenizer.model_max_length

        if len(clean_input_ids) > max_len:
            clean_input_ids = clean_input_ids[:max_len]
        else:
            clean_input_ids = clean_input_ids + [self.tokenizer.pad_token_id] * (
                max_len - len(clean_input_ids)
            )

        if len(noun_phrase_end_mask) > max_len:
            noun_phrase_end_mask = noun_phrase_end_mask[:max_len]
        else:
            noun_phrase_end_mask = noun_phrase_end_mask + [False] * (
                max_len - len(noun_phrase_end_mask)
            )

        clean_input_ids = torch.tensor(clean_input_ids, dtype=torch.long)
        noun_phrase_end_mask = torch.tensor(noun_phrase_end_mask, dtype=torch.bool)
        return clean_input_ids.unsqueeze(0), noun_phrase_end_mask.unsqueeze(0)

    @torch.no_grad()
    def preprocess(self, target_image, target_mask, object_images, object_masks, segment, id_name):
        if len(segment.keys()) > 1:
            caption = segment["caption"]
            end_pos = int(segment["end"])
            cls_id = segment["id"]
            num_objects = len(object_images)
        else:
            caption = segment["caption"]
            end_pos = 0
            num_objects = 0
            cls_id = 0

        pixel_values, transformed_segmap = self.train_transforms(target_image, target_mask)
        object_pixel_values = []
        object_segmaps = []

        prob = random.random()
        if prob < self.uncondition_prob:
            caption = ""
            replace_token = False
            num_objects = 0
        elif prob < self.uncondition_prob + self.text_only_prob:
            replace_token = False
            num_objects = 0
        else:
            replace_token = True
            if len(segment.keys()) == 1:
                replace_token = False

        # TODO: compare pixel_values and target_image
        # if (len(caption) > 0) or (num_objects > 0):
        for object_image, object_mask in zip(object_images, object_masks):
            background = self.object_processor.get_background(object_image)
            # object_mask = T.functional.resize(segmap, (self.object_resolution, self.object_resolution), interpolation=T.InterpolationMode.NEAREST)
            h, w = object_image.shape[1:]
            # TODO: edit bbox width and length
            object_image = self.object_processor(
                object_image, background, object_mask, id=cls_id, bbox=[0,0,w,h]
            )
            object_transformed_pixel_values = self.object_transforms(object_image)
            object_pixel_values.append(object_transformed_pixel_values)
            # object_segmaps.append(object_mask == cls_id)

        input_ids, image_token_mask = self._tokenize_and_mask_noun_phrases_ends(
            caption, end_pos, replace_token
        )

        # image_token_idx, image_token_idx_mask = prepare_image_token_idx(
        #     image_token_mask, self.max_num_objects
        # )

        # num_objects = image_token_idx_mask.sum().item()
        object_pixel_values = object_pixel_values[:num_objects]
        # object_segmaps = object_segmaps[:num_objects]

        if num_objects > 0:
            padding_object_pixel_values = torch.zeros_like(object_pixel_values[0])
        else:
            padding_object_pixel_values = self.object_transforms(background)
            padding_object_pixel_values[:] = 0

        if num_objects < self.max_num_objects:
            object_pixel_values += [
                torch.zeros_like(padding_object_pixel_values)
                for _ in range(self.max_num_objects - num_objects)
            ]
            # object_segmaps += [
            #     torch.zeros_like(transformed_segmap)
            #     for _ in range(self.max_num_objects - num_objects)
            # ]

        object_pixel_values = torch.stack(
            object_pixel_values
        )  # [max_num_objects, 3, 256, 256]
        object_pixel_values = object_pixel_values.to(
            memory_format=torch.contiguous_format
        ).float()

        object_segmaps = torch.stack(
            [transformed_segmap == cls_id]
        ).float()  # [max_num_objects, 256, 256]

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "image_token_mask": image_token_mask,
            "object_pixel_values": object_pixel_values,
            "object_segmaps": object_segmaps,
            "num_objects": torch.tensor(num_objects),
            "image_ids": torch.tensor(id_name),
        }

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        chunk = image_id[:5]
        image_path = os.path.join(self.root + '_1024', chunk, image_id + ".jpg")
        info_path = os.path.join(self.root, chunk, image_id + ".json")
        segmap_path = os.path.join(self.root, chunk, image_id + ".npy")
        target_image = read_image(image_path, mode=ImageReadMode.RGB)
        with open(info_path, "r") as f:
            info_dict = json.load(f)
        target_mask = torch.from_numpy(np.load(segmap_path))
        object_images = [deepcopy(target_image)]
        object_masks = [deepcopy(target_mask)]
        # if self.device is not None:
        #     target_image = target_image.to(self.device)
        #     target_mask = target_mask.to(self.device)
        #     object_images = [img.to(self.device) for img in object_images]
        #     object_masks = [mask.to(self.device) for mask in object_masks]
        segments = info_dict['segments']
        segments = [
            segment
            for segment in segments
            if segment["coco_label"] in 'person'
        ]
        if 'segment_id' in info_dict.keys():
            segment_id = info_dict['segment_id']
            segment = segments[segment_id]
            segment['caption'] = info_dict['caption']
        else:
            segment = {}
            segment['caption'] = info_dict['caption']

        return self.preprocess(target_image, target_mask, object_images, object_masks, segment, int(image_id))


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.cat([example["input_ids"] for example in examples])
    image_ids = [example["image_ids"] for example in examples]

    image_token_mask = torch.cat([example["image_token_mask"] for example in examples])
    # image_token_idx = torch.cat([example["image_token_idx"] for example in examples])
    # image_token_idx_mask = torch.cat(
    #     [example["image_token_idx_mask"] for example in examples]
    # )

    object_pixel_values = torch.stack(
        [example["object_pixel_values"] for example in examples]
    )
    object_segmaps = torch.stack([example["object_segmaps"] for example in examples])

    num_objects = torch.stack([example["num_objects"] for example in examples])
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "image_token_mask": image_token_mask,
        "object_pixel_values": object_pixel_values,
        "object_segmaps": object_segmaps,
        "num_objects": num_objects,
        "image_ids": image_ids,
    }


def get_data_loader(dataset, batch_size, shuffle=True):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=8,
    )

    return dataloader


def get_concat_final_chinese_dataset(
            dataset_name,
            tokenizer,
            train_transforms,
            object_transforms,
            object_processor,
            device,
            max_num_objects,
            object_appear_prob,
            uncondition_prob,
            text_only_prob,
            use_multi_embeds,
            use_face_mask,
            repeat_token,
            ):

    from torch.utils.data.dataset import ConcatDataset

    data_root_1 = os.path.join(dataset_name, "ffhq_wild_files")

    train_dataset_1 = IDAdapterFFHQDataset(
        data_root_1,
        tokenizer,
        train_transforms,
        object_transforms,
        object_processor,
        device=device,
        max_num_objects=max_num_objects,
        object_appear_prob=object_appear_prob,
        uncondition_prob=uncondition_prob,
        text_only_prob=text_only_prob,
        use_multi_embeds=use_multi_embeds,
    )

    train_dataset_2 = IDAdapterFileListDataset(
        dataset_name,
        tokenizer,
        train_transforms,
        object_transforms,
        object_processor,
        device=device,
        max_num_objects=max_num_objects,
        object_appear_prob=object_appear_prob,
        uncondition_prob=uncondition_prob,
        text_only_prob=text_only_prob,
        use_multi_embeds=use_multi_embeds,
        use_face_mask=use_face_mask,
        repeat_token=repeat_token,
    )

    train_dataset_3 = IDAdapterChineseFileListDataset(
        dataset_name,
        tokenizer,
        train_transforms,
        object_transforms,
        object_processor,
        device=device,
        max_num_objects=max_num_objects,
        object_appear_prob=object_appear_prob,
        uncondition_prob=uncondition_prob,
        text_only_prob=text_only_prob,
        use_multi_embeds=use_multi_embeds,
        use_face_mask=use_face_mask,
        repeat_token=repeat_token,
    )

    train_dataset = ConcatDataset([train_dataset_1, train_dataset_2, train_dataset_3])
    print(f'ffhq len: {len(train_dataset_1)} | imdb len: {len(train_dataset_2)} | chinese len: {len(train_dataset_3)} | dataset len: {len(train_dataset)}')
    return train_dataset


def test_train_concat_dataset(args, tokenizer):
    from torch.utils.data.dataset import ConcatDataset

    args.object_background_processor = "random"
    args.train_resolution = 1024
    args.object_resolution = 224
    args.pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"

    train_transforms = get_train_transforms_with_segmap(args)
    object_transforms = get_object_transforms(args)
    object_processor = get_object_processor(args)

    data_root_1 = os.path.join(args.dataset_name, "ffhq_wild_files")

    train_dataset_1 = IDAdapterFFHQDataset(
        data_root_1,
        tokenizer,
        train_transforms,
        object_transforms,
        object_processor,
        device="cpu",
        max_num_objects=4,
        object_appear_prob=0.9,
        uncondition_prob=0.1,
        text_only_prob=0,
        use_multi_embeds=True,
    )

    train_dataset_2 = IDAdapterFileListDataset(
        args.dataset_name,
        tokenizer,
        train_transforms,
        object_transforms,
        object_processor,
        device="cpu",
        max_num_objects=4,
        object_appear_prob=0.9,
        uncondition_prob=0.1,
        text_only_prob=0,
        use_multi_embeds=True,
        use_face_mask=False,
        repeat_token=True,
    )

    train_dataset_3 = IDAdapterChineseFileListDataset(
        args.dataset_name,
        tokenizer,
        train_transforms,
        object_transforms,
        object_processor,
        device="cpu",
        max_num_objects=4,
        object_appear_prob=0.9,
        uncondition_prob=0.1,
        text_only_prob=0,
        use_multi_embeds=True,
        use_face_mask=False,
        repeat_token=True,
    )

    train_dataset = ConcatDataset([train_dataset_1, train_dataset_2, train_dataset_3])
    print(f'ffhq len: {len(train_dataset_1)} | imdb len: {len(train_dataset_2)} | chinese len: {len(train_dataset_3)} | dataset len: {len(train_dataset)}')
    return train_dataset


def draw_text_on_image(text, img_size, font_size):
    from PIL import Image, ImageDraw, ImageFont
    import textwrap


    width, height = img_size
    # 创建一个白底图像
    image = Image.new("RGB", (width, height), "white")

    # 创建一个绘图对象
    draw = ImageDraw.Draw(image)

    # 设置字体
    font = ImageFont.truetype("data/calibri.ttf", font_size)
    # font = ImageFont.load_default()
    # font.getsize('test')
    # 渲染字符串
    x, y = 50, 50
    # line_height = font.getsize("hg")[1]  # 获取行高
    line_height = font.getlength("hg")  # 获取行高
    for line in textwrap.wrap(text, width=width-128):
        draw.text((x, y), line, font=font, fill="black")
        y += line_height
    
    return image

def test_test_dataset(args, tokenizer):
    test_reference_folder = "data/test"
    args.object_resolution = 224
    args.no_object_augmentation = True
    object_transforms = get_object_transforms(args)

    test_dataset = DemoSingleObjectDataset(
        test_reference_folder=test_reference_folder,
        tokenizer=tokenizer,
        object_transforms=object_transforms,
        device="cpu",
        max_num_objects=4,
    )
    
    # image_ids = os.listdir(test_reference_folder)
    # print(image_ids)
    # test_dataset.set_image_ids(image_ids)
    # test_dataset.next_caption()
    # test_dataset.next_caption()
    return test_dataset.get_data()


def test_test_multiembed_dataset(args, tokenizer):
    test_reference_folder = "data/test_multiembed/putin"
    args.object_resolution = 224
    args.no_object_augmentation = True
    object_transforms = get_object_transforms(args)

    test_dataset = DemoMultiObjectDataset(
        test_reference_folder=test_reference_folder,
        tokenizer=tokenizer,
        object_transforms=object_transforms,
        device="cpu",
        max_num_objects=4,
        padding_object_embed=True,
        repeat_token=True,
    )
    
    # image_ids = os.listdir(test_reference_folder)
    # print(image_ids)
    # test_dataset.set_image_ids(image_ids)
    test_dataset.set_reference_folder("data/test_multiembed/feifeili")
    # test_dataset.next_caption()
    # test_dataset.next_caption()
    return test_dataset.get_data()


if __name__ == "__main__":
    from transformers import CLIPTokenizer
    from idadapter.transforms import (
        get_train_transforms_with_segmap,
        get_object_transforms,
        get_object_processor,
    )
    from idadapter.utils import parse_args
    from idadapter.transforms import tensor_to_image

    args = parse_args()
    args.pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
    args.dataset_name = "./projects/IDAdapter-diffusers/data"

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        local_files_only=True,
    )

    debug_train = True
    visualize = True
    #### debug training data
    if debug_train:
        use_random = True
        dataset = test_train_concat_dataset(args, tokenizer)
        # dataset = test_train_imdb_dataset(args, tokenizer)
        if use_random:
            sample_index = random.sample(list(range(len(dataset))), 10)
        else:
            sample_index = list(range(10))

        # print(len(dataset))
        # data = dataset[27]
        for i, idx in enumerate(sample_index):
            data = dataset[idx]
            print(i, idx)
            # for i in range()
            object_pixel_values = data['object_pixel_values'][0].squeeze()
            print(data['object_pixel_values'].shape)
            print(data['input_ids'], data['image_token_mask'])
            # save visualization
            if visualize:
                tensor_to_image(object_pixel_values * 2 - 1, filename=f'dump_data/idadapter-data/object_{i}.jpg')
                tensor_to_image(data['pixel_values'], filename=f'dump_data/idadapter-data/target_{i}.jpg')

    else:
        dataset = test_test_multiembed_dataset(args, tokenizer)
        for idx, batch in enumerate(dataset):
            print(dataset[idx]['caption'])
            print(dataset[idx]['num_objects'])
            print(dataset[idx]['image_token_mask'])
            print(dataset[idx]['object_pixel_values'].shape)
            if visualize:
                object_pixel_values = dataset[idx]['object_pixel_values'][-1].squeeze()
                tensor_to_image(object_pixel_values * 2 - 1, filename=f'dump_data/idadapter-data/test_object_{idx}.jpg')
                # tensor_to_image(object_pixel_values * 2 - 1, filename=f'dump_data/idadapter-data/test_object_{i}.jpg')
        # print(dataset[0])
        # dataset = test_train_dataset(args, tokenizer)
        # print(dataset[0])
        # dataset = test_test_dataset(args, tokenizer)
        # for idx, batch in enumerate(dataset):
        #     print(batch["caption"])
        #     text = batch["caption"]
        #     image = draw_text_on_image(text, (1024,1024//2), 24)
            # torch.Size([1, 3, 224, 224])
            # torch.Size([1, 3, 224, 224])

    #### test data