import os
import torch
import torchvision
from torchvision.io import read_image, ImageReadMode
import glob
import json
import numpy as np
import random
from copy import deepcopy
from PIL import Image
import numpy as np
from torchvision import transforms as T
import torch
from collections import OrderedDict

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


class SegmentProcessor(torch.nn.Module):
    def forward(self, image, background, segmap, id, bbox):
        seg_h, seg_w = segmap.shape[0:]
        h, w = image.shape[1:]
        if seg_h != h:
            segmap = T.functional.resize(segmap.unsqueeze(0), (h, w), interpolation=T.InterpolationMode.NEAREST)
            segmap = segmap.squeeze(0)
        mask = segmap != id
        # print(background.shape, mask.shape)
        image[:, mask] = background[:, mask]
        h1, w1, h2, w2 = bbox
        return image[:, w1:w2, h1:h2]

    def get_background(self, image):
        raise NotImplementedError


class RandomSegmentProcessor(SegmentProcessor):
    # def get_background(self, image):
    #     background = torch.randint(
    #         0, 255, image.shape, dtype=image.dtype, device=image.device
    #     )
    #     return background
    def get_background(self, image):
        background = torch.zeros(
            image.shape, dtype=image.dtype, device=image.device
        )
        return background


def get_object_processor(object_background_processor="random"):
    if object_background_processor == "random":
        object_processor = RandomSegmentProcessor()
    else:
        raise ValueError(f"Unknown object processor: {object_background_processor}")
    return object_processor

class PadToSquare(torch.nn.Module):
    def __init__(self, fill=0, padding_mode="constant"):
        super().__init__()
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, image: torch.Tensor):
        _, h, w = image.shape
        if h == w:
            return image
        elif h > w:
            padding = (h - w) // 2
            image = torch.nn.functional.pad(
                image,
                (padding, padding, 0, 0),
                self.padding_mode,
                self.fill,
            )
        else:
            padding = (w - h) // 2
            image = torch.nn.functional.pad(
                image,
                (0, 0, padding, padding),
                self.padding_mode,
                self.fill,
            )
        return image


class CropTopSquare(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image: torch.Tensor):
        _, h, w = image.shape
        if h <= w:
            return image
        return image[:, :w, :]


class AlwaysCropTopSquare(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image: torch.Tensor):
        _, h, w = image.shape
        if h > w:
            return image[:, :w, :]
        else:  # h <= w
            return image[:, :, w // 2 - h // 2 : w // 2 + h // 2]


class RandomZoomIn(torch.nn.Module):
    def __init__(self, min_zoom=1.0, max_zoom=1.5):
        super().__init__()
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom

    def forward(self, image: torch.Tensor):
        zoom = torch.rand(1) * (self.max_zoom - self.min_zoom) + self.min_zoom
        original_shape = image.shape
        image = T.functional.resize(
            image,
            (int(zoom * image.shape[1]), int(zoom * image.shape[2])),
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True,
        )
        # crop top square
        image = CropTopSquare()(image)
        return image



class CenterCropOrPadSides(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image: torch.Tensor):
        _, h, w = image.shape
        if h > w:
            # pad sides with black
            padding = (h - w) // 2
            image = torch.nn.functional.pad(
                image,
                (padding, padding, 0, 0),
                "constant",
                0,
            )
            # resize to square
            image = T.functional.resize(
                image,
                (w, w),
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=True,
            )
        else:
            # center crop to square
            padding = (w - h) // 2
            image = image[:, :, padding : padding + h]
        return image

class TrainTransformWithSegmap(torch.nn.Module):
    def __init__(self, train_resolution=512):
        super().__init__()
        self.image_resize = T.Resize(
            train_resolution,
            interpolation=T.InterpolationMode.BILINEAR,
        )
        # self.segmap_resize = T.Resize(
        #     args.train_resolution,
        #     interpolation=T.InterpolationMode.NEAREST,
        # )
        self.flip = T.RandomHorizontalFlip()
        self.crop = CenterCropOrPadSides()

    def forward(self, image, segmap):
        image = self.image_resize(image)
        h, w = image.shape[1:]
        segmap = segmap.unsqueeze(0)
        # print(segmap.shape)
        segmap = T.functional.resize(segmap, (h, w), interpolation=T.InterpolationMode.NEAREST)
        # segmap = self.segmap_resize(segmap)
        # print(image.shape)
        # print(segmap.shape)
        image_and_segmap = torch.cat([image, segmap], dim=0)
        image_and_segmap = self.flip(image_and_segmap)
        image_and_segmap = self.crop(image_and_segmap)
        image = image_and_segmap[:3]
        segmap = image_and_segmap[3:]
        image = (image.float() / 127.5) - 1
        segmap = segmap.squeeze(0)
        return image, segmap

def get_object_transforms(object_resolution=512, no_object_augmentation=False):
    if no_object_augmentation:
        pre_augmentations = []
        augmentations = []
    else:
        pre_augmentations = [
            (
                "zoomin",
                T.RandomApply([RandomZoomIn(min_zoom=1.0, max_zoom=2.0)], p=0.5),
            ),
        ]

        augmentations = [
            (
                "rotate",
                T.RandomApply(
                    [
                        T.RandomAffine(
                            degrees=30, interpolation=T.InterpolationMode.BILINEAR
                        )
                    ],
                    p=0.75,
                ),
            ),
            ("jitter", T.RandomApply([T.ColorJitter(0.5, 0.5, 0.5, 0.5)], p=0.5)),
            ("blur", T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5)),
            ("gray", T.RandomGrayscale(p=0.1)),
            ("flip", T.RandomHorizontalFlip()),
            ("elastic", T.RandomApply([T.ElasticTransform()], p=0.5)),
        ]

    object_transforms = torch.nn.Sequential(
        OrderedDict(
            [
                *pre_augmentations,
                ("pad_to_square", PadToSquare(fill=0, padding_mode="constant")),
                (
                    "resize",
                    T.Resize(
                        (object_resolution, object_resolution),
                        interpolation=T.InterpolationMode.BILINEAR,
                    ),
                ),
                *augmentations,
                ("convert_to_float", T.ConvertImageDtype(torch.float32)),
            ]
        )
    )
    return object_transforms

class FFHQDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        tokenizer,
        uncondition_prob=0,
        text_drop_ratio=0.05, 
        image_drop_ratio=0.05,
        image_text_drop_ratio=0.05
    ):
        self.root = root
        self.tokenizer = tokenizer
        self.train_transforms = TrainTransformWithSegmap(train_resolution=512)
        self.object_transforms = get_object_transforms()
        self.object_processor = get_object_processor()
        self.uncondition_prob = uncondition_prob
        self.text_drop_ratio = text_drop_ratio
        self.text_drop_ratio = text_drop_ratio
        self.image_drop_ratio = image_drop_ratio
        self.image_text_drop_ratio = image_text_drop_ratio

        image_ids_path = os.path.join(root, "image_ids_insightface_highres.txt")
        with open(image_ids_path, "r") as f:
            self.image_ids = f.read().splitlines()

    def __len__(self):
        return len(self.image_ids)

    @torch.no_grad()
    def preprocess(self, target_image, target_mask, object_image, object_mask, segment, id_name):
        if len(segment.keys()) > 1:
            caption = segment["caption"]
            end_pos = int(segment["end"])
            cls_id = segment["id"]
            num_objects = len(object_image)
        else:
            caption = segment["caption"]
            end_pos = 0
            num_objects = 0
            cls_id = 0

        pixel_values, transformed_segmap = self.train_transforms(target_image, target_mask)
        object_pixel_values = []
        object_segmaps = []


        background = self.object_processor.get_background(object_image)
        h, w = object_image.shape[1:]
        object_image = self.object_processor(
            object_image, background, object_mask, id=cls_id, bbox=[0,0,w,h]
        )
        object_pixel_values = self.object_transforms(object_image)
       
        # TODO check if this is correct
        

        
        if num_objects > 0:
            padding_object_pixel_values = torch.zeros_like(object_pixel_values)
        else:
            padding_object_pixel_values = self.object_transforms(background)
            padding_object_pixel_values[:] = 0
        object_pixel_values = (object_pixel_values.float() -0.5) * 2
        object_pixel_values = object_pixel_values.to(
            memory_format=torch.contiguous_format
        ).float()
        rand_num = random.random()
        if rand_num < self.text_drop_ratio:
            caption = ""
        elif rand_num < (self.image_drop_ratio + self.text_drop_ratio):
            object_pixel_values = torch.zeros_like(object_pixel_values)
        elif rand_num < (self.image_drop_ratio + self.text_drop_ratio + self.image_text_drop_ratio):
            caption = ""
            object_pixel_values = torch.zeros_like(object_pixel_values)
        
        input_ids = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids.squeeze(0)
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "ref_pixel_values": object_pixel_values,
        }

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        chunk = image_id[:5]
        image_path = os.path.join(self.root, 'ffhq_wild_files', chunk, image_id + ".jpg")
        info_path = os.path.join(self.root, 'ffhq_wild_files', chunk, image_id + ".json")
        segmap_path = os.path.join(self.root, 'ffhq_wild_files', chunk, image_id + ".npy")
        target_image = read_image(image_path, mode=ImageReadMode.RGB)
        with open(info_path, "r") as f:
            info_dict = json.load(f)
        target_mask = torch.from_numpy(np.load(segmap_path))
        object_image = deepcopy(target_image)
        object_mask = deepcopy(target_mask)
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

        return self.preprocess(target_image, target_mask, object_image, object_mask, segment, int(image_id))
    
class FFHQProcessDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        tokenizer,
        phase,
        total,
        uncondition_prob=0,
        text_drop_ratio=0.05, 
        image_drop_ratio=0.05,
        image_text_drop_ratio=0.05
    ):
        self.root = root
        self.tokenizer = tokenizer
        self.train_transforms = TrainTransformWithSegmap(train_resolution=512)
        self.object_transforms = get_object_transforms(no_object_augmentation=True)
        self.object_processor = get_object_processor()
        self.uncondition_prob = uncondition_prob
        self.text_drop_ratio = text_drop_ratio
        self.text_drop_ratio = text_drop_ratio
        self.image_drop_ratio = image_drop_ratio
        self.image_text_drop_ratio = image_text_drop_ratio

        image_ids_path = os.path.join(root, "image_ids_insightface_highres.txt")
        with open(image_ids_path, "r") as f:
            self.image_ids = f.read().splitlines()
        
        per_devie_num = len(self.image_ids)/total
        start = int(phase*per_devie_num)
        end = int((phase+1)*per_devie_num)
        if end >= len(self.image_ids):
            self.image_ids = self.image_ids[start:]
        else:
            self.image_ids = self.image_ids[start:end]

    def __len__(self):
        return len(self.image_ids)

    @torch.no_grad()
    def preprocess(self, target_image, target_mask, object_image, object_mask, segment, id_name):
        if len(segment.keys()) > 1:
            caption = segment["caption"]
            end_pos = int(segment["end"])
            cls_id = segment["id"]
            num_objects = len(object_image)
        else:
            caption = segment["caption"]
            end_pos = 0
            num_objects = 0
            cls_id = 0

        pixel_values, transformed_segmap = self.train_transforms(target_image, target_mask)
        


        background = self.object_processor.get_background(object_image)
        h, w = object_image.shape[1:]

        object_image = self.object_processor(
            object_image, background, object_mask, id=cls_id, bbox=[0,0,w,h]
        )
        object_pixel_values = self.object_transforms(object_image)
        
        # TODO check if this is correct
        
        object_pixel_values = (object_pixel_values.float() -0.5) * 2
        object_pixel_values = object_pixel_values.to(
            memory_format=torch.contiguous_format
        ).float()
        
        input_ids = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids.squeeze(0)
        return {
            "pixel_values": pixel_values,
            "prompt": caption,
            "ref_pixel_values": object_pixel_values,
        }

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        chunk = image_id[:5]
        image_path = os.path.join(self.root, 'ffhq_wild_files', chunk, image_id + ".jpg")
        info_path = os.path.join(self.root, 'ffhq_wild_files', chunk, image_id + ".json")
        segmap_path = os.path.join(self.root, 'ffhq_wild_files', chunk, image_id + ".npy")
        target_image = read_image(image_path, mode=ImageReadMode.RGB)
        with open(info_path, "r") as f:
            info_dict = json.load(f)
        target_mask = torch.from_numpy(np.load(segmap_path))
        object_image = deepcopy(target_image)
        object_mask = deepcopy(target_mask)
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
        output = self.preprocess(target_image, target_mask, object_image, object_mask, segment, int(image_id))
        output['name'] = image_id

        return output
    


class FFHQVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root="datasets/ffhq/", 
        video_length = 16, 
        resolution = [512,512],
        with_scaling_factor=True, text_drop_ratio=0.05, image_drop_ratio=0.05, image_text_drop_ratio=0.05
    ):
        self.root = root
        self.video_length = video_length
        self.resolution = resolution
        with open(os.path.join(self.root, "processed_sd15_image_0.json"), 'r') as file:
            self.all_data = json.load(file)
        self.with_scaling_factor =with_scaling_factor
        self.text_drop_ratio = text_drop_ratio
        self.text_drop_ratio = text_drop_ratio
        self.image_drop_ratio = image_drop_ratio
        self.image_text_drop_ratio = image_text_drop_ratio

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        index = index % len(self.all_data)
        data = self.all_data[index]
        prompt = data["prompt"]
        process_data_path = data["path"]
        base_name = os.path.basename(process_data_path).split(".")[0]
        process_data_path = os.path.join(
            self.root,
            "processed_sd15",
            f"{base_name}.pt",
        )

        process_data = torch.load(process_data_path, map_location='cpu')
        frames = process_data["latent"].repeat(16, 1, 1, 1).permute(1, 0, 2, 3)
        if self.with_scaling_factor:
            frames = frames * 0.18215
        ref_images_latent = process_data["ref_images_latent"].permute(1, 0, 2, 3)
        if self.with_scaling_factor:
            ref_images_latent = ref_images_latent * 0.18215


        prompt_embeds = process_data["prompt_embeds"]
        rand_num = random.random()
        if rand_num < self.text_drop_ratio:
            prompt_embeds = torch.zeros_like(prompt_embeds)
        elif rand_num < (self.image_drop_ratio + self.text_drop_ratio):
            ref_images_latent = torch.zeros_like(ref_images_latent)
        elif rand_num < (self.image_drop_ratio + self.text_drop_ratio + self.image_text_drop_ratio):
            prompt_embeds = torch.zeros_like(prompt_embeds)
            ref_images_latent = torch.zeros_like(ref_images_latent)
        
        return {"video":frames, "prompt_embeds":prompt_embeds, "prompt":prompt, "ref_images_latent":ref_images_latent}