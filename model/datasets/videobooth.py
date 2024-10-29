import io
import os
import torch
import pandas
import decord
import random
import json

import numpy as np

import torchvision.transforms.functional as F
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
from einops import rearrange
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import torch
import random
import numbers


def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("clip should be Tensor. Got %s" % type(clip))

    if not clip.ndimension() == 4:
        raise ValueError("clip should be 4D. Got %dD" % clip.dim())

    return True


def crop(clip, i, j, h, w):
    """
    Args:
        clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
    """
    if len(clip.size()) != 4:
        raise ValueError("clip should be a 4D tensor")
    return clip[..., i : i + h, j : j + w]


def resize(clip, target_size, interpolation_mode):
    if len(target_size) != 2:
        raise ValueError(f"target size should be tuple (height, width), instead got {target_size}")
    return torch.nn.functional.interpolate(clip, size=target_size, mode=interpolation_mode, align_corners=False)


def resize_scale(clip, target_size, interpolation_mode):
    if len(target_size) != 2:
        raise ValueError(f"target size should be tuple (height, width), instead got {target_size}")
    _, _, H, W = clip.shape
    scale_ = target_size[0] / min(H, W)
    return torch.nn.functional.interpolate(clip, scale_factor=scale_, mode=interpolation_mode, align_corners=False)


def resized_crop(clip, i, j, h, w, size, interpolation_mode="bilinear"):
    """
    Do spatial cropping and resizing to the video clip
    Args:
        clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the cropped region.
        w (int): Width of the cropped region.
        size (tuple(int, int)): height and width of resized clip
    Returns:
        clip (torch.tensor): Resized and cropped clip. Size is (T, C, H, W)
    """
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    clip = crop(clip, i, j, h, w)
    clip = resize(clip, size, interpolation_mode)
    return clip


def center_crop(clip, crop_size):
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    h, w = clip.size(-2), clip.size(-1)
    th, tw = crop_size
    if h < th or w < tw:
        raise ValueError("height and width must be no smaller than crop_size")

    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))
    return crop(clip, i, j, th, tw)


def random_shift_crop(clip):
    '''
    Slide along the long edge, with the short edge as crop size
    '''
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    h, w = clip.size(-2), clip.size(-1)

    if h <= w:
        long_edge = w
        short_edge = h
    else:
        long_edge = h
        short_edge =w

    th, tw = short_edge, short_edge

    i = torch.randint(0, h - th + 1, size=(1,)).item()
    j = torch.randint(0, w - tw + 1, size=(1,)).item()
    return crop(clip, i, j, th, tw)


def to_tensor(clip):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
    """
    _is_tensor_video_clip(clip)
    if not clip.dtype == torch.uint8:
        raise TypeError("clip tensor should have data type uint8. Got %s" % str(clip.dtype))
    # return clip.float().permute(3, 0, 1, 2) / 255.0
    return clip.float() / 255.0


def normalize(clip, mean, std, inplace=False):
    """
    Args:
        clip (torch.tensor): Video clip to be normalized. Size is (T, C, H, W)
        mean (tuple): pixel RGB mean. Size is (3)
        std (tuple): pixel standard deviation. Size is (3)
    Returns:
        normalized clip (torch.tensor): Size is (T, C, H, W)
    """
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    if not inplace:
        clip = clip.clone()
    mean = torch.as_tensor(mean, dtype=clip.dtype, device=clip.device)
    print(mean)
    std = torch.as_tensor(std, dtype=clip.dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
    return clip


def hflip(clip):
    """
    Args:
        clip (torch.tensor): Video clip to be normalized. Size is (T, C, H, W)
    Returns:
        flipped clip (torch.tensor): Size is (T, C, H, W)
    """
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    return clip.flip(-1)


class RandomCropVideo:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: randomly cropped video clip.
                size is (T, C, OH, OW)
        """
        i, j, h, w = self.get_params(clip)
        return crop(clip, i, j, h, w)

    def get_params(self, clip):
        h, w = clip.shape[-2:]
        th, tw = self.size

        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()

        return i, j, th, tw

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class UCFCenterCropVideo:
    def __init__(
        self,
        size,
        interpolation_mode="bilinear",
    ):
        if isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError(f"size should be tuple (height, width), instead got {size}")
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode


    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: scale resized / center cropped video clip.
                size is (T, C, crop_size, crop_size)
        """
        clip_resize = resize_scale(clip=clip, target_size=self.size, interpolation_mode=self.interpolation_mode)
        clip_center_crop = center_crop(clip_resize, self.size)
        return clip_center_crop

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"

class KineticsRandomCropResizeVideo:
    '''
    Slide along the long edge, with the short edge as crop size. And resie to the desired size.
    '''
    def __init__(
            self,
            size,
            interpolation_mode="bilinear",
         ):
        if isinstance(size, tuple):
                if len(size) != 2:
                    raise ValueError(f"size should be tuple (height, width), instead got {size}")
                self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        clip_random_crop = random_shift_crop(clip)
        clip_resize = resize(clip_random_crop, self.size, self.interpolation_mode)
        return clip_resize


class CenterCropVideo:
    def __init__(
        self,
        size,
        interpolation_mode="bilinear",
    ):
        if isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError(f"size should be tuple (height, width), instead got {size}")
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode


    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: center cropped video clip.
                size is (T, C, crop_size, crop_size)
        """
        clip_center_crop = center_crop(clip, self.size)
        return clip_center_crop

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"


class NormalizeVideo:
    """
    Normalize the video clip by mean subtraction and division by standard deviation
    Args:
        mean (3-tuple): pixel RGB mean
        std (3-tuple): pixel RGB standard deviation
        inplace (boolean): whether do in-place normalization
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): video clip must be normalized. Size is (C, T, H, W)
        """
        return normalize(clip, self.mean, self.std, self.inplace)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, inplace={self.inplace})"


class ToTensorVideo:
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    """

    def __init__(self):
        pass

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
        """
        return to_tensor(clip)

    def __repr__(self) -> str:
        return self.__class__.__name__


class RandomHorizontalFlipVideo:
    """
    Flip the video clip along the horizontal direction with a given probability
    Args:
        p (float): probability of the clip being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Size is (T, C, H, W)
        Return:
            clip (torch.tensor): Size is (T, C, H, W)
        """
        if random.random() < self.p:
            clip = hflip(clip)
        return clip

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"

#  ------------------------------------------------------------
#  ---------------------  Sampling  ---------------------------
#  ------------------------------------------------------------
class TemporalRandomCrop(object):
	"""Temporally crop the given frame indices at a random location.

	Args:
		size (int): Desired length of frames will be seen in the model.
	"""

	def __init__(self, size):
		self.size = size

	def __call__(self, total_frames):
		rand_end = max(0, total_frames - self.size - 1)
		begin_index = random.randint(0, rand_end)
		end_index = min(begin_index + self.size, total_frames)
		return begin_index, end_index

class CenterCropResizeVideo:
    '''
    First use the short side for cropping length,
    center crop video, then resize to the specified size
    '''
    def __init__(
        self,
        size,
        interpolation_mode="bilinear",
    ):
        if isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError(f"size should be tuple (height, width), instead got {size}")
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: scale resized / center cropped video clip.
                size is (T, C, crop_size, crop_size)
        """
        clip_center_crop = center_crop_using_short_edge(clip)
        clip_center_crop_resize = resize(clip_center_crop, target_size=self.size, interpolation_mode=self.interpolation_mode)
        return clip_center_crop_resize

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"

def center_crop_using_short_edge(clip):
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    h, w = clip.size(-2), clip.size(-1)
    if h < w:
        th, tw = h, h
        i = 0
        j = int(round((w - tw) / 2.0))
    else:
        th, tw = w, w
        i = int(round((h - th) / 2.0))
        j = 0
    return crop(clip, i, j, th, tw)

class DecordInit(object):
    """Using Decord(https://github.com/dmlc/decord) to initialize the video_reader."""

    def __init__(self, num_threads=1):
        self.num_threads = num_threads
        self.ctx = decord.cpu()

    def __call__(self, filename):
        """Perform the Decord initialization.
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        reader = decord.VideoReader(filename,
                                    ctx=self.ctx,
                                    num_threads=self.num_threads)
        return reader

class VideoBoothDatasets(torch.utils.data.Dataset):
    """Load the WebVideo video files

    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(self):
        self.resolution = [512,512]
        self.video_data = self.get_videodata()
        self.transform = transforms.Compose([
            ToTensorVideo(),
            RandomHorizontalFlipVideo(),
            CenterCropResizeVideo((512,512)), # center crop using shor edge, then resize
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        self.num_frames = 16
        self.frame_stride = 4
        self.temporal_sample = TemporalRandomCrop(self.num_frames * self.frame_stride)

        self.first_frame_random_trans = transforms.Compose([
            transforms.Resize(self.resolution),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=30,
                translate=(0.1, 0.1),
                scale=(0.8, 1.2),
                fill=255)
        ])
        self.tokenizer = CLIPTokenizer.from_pretrained("./pretrain_model/stable-diffusion-v1-5", subfolder="tokenizer")
        self.parsing_dir = './datasets/VideoBoothDataset/webvid_parsing2M_final'
        self.webvid10M_dir = './datasets/WebVid/train_10M'

    def __getitem__(self, index):

        while True:
            try:
                # load video data
                video_info = self.video_data.iloc[index]
                video_id, video_page_dir, video_name, mask_dir = video_info['videoid'], video_info['page_dir'], video_info['name'], video_info['mask_dir']
                video_path = '{}/{}/{}.mp4'.format(self.webvid10M_dir, video_page_dir, video_id)

                v_reader = decord.VideoReader(video_path,ctx=decord.cpu())

                total_frames = len(v_reader)

                # Sampling video frames
                start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
                assert end_frame_ind - start_frame_ind >= self.num_frames
                frame_indice = np.linspace(start_frame_ind, end_frame_ind-1, self.num_frames, dtype=int)
                video = torch.from_numpy(v_reader.get_batch(frame_indice).asnumpy()).permute(0, 3, 1, 2).contiguous()

                # load image prompts
                parsing_mask_json_path  = f'{mask_dir}/{video_id}/mask.json'

                with open(parsing_mask_json_path) as f:
                    label_list = json.load(f)

                target_label = random.choice(label_list)
                mask_path = f'{self.parsing_dir}/{video_id}/mask_{target_label["value"]}.png'

                mask = np.array(Image.open(mask_path))
                first_frame = v_reader.get_batch([0]).asnumpy()[0]

                masked_first_frame = first_frame.copy()
                masked_first_frame[mask==0] = 255

                word_prompt = target_label['label']

                x1, y1, x2, y2 = target_label['box']

                # random crop the bbox
                augmentation_type = random.uniform(0, 1)
                if augmentation_type < 0.25:
                    y1 = y1 + random.uniform(0.01, 0.2) * (y2 - y1)
                elif augmentation_type < 0.50:
                    y2 = y2 - random.uniform(0.01, 0.2) * (y2 - y1)
                elif augmentation_type < 0.75:
                    x1 = x1 + random.uniform(0.01, 0.2) * (x2 - x1)
                elif augmentation_type < 1.0:
                    x2 = x2 - random.uniform(0.01, 0.2) * (x2 - x1)

                masked_first_frame = masked_first_frame[int(y1):int(y2), int(x1):int(x2), :]

                masked_first_frame = torch.from_numpy(masked_first_frame).permute(2, 0, 1).contiguous()
                height, width = masked_first_frame.size(1), masked_first_frame.size(2)

                if height == width:
                    pass
                elif height < width:
                    diff = width - height
                    top_pad = diff // 2
                    down_pad = diff - top_pad
                    left_pad = 0
                    right_pad = 0
                    padding_size = [left_pad, top_pad, right_pad, down_pad]
                    masked_first_frame = F.pad(masked_first_frame, padding=padding_size, fill = 255)
                else:
                    diff = height - width
                    left_pad = diff // 2
                    right_pad = diff - left_pad
                    top_pad = 0
                    down_pad = 0
                    padding_size = [left_pad, top_pad, right_pad, down_pad]
                    masked_first_frame = F.pad(masked_first_frame, padding=padding_size, fill = 255)

                masked_first_frame_ori = masked_first_frame.clone()

                aug_first_frame = self.first_frame_random_trans(masked_first_frame_ori).unsqueeze(0) / 127.5 - 1

                del v_reader
                break
            except Exception as e:
                print(e)
                index = random.randint(0, len(self.video_data) - 1)

        # videotransformer data proprecess
        video = self.transform(video) # T C H W
        video = torch.cat([video, aug_first_frame], dim = 0)
        input_ids = self.tokenizer(
            video_name, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids.squeeze(0)

        return {'video': video, 'input_ids': input_ids}

    def __len__(self):
        return len(self.video_data)

    def get_videodata(self, split='train'):

        target_animal_list = ['dog', 'cat', 'bear', 'car', 'panda', 'tiger', 'horse', 'elephant', 'lion']
        pandas_frame_list = []
        for target_animal in target_animal_list:
            parsing_dir = f'./datasets/VideoBoothDataset/{target_animal}'
            datainfo_path = f'./datasets/VideoBoothDataset/{target_animal}.csv'
            animal_video_data = pandas.read_csv(datainfo_path,  usecols=[3, 4, 6])
            animal_video_data['mask_dir'] = [parsing_dir] * len(animal_video_data)
            pandas_frame_list.append(animal_video_data)

        video_data = pandas.concat(pandas_frame_list)

        return video_data

if __name__ == '__main__':
    dataset = VideoBoothDatasets()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    for batch in dataloader:
        print(batch['video'].shape)
        print(batch['input_ids'].shape)
        torch.save(batch,'videobooth_emb.pt')
        break