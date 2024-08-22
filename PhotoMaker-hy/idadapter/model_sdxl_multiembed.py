import torch
import torch.nn as nn
import accelerate
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.image_processor import VaeImageProcessor

from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPConfig
import torch.nn.functional as F
from typing import Any, Optional, Tuple, Union, Dict, List
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import (
    CLIPTextTransformer,
    CLIPPreTrainedModel,
    CLIPModel,
)

import types
import torchvision.transforms as T
import gc
import numpy as np

inference_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        if self.use_residual:
            x = x + residual
        return x


class IDAdapterCLIPImageEncoder(CLIPPreTrainedModel):
    """
        Add a new visual projection layer for increasing the dim
    """
    @staticmethod
    def from_pretrained(
        global_model_name_or_path,
        pretrained_path=None,
        global_model_name_or_path_2=None,
        use_clip_loss=False,
    ):
        if pretrained_path is not None:
            print(f"loading clip vision model from [{pretrained_path}]")
            clip_config = CLIPConfig.from_pretrained(global_model_name_or_path)
            model = CLIPModel(clip_config)
            state_dict = torch.load(pretrained_path)
            model.load_state_dict(state_dict)
        else:
            print(f"loading pretrained clip vision model")
            model = CLIPModel.from_pretrained(global_model_name_or_path)

        if global_model_name_or_path_2 is not None:
            model_2 = CLIPModel.from_pretrained(global_model_name_or_path_2)
            vision_model_2 = model_2.vision_model
            visual_projection_2 = model_2.visual_projection
        else:
            model_2 = None
            vision_model_2 = None
            visual_projection_2 = None   

        vision_model = model.vision_model
        visual_projection = model.visual_projection

        vision_processor = T.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )
        
        return IDAdapterCLIPImageEncoder(
            vision_model,
            visual_projection,
            vision_model_2,
            visual_projection_2,
            vision_processor,
            use_clip_loss=use_clip_loss,
        )

    def __init__(
        self,
        vision_model,
        visual_projection,
        vision_model_2,
        visual_projection_2,
        vision_processor,
        use_clip_loss,
    ):
        super().__init__(vision_model.config)
        self.vision_model = vision_model
        self.visual_projection = visual_projection
        self.vision_model_2 = vision_model_2
        if visual_projection_2 is not None:
            self.visual_projection_2 = visual_projection_2
        else:
            self.visual_projection_2 = nn.Linear(1024, 1280, bias=False)
        self.vision_processor = vision_processor

        self.image_size = vision_model.config.image_size
        self.use_clip_loss = use_clip_loss

    def forward(self, object_pixel_values):
        b, num_objects, c, h, w = object_pixel_values.shape

        object_pixel_values = object_pixel_values.view(b * num_objects, c, h, w)

        if h != self.image_size or w != self.image_size:
            h, w = self.image_size, self.image_size
            object_pixel_values = F.interpolate(
                object_pixel_values, (h, w), mode="bilinear", antialias=True
            )

        # Normalization
        object_pixel_values = self.vision_processor(object_pixel_values)

        shared_object_embeds = self.vision_model(object_pixel_values)[1]
        object_embeds = self.visual_projection(shared_object_embeds)
        object_embeds = object_embeds.view(b, num_objects, 1, -1)

        if self.vision_model_2 is not None:
            object_embeds_2 = self.vision_model_2(object_pixel_values)[1]
            object_embeds_2 = self.visual_projection_2(object_embeds_2)
        else:
            object_embeds_2 = self.visual_projection_2(shared_object_embeds)

        object_embeds_2 = object_embeds_2.view(b, num_objects, 1, -1)      
        if self.use_clip_loss:
            return torch.cat((object_embeds, object_embeds_2), dim=-1), shared_object_embeds
        return torch.cat((object_embeds, object_embeds_2), dim=-1)


def scatter_object_embeddings(
    inputs_embeds,
    image_token_mask,
    object_embeds,
    num_objects,
    image_embedding_transform=None,
):
    object_embeds = object_embeds.to(inputs_embeds.dtype)

    batch_size, max_num_objects = object_embeds.shape[:2]
    seq_length = inputs_embeds.shape[1]
    flat_object_embeds = object_embeds.view(
        -1, object_embeds.shape[-2], object_embeds.shape[-1]
    )

    valid_object_mask = (
        torch.arange(max_num_objects, device=flat_object_embeds.device)[None, :]
        < num_objects[:, None]
    )

    valid_object_embeds = flat_object_embeds[valid_object_mask.flatten()]

    if image_embedding_transform is not None:
        valid_object_embeds = image_embedding_transform(valid_object_embeds)

    inputs_embeds = inputs_embeds.view(-1, inputs_embeds.shape[-1])
    image_token_mask = image_token_mask.view(-1)
    valid_object_embeds = valid_object_embeds.view(-1, valid_object_embeds.shape[-1])

    inputs_embeds.masked_scatter_(image_token_mask[:, None], valid_object_embeds)
    inputs_embeds = inputs_embeds.view(batch_size, seq_length, -1)
    return inputs_embeds


def fuse_object_embeddings(
    inputs_embeds,
    image_token_mask,
    object_embeds,
    num_objects,
    fuse_fn=torch.add,
    pre_fn=None,
):
    # import pdb; pdb.set_trace()
    # object_embeds shape: [b, max_num_objects, 1, 2048]
    object_embeds = object_embeds.to(inputs_embeds.dtype)

    batch_size, max_num_objects = object_embeds.shape[:2]
    # seq_length: 77
    seq_length = inputs_embeds.shape[1]
    # flat_object_embeds shape: [b*max_num_objects, 1, 2048]
    flat_object_embeds = object_embeds.view(
        -1, object_embeds.shape[-2], object_embeds.shape[-1]
    )
    # valid_object_mask [b*max_num_objects]
    valid_object_mask = (
        torch.arange(max_num_objects, device=flat_object_embeds.device)[None, :]
        < num_objects[:, None]
    )
    # valid_object_embeds [num_o1+o2+o3]
    valid_object_embeds = flat_object_embeds[valid_object_mask.flatten()]

    splited_object_embeds = torch.split(valid_object_embeds, num_objects.tolist(), dim=0)
    splited_object_embeds = [torch.mean(embed, dim=0, keepdim=True) for embed in splited_object_embeds if embed.shape[0] > 0]
    if len(splited_object_embeds) > 0:
        valid_object_embeds = torch.cat(splited_object_embeds, dim=0)
    inputs_embeds = inputs_embeds.view(-1, inputs_embeds.shape[-1])
    image_token_mask = image_token_mask.view(-1)
    valid_object_embeds = valid_object_embeds.view(-1, valid_object_embeds.shape[-1])
    # slice out the image token embeddings
    image_token_embeds = inputs_embeds[image_token_mask]
    valid_object_embeds = fuse_fn(image_token_embeds, valid_object_embeds)
    # TODO: check dtype is identical for training original model
    assert image_token_mask.sum() == valid_object_embeds.shape[0], f"{image_token_mask.sum()} != {valid_object_embeds.shape[0]}"
    inputs_embeds.masked_scatter_(image_token_mask[:, None], valid_object_embeds.to(inputs_embeds.dtype))
    inputs_embeds = inputs_embeds.view(batch_size, seq_length, -1)
    return inputs_embeds


def linear_object_embeddings(
    inputs_embeds,
    image_token_mask,
    object_embeds,
    num_objects,
    fuse_fn=torch.add,
    pre_fn=None,
):
    # import pdb; pdb.set_trace()
    # object_embeds shape: [b, max_num_objects, 1, 2048]
    object_embeds = object_embeds.to(inputs_embeds.dtype)

    batch_size, max_num_objects = object_embeds.shape[:2]
    # seq_length: 77
    seq_length = inputs_embeds.shape[1]
    # flat_object_embeds shape: [b*max_num_objects, 1, 2048]
    flat_object_embeds = object_embeds.view(
        -1, object_embeds.shape[-2], object_embeds.shape[-1]
    )
    # valid_object_mask [b, max_num_objects]
    valid_object_mask = (
        torch.arange(max_num_objects, device=flat_object_embeds.device)[None, :]
        < num_objects[:, None]
    )
    # valid_object_embeds [num_o1+o2+o3]
    # valid_object_embeds = flat_object_embeds[valid_object_mask.flatten()]

    masked_object_embeds = object_embeds * valid_object_mask[...,None,None]
    masked_object_embeds = masked_object_embeds[num_objects.bool()]
    # if masked_object_embeds.shape[0] > 0:
    masked_object_embeds = masked_object_embeds.permute(0, 2, 3, 1)
    print(num_objects.bool(), masked_object_embeds.shape)
    masked_object_embeds = masked_object_embeds.reshape(masked_object_embeds.shape[0], masked_object_embeds.shape[-2], masked_object_embeds.shape[-1])
    valid_object_embeds = pre_fn(masked_object_embeds)
    valid_object_embeds = valid_object_embeds.permute(0, 2, 1)
    print(valid_object_embeds.shape)
    # else:
        # valid_object_embeds = flat_object_embeds[valid_object_mask.flatten()]
    valid_object_embeds = valid_object_embeds.view(-1, valid_object_embeds.shape[-1])

    inputs_embeds = inputs_embeds.view(-1, inputs_embeds.shape[-1])
    image_token_mask = image_token_mask.view(-1)
    valid_object_embeds = valid_object_embeds.reshape(-1, valid_object_embeds.shape[-1])
    # slice out the image token embeddings
    image_token_embeds = inputs_embeds[image_token_mask]
    valid_object_embeds = fuse_fn(image_token_embeds, valid_object_embeds)
    # TODO: check dtype is identical for training original model
    assert image_token_mask.sum() == valid_object_embeds.shape[0], f"{image_token_mask.sum()} != {valid_object_embeds.shape[0]}"
    inputs_embeds.masked_scatter_(image_token_mask[:, None], valid_object_embeds.to(inputs_embeds.dtype))
    inputs_embeds = inputs_embeds.view(batch_size, seq_length, -1)
    return inputs_embeds


def linearchannel_object_embeddings(
    inputs_embeds,
    image_token_mask,
    object_embeds,
    num_objects,
    fuse_fn=torch.add,
    pre_fn=None,
):
    # import pdb; pdb.set_trace()
    # object_embeds shape: [b, max_num_objects, 1, 2048]
    object_embeds = object_embeds.to(inputs_embeds.dtype)

    batch_size, max_num_objects = object_embeds.shape[:2]
    # seq_length: 77
    seq_length = inputs_embeds.shape[1]
    # flat_object_embeds shape: [b*max_num_objects, 1, 2048]
    flat_object_embeds = object_embeds.view(
        -1, object_embeds.shape[-2], object_embeds.shape[-1]
    )
    # valid_object_mask [b, max_num_objects]
    valid_object_mask = (
        torch.arange(max_num_objects, device=flat_object_embeds.device)[None, :]
        < num_objects[:, None]
    )
    # valid_object_embeds [num_o1+o2+o3]
    # valid_object_embeds = flat_object_embeds[valid_object_mask.flatten()]

    masked_object_embeds = object_embeds * valid_object_mask[...,None,None]
    masked_object_embeds = masked_object_embeds[num_objects.bool()]
    # if masked_object_embeds.shape[0] > 0:
    masked_object_embeds = masked_object_embeds.view(masked_object_embeds.shape[0], masked_object_embeds.shape[1] *  masked_object_embeds.shape[-1])
    print(num_objects.bool(), masked_object_embeds.shape)
    valid_object_embeds = pre_fn(masked_object_embeds)
    # else:
        # valid_object_embeds = flat_object_embeds[valid_object_mask.flatten()]
    # valid_object_embeds = valid_object_embeds.view(-1, valid_object_embeds.shape[-1])

    inputs_embeds = inputs_embeds.view(-1, inputs_embeds.shape[-1])
    image_token_mask = image_token_mask.view(-1)
    # slice out the image token embeddings
    image_token_embeds = inputs_embeds[image_token_mask]
    valid_object_embeds = fuse_fn(image_token_embeds, valid_object_embeds)
    # TODO: check dtype is identical for training original model
    assert image_token_mask.sum() == valid_object_embeds.shape[0], f"{image_token_mask.sum()} != {valid_object_embeds.shape[0]}"
    inputs_embeds.masked_scatter_(image_token_mask[:, None], valid_object_embeds.to(inputs_embeds.dtype))
    inputs_embeds = inputs_embeds.view(batch_size, seq_length, -1)
    return inputs_embeds


def postlinear_object_embeddings(
    inputs_embeds,
    image_token_mask,
    object_embeds,
    num_objects,
    fuse_fn=torch.add,
    pre_fn=None,
):
    # print(inputs_embeds, image_token_mask, object_embeds, num_objects)
    # import pdb; pdb.set_trace()
    # object_embeds shape: [b, max_num_objects, 1, 2048]
    object_embeds = object_embeds.to(inputs_embeds.dtype)

    batch_size, max_num_objects = object_embeds.shape[:2]
    # seq_length: 77
    seq_length = inputs_embeds.shape[1]
    # valid_object_mask [b, max_num_objects]
    valid_object_mask = (
        torch.arange(max_num_objects, device=object_embeds.device)[None, :]
        < num_objects[:, None]
    )
    
    # valid_object_embeds [num_o1+o2+o3]
    # valid_object_embeds = flat_object_embeds[valid_object_mask.flatten()]
    # filter out valid object embedding (based on num_objects)
    valid_object_mask = valid_object_mask[num_objects.bool()].flatten()
    object_embeds = object_embeds[num_objects.bool()]
    # flat_object_embeds shape: [b*max_num_objects, 1, 2048]
    flat_object_embeds = object_embeds.view(
        -1, object_embeds.shape[-2], object_embeds.shape[-1]
    )
    valid_object_embeds = flat_object_embeds.view(-1, flat_object_embeds.shape[-1])

    # filttr out input embeds
    inputs_embeds = inputs_embeds.view(-1, inputs_embeds.shape[-1])
    image_token_mask = image_token_mask.view(-1)
    # slice out the image token embeddings
    image_token_embeds = inputs_embeds[image_token_mask]

    # repeat image tokens
    # filter out valid object embedding (based on num_objects)
    splited_token_embeds = torch.split(image_token_embeds, (num_objects>0).int().tolist(), dim=0)
    splited_token_embeds = [embed.repeat(max_num_objects, 1) for embed in splited_token_embeds if embed.shape[0] > 0]  
    # concat splited tokens
    if len(splited_token_embeds) > 0:
        image_token_embeds = torch.cat(splited_token_embeds, dim=0)

    # fuse function
    valid_object_embeds = fuse_fn(image_token_embeds, valid_object_embeds)
    valid_object_embeds = valid_object_embeds * valid_object_mask[:, None]
    valid_object_embeds = valid_object_embeds.view(-1, 4, valid_object_embeds.shape[-1])
    valid_object_embeds = pre_fn[0](valid_object_embeds.transpose(1, 2))
    valid_object_embeds = pre_fn[1](valid_object_embeds.transpose(1, 2))
    valid_object_embeds = valid_object_embeds.view(-1, valid_object_embeds.shape[-1])   

    assert image_token_mask.sum() == valid_object_embeds.shape[0], f"{image_token_mask.sum()} != {valid_object_embeds.shape[0]}"
    inputs_embeds.masked_scatter_(image_token_mask[:, None], valid_object_embeds.to(inputs_embeds.dtype))
    inputs_embeds = inputs_embeds.view(batch_size, seq_length, -1)
    return inputs_embeds


def postlinearchannel_object_embeddings(
    inputs_embeds,
    image_token_mask,
    object_embeds,
    num_objects,
    fuse_fn=torch.add,
    pre_fn=None,
):
    # print(inputs_embeds, image_token_mask, object_embeds, num_objects)
    # import pdb; pdb.set_trace()
    # object_embeds shape: [b, max_num_objects, 1, 2048]
    object_embeds = object_embeds.to(inputs_embeds.dtype)

    batch_size, max_num_objects = object_embeds.shape[:2]
    # seq_length: 77
    seq_length = inputs_embeds.shape[1]
    # valid_object_mask [b, max_num_objects]
    valid_object_mask = (
        torch.arange(max_num_objects, device=object_embeds.device)[None, :]
        < num_objects[:, None]
    )
    
    # valid_object_embeds [num_o1+o2+o3]
    # valid_object_embeds = flat_object_embeds[valid_object_mask.flatten()]
    # filter out valid object embedding (based on num_objects)
    valid_object_mask = valid_object_mask[num_objects.bool()].flatten()
    object_embeds = object_embeds[num_objects.bool()]
    # flat_object_embeds shape: [b*max_num_objects, 1, 2048]
    flat_object_embeds = object_embeds.view(
        -1, object_embeds.shape[-2], object_embeds.shape[-1]
    )
    valid_object_embeds = flat_object_embeds.view(-1, flat_object_embeds.shape[-1])

    # filttr out input embeds
    inputs_embeds = inputs_embeds.view(-1, inputs_embeds.shape[-1])
    image_token_mask = image_token_mask.view(-1)
    # slice out the image token embeddings
    image_token_embeds = inputs_embeds[image_token_mask]

    # repeat image tokens
    # filter out valid object embedding (based on num_objects)
    splited_token_embeds = torch.split(image_token_embeds, (num_objects>0).int().tolist(), dim=0)
    splited_token_embeds = [embed.repeat(max_num_objects, 1) for embed in splited_token_embeds if embed.shape[0] > 0]  
    # concat splited tokens
    if len(splited_token_embeds) > 0:
        image_token_embeds = torch.cat(splited_token_embeds, dim=0)

    # fuse function
    valid_object_embeds = fuse_fn(image_token_embeds, valid_object_embeds)
    valid_object_embeds = valid_object_embeds * valid_object_mask[:, None]
    valid_object_embeds = valid_object_embeds.view(-1, 4, valid_object_embeds.shape[-1])
    valid_object_embeds = valid_object_embeds.view(-1, 4*valid_object_embeds.shape[-1])
    valid_object_embeds = pre_fn[0](valid_object_embeds)
    valid_object_embeds = pre_fn[1](valid_object_embeds)
    assert image_token_mask.sum() == valid_object_embeds.shape[0], f"{image_token_mask.sum()} != {valid_object_embeds.shape[0]}"
    inputs_embeds.masked_scatter_(image_token_mask[:, None], valid_object_embeds.to(inputs_embeds.dtype))
    inputs_embeds = inputs_embeds.view(batch_size, seq_length, -1)
    return inputs_embeds


def postaverage_object_embeddings(
    inputs_embeds,
    image_token_mask,
    object_embeds,
    num_objects,
    fuse_fn=torch.add,
    pre_fn=None,
):
    # import pdb; pdb.set_trace()
    # object_embeds shape: [b, max_num_objects, 1, 2048]
    object_embeds = object_embeds.to(inputs_embeds.dtype)

    batch_size, max_num_objects = object_embeds.shape[:2]
    # seq_length: 77
    seq_length = inputs_embeds.shape[1]
    # flat_object_embeds shape: [b*max_num_objects, 1, 2048]
    flat_object_embeds = object_embeds.view(
        -1, object_embeds.shape[-2], object_embeds.shape[-1]
    )
    # valid_object_mask [b*max_num_objects]
    valid_object_mask = (
        torch.arange(max_num_objects, device=flat_object_embeds.device)[None, :]
        < num_objects[:, None]
    )
    # valid_object_embeds [num_o1+o2+o3]
    valid_object_embeds = flat_object_embeds[valid_object_mask.flatten()]

    # inputs_embeds => [b*77, n]
    inputs_embeds = inputs_embeds.view(-1, inputs_embeds.shape[-1])
    image_token_mask = image_token_mask.view(-1)
    valid_object_embeds = valid_object_embeds.view(-1, valid_object_embeds.shape[-1])
    # slice out the image token embeddings
    image_token_embeds = inputs_embeds[image_token_mask]

    # repeat image tokens
    object_list = num_objects.tolist()
    # first type implementation
    # splited_token_embeds = []
    # select_b = 0
    # for cur_b in range(batch_size):
    #     cur_num_object = object_list[cur_b]
    #     if cur_num_object > 0:
    #         splited_token_embeds.append(image_token_embeds[select_b].repeat(cur_num_object))
    #         select_b += 1
    # second type implementation
    splited_token_embeds = torch.split(image_token_embeds, (num_objects>0).int().tolist(), dim=0)
    splited_token_embeds = [embed.repeat(num_o, 1) for num_o, embed in zip(object_list, splited_token_embeds) if embed.shape[0] > 0]  

    # concat splited tokens
    if len(splited_token_embeds) > 0:
        image_token_embeds = torch.cat(splited_token_embeds, dim=0)
    
    valid_object_embeds = fuse_fn(image_token_embeds, valid_object_embeds)
    splited_object_embeds = torch.split(valid_object_embeds, object_list, dim=0)
    splited_object_embeds = [torch.mean(embed, dim=0, keepdim=True) for embed in splited_object_embeds if embed.shape[0] > 0]
    if len(splited_object_embeds) > 0:
        valid_object_embeds = torch.cat(splited_object_embeds, dim=0)
    # TODO: check dtype is identical for training original model
    assert image_token_mask.sum() == valid_object_embeds.shape[0], f"{image_token_mask.sum()} != {valid_object_embeds.shape[0]}"
    inputs_embeds.masked_scatter_(image_token_mask[:, None], valid_object_embeds.to(inputs_embeds.dtype))
    inputs_embeds = inputs_embeds.view(batch_size, seq_length, -1)
    return inputs_embeds

def append_object_embeddings(
    inputs_embeds,
    image_token_mask,
    object_embeds,
    num_objects,
    fuse_fn=torch.add,
    pre_fn=None,
):
    # import pdb; pdb.set_trace()
    # object_embeds shape: [b, max_num_objects, 1, 2048]
    object_embeds = object_embeds.to(inputs_embeds.dtype)

    batch_size, max_num_objects = object_embeds.shape[:2]
    # seq_length: 77
    seq_length = inputs_embeds.shape[1]
    # flat_object_embeds shape: [b*max_num_objects, 1, 2048]
    flat_object_embeds = object_embeds.view(
        -1, object_embeds.shape[-2], object_embeds.shape[-1]
    )
    # valid_object_mask [b*max_num_objects]
    valid_object_mask = (
        torch.arange(max_num_objects, device=flat_object_embeds.device)[None, :]
        < num_objects[:, None]
    )
    valid_object_embeds = flat_object_embeds[valid_object_mask.flatten()]

    inputs_embeds = inputs_embeds.view(-1, inputs_embeds.shape[-1])
    image_token_mask = image_token_mask.view(-1)
    valid_object_embeds = valid_object_embeds.view(-1, valid_object_embeds.shape[-1])
    # slice out the image token embeddings
    image_token_embeds = inputs_embeds[image_token_mask]
    valid_object_embeds = fuse_fn(image_token_embeds, valid_object_embeds)
    # TODO: check dtype is identical for training original model
    assert image_token_mask.sum() == valid_object_embeds.shape[0], f"{image_token_mask.sum()} != {valid_object_embeds.shape[0]}"
    inputs_embeds.masked_scatter_(image_token_mask[:, None], valid_object_embeds.to(inputs_embeds.dtype))
    inputs_embeds = inputs_embeds.view(batch_size, seq_length, -1)
    return inputs_embeds

class IDAdapterPostfuseModule(nn.Module):
    def __init__(self, embed_dim, fuse_type, use_embed_loss=False, use_embed_clip_loss=False):
        super().__init__()
        self.mlp1 = MLP(embed_dim * 2, embed_dim, embed_dim, use_residual=False)
        self.mlp2 = MLP(embed_dim, embed_dim, embed_dim, use_residual=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.fuse_type = fuse_type
        self.use_embed_loss = use_embed_loss
        self.use_embed_clip_loss = use_embed_clip_loss

        self.pre_fn = None
        print(f"Use fuse type: {fuse_type}")
        if self.fuse_type == 'average':
            self.general_fuse_function = fuse_object_embeddings
        elif self.fuse_type == 'append':
            self.general_fuse_function = append_object_embeddings
        elif self.fuse_type == 'postaverage': 
            self.general_fuse_function = postaverage_object_embeddings     
        elif self.fuse_type == 'linear': 
            self.general_fuse_function = linear_object_embeddings   
            self.pre_fn = MLP(4, 1, 16, use_residual=False)
        elif self.fuse_type == 'postlinear': 
            self.general_fuse_function = postlinear_object_embeddings   
            self.pre_fn = nn.ModuleList([
                                MLP(4, 1, 16, use_residual=False),
                                nn.LayerNorm(embed_dim)]
                            ) 
        elif self.fuse_type == 'linearchannel': 
            self.general_fuse_function = linearchannel_object_embeddings   
            self.pre_fn = MLP(embed_dim * 4, embed_dim, embed_dim, use_residual=False)
        elif self.fuse_type == 'postlinearchannel': 
            self.general_fuse_function = postlinearchannel_object_embeddings   
            self.pre_fn = nn.ModuleList([
                                MLP(embed_dim * 4, embed_dim, embed_dim, use_residual=False),
                                nn.LayerNorm(embed_dim)]
                            ) 
        else:
            raise NotImplementedError(f"{self.fuse_type} has not found")        

    def fuse_fn(self, text_embeds, object_embeds):
        print(text_embeds.shape, object_embeds.shape)
        text_object_embeds = torch.cat([text_embeds, object_embeds], dim=-1)
        text_object_embeds = self.mlp1(text_object_embeds) + text_embeds
        text_object_embeds = self.mlp2(text_object_embeds)
        text_object_embeds = self.layer_norm(text_object_embeds)
        return text_object_embeds

    def forward(
        self,
        text_embeds,
        object_embeds,
        image_token_mask,
        num_objects,
    ) -> torch.Tensor:
            # print([F.pdist(embed.squeeze().contiguous(), p=2) for embed in splited_object_embeds if embed.shape[0]>1])
        splited_object_embeds = []

        text_object_embeds = self.general_fuse_function(
            text_embeds, image_token_mask, object_embeds, num_objects, self.fuse_fn, self.pre_fn
        )

        if self.use_embed_loss or self.use_embed_clip_loss:
            batch_size, max_num_objects = object_embeds.shape[:2]

            flat_object_embeds = object_embeds.view(
                -1, object_embeds.shape[-2], object_embeds.shape[-1]
            )
            valid_object_mask = (
                torch.arange(max_num_objects, device=flat_object_embeds.device)[None, :]
                < num_objects[:, None]
            )
            # valid_object_embeds [num_o1+o2+o3]
            valid_object_embeds = flat_object_embeds[valid_object_mask.flatten()]
            if self.use_embed_loss:
                splited_object_embeds = torch.split(valid_object_embeds, num_objects.tolist(), dim=0)
                splited_object_embeds = [ F.normalize(embed.squeeze().contiguous(), p=2, dim=1) for embed in splited_object_embeds if embed.shape[0] > 1 ]
            elif self.use_embed_clip_loss:
                splited_object_embeds = valid_object_embeds.squeeze().contiguous()

        # if self.use_embed_loss or self.use_embed_clip_loss
        # print(text_object_embeds.shape, len(splited_object_embeds))
        return text_object_embeds, splited_object_embeds

def unet_store_cross_attention_scores(unet, attention_scores, layers=5):
    from diffusers.models.attention_processor import (
        Attention,
        AttnProcessor,
        AttnProcessor2_0,
    )

    UNET_LAYER_NAMES = [
        "down_blocks.0",
        "down_blocks.1",
        "down_blocks.2",
        "mid_block",
        "up_blocks.1",
        "up_blocks.2",
        "up_blocks.3",
    ]

    start_layer = (len(UNET_LAYER_NAMES) - layers) // 2
    end_layer = start_layer + layers
    applicable_layers = UNET_LAYER_NAMES[start_layer:end_layer]

    def make_new_get_attention_scores_fn(name):
        def new_get_attention_scores(module, query, key, attention_mask=None):
            attention_probs = module.old_get_attention_scores(
                query, key, attention_mask
            )
            attention_scores[name] = attention_probs
            return attention_probs

        return new_get_attention_scores

    for name, module in unet.named_modules():
        if isinstance(module, Attention) and "attn2" in name:
            if not any(layer in name for layer in applicable_layers):
                continue
            if isinstance(module.processor, AttnProcessor2_0):
                module.set_processor(AttnProcessor())
            module.old_get_attention_scores = module.get_attention_scores
            module.get_attention_scores = types.MethodType(
                make_new_get_attention_scores_fn(name), module
            )

    return unet


class BalancedL1Loss(nn.Module):
    def __init__(self, threshold=1.0, normalize=False):
        super().__init__()
        self.threshold = threshold
        self.normalize = normalize

    def forward(self, object_token_attn_prob, object_segmaps):
        if self.normalize:
            object_token_attn_prob = object_token_attn_prob / (
                object_token_attn_prob.max(dim=2, keepdim=True)[0] + 1e-5
            )
        background_segmaps = 1 - object_segmaps
        background_segmaps_sum = background_segmaps.sum(dim=2) + 1e-5
        object_segmaps_sum = object_segmaps.sum(dim=2) + 1e-5

        background_loss = (object_token_attn_prob * background_segmaps).sum(
            dim=2
        ) / background_segmaps_sum

        object_loss = (object_token_attn_prob * object_segmaps).sum(
            dim=2
        ) / object_segmaps_sum

        return background_loss - object_loss


def get_object_localization_loss_for_one_layer(
    cross_attention_scores,
    object_segmaps,
    object_token_idx,
    object_token_idx_mask,
    loss_fn,
):
    bxh, num_noise_latents, num_text_tokens = cross_attention_scores.shape
    b, max_num_objects, _, _ = object_segmaps.shape
    size = int(num_noise_latents**0.5)

    # Resize the object segmentation maps to the size of the cross attention scores
    object_segmaps = F.interpolate(
        object_segmaps, size=(size, size), mode="bilinear", antialias=True
    )  # (b, max_num_objects, size, size)

    object_segmaps = object_segmaps.view(
        b, max_num_objects, -1
    )  # (b, max_num_objects, num_noise_latents)

    num_heads = bxh // b

    cross_attention_scores = cross_attention_scores.view(
        b, num_heads, num_noise_latents, num_text_tokens
    )

    # Gather object_token_attn_prob
    object_token_attn_prob = torch.gather(
        cross_attention_scores,
        dim=3,
        index=object_token_idx.view(b, 1, 1, max_num_objects).expand(
            b, num_heads, num_noise_latents, max_num_objects
        ),
    )  # (b, num_heads, num_noise_latents, max_num_objects)

    object_segmaps = (
        object_segmaps.permute(0, 2, 1)
        .unsqueeze(1)
        .expand(b, num_heads, num_noise_latents, max_num_objects)
    )

    loss = loss_fn(object_token_attn_prob, object_segmaps)

    loss = loss * object_token_idx_mask.view(b, 1, max_num_objects)
    object_token_cnt = object_token_idx_mask.sum(dim=1).view(b, 1) + 1e-5
    loss = (loss.sum(dim=2) / object_token_cnt).mean()

    return loss


def get_object_localization_loss(
    cross_attention_scores,
    object_segmaps,
    image_token_idx,
    image_token_idx_mask,
    loss_fn,
):
    num_layers = len(cross_attention_scores)
    loss = 0
    for k, v in cross_attention_scores.items():
        layer_loss = get_object_localization_loss_for_one_layer(
            v, object_segmaps, image_token_idx, image_token_idx_mask, loss_fn
        )
        loss += layer_loss
    return loss / num_layers


class IDAdapterModel(nn.Module):
    def __init__(self, text_encoder, text_encoder_2, image_encoder, freeze_image_encoder, vae, unet, args):
        super().__init__()
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.image_encoder = image_encoder
        self.freeze_image_encoder = freeze_image_encoder
        self.vae = vae
        self.unet = unet
        self.use_ema = False
        self.ema_param = None
        self.pretrained_model_name_or_path = args.pretrained_model_name_or_path
        self.revision = args.revision
        self.non_ema_revision = args.non_ema_revision
        self.object_localization = args.object_localization
        self.object_localization_weight = args.object_localization_weight
        self.localization_layers = args.localization_layers
        self.mask_loss = args.mask_loss
        self.mask_loss_prob = args.mask_loss_prob

        self.args = args
        self.use_embed_loss = args.use_embed_loss
        self.use_id_loss = args.use_id_loss
        self.use_clip_loss = args.use_clip_loss
        self.use_embed_clip_loss = args.use_embed_clip_loss

        self.embed_loss_weight = args.embed_loss_weight
        if self.use_id_loss:
            from idadapter.id_loss import IDLoss
            self.id_loss_module = IDLoss()
            self.id_loss_weight = args.id_loss_weight
        if self.use_clip_loss:
            self.logit_scale = nn.Parameter(torch.tensor(2.6592)) # from clip_model.config.logit_scale_init_value
            self.clip_loss_weight = args.clip_loss_weight
        if self.use_embed_clip_loss:
            # self.logit_embed_scale = nn.Parameter(torch.tensor(2.6592)) # from clip_model.config.logit_scale_init_value
            self.embed_clip_loss_weight = args.embed_clip_loss_weight
        embed_dim = text_encoder.config.hidden_size
        embed_dim_2 = text_encoder_2.config.hidden_size
        if args.image_encoder_2_name_or_path is None:
            self.postfuse_module = IDAdapterPostfuseModule(embed_dim + embed_dim_2, fuse_type=args.fuse_type, use_embed_loss=self.use_embed_loss, use_embed_clip_loss=self.use_embed_clip_loss)
        else:
            self.postfuse_module = IDAdapterPostfuseModule(embed_dim, fuse_type=args.fuse_type)
            self.postfuse_module_2 = IDAdapterPostfuseModule(embed_dim_2, fuse_type=args.fuse_type)

        if self.object_localization:
            self.cross_attention_scores = {}
            self.unet = unet_store_cross_attention_scores(
                self.unet, self.cross_attention_scores, self.localization_layers
            )
            self.object_localization_loss_fn = BalancedL1Loss(
                args.object_localization_threshold,
                args.object_localization_normalize,
            )

    def _clear_cross_attention_scores(self):
        if hasattr(self, "cross_attention_scores"):
            keys = list(self.cross_attention_scores.keys())
            for k in keys:
                del self.cross_attention_scores[k]

        gc.collect()

    @staticmethod
    def from_pretrained(args):
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae",
            # local_files_only=True, 
            use_safetensors=True, 
            # variant="fp16"
        )

        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            # local_files_only=True,
            use_safetensors=True, 
            # variant="fp16"
        )

        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            # local_files_only=True, 
            use_safetensors=True, 
            # variant="fp16"
        )

        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            # local_files_only=True, 
            use_safetensors=True, 
            # variant="fp16"
        )

        image_encoder = IDAdapterCLIPImageEncoder.from_pretrained(
            args.image_encoder_name_or_path,
            args.image_encoder_pretrained_path,
            args.image_encoder_2_name_or_path,
            use_clip_loss=args.use_clip_loss,
        )

        if args.use_clip_loss:
            freeze_image_encoder = IDAdapterCLIPImageEncoder.from_pretrained(
                args.image_encoder_name_or_path,
                args.freeze_image_encoder_pretrained_path,
                use_clip_loss=args.use_clip_loss,
            )
        else:
            freeze_image_encoder = None 

        return IDAdapterModel(text_encoder, text_encoder_2, image_encoder, freeze_image_encoder, vae, unet, args)

    def to_pipeline(self):
        pipe = StableDiffusionXLPipeline.from_pretrained(
            self.pretrained_model_name_or_path,
            revision=self.revision,
            non_ema_revision=self.non_ema_revision,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            vae=self.vae,
            unet=self.unet,
        )
        pipe.safety_checker = None

        pipe.image_encoder = self.image_encoder

        pipe.postfuse_module = self.postfuse_module


        return pipe

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + self.text_encoder_2.config.projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids
    
    def encode_prompt(self, input_ids, input_ids_2):
        prompt_embeds_list = []
        prompt_embeds = self.text_encoder(
            input_ids,
            output_hidden_states=True,
        )
        prompt_embeds_list.append(prompt_embeds.hidden_states[-2])
        prompt_embeds = self.text_encoder_2(
            input_ids_2,
            output_hidden_states=True,
        )        
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds_list.append(prompt_embeds.hidden_states[-2])
        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        return prompt_embeds, pooled_prompt_embeds

    def forward(self, batch, noise_scheduler):
        image_token_mask = batch["image_token_mask"]
        object_pixel_values = batch["object_pixel_values"]
        num_objects = batch["num_objects"]
        latents = batch["latents"]
        prompt_embeds = batch["prompt_embeds"]
        pooled_prompt_embeds = batch["pooled_prompt_embeds"]

        original_size = (1024, 1024)
        crops_coords_top_left = (0, 0)
        target_size = (1024, 1024)

        # with torch.no_grad():
        #     self.vae = self.vae.to(torch.float32)
        #     latents = self.vae.encode(batch["pixel_values"].to(torch.float32)).latent_dist.sample()
        #     latents = latents * self.vae.config.scaling_factor
        #     prompt_embeds, pooled_prompt_embeds = self.encode_prompt(batch["input_ids"], batch["input_ids_2"])
            # batch["latents"], batch["prompt_embeds"], batch["pooled_prompt_embeds"] = latents, prompt_embeds, pooled_prompt_embeds
        # import pdb; pdb.set_trace()
        # vae_dtype = self.vae.parameters().__next__().dtype
        # vae_input = pixel_values.to(vae_dtype)

        # latents = self.vae.encode(vae_input).latent_dist.sample()
        # import pdb; pdb.set_trace()
        if torch.any(torch.isnan(latents)):
            print("NaN found in latents, replacing with zeros")
            latents = torch.where(torch.isnan(latents), torch.zeros_like(latents), latents)
        # latents = latents * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device
        )
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # (bsz, max_num_objects, num_image_tokens, dim)
        if self.use_clip_loss:
            object_embeds, shared_object_embeds = self.image_encoder(object_pixel_values)
            bsz, max_num_objects = object_embeds.shape[:2]
            valid_object_mask = (
                torch.arange(max_num_objects, device=object_embeds.device)[None, :]
                < num_objects[:, None]
            )
            shared_object_embeds = shared_object_embeds[valid_object_mask.flatten()]
            with torch.no_grad():
                _, freeze_shared_object_embeds = self.freeze_image_encoder(object_pixel_values)
                freeze_shared_object_embeds = freeze_shared_object_embeds[valid_object_mask.flatten()]
        else:
            object_embeds = self.image_encoder(object_pixel_values)
        batch_size = prompt_embeds.shape[0]
        
        if self.args.image_encoder_2_name_or_path is None:
            prompt_embeds, splited_object_embeds = self.postfuse_module(
                prompt_embeds,
                object_embeds,
                image_token_mask,
                num_objects,
            )
        else:
            prompt_embeds, prompt_embeds_2 = torch.split(prompt_embeds, [768, 1280], dim=-1)
            object_embeds, object_embeds_2 = torch.split(object_embeds, [768, 1280], dim=-1)
            # print(prompt_embeds.shape, prompt_embeds_2.shape, object_embeds.shape, object_embeds_2.shape)
            # exit()
            prompt_embeds = self.postfuse_module(
                prompt_embeds,
                object_embeds,
                image_token_mask,
                num_objects,
            )
            prompt_embeds_2 = self.postfuse_module(
                prompt_embeds_2,
                object_embeds_2,
                image_token_mask,
                num_objects,
            )
            prompt_embeds = torch.cat((prompt_embeds, prompt_embeds_2), dim=-1)

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {noise_scheduler.config.prediction_type}"
            )
            
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
        )
        add_time_ids = add_time_ids.to(device=latents.device).repeat(batch_size, 1)       
        added_cond_kwargs = {"text_embeds": add_text_embeds.to(noisy_latents.dtype), "time_ids": add_time_ids.to(noisy_latents.dtype)}
        # prompt_embeds.to(noisy_latents.dtype)
        pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=prompt_embeds, added_cond_kwargs=added_cond_kwargs).sample

        if self.mask_loss and torch.rand(1) < self.mask_loss_prob:
            object_segmaps = batch["object_segmaps"]
            mask = (object_segmaps.sum(dim=1) > 0).float()
            mask = F.interpolate(
                mask.unsqueeze(1),
                size=(pred.shape[-2], pred.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            pred = pred * mask
            target = target * mask

        denoise_loss = F.mse_loss(pred.float(), target.float(), reduction="mean")

        return_dict = {"denoise_loss": denoise_loss}

        if self.use_id_loss:
            if sum(num_objects.bool()) > 0:
                id_loss = self.id_loss_weight * id_loss
            else:
                id_loss = torch.zeros_like(denoise_loss)
                id_loss = self.id_loss_weight * id_loss

            return_dict["id_loss"] = id_loss
            loss = id_loss + denoise_loss                
        else:
            loss = denoise_loss            

        if self.use_embed_loss:
            if len(splited_object_embeds) > 0:
                embed_loss = 0
                for embed in splited_object_embeds:
                    pairwise_distance = F.pdist(embed, p=2)
                    embed_loss += pairwise_distance.mean()
                embed_loss = self.embed_loss_weight * (embed_loss / len(splited_object_embeds))
                # return_dict["embed_loss"] = embed_loss
                # loss = embed_loss + denoise_loss
            else:
                embed_loss = torch.zeros_like(denoise_loss)
                embed_loss = self.embed_loss_weight * (embed_loss)

            return_dict["embed_loss"] = embed_loss
            loss = embed_loss + denoise_loss
        else:
            loss = denoise_loss

        return_dict["loss"] = loss
        return return_dict


if __name__ == "__main__":
    fuse_type = 'postaverage'
    case = 3
    device = 'cuda'
    import random
    
    embed_dim = 2048
    postmodule = IDAdapterPostfuseModule(embed_dim=2048, fuse_type=fuse_type).to(device)
    
    bs = 3
    seq_len = 77
    max_objects = 4

    text_embeds = torch.randn((bs, seq_len, embed_dim), dtype=torch.float32, device='cuda')
    object_embeds = torch.randn((bs, max_objects, 1, embed_dim), dtype=torch.float32, device='cuda')
    image_token_mask = torch.zeros((bs, seq_len), device='cuda').to(torch.bool)
    num_objects = torch.zeros((bs, ), dtype=torch.long, device='cuda')
    if case > 0:
        for i in range(case):
            image_token_mask[i][random.randint(0, 10)] = True
            num_objects[i] = random.randint(1, 4)

    # object_embeds = torch.randn((), dtype=, device='cuda')
    # image_token_mask = torch.randn((), dtype=, device='cuda')
    # num_objects = torch.tensor((), dtype=, device='cuda')
    #     object_embeds,
    #     image_token_mask,
    #     num_objects,

    output = postmodule(text_embeds, object_embeds, image_token_mask, num_objects)
    print(output.shape)

"""
# import torch

# def pairwise_distances(x):
#     
#     Compute the pairwise Euclidean distances between the rows of a tensor.

#     :param x: A tensor of shape (batch_size, embedding_dim).
#     :return: A tensor of shape (batch_size, batch_size) containing the pairwise distances.
#     
#     # Compute the dot product between each pair of rows
#     dot_product = x @ x.t()

#     # Compute the squared norm of each row
#     squared_norms = dot_product.diag()

#     # Compute the pairwise Euclidean distances
#     distances = squared_norms.unsqueeze(1) - 2.0 * dot_product + squared_norms.unsqueeze(0)
#     distances = torch.clamp(distances, min=0.0)  # Ensure that the distances are non-negative
#     distances = torch.sqrt(distances)

#     return distances

# # Example usage
# x = torch.tensor([[1.0, 2.0, 3.0],
#                   [4.0, 5.0, 6.0],
#                   [7.0, 8.0, 9.0]])

# distances = pairwise_distances(x)
# print(distances)
"""