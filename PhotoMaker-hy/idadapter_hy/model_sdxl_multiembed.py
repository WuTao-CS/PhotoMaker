import torch
import torch.nn as nn
import accelerate
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel, StableDiffusionPipeline, DDPMScheduler
from diffusers.image_processor import VaeImageProcessor
from transformers import BertModel, BertConfig, BertTokenizer
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPConfig
import torch.nn.functional as F
from typing import Any, Optional, Tuple, Union, Dict, List
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import (
    CLIPTextTransformer,
    CLIPPreTrainedModel,
    CLIPModel,
)
import os
import types
import torchvision.transforms as T
import gc
import numpy as np
import sys; sys.path.append("/apdcephfs/share_1367250/liwenyue/adapter/mutiscalebigmodel/0_hunyuan_text2image/tools/")
from annotator.img_clip_emb.img_clip_emb import ImgClipEmbDetector

inference_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

class AttentionPool(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.permute(2, 0, 1)  # NCL -> LNC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (L+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (L+1)NC
        # import pdb; pdb.set_trace()
        # x = x.float()
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


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

class IDAdapterHunyuanImageEncoder(nn.Module):
    @staticmethod
    def from_pretrained(
        global_model_name_or_path,
        pretrained_path=None,
        global_model_name_or_path_2=None,
        use_clip_loss=False,
        use_hunyuan_image_encoder=False,
    ):
        vision_processor = T.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )


        vision_model = ImgClipEmbDetector()
        vision_model.config=vision_model.cfg
        return IDAdapterHunyuanImageEncoder(
            vision_model,
            None,
            None,
            None,
            vision_processor,
            use_clip_loss=use_clip_loss,
            use_hunyuan_image_encoder=use_hunyuan_image_encoder,
        )

        
    def __init__(
        self,
        vision_model,
        visual_projection,
        vision_model_2,
        visual_projection_2,
        vision_processor,
        use_clip_loss,
        use_hunyuan_image_encoder=False
    ):
        
        super().__init__()

        self.vision_model = vision_model
        # self.visual_projection = visual_projection
        self.vision_processor = vision_processor

        self.image_size = 224
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
        shared_object_embeds = self.vision_model(object_pixel_values)
        out_embeds = shared_object_embeds.view(b, num_objects, 1, -1)
        return out_embeds

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
        use_hunyuan_image_encoder=False,
    ):
        vision_processor = T.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )

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
        use_hunyuan_image_encoder=False
    ):
        super().__init__(vision_model.config)
        self.use_hunyuan_image_encoder= use_hunyuan_image_encoder
        self.vision_model = vision_model
        # self.visual_projection = visual_projection
        self.vision_processor = vision_processor

        self.image_size = 224
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
        out_embeds = shared_object_embeds.view(b, num_objects, 1, -1)
        return out_embeds



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
        if self.fuse_type == 'append':
            self.general_fuse_function = append_object_embeddings
        else:
            raise NotImplementedError(f"{self.fuse_type} has not found")        

    def fuse_fn(self, text_embeds, object_embeds):
        # print(text_embeds.shape, object_embeds.shape)
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
    def to_cuda(self, idx=0, dtype=torch.float16):
        self.text_encoder.cuda(idx).to(dtype)
        self.image_encoder.cuda(idx).to(dtype)
        self.vae.cuda(idx).to(dtype)
        self.unet.cuda(idx).to(dtype)
        self.text_attn_pool.cuda(idx).to(dtype)
        self.postfuse_module.cuda(idx).to(dtype)


    def __init__(self, text_encoder, image_encoder, freeze_image_encoder, vae, unet, text_attn_pool, args):
        super().__init__()
        self.text_encoder = text_encoder
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
        self.text_attn_pool = text_attn_pool

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
        self.postfuse_module = IDAdapterPostfuseModule(embed_dim, fuse_type=args.fuse_type, use_embed_loss=self.use_embed_loss, use_embed_clip_loss=self.use_embed_clip_loss)

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
        # vae = AutoencoderKL.from_pretrained(
        #     args.pretrained_model_name_or_path, subfolder="vae",
        #     # local_files_only=True, 
        #     use_safetensors=True, 
        #     # variant="fp16"
        # )
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            # torch_dtype=torch_dtype
        )

        unet = UNet2DConditionModel.from_pretrained(
            args.unet_path,
        )

        # text_encoder = CLIPTextModel.from_pretrained(
        #     args.pretrained_model_name_or_path,
        #     subfolder="text_encoder",
        #     # local_files_only=True, 
        #     use_safetensors=True, 
        #     # variant="fp16"
        # )
        text_encoder = BertModel.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, "text_encoder")
            # torch_dtype=torch_dtype
        )

        if args.use_hunyuan_image_encoder:
            image_encoder = IDAdapterHunyuanImageEncoder.from_pretrained(
                args.image_encoder_name_or_path,
                args.image_encoder_pretrained_path,
                args.image_encoder_2_name_or_path,
                use_clip_loss=args.use_clip_loss,
                use_hunyuan_image_encoder=args.use_hunyuan_image_encoder
            )
        else:
            image_encoder = IDAdapterCLIPImageEncoder.from_pretrained(
                args.image_encoder_name_or_path,
                args.image_encoder_pretrained_path,
                args.image_encoder_2_name_or_path,
                use_clip_loss=args.use_clip_loss,
                use_hunyuan_image_encoder=args.use_hunyuan_image_encoder
            )

        text_attn_pool = AttentionPool(spacial_dim=77, embed_dim=1024, num_heads=8, output_dim=1024)
        attn_pool_weight = torch.load(args.pooling_path, map_location='cpu')
        text_attn_pool.load_state_dict(attn_pool_weight)

        if args.use_clip_loss:
            freeze_image_encoder = IDAdapterCLIPImageEncoder.from_pretrained(
                args.image_encoder_name_or_path,
                args.freeze_image_encoder_pretrained_path,
                use_clip_loss=args.use_clip_loss,
            )
        else:
            freeze_image_encoder = None 

        return IDAdapterModel(text_encoder, image_encoder, freeze_image_encoder, vae, unet, text_attn_pool, args)

    def to_pipeline(self):
        pipe = StableDiffusionPipeline.from_pretrained(
            self.pretrained_model_name_or_path,
            revision=self.revision,
            non_ema_revision=self.non_ema_revision,
            text_encoder=self.text_encoder,
            vae=self.vae,
            unet=self.unet,
        )
        pipe.safety_checker = None

        pipe.image_encoder = self.image_encoder

        pipe.postfuse_module = self.postfuse_module

        pipe.text_attn_pool = self.text_attn_pool


        return pipe

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + 1024
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids
    
    def encode_prompt(self, input_ids):
        
        prompt_embeds = self.text_encoder(
            input_ids,
            output_hidden_states=True,
        )
        prompt_embeds = prompt_embeds[0]
        pooled_prompt_embeds = self.text_attn_pool(prompt_embeds.transpose(2, 1).contiguous())
    
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
        
        prompt_embeds, splited_object_embeds = self.postfuse_module(
            prompt_embeds,
            object_embeds,
            image_token_mask,
            num_objects,
        )
        
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


def test_idadpater():
    fuse_type = 'append'
    case = 3
    device = 'cuda'
    import random
    
    embed_dim = 1024
    postmodule = IDAdapterPostfuseModule(embed_dim=embed_dim, fuse_type=fuse_type).to(device)
    
    bs = 3
    seq_len = 77
    max_objects = 4

    text_embeds = torch.randn((bs, seq_len, embed_dim), dtype=torch.float32, device='cuda')
    object_embeds = torch.randn((bs, max_objects, 1, embed_dim), dtype=torch.float32, device='cuda')
    image_token_mask = torch.zeros((bs, seq_len), device='cuda').to(torch.bool)
    num_objects = torch.zeros((bs, ), dtype=torch.long, device='cuda')
    if case > 0:
        for i in range(case):
            num_objects[i] = random.randint(1, 4)
            start_pos = random.randint(0, 10)
            for ii in range(num_objects[i]):
                image_token_mask[i][start_pos+ii] = True


    output = postmodule(text_embeds, object_embeds, image_token_mask, num_objects)
    print(output[0].shape)

def test_model():
    # pretrained = "openai/clip-vit-large-patch14"
    # image_encoder = IDAdapterCLIPImageEncoder.from_pretrained(pretrained)
    # inp = torch.randn(1,4,3,224,224)
    # import pdb; pdb.set_trace()
    # out = model(inp)
    # print()


    from idadapter import utils
    args = utils.parse_args()
    unet_path="/apdcephfs/share_1367250/rongweiquan/hunyuan_latent_zh/unet_v1.4_human"
    pooling_path="/apdcephfs/share_1367250/rongweiquan/hunyuan_latent_zh/unet_v1.4_human/pooling_weight.pt"
    args.unet_path = unet_path
    args.pooling_path = pooling_path
    args.pretrained_model_name_or_path = '/apdcephfs/share_1367250/rongweiquan/ControlNet_SR/models/controlnet_sr_zh'
    args.image_encoder_name_or_path = "openai/clip-vit-large-patch14"
    args.image_encoder_pretrained_path = None
    args.use_hunyuan_image_encoder = True
    model = IDAdapterModel.from_pretrained(args)
    model.to_cuda()
    
    case = 2
    device = 'cuda'
    import random
    embed_dim = 1024
    bs = 2
    seq_len = 77
    max_objects = 4

    text_embeds = torch.randn((bs, seq_len, embed_dim), dtype=torch.float16, device='cuda')
    object_embeds = torch.randn((bs, max_objects, 1, embed_dim), dtype=torch.float32, device='cuda').to(torch.float16)
    image_token_mask = torch.zeros((bs, seq_len), device='cuda').to(torch.bool)
    num_objects = torch.zeros((bs, ), dtype=torch.long, device='cuda')
    if case > 0:
        for i in range(case):
            num_objects[i] = random.randint(1, max_objects)
            start_pos = random.randint(0, 10)
            for ii in range(num_objects[i]):
                image_token_mask[i][start_pos+ii] = True
    batch = {}
    latents = model.vae.encode(torch.randn(bs,3,1024,1024).to(torch.float16).cuda()).latent_dist.sample()
    latents = latents * model.vae.config.scaling_factor
    # print(latents.shape)
    tokenizer = BertTokenizer.from_pretrained("/apdcephfs/share_1367250/zhiminli/0_project/stable-diffusion-2-1_hunyuanclip_angelptm/tokenizer")
    prompt = ["i am nick"] * bs
    text_inputs = tokenizer(prompt,padding="max_length",max_length=77,truncation=True,return_tensors="pt",)['input_ids'].cuda()
    prompt_embeds, pooled_prompt_embeds = model.encode_prompt(text_inputs.cuda())
    batch["latents"], batch["prompt_embeds"], batch["pooled_prompt_embeds"] = latents, prompt_embeds.to(torch.float16), pooled_prompt_embeds.to(torch.float16)
    batch["image_token_mask"] = image_token_mask
    batch["object_pixel_values"] = torch.randn(bs,max_objects,3,224,224).cuda().to(torch.float16)
    batch["num_objects"] = num_objects
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    res = model(batch, noise_scheduler)
    import pdb; pdb.set_trace()
    print()



if __name__ == "__main__":
    test_model()
