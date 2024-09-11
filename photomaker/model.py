# Merge image encoder and fuse module to create an ID Encoder
# send multiple ID images, we can directly obtain the updated text encoder containing a stacked ID embedding
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import torch
import torch.nn as nn
from transformers.models.clip.modeling_clip import CLIPVisionModelWithProjection
from transformers.models.clip.configuration_clip import CLIPVisionConfig
from transformers import CLIPImageProcessor, CLIPTokenizer
from transformers import PretrainedConfig
from safetensors import safe_open
import torch.nn.functional as F
from diffusers import AutoencoderKL,MotionAdapter, UNet2DConditionModel, UNetMotionModel
from diffusers.loaders import StableDiffusionXLLoraLoaderMixin
from diffusers.loaders.ip_adapter import IPAdapterMixin
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
# from .attention_processor import MixIPAdapterAttnProcessor2_0
from einops import rearrange, repeat
from diffusers.utils import (
    USE_PEFT_BACKEND,
    _get_model_file,
    is_accelerate_available,
    is_torch_version,
    is_transformers_available,
    logging,
)
try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False
from packaging import version
if version.parse(torch.__version__) >= version.parse("2.0.0"):
    SDP_IS_AVAILABLE = True
    from torch.backends.cuda import SDPBackend, sdp_kernel

    BACKEND_MAP = {
        SDPBackend.MATH: {
            "enable_math": True,
            "enable_flash": False,
            "enable_mem_efficient": False,
        },
        SDPBackend.FLASH_ATTENTION: {
            "enable_math": False,
            "enable_flash": True,
            "enable_mem_efficient": False,
        },
        SDPBackend.EFFICIENT_ATTENTION: {
            "enable_math": False,
            "enable_flash": False,
            "enable_mem_efficient": True,
        },
        None: {"enable_math": True, "enable_flash": True, "enable_mem_efficient": True},
    }
else:
    from contextlib import nullcontext

    SDP_IS_AVAILABLE = False
    sdp_kernel = nullcontext
    BACKEND_MAP = {}
# from diffusers.models.attention_processor import Attention
VISION_CONFIG_DICT = {
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "patch_size": 14,
    "projection_dim": 768
}
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = zero_module(nn.Linear(hidden_dim, out_dim))
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


class FuseModule(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.mlp1 = MLP(embed_dim * 2, embed_dim, embed_dim, use_residual=False)
        self.mlp2 = MLP(embed_dim, embed_dim, embed_dim, use_residual=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def fuse_fn(self, prompt_embeds, id_embeds):
        stacked_id_embeds = torch.cat([prompt_embeds, id_embeds], dim=-1)
        stacked_id_embeds = self.mlp1(stacked_id_embeds) + prompt_embeds
        stacked_id_embeds = self.mlp2(stacked_id_embeds)
        stacked_id_embeds = self.layer_norm(stacked_id_embeds)
        return stacked_id_embeds

    def forward(
        self,
        prompt_embeds,
        id_embeds,
        class_tokens_mask,
    ) -> torch.Tensor:
        # id_embeds shape: [b, max_num_inputs, 1, 2048]
        id_embeds = id_embeds.to(prompt_embeds.dtype)
        num_inputs = class_tokens_mask.sum().unsqueeze(0) # TODO: check for training case
        batch_size, max_num_inputs = id_embeds.shape[:2]
        # seq_length: 77
        seq_length = prompt_embeds.shape[1]
        # flat_id_embeds shape: [b*max_num_inputs, 1, 2048]
        flat_id_embeds = id_embeds.view(
            -1, id_embeds.shape[-2], id_embeds.shape[-1]
        )
        # valid_id_mask [b*max_num_inputs]
        valid_id_mask = (
            torch.arange(max_num_inputs, device=flat_id_embeds.device)[None, :]
            < num_inputs[:, None]
        )
        valid_id_embeds = flat_id_embeds[valid_id_mask.flatten()]

        prompt_embeds = prompt_embeds.view(-1, prompt_embeds.shape[-1])
        class_tokens_mask = class_tokens_mask.view(-1)
        valid_id_embeds = valid_id_embeds.view(-1, valid_id_embeds.shape[-1])
        # slice out the image token embeddings
        image_token_embeds = prompt_embeds[class_tokens_mask]
        stacked_id_embeds = self.fuse_fn(image_token_embeds, valid_id_embeds)
        assert class_tokens_mask.sum() == stacked_id_embeds.shape[0], f"{class_tokens_mask.sum()} != {stacked_id_embeds.shape[0]}"
        prompt_embeds.masked_scatter_(class_tokens_mask[:, None], stacked_id_embeds.to(prompt_embeds.dtype))
        updated_prompt_embeds = prompt_embeds.view(batch_size, seq_length, -1)
        return updated_prompt_embeds

class PhotoMakerIDEncoder(CLIPVisionModelWithProjection):
    def __init__(self):
        super().__init__(CLIPVisionConfig(**VISION_CONFIG_DICT))
        self.visual_projection_2 = nn.Linear(1024, 1280, bias=False)
        self.fuse_module = FuseModule(2048)

    def forward(self, id_pixel_values, prompt_embeds, class_tokens_mask):
        b, num_inputs, c, h, w = id_pixel_values.shape
        id_pixel_values = id_pixel_values.view(b * num_inputs, c, h, w)

        vision_outputs= self.vision_model(id_pixel_values)
        shared_id_embeds = vision_outputs[1] # 768
        id_embeds = self.visual_projection(shared_id_embeds)
        id_embeds_2 = self.visual_projection_2(shared_id_embeds)

        id_embeds = id_embeds.view(b, num_inputs, 1, -1)
        id_embeds_2 = id_embeds_2.view(b, num_inputs, 1, -1)    

        id_embeds = torch.cat((id_embeds, id_embeds_2), dim=-1) #[b, num_inputs, 1, 2048]
        updated_prompt_embeds = self.fuse_module(prompt_embeds, id_embeds, class_tokens_mask)

        return updated_prompt_embeds, id_embeds_2
    
    def load_from_pretrained(self, model_path):
        if model_path.endswith(".safetensors"):
            state_dict = {"id_encoder": {}, "lora_weights": {}}
            with safe_open(model_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("id_encoder."):
                        state_dict["id_encoder"][key.replace("id_encoder.", "")] = f.get_tensor(key)
                    elif key.startswith("lora_weights."):
                        state_dict["lora_weights"][key.replace("lora_weights.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(model_path, map_location="cpu")
        self.load_state_dict(state_dict["id_encoder"], strict=True)

        

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def calc_mean_std(feat, eps: float = 1e-5):
    feat_std = (feat.var(dim=-2, keepdims=True) + eps).sqrt()
    feat_mean = feat.mean(dim=-2, keepdims=True)
    return feat_mean, feat_std

def adain(feat, ref_feat) -> torch.Tensor:
    feat_mean, feat_std = calc_mean_std(feat)
    ref_mean, ref_std = calc_mean_std(ref_feat)
    feat = (feat - feat_mean) / feat_std
    feat = ref_std * (feat - feat_mean) / feat_std + ref_mean
    return feat

# class RelativePosition(nn.Module):
#     """ https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py """

#     def __init__(self, num_units, max_relative_position):
#         super().__init__()
#         self.num_units = num_units
#         self.max_relative_position = max_relative_position
#         self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
#         nn.init.xavier_uniform_(self.embeddings_table)

#     def forward(self, length_q, length_k):
#         device = self.embeddings_table.device
#         range_vec_q = torch.arange(length_q, device=device)
#         range_vec_k = torch.arange(length_k, device=device)
#         distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
#         distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
#         final_mat = distance_mat_clipped + self.max_relative_position
#         # final_mat = th.LongTensor(final_mat).to(self.embeddings_table.device)
#         # final_mat = th.tensor(final_mat, device=self.embeddings_table.device, dtype=torch.long)
#         final_mat = final_mat.long()
#         embeddings = self.embeddings_table[final_mat]
#         return embeddings

# class CrossAttention(nn.Module):

#     def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., 
#                  relative_position=False, temporal_length=None, qkv_bias=False):
#         super().__init__()
#         inner_dim = dim_head * heads

#         context_dim = context_dim if context_dim is not None else query_dim

#         self.scale = dim_head**-0.5
#         self.heads = heads
#         self.dim_head = dim_head
#         self.qkv_bias = qkv_bias
#         self.to_q = nn.Linear(query_dim, inner_dim, bias=self.qkv_bias)
#         self.to_k = nn.Linear(context_dim, inner_dim, bias=self.qkv_bias)
#         self.to_v = nn.Linear(context_dim, inner_dim, bias=self.qkv_bias)

#         self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        
#         self.relative_position = relative_position
#         if self.relative_position:
#             assert(temporal_length is not None)
#             self.relative_position_k = RelativePosition(num_units=dim_head, max_relative_position=temporal_length)
#             self.relative_position_v = RelativePosition(num_units=dim_head, max_relative_position=temporal_length)
#         else:
#             ## only used for spatial attention, while NOT for temporal attention
#             if XFORMERS_IS_AVAILBLE and temporal_length is None:
#                 self.forward = self.efficient_forward

#     def forward(self, x, context=None, mask=None):
#         h = self.heads

#         q = self.to_q(x)
#         context = context if context is not None else x
#         k = self.to_k(context)
#         v = self.to_v(context)

#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
#         sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
#         if self.relative_position:
#             len_q, len_k, len_v = q.shape[1], k.shape[1], v.shape[1]
#             k2 = self.relative_position_k(len_q, len_k)
#             sim2 = torch.einsum('b t d, t s d -> b t s', q, k2) * self.scale # TODO check 
#             sim += sim2
#         del q, k

#         if mask is not None:
#             ## feasible for causal attention mask only
#             max_neg_value = -torch.finfo(sim.dtype).max
#             mask = repeat(mask, 'b i j -> (b h) i j', h=h)
#             sim.masked_fill_(~(mask>0.5), max_neg_value)

#         # attention, what we cannot get enough of
#         sim = sim.softmax(dim=-1)

#         out = torch.einsum('b i j, b j d -> b i d', sim, v)
#         if self.relative_position:
#             v2 = self.relative_position_v(len_q, len_v)
#             out2 = torch.einsum('b t s, t s d -> b t d', sim, v2) # TODO check
#             out += out2
#         out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
#         return self.to_out(out)
    
#     def efficient_forward(self, x, context=None, mask=None):
#         q = self.to_q(x)
#         context = context if context is not None else x
#         k = self.to_k(context)
#         v = self.to_v(context)

#         b, _, _ = q.shape
#         q, k, v = map(
#             lambda t: t.unsqueeze(3)
#             .reshape(b, t.shape[1], self.heads, self.dim_head)
#             .permute(0, 2, 1, 3)
#             .reshape(b * self.heads, t.shape[1], self.dim_head)
#             .contiguous(),
#             (q, k, v),
#         )
#         # actually compute the attention, what we cannot get enough of
#         out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=None)

#         if mask is not None:
#             raise NotImplementedError
#         out = (
#             out.unsqueeze(0)
#             .reshape(b, self.heads, out.shape[1], self.dim_head)
#             .permute(0, 2, 1, 3)
#             .reshape(b, out.shape[1], self.heads * self.dim_head)
#         )
#         return self.to_out(out)

class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        backend=None,
        qkv_bias=False
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim

        self.scale = dim_head**-0.5
        self.heads = heads
        self.qkv_bias = qkv_bias
        self.to_q = nn.Linear(query_dim, inner_dim, bias=self.qkv_bias)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=self.qkv_bias)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=self.qkv_bias)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.backend = backend

    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        h = self.heads

        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat([additional_tokens, x], dim=1)

        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)

        if n_times_crossframe_attn_in_self:
            # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            n_cp = x.shape[0] // n_times_crossframe_attn_in_self
            k = repeat(
                k[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
            )
            v = repeat(
                v[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
            )

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        ## old
        """
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        del q, k

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        """
        ## new
        with sdp_kernel(**BACKEND_MAP[self.backend]):
            # print("dispatching into backend", self.backend, "q/k/v shape: ", q.shape, k.shape, v.shape)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask
            )  # scale is dim_head ** -0.5 per default

        del q, k, v
        out = rearrange(out, "b h n d -> b n (h d)", h=h)

        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]
        return self.to_out(out)


# class MemoryEfficientCrossAttention(nn.Module):
#     # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
#     def __init__(
#         self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, **kwargs
#     ):
#         super().__init__()
#         logpy.debug(
#             f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, "
#             f"context_dim is {context_dim} and using {heads} heads with a "
#             f"dimension of {dim_head}."
#         )
#         inner_dim = dim_head * heads
#         context_dim = default(context_dim, query_dim)

#         self.heads = heads
#         self.dim_head = dim_head

#         self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
#         self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
#         self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
#         )
#         self.attention_op: Optional[Any] = None

#     def forward(
#         self,
#         x,
#         context=None,
#         mask=None,
#         additional_tokens=None,
#         n_times_crossframe_attn_in_self=0,
#     ):
#         if additional_tokens is not None:
#             # get the number of masked tokens at the beginning of the output sequence
#             n_tokens_to_mask = additional_tokens.shape[1]
#             # add additional token
#             x = torch.cat([additional_tokens, x], dim=1)
#         q = self.to_q(x)
#         context = default(context, x)
#         k = self.to_k(context)
#         v = self.to_v(context)

#         if n_times_crossframe_attn_in_self:
#             # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
#             assert x.shape[0] % n_times_crossframe_attn_in_self == 0
#             # n_cp = x.shape[0]//n_times_crossframe_attn_in_self
#             k = repeat(
#                 k[::n_times_crossframe_attn_in_self],
#                 "b ... -> (b n) ...",
#                 n=n_times_crossframe_attn_in_self,
#             )
#             v = repeat(
#                 v[::n_times_crossframe_attn_in_self],
#                 "b ... -> (b n) ...",
#                 n=n_times_crossframe_attn_in_self,
#             )

#         b, _, _ = q.shape
#         q, k, v = map(
#             lambda t: t.unsqueeze(3)
#             .reshape(b, t.shape[1], self.heads, self.dim_head)
#             .permute(0, 2, 1, 3)
#             .reshape(b * self.heads, t.shape[1], self.dim_head)
#             .contiguous(),
#             (q, k, v),
#         )

#         # actually compute the attention, what we cannot get enough of
#         if version.parse(xformers.__version__) >= version.parse("0.0.21"):
#             # NOTE: workaround for
#             # https://github.com/facebookresearch/xformers/issues/845
#             max_bs = 32768
#             N = q.shape[0]
#             n_batches = math.ceil(N / max_bs)
#             out = list()
#             for i_batch in range(n_batches):
#                 batch = slice(i_batch * max_bs, (i_batch + 1) * max_bs)
#                 out.append(
#                     xformers.ops.memory_efficient_attention(
#                         q[batch],
#                         k[batch],
#                         v[batch],
#                         attn_bias=None,
#                         op=self.attention_op,
#                     )
#                 )
#             out = torch.cat(out, 0)
#         else:
#             out = xformers.ops.memory_efficient_attention(
#                 q, k, v, attn_bias=None, op=self.attention_op
#             )

#         # TODO: Use this directly in the attention operation, as a bias
#         if exists(mask):
#             raise NotImplementedError
#         out = (
#             out.unsqueeze(0)
#             .reshape(b, self.heads, out.shape[1], self.dim_head)
#             .permute(0, 2, 1, 3)
#             .reshape(b, out.shape[1], self.heads * self.dim_head)
#         )
#         if additional_tokens is not None:
#             # remove additional token
#             out = out[:, n_tokens_to_mask:]
#         return self.to_out(out)


class IDAttentionFusionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads=8, mlp_ratio=4.0,time_ebed_size=1280, num_frame=16, **block_kwargs):
        super().__init__()
        self.num_frame = num_frame
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = CrossAttention(hidden_size, heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(1280 * mlp_ratio)
        self.mlp = MLP(in_dim=hidden_size, out_dim=hidden_size, hidden_dim=mlp_hidden_dim, use_residual=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            zero_module(nn.Linear(time_ebed_size, 6 * hidden_size, bias=True)),
        )
        # self.adaLN_modulation_ratio = nn.Sequential(
        #     nn.Linear(time_ebed_size, 3, bias=True),
        #     nn.SiLU(),
        # )
        self.adaLN_modulation_ratio = nn.Sequential(
            nn.Linear(time_ebed_size, 3 * hidden_size, bias=True),
            nn.SiLU(),
        )

    def forward(self, text_feature, clipi_feature, faceid_feature, time_emb):
        time_emb = time_emb.repeat(self.num_frame, 1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(time_emb).chunk(6, dim=1)
        ratio_face, ratio_clipi, ratio_text = self.adaLN_modulation_ratio(time_emb).chunk(3, dim=1)
        # input_key = adain(clipi_feature,faceid_feature) + 
        id_feature  = ratio_face.unsqueeze(1) *faceid_feature + ratio_clipi.unsqueeze(1) * clipi_feature
        x = text_feature + gate_msa.unsqueeze(1) * self.attn(text_feature, modulate(self.norm1(id_feature), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = ratio_text.unsqueeze(1)*x + id_feature
        # print(ratio_face.shape)
        # print(faceid_feature.shape)
        # id_feature  = ratio_face *faceid_feature + ratio_clipi * clipi_feature
        # x = text_feature + gate_msa.unsqueeze(1) * self.attn(text_feature, modulate(self.norm1(id_feature), shift_msa, scale_msa))
        # x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        # x = ratio_text*x + id_feature
        return x

class IDAttentiontestFusionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads=8, mlp_ratio=4.0,time_ebed_size=1280, num_frame=16, **block_kwargs):
        super().__init__()
        self.num_frame = num_frame
        self.adaLN_modulation_ratio = nn.Sequential(
            nn.Linear(time_ebed_size, 3 * hidden_size, bias=True),
            nn.SiLU(),
        )

    def forward(self, text_feature, clipi_feature, faceid_feature, time_emb):
        time_emb = time_emb.repeat(self.num_frame, 1)
        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(time_emb).chunk(6, dim=1)
        ratio_face, ratio_clipi, ratio_text = self.adaLN_modulation_ratio(time_emb).chunk(3, dim=1)
        # input_key = adain(clipi_feature,faceid_feature) + 
        x = ratio_text.unsqueeze(1)*text_feature + ratio_face.unsqueeze(1) *faceid_feature + ratio_clipi.unsqueeze(1) * clipi_feature
        return x
if __name__ == "__main__":
    PhotoMakerIDEncoder()