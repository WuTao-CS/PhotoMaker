import inspect
import math
from importlib import import_module
from typing import Callable, List, Optional, Union
from einops import rearrange, repeat
try:
    import xformers
    import xformers.ops
    xformers_available = True
except Exception as e: 
    xformers_available = False
import torch
import torch.nn.functional as F
from torch import nn
from diffusers.utils import deprecate, logging
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from diffusers.models.attention import FeedForward
# xformers_available = False
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

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
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.heads = heads
        self.qkv_bias = qkv_bias
        self.to_q = nn.Linear(query_dim, inner_dim, bias=self.qkv_bias)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=self.qkv_bias)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=self.qkv_bias)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        if xformers_available:
            self.forward = self.efficient_forward
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
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask
        )  # scale is dim_head ** -0.5 per default

        del q, k, v
        out = rearrange(out, "b h n d -> b n (h d)", h=h)

        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]
        return self.to_out(out)

    def efficient_forward(self, x, context=None, mask=None ,additional_tokens=None, n_times_crossframe_attn_in_self=0):
        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )
        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=None)

        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)
    
def process_latent_tensor(tensor): 
    # concat self attention with last frame
    # [b, f, t, c]
    _, f, t, _ = tensor.shape
    n_visual_with_condition = f*t
    last_frame = tensor[:, -1:, :, :]  # [b, 1, t, c]
    repeated_last_frame = last_frame.repeat(1, f-1, 1, 1)  # [b, f-1, t, c]
    remaining_frames = tensor[:, :-1, :, :]  #[b, f-1, t, c]
    n_visual = remaining_frames.shape[2]
    result = torch.cat((remaining_frames, repeated_last_frame), dim=2) #[b, f-1, 2*t, c]
    result = rearrange(result, 'b f t c -> (b f) t c').contiguous() 
    return result, n_visual

def process_latent_tensor_story(tensor): 
    # concat self attention with last frame
    # [b, f, t, c]
    _, f, t, _ = tensor.shape
    n_visual_with_condition = f*t
    last_frame = tensor[:, -1:, :, :]  # [b, 1, t, c]
    repeated_last_frame = last_frame.repeat(1, f, 1, 1)  # [b, f, t, c]
    n_visual = tensor.shape[2]
    result = torch.cat((tensor, repeated_last_frame), dim=2) #[b, f, 2*t, c]
    result = rearrange(result, 'b f t c -> (b f) t c').contiguous() 
    return result, n_visual

def process_conditon_latent_tensor(tensor): 
    # concat self attention with last frame
    # [b, f, t, c]
    _, f, t, _ = tensor.shape
    n_visual_with_condition = f*t
    last_frame = tensor[:, -1:, :, :]  # [b, 1, t, c]
    repeated_last_frame = last_frame.repeat(1, f-1, 1, 1)  # [b, f-1, t, c]
    remaining_frames = tensor[:, :-1, :, :]  #[b, f-1, t, c]
    result = torch.cat((remaining_frames, repeated_last_frame), dim=2) #[b, f-1, 2*t, c]
    result = rearrange(result, 'b f t c -> (b f) t c').contiguous() 
    return result, n_visual_with_condition

def process_conditon_video_latent_tensor(tensor): 
    # concat self attention with last frame
    # [b, f, t, c]
    _, f, t, _ = tensor.shape
    n_visual = t
    remaining_frames, ref_frame = torch.chunk(tensor, 2, dim=1)
    result = torch.cat((remaining_frames, ref_frame), dim=2) #[b, f, 2*t, c]
    result = rearrange(result, 'b f t c -> (b f) t c').contiguous() 
    return result, n_visual
def recover_tensor(tensor, ratio):
    chunks = torch.chunk(tensor, ratio, dim=1)
    # 对每组进行平均
    averaged_chunks = [chunk.mean(dim=1, keepdim=True) for chunk in chunks]
    # 将平均后的结果拼接起来
    result = torch.cat(averaged_chunks, dim=1)
    return result
def process_gap_conditon_video_latent_tensor(tensor, ref_tensor, ratio): 
    # concat self attention with last frame
    # [b, f, t, c]
    _, f, t, _ = tensor.shape
    n_visual = t
    repeated_ref_tensor = ref_tensor.repeat_interleave(ratio, dim=1)
    result = torch.cat((tensor, repeated_ref_tensor), dim=2) #[b, f, 2*t, c]
    result = rearrange(result, 'b f t c -> (b f) t c').contiguous() 
    return result, n_visual

class LastFrameProjectAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_frames=16):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_frames = num_frames

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        num_frames=self.num_frames
        residual = hidden_states # [b*f,h*w,c]
        hidden_states = rearrange(hidden_states, '(b f) t c -> b f t c',f=num_frames+1)
        ref_feature = hidden_states[:,-1:,:,:]
        if encoder_hidden_states is None:
            is_cross_attention = False
            hidden_states, n_visual = process_latent_tensor(hidden_states) # [b*(f-1),h*w*2,c]
        else:
            is_cross_attention = True
            encoder_hidden_states = rearrange(encoder_hidden_states,'(b f) t c -> b f t c',f=num_frames+1)
            encoder_hidden_states = encoder_hidden_states[:,:num_frames,:,:]
            encoder_hidden_states = rearrange(encoder_hidden_states,'b f t c -> (b f) t c')
            hidden_states = hidden_states[:,:num_frames,:,:]
            hidden_states = rearrange(hidden_states, 'b f t c -> (b f) t c')

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        if xformers_available:
            hidden_states = xformers.ops.memory_efficient_attention(
                query, key, value, attn_bias=attention_mask, scale=attn.scale
            )
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        
        if is_cross_attention:
            hidden_states = rearrange(hidden_states, '(b f) t c -> b f t c', f=num_frames)
            hidden_states = torch.cat((hidden_states, ref_feature), dim=1)
            hidden_states = rearrange(hidden_states, 'b f t c -> (b f) t c')
        else:
            hidden_states = hidden_states[:,:n_visual,:]
            
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class SkipLastFrameAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_frames=16):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_frames = num_frames

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        num_frames=self.num_frames
        residual = hidden_states # [b*f,h*w,c]
        hidden_states = rearrange(hidden_states, '(b f) t c -> b f t c',f=num_frames+1)
        ref_feature = hidden_states[:,-1:,:,:]
        if encoder_hidden_states is None:
            is_cross_attention = False
            hidden_states = hidden_states[:,:num_frames,:,:]
            hidden_states = rearrange(hidden_states, 'b f t c -> (b f) t c')
        else:
            is_cross_attention = True
            encoder_hidden_states = rearrange(encoder_hidden_states,'(b f) t c -> b f t c',f=num_frames+1)
            encoder_hidden_states = encoder_hidden_states[:,:num_frames,:,:]
            encoder_hidden_states = rearrange(encoder_hidden_states,'b f t c -> (b f) t c')
            hidden_states = hidden_states[:,:num_frames,:,:]
            hidden_states = rearrange(hidden_states, 'b f t c -> (b f) t c')

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        if xformers_available:
            hidden_states = xformers.ops.memory_efficient_attention(
                query, key, value, attn_bias=attention_mask, scale=attn.scale
            )
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        
        if is_cross_attention:
            hidden_states = rearrange(hidden_states, '(b f) t c -> b f t c', f=num_frames)
            hidden_states = torch.cat((hidden_states, ref_feature), dim=1)
            hidden_states = rearrange(hidden_states, 'b f t c -> (b f) t c')
        else:
            hidden_states = rearrange(hidden_states, '(b f) t c -> b f t c', f=num_frames)
            hidden_states = torch.cat((hidden_states, ref_feature), dim=1)
            hidden_states = rearrange(hidden_states, 'b f t c -> (b f) t c')
            
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class LastFrameGatedAttnProcessor2_0(nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_frames=16):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_frames = num_frames
        self.fuser_gate_attention = CrossAttention(hidden_size)
        self.fuser_gate_norm1 = nn.LayerNorm(hidden_size)
        self.fuser_gate_alpha_attn=nn.Parameter(torch.tensor(0.0))
        
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        num_frames=self.num_frames
        residual = hidden_states # [b*f,h*w,c]
        hidden_states = rearrange(hidden_states, '(b f) t c -> b f t c',f=num_frames+1)
        ref_feature = hidden_states[:,-1:,:,:]
        if encoder_hidden_states is None:
            is_cross_attention = False
            hidden_states = hidden_states[:,:num_frames,:,:]
            hidden_states = rearrange(hidden_states, 'b f t c -> (b f) t c')
        else:
            is_cross_attention = True
            encoder_hidden_states = rearrange(encoder_hidden_states,'(b f) t c -> b f t c',f=num_frames+1)
            encoder_hidden_states = encoder_hidden_states[:,:num_frames,:,:]
            encoder_hidden_states = rearrange(encoder_hidden_states,'b f t c -> (b f) t c')
            hidden_states = hidden_states[:,:num_frames,:,:]
            hidden_states = rearrange(hidden_states, 'b f t c -> (b f) t c')

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        if xformers_available:
            hidden_states = xformers.ops.memory_efficient_attention(
                query, key, value, attn_bias=attention_mask, scale=attn.scale
            )
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        

        hidden_states = rearrange(hidden_states, '(b f) t c -> b f t c', f=num_frames)
        hidden_states = torch.cat((hidden_states, ref_feature), dim=1)
        hidden_states = rearrange(hidden_states, 'b f t c -> (b f) t c')
            
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        
        # GateAttention
        hidden_states = rearrange(hidden_states, '(b f) t c -> b f t c',f=num_frames+1)
        ref_feature = hidden_states[:,-1:,:,:]
        hidden_states, n_visual = process_latent_tensor(hidden_states) # [b*(f-1),h*w*2,c]
        hidden_states = hidden_states + self.fuser_gate_alpha_attn.tanh() * self.fuser_gate_attention(self.fuser_gate_norm1(hidden_states))
        # hidden_states = hidden_states + self.fuser_gate_alpha_dense.tanh() * self.fuser_gate_ff(self.fuser_gate_norm2(hidden_states))
        hidden_states = rearrange(hidden_states, '(b f) t c -> b f t c', f=num_frames)
        hidden_states = hidden_states[:,:,:n_visual,:]

        hidden_states = torch.cat((hidden_states, ref_feature), dim=1)
        hidden_states = rearrange(hidden_states, 'b f t c -> (b f) t c')
        return hidden_states

class SkipMotionAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, num_frames=16):
        self.num_frames = num_frames
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        ref_token = hidden_states[:,-1:,:]
        hidden_states = hidden_states[:,:self.num_frames,:]
        residual = hidden_states
        
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        if xformers_available:
            hidden_states = xformers.ops.memory_efficient_attention(
                query, key, value, attn_bias=attention_mask, scale=attn.scale
            )
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        
        
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        hidden_states = torch.cat([hidden_states,ref_token],dim=1)
        return hidden_states

class SkipLastMotionAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, num_frames=16):
        self.num_frames = num_frames
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        ref_token = hidden_states[:,-1:,:]
        residual = hidden_states
        
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        if xformers_available:
            hidden_states = xformers.ops.memory_efficient_attention(
                query, key, value, attn_bias=attention_mask, scale=attn.scale
            )
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        
        
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        hidden_states = hidden_states[:,:self.num_frames,:]
        hidden_states = torch.cat([hidden_states,ref_token],dim=1)
        return hidden_states

class XFormersAttnProcessor:
    r"""
    Processor for implementing memory efficient attention using xFormers.

    Args:
        attention_op (`Callable`, *optional*, defaults to `None`):
            The base
            [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to
            use as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best
            operator.
    """

    def __init__(self, attention_op: Optional[Callable] = None):
        self.attention_op = attention_op

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, key_tokens, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    
class NewLastFrameProjectAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_frames=16):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_frames = num_frames

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        use_origin_attn=False,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        num_frames=self.num_frames
        residual = hidden_states # [b*f,h*w,c]
        if use_origin_attn is False:
            hidden_states = rearrange(hidden_states, '(b f) t c -> b f t c',f=num_frames+1)
            ref_feature = hidden_states[:,-1:,:,:]
        if encoder_hidden_states is None and use_origin_attn is False:
            is_cross_attention = False
            hidden_states, n_visual = process_latent_tensor(hidden_states) # [b*(f-1),h*w*2,c]
        elif encoder_hidden_states is None and use_origin_attn is True:
            is_cross_attention = False
        else:
            is_cross_attention = True
            encoder_hidden_states = rearrange(encoder_hidden_states,'(b f) t c -> b f t c',f=num_frames+1)
            encoder_hidden_states = encoder_hidden_states[:,:num_frames,:,:]
            encoder_hidden_states = rearrange(encoder_hidden_states,'b f t c -> (b f) t c')
            hidden_states = hidden_states[:,:num_frames,:,:]
            hidden_states = rearrange(hidden_states, 'b f t c -> (b f) t c')

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        if xformers_available:
            hidden_states = xformers.ops.memory_efficient_attention(
                query, key, value, attn_bias=attention_mask, scale=attn.scale
            )
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        
        if is_cross_attention:
            hidden_states = rearrange(hidden_states, '(b f) t c -> b f t c', f=num_frames)
            hidden_states = torch.cat((hidden_states, ref_feature), dim=1)
            hidden_states = rearrange(hidden_states, 'b f t c -> (b f) t c')
        elif use_origin_attn is False:
            hidden_states = rearrange(hidden_states, '(b f) t c -> b f t c', f=num_frames)
            hidden_states, ref_frame_result = hidden_states[:,:,:n_visual,:], hidden_states[:,:,n_visual:,:]
            ref_frame_result = torch.mean(ref_frame_result, dim=1, keepdim=True)
            hidden_states = torch.cat((hidden_states, ref_frame_result), dim=1)
            hidden_states = rearrange(hidden_states, 'b f t c -> (b f) t c')
            
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class NewLastFrameGatedAttnProcessor2_0(nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_frames=16):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_frames = num_frames
        self.fuser_gate_attention = CrossAttention(hidden_size)
        self.fuser_gate_norm1 = nn.LayerNorm(hidden_size)
        self.fuser_gate_alpha_attn=nn.Parameter(torch.tensor(0.0))
        
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        num_frames=self.num_frames
        residual = hidden_states # [b*f,h*w,c]
        hidden_states = rearrange(hidden_states, '(b f) t c -> b f t c',f=num_frames+1)
        ref_feature = hidden_states[:,-1:,:,:]
        if encoder_hidden_states is None:
            is_cross_attention = False
            hidden_states = hidden_states[:,:num_frames,:,:]
            hidden_states = rearrange(hidden_states, 'b f t c -> (b f) t c')
        else:
            is_cross_attention = True
            encoder_hidden_states = rearrange(encoder_hidden_states,'(b f) t c -> b f t c',f=num_frames+1)
            encoder_hidden_states = encoder_hidden_states[:,:num_frames,:,:]
            encoder_hidden_states = rearrange(encoder_hidden_states,'b f t c -> (b f) t c')
            hidden_states = hidden_states[:,:num_frames,:,:]
            hidden_states = rearrange(hidden_states, 'b f t c -> (b f) t c')

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        if xformers_available:
            hidden_states = xformers.ops.memory_efficient_attention(
                query, key, value, attn_bias=attention_mask, scale=attn.scale
            )
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        

        hidden_states = rearrange(hidden_states, '(b f) t c -> b f t c', f=num_frames)
        hidden_states = torch.cat((hidden_states, ref_feature), dim=1)
        hidden_states = rearrange(hidden_states, 'b f t c -> (b f) t c')
            
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        
        # GateAttention
        hidden_states = rearrange(hidden_states, '(b f) t c -> b f t c',f=num_frames+1)
        ref_feature = hidden_states[:,-1:,:,:]
        hidden_states, n_visual = process_latent_tensor(hidden_states) # [b*(f-1),h*w*2,c]
        hidden_states = hidden_states + self.fuser_gate_alpha_attn.tanh() * self.fuser_gate_attention(self.fuser_gate_norm1(hidden_states))
        # hidden_states = hidden_states + self.fuser_gate_alpha_dense.tanh() * self.fuser_gate_ff(self.fuser_gate_norm2(hidden_states))
        hidden_states = rearrange(hidden_states, '(b f) t c -> b f t c', f=num_frames)
        hidden_states, ref_frame_result = hidden_states[:,:,:n_visual,:], hidden_states[:,:,n_visual:,:]
        ref_frame_result = torch.mean(ref_frame_result, dim=1, keepdim=True)
        hidden_states = torch.cat((hidden_states, ref_frame_result), dim=1)
        hidden_states = rearrange(hidden_states, 'b f t c -> (b f) t c')
        return hidden_states
    
    
class ReferVideoAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_frames=16):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_frames = num_frames

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        num_frames=self.num_frames
        residual = hidden_states # [b*f,h*w,c]
        hidden_states = rearrange(hidden_states, '(b f) t c -> b f t c',f=2*num_frames)
        ref_feature = hidden_states[:,num_frames:,:,:]
        if encoder_hidden_states is None:
            is_cross_attention = False
            hidden_states, n_visual = process_conditon_video_latent_tensor(hidden_states) # [b*(f-1),h*w*2,c]
        else:
            raise NotImplementedError

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        if xformers_available:
            hidden_states = xformers.ops.memory_efficient_attention(
                query, key, value, attn_bias=attention_mask, scale=attn.scale
            )
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        
        if is_cross_attention:
            raise NotImplementedError
        else:
            hidden_states = rearrange(hidden_states, '(b f) t c -> b f t c', f=num_frames)
            hidden_states, ref_frame_result = hidden_states[:,:,:n_visual,:], hidden_states[:,:,n_visual:,:]
            hidden_states = torch.cat((hidden_states, ref_frame_result), dim=1)
            hidden_states = rearrange(hidden_states, 'b f t c -> (b f) t c')
            
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    
    
class ReferSelectVideoAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_frames=16, num_reference_frame=4):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_frames = num_frames
        self.num_reference_frame = num_reference_frame

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        num_frames=self.num_frames
        residual = hidden_states # [b*f,h*w,c]
        hidden_states = rearrange(hidden_states, '(b f) t c -> b f t c',f=num_frames + self.num_reference_frame)
        feature = hidden_states[:,:num_frames,:,:]
        ref_feature = hidden_states[:,num_frames:,:,:]
        ratio = self.num_frames // self.num_reference_frame  # N = 16 // 4 = 4
        if encoder_hidden_states is None:
            is_cross_attention = False
            hidden_states, n_visual = process_gap_conditon_video_latent_tensor(feature, ref_feature, ratio)
        else:
            raise NotImplementedError

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        if xformers_available:
            hidden_states = xformers.ops.memory_efficient_attention(
                query, key, value, attn_bias=attention_mask, scale=attn.scale
            )
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        
        
        if is_cross_attention:
            raise NotImplementedError
        else:
            hidden_states = rearrange(hidden_states, '(b f) t c -> b f t c', f=num_frames)
            hidden_states, ref_frame_result = hidden_states[:,:,:n_visual,:], hidden_states[:,:,n_visual:,:]
            ref_frame_result = recover_tensor(ref_frame_result, self.num_reference_frame)
            hidden_states = torch.cat((hidden_states, ref_frame_result), dim=1)
            hidden_states = rearrange(hidden_states, 'b f t c -> (b f) t c')
            
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class StoryDiffusionAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_frames=16):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_frames = num_frames

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        num_frames=self.num_frames
        residual = hidden_states # [b*f,h*w,c]
        hidden_states = rearrange(hidden_states, '(b f) t c -> b f t c',f=num_frames+1)
        encoder_hidden_states = hidden_states
        encoder_hidden_states, n_visual = process_latent_tensor_story(encoder_hidden_states) # [b*(f-1),h*w*2,c]
        hidden_states = rearrange(hidden_states, 'b f t c -> (b f) t c')

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        if xformers_available:
            hidden_states = xformers.ops.memory_efficient_attention(
                query, key, value, attn_bias=attention_mask, scale=attn.scale
            )
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        

        # hidden_states = rearrange(hidden_states, '(b f) t c -> b f t c', f=num_frames)
        # hidden_states = torch.cat((hidden_states, ref_feature), dim=1)
        # hidden_states = rearrange(hidden_states, 'b f t c -> (b f) t c')
            
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states