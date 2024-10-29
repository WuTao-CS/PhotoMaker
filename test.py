
import torch
from diffusers import AutoencoderKL
from diffusers.video_processor import VideoProcessor
import os
import numpy as np
from PIL import Image
from einops import rearrange



def img_resize(img_folder, resize_h, resize_w):
    for img_path in os.listdir(img_folder):
        Image.open(os.path.join(img_folder, img_path)).resize((resize_h, resize_w)). \
            save(os.path.join(img_folder, img_path))


def dct(x, norm=None):
    '''
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter 'norm', see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    '''
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)
    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    Vc = torch.view_as_real(torch.fft.fft(v, dim=1))
    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)
    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2
    V = 2 * V.view(*x_shape)
    return V


def idct(X, norm=None):
    '''
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter 'norm', see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    '''
    x_shape = X.shape
    N = x_shape[-1]
    X_v = X.contiguous().view(-1, x_shape[-1]) / 2
    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2
    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)
    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)
    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r
    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]
    return x.view(*x_shape)


def dct_2d(x, norm=None):
    '''
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter 'norm', see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT_II of the signal over the last 2 dimensions
    '''
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    '''
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_2d(dct_2d(x)) == x
    For the meaning of the parameter 'norm', see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimension
    '''
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def dct_3d(x, norm=None):
    '''
    3-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter 'norm', see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT_II of the signal over the last 3 dimensions
    '''
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    X3 = dct(X2.transpose(-1, -3), norm=norm)
    return X3.transpose(-1, -3).transpose(-1, -2)


def idct_3d(X, norm=None):
    '''
    The inverse to 3D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_3d(dct_3d(x)) == x
    For the meaning of the parameter 'norm', see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimension
    '''
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    x3 = idct(x2.transpose(-1, -3), norm=norm)
    return x3.transpose(-1, -3).transpose(-1, -2)


def low_pass(dct, threshold):
    '''
    dct: tensor of ... h, w shape
    threshold: integer number above which to zero out
    '''
    h, w = dct.shape[-2], dct.shape[-1]
    assert (threshold >= 0) and (threshold <= h + w - 2), 'invalid value of threshold'
    vertical = torch.range(0, h-1)[..., None].repeat(1, w).cuda()
    horizontal = torch.range(0, w-1)[None, ...].repeat(h, 1).cuda()
    mask = vertical + horizontal
    while len(mask.shape) != len(dct.shape):
        mask = mask[None, ...]
    dct = torch.where(mask > threshold, torch.zeros_like(dct), dct)
    return dct



def high_pass(dct, threshold):
    '''
    dct: tensor of ... h, w shape
    threshold: integer number below which to zero out
    '''
    h, w = dct.shape[-2], dct.shape[-1]
    assert (threshold >= 0) and (threshold <= h + w - 1), 'invalid value of threshold'
    vertical = torch.range(0, h-1)[..., None].repeat(1, w).cuda()
    horizontal = torch.range(0, w-1)[None, ...].repeat(h, 1).cuda()
    mask = vertical + horizontal
    print(torch.max(mask),torch.min(mask))
    while len(mask.shape) != len(dct.shape):
        mask = mask[None, ...]
    dct = torch.where(mask < threshold, torch.zeros_like(dct), dct)
    
    return dct

def encode_video_with_vae(vae, video):
    video_length = video.shape[1]
    pixel_values = rearrange(video, "b f c h w -> (b f) c h w")
    with torch.no_grad():
        latents = vae.encode(pixel_values).latent_dist
    latents = latents.sample()
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
    return latents

latent = torch.load("/group/40007/public_datasets/CeleV-Text/processed_sd15/ApLhtzsal8k_16_0.pt",map_location='cpu')
ref_images_latent = latent['new_ref_images_latent'].to('cuda').unsqueeze(0)
latent = latent['latent'].to('cuda').unsqueeze(0)
latent = ref_images_latent



def decode_latents(vae, latents):
    latents = 1 / vae.config.scaling_factor * latents

    batch_size, channels, num_frames, height, width = latents.shape
    latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)
    print(latents.shape)
    with torch.no_grad():
        image = vae.decode(latents).sample
    video = image[None, :].reshape((batch_size, num_frames, -1) + image.shape[2:]).permute(0, 2, 1, 3, 4)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    video = video.float().detach()
    return video

vae = AutoencoderKL.from_pretrained("./pretrain_model/Realistic_Vision_V5.1_noVAE", subfolder="vae")
vae = vae.to("cuda")
vae.enable_slicing()
vae.enable_tiling()
# latent = encode_video_with_vae(vae, latent)
video_processor = VideoProcessor(vae_scale_factor=vae.config.scaling_factor)
latents = latent.to("cuda")
# dct_latents = dct_2d(latents, norm='ortho')
# print(torch.max(dct_latents),torch.min(dct_latents))
# dct_latents= high_pass(dct_latents,threshold=40)
# low_dct_latents = idct_2d(dct_latents, norm='ortho')
print("vae.config.scaling_factor",vae.config.scaling_factor)
latents = latents * vae.config.scaling_factor

video = decode_latents(vae,latents)
print(video.shape)
# video, ref = video[:,:,:16,:,:],video[:,:,-1:,:,:]
video = video_processor.postprocess_video(video=video, output_type='pil')[0]
# ref = video_processor.postprocess_video(video=ref, output_type='pil')
# video = video[0]
# ref = ref[0]
from diffusers.utils import export_to_gif, load_image
export_to_gif(video, "video_source.gif")
# export_to_gif(ref, "video_ref.gif")
