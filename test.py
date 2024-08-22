import torch
from diffusers import AutoencoderKL
from diffusers.video_processor import VideoProcessor

latent = torch.load("/group/40034/jackeywu/code/PhotoMaker/datasets/CeleV-Text/processed/___5yD2BVx8_4_0.pt",map_location='cpu')
latent = latent['latent']
latent=latent.unsqueeze(0)
print(latent.shape)


def decode_latents(vae, latents):
    latents = 1 / vae.config.scaling_factor * latents

    batch_size, channels, num_frames, height, width = latents.shape
    latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)

    image = vae.decode(latents).sample
    video = image[None, :].reshape((batch_size, num_frames, -1) + image.shape[2:]).permute(0, 2, 1, 3, 4)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    video = video.float()
    return video

vae = AutoencoderKL.from_pretrained("./pretrain_model/RealVisXL_V4.0", subfolder="vae",use_safetensors=True)
vae = vae.to("cuda")
# vae.enable_slicing()
# vae.enable_tiling()
video_processor = VideoProcessor(vae_scale_factor=2 ** (len(vae.config.block_out_channels) - 1))
latents = latent.to("cuda")
# video_tensor = decode_latents(vae,latent)

print("vae.config.scaling_factor",vae.config.scaling_factor)
# latents = 1 / vae.config.scaling_factor * latents
latents = latents[:,:,:1,:,:]
batch_size, channels, num_frames, height, width = latents.shape
print(batch_size, channels, num_frames, height, width)
latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)
print(latents.shape)
image = vae.decode(latents).sample
video = image[None, :].reshape((batch_size, num_frames, -1) + image.shape[2:]).permute(0, 2, 1, 3, 4)
# we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
video = video.float().detach()
video = video_processor.postprocess_video(video=video, output_type='pil')
video = video[0]
from diffusers.utils import export_to_gif, load_image
export_to_gif(video, "video.gif")