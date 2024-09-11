
import torch
from diffusers import AutoencoderKL
from diffusers.video_processor import VideoProcessor

latent = torch.load("datasets/CeleV-Text/processed_sd15_fix/--qMwBtoejo_0_0.pt",map_location='cpu')
latent = latent['ref_images_latent']
latent=latent.unsqueeze(0)
print(latent.shape)


def decode_latents(vae, latents):
    latents = 1 / vae.config.scaling_factor * latents

    batch_size, channels, num_frames, height, width = latents.shape
    latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)
    with torch.no_grad():
        image = vae.decode(latents).sample
    video = image[None, :].reshape((batch_size, num_frames, -1) + image.shape[2:]).permute(0, 2, 1, 3, 4)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    video = video.float().detach()
    return video

vae = AutoencoderKL.from_pretrained("./pretrain_model/stable-diffusion-v1-5", subfolder="vae",use_safetensors=True)
vae = vae.to("cuda")
vae.enable_slicing()
vae.enable_tiling()
video_processor = VideoProcessor(vae_scale_factor=vae.config.scaling_factor)
latents = latent.to("cuda")

print("vae.config.scaling_factor",vae.config.scaling_factor)
latents = latents * vae.config.scaling_factor

video = decode_latents(vae,latents)
video = video_processor.postprocess_video(video=video, output_type='pil')
video = video[0]
from diffusers.utils import export_to_gif, load_image
export_to_gif(video, "video.gif")