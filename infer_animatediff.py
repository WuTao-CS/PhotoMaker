import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif

# Load the motion adapter
adapter = MotionAdapter.from_pretrained("./pretrain_model/animatediff-motion-adapter-v1-5-2")
# load SD 1.5 based finetuned model
base_model_path = './pretrain_model/stable-diffusion-v1-5'
pipe = AnimateDiffPipeline.from_pretrained(
    base_model_path,
    motion_adapter=adapter,
    variant="fp16",
).to("cuda")
scheduler = DDIMScheduler.from_pretrained(
    base_model_path,
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)
pipe.scheduler = scheduler
latent = torch.load("datasets/CeleV-Text/processed_sd15/--qMwBtoejo_0_0.pt",map_location='cpu')
# enable memory savings
pipe.enable_vae_slicing()
# pipe.enable_model_cpu_offload()

output = pipe(
    negative_prompt="bad quality, worse quality",
    num_frames=16,
    prompt_embeds=latent['prompt_embeds'].unsqueeze(dim=0),
    guidance_scale=7.5,
    num_inference_steps=25,
    generator=torch.Generator("cpu").manual_seed(128),
)
frames = output.frames[0]
export_to_gif(frames, "animation_test_prompt.gif")