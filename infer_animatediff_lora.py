import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter, AnimateDiffSDXLPipeline
from diffusers.utils import export_to_gif

# Load the motion adapter
adapter = MotionAdapter.from_pretrained("./pretrain_model/animatediff-motion-adapter-sdxl-beta")
# load SD 1.5 based finetuned model
base_model_path = './pretrain_model/stable-diffusion-xl-base-1.0'
pipe = AnimateDiffSDXLPipeline.from_pretrained(
    base_model_path,
    motion_adapter=adapter,
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
pipe.load_lora_weights('checkpoints/train_sdxl_lora/checkpoint-30000/pytorch_lora_weights.safetensors')
# enable memory savings
pipe.enable_vae_slicing()
# pipe.enable_model_cpu_offload()

output = pipe(
    prompt="a man is reading a book",
    negative_prompt="bad quality, worse quality",
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=25,
    generator=torch.Generator("cpu").manual_seed(128),
)
frames = output.frames[0]
export_to_gif(frames, "animation_test_prompt.gif")

# load_lora_weights