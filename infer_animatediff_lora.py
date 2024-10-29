import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter, AnimateDiffSDXLPipeline, EulerDiscreteScheduler
from diffusers.utils import export_to_gif

# Load the motion adapter
adapter = MotionAdapter.from_pretrained("./pretrain_model/animatediff-motion-adapter-sdxl-beta")
# load SD 1.5 based finetuned model
base_model_path = './pretrain_model/RealVisXL_V4.0'
# base_model_path = './pretrain_model/stable-diffusion-xl-base-1.0'
# scheduler = DDIMScheduler.from_pretrained(
#     base_model_path,
#     subfolder="scheduler",
#     clip_sample=False,
#     timestep_spacing="linspace",
#     beta_schedule="linear",
#     steps_offset=1,
# )
scheduler = EulerDiscreteScheduler.from_pretrained(
    base_model_path,
    subfolder="scheduler",
    timestep_spacing='leading', 
    steps_offset=1,	
    beta_schedule="scaled_linear",
    beta_start=0.00085,
    beta_end=0.020
)
pipe = AnimateDiffSDXLPipeline.from_pretrained(
    base_model_path,
    motion_adapter=adapter,
    scheduler = scheduler,
).to("cuda")

# pipe.load_lora_weights('checkpoints/train_sdxl_lora/checkpoint-85000/pytorch_lora_weights.safetensors')
# enable memory savings
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()
# pipe.enable_model_cpu_offload()

output = pipe(
    prompt="a man is dancing in the rain. bust shot",
    negative_prompt="bad quality, worse quality",
    height=1024,
    width=1024,
    num_frames=16,
    guidance_scale=8,
    num_inference_steps=25,
    generator=torch.Generator("cpu").manual_seed(42),
)
frames = output.frames[0]
export_to_gif(frames, "animation_test_prompt_bost_shot_1024.gif")

# load_lora_weights