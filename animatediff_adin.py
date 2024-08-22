import torch
from diffusers.models import MotionAdapter
from diffusers import AnimateDiffSDXLPipeline, DDIMScheduler, AnimateDiffPipeline
from diffusers.utils import export_to_gif, load_image
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from diffusers.pipelines.animatediff import AnimateDiffPipelineOutput
import sa_handler
adapter = MotionAdapter.from_pretrained("/mnt/nfs/data/pretrained_models/animatediff-motion-adapter-v1-5-3", torch_dtype=torch.float16)

model_id = "./pretrain_model/Realistic_Vision_V5.1_noVAE/"
scheduler = DDIMScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)
pipe = AnimateDiffPipeline.from_pretrained(
    model_id,
    motion_adapter=adapter,
    scheduler=scheduler,
    torch_dtype=torch.float16,
).to("cuda")


# pipe.load_ip_adapter("./pretrain_model/IP-Adapter", subfolder="models", weight_name='ip-adapter-plus-face_sd15.bin')

handler = sa_handler.Handler(pipe)
sa_args = sa_handler.StyleAlignedArgs(share_group_norm=False,
                                      share_layer_norm=False,
                                      share_attention=False,
                                      adain_queries=True,
                                      adain_keys=True,
                                      adain_values=True,
                                      adain_hidden=True,
                                      with_motion_layer=True,
                                      with_self_attn= False,
                                     )

handler.register(sa_args, )
# # enable memory savings
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()
# pipe.enable_xformers_memory_efficient_attention()
# pipe.enable_sequential_cpu_offload()
pipe.enable_model_cpu_offload()

# ref_img = load_image("examples/scarletthead_woman/scarlett_0.jpg")
# ref_img_emb = pipe.prepare_ip_adapter_image_embeds([ref_img], None, torch.device("cuda"), 1, True)[0]

sets_of_prompts = [
  "a car is running in the road.",
  "a ship is running in the road.",
#   "a panda is running in the forest. realistic, high quality",
]
seed = 1234
output = pipe(
    prompt=sets_of_prompts,
    num_inference_steps=25,
    guidance_scale=9,
    negative_prompt=["bad quality, worse quality","bad quality, worse quality"],
    num_frames=16,
    generator=torch.Generator("cpu").manual_seed(seed),
)
export_to_gif(output.frames[0], "animation_0_base_car_{}.gif".format(seed))
export_to_gif(output.frames[1], "animation_1_qkv_tmp_attn_ship_mean_var_{}.gif".format(seed))