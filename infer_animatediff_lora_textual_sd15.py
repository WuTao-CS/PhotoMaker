import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter, AnimateDiffSDXLPipeline, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
import argparse
import os

def load_prompts(prompt_file):
    f = open(prompt_file, 'r')
    prompt_list = []
    for idx, line in enumerate(f.readlines()):
        l = line.strip()
        if len(l) != 0:
            prompt_list.append(l)
        f.close()
    return prompt_list
def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-s", "--seed", type=int, nargs='+',default=[42,128], help="seed for seed_everything")
    parser.add_argument("-p", "--prompt", type=str, default='emjoy.txt', help="prompt file path")
    parser.add_argument("-o","--output", type=str, default='outputs/sd15_lora-textual-1007-cross/checkpoint-30000/', help="output dir")
    parser.add_argument("--model_fold_path", type=str, help="image", default="./checkpoints/sd15_lora-textual-1007-cross/checkpoint-30000")
    parser.add_argument("--name", type=str, default='sd15_lora_textual', help="output name")
    return parser

parser = get_parser()
args = parser.parse_args()
base_model_path = './pretrain_model/Realistic_Vision_V5.1_noVAE'
device = "cuda"
adapter = MotionAdapter.from_pretrained("./pretrain_model/animatediff-motion-adapter-v1-5-3")
# base_model_path = './pretrain_model/stable-diffusion-xl-base-1.0'
scheduler = DDIMScheduler.from_pretrained(
    base_model_path,
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)

pipe = AnimateDiffPipeline.from_pretrained(
    base_model_path,
    motion_adapter=adapter,
    scheduler = scheduler,
).to("cuda")

pipe.load_lora_weights(os.path.join(args.model_fold_path,'pytorch_lora_weights.safetensors'))
# enable memory savings
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()
# pipe.enable_model_cpu_offload()

prompts = load_prompts(args.prompt)
negative_prompt = "asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch, multiple people, text on the screen, no people, unclear faces"
# print(input_id_images[0] if args.ip_adapter else None)
seed_list = args.seed
inject_learnable_token = torch.load(os.path.join(args.model_fold_path,'learnable_token.pt'),map_location='cpu')
for prompt in prompts:
    for seed in seed_list:
        generator = torch.Generator(device=device).manual_seed(seed)
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(prompt=prompt,device=pipe._execution_device,num_images_per_prompt=1,do_classifier_free_guidance=True,negative_prompt=negative_prompt)
        inject_learnable_token = inject_learnable_token.to(device=prompt_embeds.device,dtype=prompt_embeds.dtype)
        prompt_embeds = torch.cat([prompt_embeds[:,:1,:], inject_learnable_token.repeat(prompt_embeds.shape[0], 1, 1), prompt_embeds[:,1:,:]], dim=1)
        prompt_embeds = prompt_embeds[:, :-2, :]
        frames = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_frames=16,
            guidance_scale=8,
            num_videos_per_prompt=1,
            generator=generator,
        ).frames[0]
        os.makedirs(args.output, exist_ok=True)
        export_to_gif(frames, "{}/{}_{}_seed_{}.gif".format(args.output, args.name, prompt.replace(' ','_'),seed))
# load_lora_weights