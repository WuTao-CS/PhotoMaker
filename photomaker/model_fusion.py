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
from diffusers.loaders.lora import StableDiffusionXLLoraLoaderMixin
from diffusers.loaders.ip_adapter import IPAdapterMixin
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
import itertools
from diffusers.loaders import AttnProcsLayers
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
from .model import PhotoMakerIDEncoder
from .attention_processor import MixIPAdapterAttnProcessor2_0
from diffusers.models.attention_processor import (
    AttnProcessor,
    AttnProcessor2_0,
)
class MixIDModel(nn.Module, StableDiffusionXLLoraLoaderMixin,IPAdapterMixin):
    def __init__(self, text_encoder, tokenizer, tokenizer_2, text_encoder_2, image_encoder, vae, unet):
        super().__init__()
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.text_encoder_2 = text_encoder_2
        self.image_encoder = image_encoder
        self.vae = vae
        self.unet = unet
        self.use_ema = False
        self.ema_param = None
        
        self.id_image_processor = None
        self.id_encoder = None

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path):
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae",
            # local_files_only=True, 
            use_safetensors=True, 
            # variant="fp16"
        )
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
            # local_files_only=True,
            use_safetensors=True, 
            # variant="fp16"
        )
        motion_adapter = MotionAdapter.from_pretrained("pretrain_model/animatediff-motion-adapter-sdxl-beta")
        unet = UNetMotionModel.from_unet2d(unet, motion_adapter)
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            # local_files_only=True, 
            use_safetensors=True, 
            # variant="fp16"
        )
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            # local_files_only=True, 
            use_safetensors=True, 
            # variant="fp16"
        )
        tokenizer_2 = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_2")
        # TODO: can skip
        image_encoder_path = "./pretrain_model/CLIP-ViT-H-14-laion2B-s32B-b79K"
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path, torch_dtype=torch.float16)
        


        # if args.use_clip_loss:
        #     freeze_image_encoder = IDAdapterCLIPImageEncoder.from_pretrained(
        #         args.image_encoder_name_or_path,
        #         args.freeze_image_encoder_pretrained_path,
        #         use_clip_loss=args.use_clip_loss,
        #     )
        # else:
        #     freeze_image_encoder = None 

        return MixIDModel(text_encoder, tokenizer, tokenizer_2, text_encoder_2, image_encoder, vae, unet)
    
    def load_photomaker_adapter(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        weight_name: str,
        subfolder: str = '',
        trigger_word: str = 'img',
        **kwargs,
    ):

        # Load the main state dict first.
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }

        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            model_file = _get_model_file(
                pretrained_model_name_or_path_or_dict,
                weights_name=weight_name,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
            )
            if weight_name.endswith(".safetensors"):
                state_dict = {"id_encoder": {}, "lora_weights": {}}
                with safe_open(model_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key.startswith("id_encoder."):
                            state_dict["id_encoder"][key.replace("id_encoder.", "")] = f.get_tensor(key)
                        elif key.startswith("lora_weights."):
                            state_dict["lora_weights"][key.replace("lora_weights.", "")] = f.get_tensor(key)
            else:
                state_dict = torch.load(model_file, map_location="cpu")
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        keys = list(state_dict.keys())
        if keys != ["id_encoder", "lora_weights"]:
            raise ValueError("Required keys are (`id_encoder` and `lora_weights`) missing from the state dict.")

        self.trigger_word = trigger_word
        # load finetuned CLIP image encoder and fuse module here if it has not been registered to the pipeline yet
        print(f"Loading PhotoMaker components [1] id_encoder from [{pretrained_model_name_or_path_or_dict}]...")
        id_encoder = PhotoMakerIDEncoder()
        id_encoder.load_state_dict(state_dict["id_encoder"], strict=True)
        id_encoder = id_encoder.to(self.device, dtype=self.unet.dtype)    
        self.id_encoder = id_encoder
        self.id_image_processor = CLIPImageProcessor()

        # load lora into models
        print(f"Loading PhotoMaker components [2] lora_weights from [{pretrained_model_name_or_path_or_dict}]")
        self.load_lora_weights(state_dict["lora_weights"], adapter_name="photomaker")

        # Add trigger word token
        if self.tokenizer is not None: 
            self.tokenizer.add_tokens([self.trigger_word], special_tokens=True)
        
        self.tokenizer_2.add_tokens([self.trigger_word], special_tokens=True)

    def init_model(self):
        self.load_photomaker_adapter(
            "./pretrain_model/PhotoMaker",
            subfolder="",
            weight_name="photomaker-v1.bin",
            trigger_word="img"
        )
        self.load_ip_adapter(["./pretrain_model/IP-Adapter","./pretrain_model/IP-Adapter-FaceID/"], subfolder=["sdxl_models",None], weight_name=['ip-adapter-plus-face_sdxl_vit-h.bin',"ip-adapter-faceid-portrait_sdxl.bin"], image_encoder_folder=None)
        attn_procs = {}
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
            if cross_attention_dim is None or "motion_modules" in name or "encoder_hid_proj" in name:
                attn_procs[name] = AttnProcessor2_0()
            elif "fusion" in name:
                continue
            else:
                attn_procs[name] = MixIPAdapterAttnProcessor2_0(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.unet.attn_processors[name].num_tokens,
                ).to(self.device, dtype=torch.float16)
        self.unet.set_attn_processor(attn_procs)

        return

    def encode_prompt_with_trigger_word(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        num_id_images: int = 1,
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        class_tokens_mask: Optional[torch.LongTensor] = None,
    ):
        device = device or self._execution_device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Find the token id of the trigger word
        image_token_id = self.tokenizer_2.convert_tokens_to_ids(self.trigger_word)

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_embeds_list = []
            prompts = [prompt, prompt_2]
            for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
                input_ids = tokenizer.encode(prompt) # TODO: batch encode
                clean_index = 0
                clean_input_ids = []
                class_token_index = []
                # Find out the corresponding class word token based on the newly added trigger word token
                for i, token_id in enumerate(input_ids):
                    if token_id == image_token_id:
                        class_token_index.append(clean_index - 1)
                    else:
                        clean_input_ids.append(token_id)
                        clean_index += 1

                # if len(class_token_index) != 1:
                #     raise ValueError(
                #         f"PhotoMaker currently does not support multiple trigger words in a single prompt.\
                #             Trigger word: {self.trigger_word}, Prompt: {prompt}."
                #     )
                class_token_index = class_token_index[0]

                # Expand the class word token and corresponding mask
                class_token = clean_input_ids[class_token_index]
                clean_input_ids = clean_input_ids[:class_token_index] + [class_token] * num_id_images + \
                    clean_input_ids[class_token_index+1:]                
                    
                # Truncation or padding
                max_len = tokenizer.model_max_length
                if len(clean_input_ids) > max_len:
                    clean_input_ids = clean_input_ids[:max_len]
                else:
                    clean_input_ids = clean_input_ids + [tokenizer.pad_token_id] * (
                        max_len - len(clean_input_ids)
                    )

                class_tokens_mask = [True if class_token_index <= i < class_token_index+num_id_images else False \
                     for i in range(len(clean_input_ids))]
                
                clean_input_ids = torch.tensor(clean_input_ids, dtype=torch.long).unsqueeze(0)
                class_tokens_mask = torch.tensor(class_tokens_mask, dtype=torch.bool).unsqueeze(0)
                
                prompt_embeds = text_encoder(
                    clean_input_ids.to(device),
                    output_hidden_states=True,
                )

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]
                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        class_tokens_mask = class_tokens_mask.to(device=device) # TODO: ignoring two-prompt case

        return prompt_embeds, pooled_prompt_embeds, class_tokens_mask
    
    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids
    
    def forward(self, batch, noise_scheduler):
        latents = batch["latents"]
        prompt = batch["prompt"]
        # prompt_embeds = batch["prompt_embeds"]
        # pooled_prompt_embeds = batch["pooled_prompt_embeds"]
        input_id_images = batch["input_id_images"]
        clip_image_embed = batch["clip_image_embed"]
        face_id_embed = batch["face_id_embed"]

        original_size = (1024, 1024)
        crops_coords_top_left = (0, 0)
        target_size = (1024, 1024)
        if torch.any(torch.isnan(latents)):
            print("NaN found in latents, replacing with zeros")
            latents = torch.where(torch.isnan(latents), torch.zeros_like(latents), latents)
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device
        )
        timesteps = timesteps.long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {noise_scheduler.config.prediction_type}"
            )
        
        if not isinstance(input_id_images[0], torch.Tensor):
            id_pixel_values = self.id_image_processor(input_id_images, return_tensors="pt").pixel_values

        id_pixel_values = id_pixel_values.unsqueeze(0).to(device=self.device, dtype=self.dtype) # TODO: multiple prompts
        with torch.no_grad():
        # Get the update text embedding with the stacked ID embedding
            (
                prompt_embeds,
                pooled_prompt_embeds,
                class_tokens_mask,
            ) = self.encode_prompt_with_trigger_word(
                prompt=prompt,
                prompt_2=prompt,
                device=self.device,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                class_tokens_mask=class_tokens_mask,
            )
            prompt_embeds, _ip_adapter_image_embeds = self.id_encoder(id_pixel_values, prompt_embeds, class_tokens_mask)

        batch_size = prompt_embeds.shape[0]
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
        )
        add_time_ids = add_time_ids.to(device=latents.device).repeat(batch_size, 1)       
        added_cond_kwargs = {"text_embeds": add_text_embeds.to(noisy_latents.dtype), "time_ids": add_time_ids.to(noisy_latents.dtype)}
        added_cond_kwargs["image_embeds"] = [clip_image_embed, face_id_embed]

        num_frames = noisy_latents.shape[2]
        timesteps = timesteps.expand(noisy_latents.shape[0])

        t_emb = self.unet.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        timestep_cond = None
        t_emb = t_emb.to(dtype=noisy_latents.dtype)
        t_emb = self.unet.time_embedding(t_emb, timestep_cond)
        cross_attention_kwargs={}
        cross_attention_kwargs['temb'] = t_emb

        # prompt_embeds.to(noisy_latents.dtype)
        pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=prompt_embeds, added_cond_kwargs=added_cond_kwargs,cross_attention_kwargs=cross_attention_kwargs).sample
        denoise_loss = F.mse_loss(pred.float(), target.float(), reduction="mean")

        return_dict = {"denoise_loss": denoise_loss}
        return return_dict

class MixIDTrainModel(nn.Module,StableDiffusionXLLoraLoaderMixin,IPAdapterMixin):
    def __init__(self, unet, video_length=16):
        super().__init__()
        self.unet = unet
        self.adapter_modules = None
        self.video_length = video_length
        self.init_model()
    
    def from_pretrained(pretrained_model_name_or_path, video_length=16):
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
            # local_files_only=True,
            use_safetensors=True, 
            # variant="fp16"
        )
        motion_adapter = MotionAdapter.from_pretrained("pretrain_model/animatediff-motion-adapter-sdxl-beta")
        unet = UNetMotionModel.from_unet2d(unet, motion_adapter)
        
        return MixIDTrainModel(unet, video_length=video_length)
    
    def load_photomaker_adapter(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        weight_name: str,
        subfolder: str = '',
        trigger_word: str = 'img',
        **kwargs,
    ):
        # Load the main state dict first.
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }

        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            model_file = _get_model_file(
                pretrained_model_name_or_path_or_dict,
                weights_name=weight_name,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
            )
            if weight_name.endswith(".safetensors"):
                state_dict = {"id_encoder": {}, "lora_weights": {}}
                with safe_open(model_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key.startswith("id_encoder."):
                            state_dict["id_encoder"][key.replace("id_encoder.", "")] = f.get_tensor(key)
                        elif key.startswith("lora_weights."):
                            state_dict["lora_weights"][key.replace("lora_weights.", "")] = f.get_tensor(key)
            else:
                state_dict = torch.load(model_file, map_location="cpu")
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        keys = list(state_dict.keys())
        if keys != ["id_encoder", "lora_weights"]:
            raise ValueError("Required keys are (`id_encoder` and `lora_weights`) missing from the state dict.")

        self.trigger_word = trigger_word

        # load lora into models
        print(f"Loading PhotoMaker components [1] lora_weights from [{pretrained_model_name_or_path_or_dict}]")
        self.unet.load_attn_procs(state_dict["lora_weights"], adapter_name="photomaker")
        # self.load_lora_weights(state_dict["lora_weights"], adapter_name="photomaker")
    
    def init_model(self):
        self.load_photomaker_adapter(
            "./pretrain_model/PhotoMaker",
            subfolder="",
            weight_name="photomaker-v1.bin",
            trigger_word="img"
        )
        self.load_ip_adapter(["./pretrain_model/IP-Adapter","./pretrain_model/IP-Adapter-FaceID/"], subfolder=["sdxl_models",None], weight_name=['ip-adapter-plus-face_sdxl_vit-h.bin',"ip-adapter-faceid-portrait_sdxl.bin"], image_encoder_folder=None)
        attn_procs = {}
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
            if cross_attention_dim is None or "motion_modules" in name or "encoder_hid_proj" in name:
                attn_procs[name] = AttnProcessor2_0()
            elif "fusion" in name:
                continue
            else:
                attn_procs[name] = MixIPAdapterAttnProcessor2_0(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.unet.attn_processors[name].num_tokens,
                    video_length=self.video_length
                ).to(dtype=torch.float16)
        self.unet.set_attn_processor(attn_procs)
        # self.adapter_modules = AttnProcsLayers(self.unet.attn_processors)
        self.adapter_modules = self.unet.attn_processors.values()
        return

    def get_learnable_parameters(self):
        learn_params = []
        for proc in self.adapter_modules:
            if isinstance(proc, (MixIPAdapterAttnProcessor2_0)):
                learn_params.append(list(proc.fusion.parameters()))
        return list(itertools.chain(*learn_params))
    
    def freeze_parameters(self):
        for name, parm in self.named_parameters():
            if "attn2.processor.fusion" in name:
                parm.requires_grad = True
            else:
                parm.requires_grad = False
    # def get_learnable_parameters(self):
    #     return self.adapter_modules.parameters()
    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=1280
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids
    
    def forward(self, batch, noise_scheduler):
        latents = batch["video"]
        clip_emb = batch["clip_emb"]
        face_id_embed = batch["faces_id"]
        prompt_embeds = batch["prompt_embeds"]
        pooled_prompt_embeds = batch["pooled_prompt_embeds"]
    
        original_size = (1024, 1024)
        crops_coords_top_left = (0, 0)
        target_size = (1024, 1024)
        if torch.any(torch.isnan(latents)):
            print("NaN found in latents, replacing with zeros")
            latents = torch.where(torch.isnan(latents), torch.zeros_like(latents), latents)
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device
        )
        timesteps = timesteps.long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        
        batch_size = prompt_embeds.shape[0]
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
        )
        add_time_ids = add_time_ids.to(device=latents.device).repeat(batch_size, 1)       
        added_cond_kwargs = {"text_embeds": add_text_embeds.to(noisy_latents.dtype), "time_ids": add_time_ids.to(noisy_latents.dtype)}
        added_cond_kwargs["image_embeds"] = [clip_emb, face_id_embed]
        timesteps = timesteps.expand(noisy_latents.shape[0])

        t_emb = self.unet.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        timestep_cond = None
        t_emb = t_emb.to(device=latents.device,dtype=noisy_latents.dtype)
        t_emb = self.unet.time_embedding(t_emb, timestep_cond)
        cross_attention_kwargs={}
        cross_attention_kwargs['temb'] = t_emb
        # prompt_embeds.to(noisy_latents.dtype)
        pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=prompt_embeds, added_cond_kwargs=added_cond_kwargs,cross_attention_kwargs=cross_attention_kwargs,return_dict=False)[0]

        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        elif noise_scheduler.config.prediction_type == "sample":
            # We set the target to latents here, but the model_pred will return the noise sample prediction.
            target = latents
            pred = pred - noise
        else:
            raise ValueError(
                f"Unknown prediction type {noise_scheduler.config.prediction_type}"
            )
        denoise_loss = F.mse_loss(pred.float(), target.float(), reduction="mean")

        return_dict = {"denoise_loss": denoise_loss}
        return return_dict