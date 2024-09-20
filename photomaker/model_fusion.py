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
from diffusers.loaders import StableDiffusionXLLoraLoaderMixin
from diffusers.loaders.ip_adapter import IPAdapterMixin
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
import itertools
from diffusers.loaders import AttnProcsLayers
# from .attention_processor import MixIPAdapterAttnProcessor2_0
from diffusers.training_utils import EMAModel, compute_snr
from einops import rearrange, repeat
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import (
    USE_PEFT_BACKEND,
    _get_model_file,
    is_accelerate_available,
    is_torch_version,
    is_transformers_available,
    logging,
)
from pathlib import Path
from diffusers.models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT, load_state_dict
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
from huggingface_hub.utils import validate_hf_hub_args
logger = logging.get_logger(__name__)

class MixIDTrainModel(nn.Module,StableDiffusionXLLoraLoaderMixin,IPAdapterMixin):
    def __init__(self, unet, video_length=16):
        super().__init__()
        self.unet = unet
        self.video_length = video_length
        self.hf_device_map = "auto"
        self.weight_dtype = None
        # self.init_model()
    @validate_hf_hub_args
    def load_ip_adapter(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, List[str], Dict[str, torch.Tensor]],
        subfolder: Union[str, List[str]],
        weight_name: Union[str, List[str]],
        image_encoder_folder: Optional[str] = "image_encoder",
        **kwargs,
    ):
        """
        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `List[str]` or `os.PathLike` or `List[os.PathLike]` or `dict` or `List[dict]`):
                Can be either:

                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                      with [`ModelMixin.save_pretrained`].
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).
            subfolder (`str` or `List[str]`):
                The subfolder location of a model file within a larger model repository on the Hub or locally. If a
                list is passed, it should have the same length as `weight_name`.
            weight_name (`str` or `List[str]`):
                The name of the weight file to load. If a list is passed, it should have the same length as
                `weight_name`.
            image_encoder_folder (`str`, *optional*, defaults to `image_encoder`):
                The subfolder location of the image encoder within a larger model repository on the Hub or locally.
                Pass `None` to not load the image encoder. If the image encoder is located in a folder inside
                `subfolder`, you only need to pass the name of the folder that contains image encoder weights, e.g.
                `image_encoder_folder="image_encoder"`. If the image encoder is located in a folder other than
                `subfolder`, you should pass the path to the folder that contains image encoder weights, for example,
                `image_encoder_folder="different_subfolder/image_encoder"`.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.

            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
        """

        # handle the list inputs for multiple IP Adapters
        if not isinstance(weight_name, list):
            weight_name = [weight_name]

        if not isinstance(pretrained_model_name_or_path_or_dict, list):
            pretrained_model_name_or_path_or_dict = [pretrained_model_name_or_path_or_dict]
        if len(pretrained_model_name_or_path_or_dict) == 1:
            pretrained_model_name_or_path_or_dict = pretrained_model_name_or_path_or_dict * len(weight_name)

        if not isinstance(subfolder, list):
            subfolder = [subfolder]
        if len(subfolder) == 1:
            subfolder = subfolder * len(weight_name)

        if len(weight_name) != len(pretrained_model_name_or_path_or_dict):
            raise ValueError("`weight_name` and `pretrained_model_name_or_path_or_dict` must have the same length.")

        if len(weight_name) != len(subfolder):
            raise ValueError("`weight_name` and `subfolder` must have the same length.")

        # Load the main state dict first.
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)

        if low_cpu_mem_usage and not is_accelerate_available():
            low_cpu_mem_usage = False
            logger.warning(
                "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
                " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
                " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
                " install accelerate\n```\n."
            )

        if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `low_cpu_mem_usage=False`."
            )

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }
        state_dicts = []
        for pretrained_model_name_or_path_or_dict, weight_name, subfolder in zip(
            pretrained_model_name_or_path_or_dict, weight_name, subfolder
        ):
            if not isinstance(pretrained_model_name_or_path_or_dict, dict):
                model_file = _get_model_file(
                    pretrained_model_name_or_path_or_dict,
                    weights_name=weight_name,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                )
                if weight_name.endswith(".safetensors"):
                    state_dict = {"image_proj": {}, "ip_adapter": {}}
                    with safe_open(model_file, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            if key.startswith("image_proj."):
                                state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                            elif key.startswith("ip_adapter."):
                                state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
                else:
                    state_dict = load_state_dict(model_file)
            else:
                state_dict = pretrained_model_name_or_path_or_dict

            keys = list(state_dict.keys())
            if keys != ["image_proj", "ip_adapter"]:
                raise ValueError("Required keys are (`image_proj` and `ip_adapter`) missing from the state dict.")

            state_dicts.append(state_dict)

            # load CLIP image encoder here if it has not been registered to the pipeline yet
            if hasattr(self, "image_encoder") and getattr(self, "image_encoder", None) is None:
                if image_encoder_folder is not None:
                    if not isinstance(pretrained_model_name_or_path_or_dict, dict):
                        logger.info(f"loading image_encoder from {pretrained_model_name_or_path_or_dict}")
                        if image_encoder_folder.count("/") == 0:
                            image_encoder_subfolder = Path(subfolder, image_encoder_folder).as_posix()
                        else:
                            image_encoder_subfolder = Path(image_encoder_folder).as_posix()

                        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                            pretrained_model_name_or_path_or_dict,
                            subfolder=image_encoder_subfolder,
                            low_cpu_mem_usage=low_cpu_mem_usage,
                        ).to(self.device, dtype=self.dtype)
                        self.register_modules(image_encoder=image_encoder)
                    else:
                        raise ValueError(
                            "`image_encoder` cannot be loaded because `pretrained_model_name_or_path_or_dict` is a state dict."
                        )
                else:
                    logger.warning(
                        "image_encoder is not loaded since `image_encoder_folder=None` passed. You will not be able to use `ip_adapter_image` when calling the pipeline with IP-Adapter."
                        "Use `ip_adapter_image_embeds` to pass pre-generated image embedding instead."
                    )

            # create feature extractor if it has not been registered to the pipeline yet
            if hasattr(self, "feature_extractor") and getattr(self, "feature_extractor", None) is None:
                clip_image_size = self.image_encoder.config.image_size
                feature_extractor = CLIPImageProcessor(size=clip_image_size, crop_size=clip_image_size)
                self.register_modules(feature_extractor=feature_extractor)

        # load ip-adapter into unet
        unet = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
        unet._load_ip_adapter_weights(state_dicts, low_cpu_mem_usage=low_cpu_mem_usage)


    def from_pretrained(pretrained_model_name_or_path, video_length=16):
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
            use_safetensors=True, 
            variant="fp16"
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
        # self.unet.load_attn_procs(state_dict["lora_weights"], adapter_name="photomaker")
        self.load_lora_weights(state_dict["lora_weights"], adapter_name="photomaker")
        del self.hf_device_map
    
    def init_model(self, unet_inject_block=None, new_ip_adapter=False):
        self.load_photomaker_adapter(
            "./pretrain_model/PhotoMaker",
            subfolder="",
            weight_name="photomaker-v1.bin",
            trigger_word="img"
        )
        if new_ip_adapter:
            self.load_ip_adapter(["./pretrain_model/IP-Adapter","./pretrain_model/IP-Adapter-FaceID/"], subfolder=["sdxl_models",None], weight_name=['ip-adapter-plus-face_sdxl_vit-h.bin',"ip-adapter-faceid_sdxl.bin"], image_encoder_folder=None)
        else:
            self.load_ip_adapter(["./pretrain_model/IP-Adapter","./pretrain_model/IP-Adapter-FaceID/"], subfolder=["sdxl_models",None], weight_name=['ip-adapter-plus-face_sdxl_vit-h.bin',"ip-adapter-faceid-portrait_sdxl.bin"], image_encoder_folder=None)
        attn_procs = {}
        inject_block = unet_inject_block
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
                attn_procs[name] = self.unet.attn_processors[name]
            elif name in inject_block:
                print(name)
                attn_procs[name] = MixIPAdapterAttnProcessor2_0(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.unet.attn_processors[name].num_tokens,
                    video_length=self.video_length
                ).to(dtype=torch.float16)
            else:
                attn_procs[name] = self.unet.attn_processors[name]
        self.unet.set_attn_processor(attn_procs)
        return

    def get_learnable_parameters(self):
        learn_params = []
        for proc in self.unet.attn_processors.values():
            if isinstance(proc, (MixIPAdapterAttnProcessor2_0)):
                learn_params.append(list(proc.fusion.parameters()))
        return list(itertools.chain(*learn_params))
    
    def freeze_parameters(self):
        for name, parm in self.named_parameters():
            if "attn2.processor.fusion" in name:
                parm.requires_grad = True
            else:
                parm.requires_grad = False
    
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


    def forward(self, batch, noise_scheduler, resolution=512, snr_gamma=None):
        latents = batch["video"].to(self.weight_dtype)
        clip_emb = batch["clip_emb"].to(self.weight_dtype)
        face_id_embed = batch["faces_id"].to(self.weight_dtype)
        prompt_embeds = batch["prompt_embeds"].to(self.weight_dtype)
        pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(self.weight_dtype)
        original_size = (resolution,resolution)
        crops_coords_top_left = (0, 0)
        target_size = (resolution,resolution)
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

        
        
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
        )
        add_time_ids = add_time_ids.to(device=latents.device).repeat(bsz, 1)       
        added_cond_kwargs = {"text_embeds": add_text_embeds.to(noisy_latents.dtype), "time_ids": add_time_ids.to(noisy_latents.dtype)}
        added_cond_kwargs["image_embeds"] = [clip_emb, face_id_embed]
        timesteps_tmp = timesteps.expand(noisy_latents.shape[0]).to(self.weight_dtype)

        t_emb = self.unet.time_proj(timesteps_tmp)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        timestep_cond = None
        t_emb = t_emb.to(device=noisy_latents.device,dtype=noisy_latents.dtype)
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
        # denoise_loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
        if snr_gamma is None:
            loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(noise_scheduler, timesteps)
            mse_loss_weights = torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                dim=1
            )[0]
            if noise_scheduler.config.prediction_type == "epsilon":
                mse_loss_weights = mse_loss_weights / snr
            elif noise_scheduler.config.prediction_type == "v_prediction":
                mse_loss_weights = mse_loss_weights / (snr + 1)

            loss = F.mse_loss(pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()
        return_dict = {"denoise_loss": loss}
        return return_dict



   