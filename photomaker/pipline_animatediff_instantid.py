from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from collections import OrderedDict
import os
import PIL
import numpy as np 
import inspect
import torch
import torch.nn as nn
from torchvision import transforms as T
from diffusers.utils.import_utils import is_xformers_available
from safetensors import safe_open
from huggingface_hub.utils import validate_hf_hub_args
from transformers import CLIPImageProcessor, CLIPTokenizer
from diffusers import StableDiffusionXLPipeline, AnimateDiffSDXLPipeline
from diffusers.models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT, load_state_dict
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
from diffusers.pipelines.animatediff import AnimateDiffPipelineOutput
import math

from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils import (
    USE_PEFT_BACKEND,
    _get_model_file,
    is_accelerate_available,
    is_torch_version,
    is_transformers_available,
    logging,
)
if is_transformers_available():
    from transformers import (
        CLIPImageProcessor,
        CLIPVisionModelWithProjection,
    )

    from diffusers.models.attention_processor import (
        AttnProcessor,
        AttnProcessor2_0,
        IPAdapterAttnProcessor,
        IPAdapterAttnProcessor2_0,
    )
from . import PhotoMakerIDEncoder
try:
    import xformers
    import xformers.ops

    xformers_available = True
except Exception:
    xformers_available = False
logger = logging.get_logger(__name__)
PipelineImageInput = Union[
    PIL.Image.Image,
    torch.FloatTensor,
    List[PIL.Image.Image],
    List[torch.FloatTensor],
]
