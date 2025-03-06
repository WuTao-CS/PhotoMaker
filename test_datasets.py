import argparse
import functools
import gc
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from einops import rearrange
import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed

from packaging import version
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
import diffusers
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, EulerDiscreteScheduler, MotionAdapter, UNet2DConditionModel, UNetMotionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
import torch.nn as nn
from model.attention_processor import ReferVideoAttnProcessor2_0, NewLastFrameProjectAttnProcessor2_0
from model.datasets.celebv_text_canny import CelebVTextCannySD15Dataset,CelebVCannySD15LatentDataset

train_dataset = CelebVTextCannySD15Dataset(root='./datasets/CeleV-Text')
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    shuffle=False,
    batch_size=1,
    num_workers=4,
)

for step, batch in enumerate(train_dataloader):
    print(step)
    continue