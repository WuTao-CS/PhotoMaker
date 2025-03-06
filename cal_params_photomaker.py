import torch
import numpy as np
import random
import os
from PIL import Image


from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler, DDIMScheduler
from huggingface_hub import hf_hub_download
import sys
import torch
import clip
from decord import VideoReader, cpu
import json
from eval.eval_clip import ClipEval
from eval.eval_dino import DINOEvaluator
import pandas as pd
import argparse
import json
from diffusers import DiffusionPipeline, AnimateDiffPipeline, MotionAdapter, DDIMScheduler,AnimateDiffSDXLPipeline
from diffusers.utils import export_to_gif, load_image
import argparse
from transformers import CLIPVisionModelWithProjection
import cv2
from insightface.app import FaceAnalysis
from insightface.utils import face_align

from photomaker import PhotoMakerAnimateDiffXLPipline

base_model_path = './pretrain_model/RealVisXL_V4.0'
device = "cuda"
save_path = "./outputs"
adapter = MotionAdapter.from_pretrained("pretrain_model/animatediff-motion-adapter-sdxl-beta")
scheduler = DDIMScheduler.from_pretrained(
    base_model_path,
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)
pipe = PhotoMakerAnimateDiffXLPipline.from_pretrained(
    base_model_path,
    motion_adapter=adapter,
    scheduler=scheduler,
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")


pipe.load_photomaker_adapter(
    "./pretrain_model/PhotoMaker",
    subfolder="",
    weight_name="photomaker-v1.bin",
    trigger_word="img"
)
pipe.id_encoder.to(device)

print(pipe.id_encoder.num_parameters())