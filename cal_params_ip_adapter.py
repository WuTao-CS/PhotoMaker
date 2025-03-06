import torch
import numpy as np
import random
import os
from PIL import Image


from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler, DDIMScheduler, AnimateDiffPipeline
from huggingface_hub import hf_hub_download
import sys
import torch
import clip
from decord import VideoReader, cpu
import json
from eval.eval_clip import ClipEval
from eval.eval_dino import DINOEvaluator
import pandas as pd
from model.pipline import VAEProjectAnimateDiffPipeline, VAETimeProjectAnimateDiffPipeline
from diffusers import MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_gif, load_image
from insightface.app import FaceAnalysis
from transformers import CLIPVisionModelWithProjection
import cv2
from insightface.utils import face_align
# gloal variable and function
import argparse

base_model_path = './pretrain_model/Realistic_Vision_V5.1_noVAE'
device = "cuda"
adapter = MotionAdapter.from_pretrained("./pretrain_model/animatediff-motion-adapter-v1-5-2")

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
    scheduler=scheduler,
)


pipe.load_ip_adapter("./pretrain_model/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.bin")
print(pipe.unet.num_parameters()+pipe.text_encoder.num_parameters()+pipe.vae.num_parameters())
print(pipe.unet.num_parameters()+pipe.text_encoder.num_parameters()+pipe.vae.num_parameters()+pipe.image_encoder.num_parameters())
