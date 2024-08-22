# Part of this script are derived from the official example script of diffusers
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");


import logging
import math
import os
import shutil
import torch
import torch.utils.checkpoint
import sys
import datasets
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.loaders import (
    LoraLoaderMixin,
)
from typing import Dict
from packaging import version

from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    SlicedAttnAddedKVProcessor,
)
from torch.utils.data import Subset
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms
import torch.nn.functional as F
from transformers.optimization import Adafactor

import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# os.chdir(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append('../')
from idadapter.utils import parse_args

# from idadapter.model_sdxl import IDAdapterModel

from idadapter.transforms import (
    get_train_transforms_with_segmap,
    get_object_transforms,
    get_object_processor,
)

from diffusers.utils import check_min_version
check_min_version("0.20.0.dev0")


logger = get_logger(__name__)


def train():
    args = parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=args.logging_dir,
    )

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.logging_dir is not None:
            os.makedirs(args.logging_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Make one log on every process with the configuration for debugging.
    t = time.localtime()
    str_m_d_y_h_m_s = time.strftime("%m-%d-%Y_%H-%M-%S", t)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(args.logging_dir, f"{str_m_d_y_h_m_s}.log")
            ),
        ]
        if accelerator.is_main_process
        else [],
    )

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer"
    )

    tokenizer_2 = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
    )

    if args.image_encoder_name_or_path == "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k":
        from idadapter.model_sdxl_vitbigg import IDAdapterModel
        logger.info("from idadapter.model_sdxl_vitbigg import IDAdapterModel")
    else:
        from idadapter.model_sdxl_multiembed import IDAdapterModel
        logger.info("from idadapter.model_sdxl_multiembed import IDAdapterModel")
    # elif args.use_multi_embeds:
    #     from idadapter.model_sdxl_multiembed import IDAdapterModel
    #     logger.info("from idadapter.model_sdxl_multiembed import IDAdapterModel")
    # else:
    #     from idadapter.model_sdxl import IDAdapterModel
    #     logger.info("from idadapter.model_sdxl import IDAdapterModel")
    model = IDAdapterModel.from_pretrained(args)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    # freeze all params in the model
    for param in model.parameters():
        param.requires_grad = False
        param.data = param.data.to(weight_dtype)

    print(accelerator.device)
    # exit()
    # Set correct lora layers
    unet_lora_attn_procs = {}
    unet_lora_parameters = []
    for name, attn_processor in model.unet.attn_processors.items():
        cross_attention_dim = None if name.endswith("attn1.processor") else model.unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = model.unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(model.unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = model.unet.config.block_out_channels[block_id]

        if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
            lora_attn_processor_class = LoRAAttnAddedKVProcessor
        else:
            lora_attn_processor_class = (
                LoRAAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else LoRAAttnProcessor
            )

        module = lora_attn_processor_class(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=args.rank, network_alpha=args.unet_lora_alpha
        )
        unet_lora_attn_procs[name] = module
        unet_lora_parameters.extend(module.parameters())

    model.unet.set_attn_processor(unet_lora_attn_procs)
        
    if args.load_model is not None:
        if os.path.exists(Path(args.load_model) / "pytorch_model.bin"):
            model.load_state_dict(
                torch.load(Path(args.load_model) / "pytorch_model.bin", map_location="cpu")
            )
        elif os.path.exists(Path(args.load_model) / "model.safetensors"):
            model.load_state_dict(
                torch.load(Path(args.load_model) / "model.safetensors", map_location="cpu")
            )       
        else:
            raise NotImplementedError("Not found bin or safetensors")

        
    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    model.vae.to(accelerator.device, dtype=torch.float32)
    model.unet.to(accelerator.device, dtype=weight_dtype)
    model.text_encoder.to(accelerator.device, dtype=weight_dtype)
    model.text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    if args.use_id_loss:
        model.id_loss_module.to(accelerator.device, dtype=torch.float32)

    if args.text_image_linking in ["postfuse"] and not args.freeze_postfuse_module:
        model.postfuse_module.requires_grad_(True)
        model.postfuse_module.to(accelerator.device, dtype=torch.float32)

    if args.train_text_encoder:
        model.text_encoder.requires_grad_(True)
        model.text_encoder.to(accelerator.device, dtype=torch.float32)

        model.text_encoder_2.requires_grad_(True)
        model.text_encoder_2.to(accelerator.device, dtype=torch.float32)   

    if args.image_encoder_2_name_or_path is None:
        model.image_encoder.visual_projection_2.requires_grad_(True)
        model.image_encoder.visual_projection_2.to(accelerator.device, dtype=torch.float32)
        
    if args.train_image_encoder:
        if args.image_encoder_trainable_layers > 0:
            for idx in range(args.image_encoder_trainable_layers):
                model.image_encoder.vision_model.encoder.layers[
                    -1 - idx
                ].requires_grad_(True)
                model.image_encoder.vision_model.encoder.layers[-1 - idx].to(
                    accelerator.device, dtype=torch.float32
                )
                
                if args.image_encoder_2_name_or_path is not None:
                    model.image_encoder.vision_model_2.encoder.layers[
                        -1 - idx
                    ].requires_grad_(True)
                    model.image_encoder.vision_model_2.encoder.layers[-1 - idx].to(
                        accelerator.device, dtype=torch.float32
                    )
        else:
            model.image_encoder.requires_grad_(True)
            model.image_encoder.to(accelerator.device, dtype=torch.float32)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=args.revision,
        )
        model.load_ema(ema_unet)
        if args.load_model is not None:
            model.ema_param.load_state_dict(
                torch.load(
                    Path(args.load_model) / "custom_checkpoint_0.pkl",
                    map_location="cpu",
                )
            )

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            print("Enable Xformers memory efficient attention!!!!!!!!!!!!!!!!!!!!!!!")
            
            model.unet.enable_xformers_memory_efficient_attention()
            model.vae.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        model.unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            # model.text_encoder.gradient_checkpointing_enable()
            model.text_encoder.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    optimizer_cls = torch.optim.AdamW

    encoder_params = list(
        [p for n, p in model.named_parameters() if p.requires_grad and ("unet" not in n) and ("postfuse_module" not in n) and ("visual_projection_2" not in n)]
    )

    postfuse_params = list(
        [p for n, p in model.named_parameters() if p.requires_grad and ("postfuse_module" in n)]
    )
    projection_params = list(
        [p for n, p in model.named_parameters() if p.requires_grad and ("visual_projection_2" in n)]
    )

    # parameters = unet_lora_parameters + other_params
    parameters = unet_lora_parameters + encoder_params + postfuse_params + projection_params

    logger.info(f"Rank: {args.rank} | Alpha: {args.unet_lora_alpha} | Use postfuse: {args.large_step_postfuse}")
    if args.large_step_postfuse:
        optimizer = optimizer_cls(
            [
                {"params": unet_lora_parameters + projection_params + postfuse_params, "lr": args.learning_rate * args.unet_lr_scale}, # TODO: 
                {"params": encoder_params, "lr": args.learning_rate},
            ],
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )    
    else:
        optimizer = optimizer_cls(
            [
                {"params": unet_lora_parameters + projection_params, "lr": args.learning_rate * args.unet_lr_scale}, # TODO: 
                {"params": encoder_params + postfuse_params, "lr": args.learning_rate},
            ],
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    train_transforms = get_train_transforms_with_segmap(args)
    object_transforms = get_object_transforms(args)
    object_processor = get_object_processor(args)

    if args.object_types is None or args.object_types == "all":
        object_types = None  # all object types
    else:
        object_types = args.object_types.split("_")
        logger.info(f"Using object types: {object_types}")

    repeat_token = True if args.fuse_type == 'append' else False
    # if args.use_high_resolution:
    if args.use_imdb:
        from idadapter.data_sdxl_multiembed import get_data_loader, IDAdapterIMDBDataset
        logger.info(f"Using IMDB dataset...")
        train_dataset = IDAdapterIMDBDataset(
            args.dataset_name,
            tokenizer,
            tokenizer_2,
            train_transforms,
            object_transforms,
            object_processor,
            device=accelerator.device,
            max_num_objects=args.max_num_objects,
            object_appear_prob=args.object_appear_prob,
            uncondition_prob=args.uncondition_prob,
            text_only_prob=args.text_only_prob,
            use_multi_embeds=args.use_multi_embeds,
            use_face_mask=args.use_face_mask,
            repeat_token=repeat_token,
        )
    elif args.use_ablation_concat_single:
        from idadapter.data_sdxl_multiembed import get_data_loader, get_concat_full_single_dataset
        logger.info(f"Using concat single dataset...")
        train_dataset = get_concat_full_single_dataset(args.dataset_name,
            tokenizer,
            tokenizer_2,
            train_transforms,
            object_transforms,
            object_processor,
            device=accelerator.device,
            max_num_objects=args.max_num_objects,
            object_appear_prob=args.object_appear_prob,
            uncondition_prob=args.uncondition_prob,
            text_only_prob=args.text_only_prob,
            use_multi_embeds=args.use_multi_embeds,
            use_face_mask=args.use_face_mask,
            repeat_token=repeat_token,
        )        
    elif args.use_ffhq_single:
        from idadapter.data_sdxl_multiembed import get_data_loader, IDAdapterFFHQDataset
        logger.info(f"Using single ffhq dataset...")
        train_dataset = IDAdapterFFHQDataset(
            args.dataset_name,
            tokenizer,
            tokenizer_2,
            train_transforms,
            object_transforms,
            object_processor,
            device=accelerator.device,
            max_num_objects=args.max_num_objects,
            object_appear_prob=args.object_appear_prob,
            uncondition_prob=args.uncondition_prob,
            text_only_prob=args.text_only_prob,
            use_multi_embeds=args.use_multi_embeds,
        )          
    elif args.use_concat_dataset:
        from idadapter.data_sdxl_multiembed import get_data_loader, get_concat_dataset
        logger.info(f"Using concat dataset...")
        train_dataset = get_concat_dataset(args.dataset_name,
            tokenizer,
            tokenizer_2,
            train_transforms,
            object_transforms,
            object_processor,
            device=accelerator.device,
            max_num_objects=args.max_num_objects,
            object_appear_prob=args.object_appear_prob,
            uncondition_prob=args.uncondition_prob,
            text_only_prob=args.text_only_prob,
            use_multi_embeds=args.use_multi_embeds,
            use_face_mask=args.use_face_mask,
            repeat_token=repeat_token,
        )
    elif args.use_concat_full_dataset:
        from idadapter.data_sdxl_multiembed import get_data_loader, get_concat_full_dataset
        logger.info(f"Using concat full dataset...")
        train_dataset = get_concat_full_dataset(args.dataset_name,
            tokenizer,
            tokenizer_2,
            train_transforms,
            object_transforms,
            object_processor,
            device=accelerator.device,
            max_num_objects=args.max_num_objects,
            object_appear_prob=args.object_appear_prob,
            uncondition_prob=args.uncondition_prob,
            text_only_prob=args.text_only_prob,
            use_multi_embeds=args.use_multi_embeds,
            use_face_mask=args.use_face_mask,
            repeat_token=repeat_token,
        )
    elif args.use_concat_final_dataset:
        from idadapter.data_sdxl_multiembed import get_data_loader, get_concat_final_dataset
        logger.info(f"Using concat final dataset (FFHQ+IMDB+Wallhaven)...")
        train_dataset = get_concat_final_dataset(args.dataset_name,
            tokenizer,
            tokenizer_2,
            train_transforms,
            object_transforms,
            object_processor,
            device=accelerator.device,
            max_num_objects=args.max_num_objects,
            object_appear_prob=args.object_appear_prob,
            uncondition_prob=args.uncondition_prob,
            text_only_prob=args.text_only_prob,
            use_multi_embeds=args.use_multi_embeds,
            use_face_mask=args.use_face_mask,
            repeat_token=repeat_token,
        )
    elif args.use_concat_imdbffhq_dataset:
        from idadapter.data_sdxl_multiembed import get_data_loader, get_concat_imdbffhq_dataset
        logger.info(f"Using concat final dataset (FFHQ+IMDB)...")
        train_dataset = get_concat_imdbffhq_dataset(args.dataset_name,
            tokenizer,
            tokenizer_2,
            train_transforms,
            object_transforms,
            object_processor,
            device=accelerator.device,
            max_num_objects=args.max_num_objects,
            object_appear_prob=args.object_appear_prob,
            uncondition_prob=args.uncondition_prob,
            text_only_prob=args.text_only_prob,
            use_multi_embeds=args.use_multi_embeds,
            use_face_mask=args.use_face_mask,
            repeat_token=repeat_token,
        )
    elif args.use_file_list_dataset:
        from idadapter.data_sdxl_multiembed import get_data_loader, IDAdapterFileListDataset
        logger.info(f"Using file list dataset...")
        train_dataset = IDAdapterFileListDataset(
            args.dataset_name,
            tokenizer,
            tokenizer_2,
            train_transforms,
            object_transforms,
            object_processor,
            device=accelerator.device,
            max_num_objects=args.max_num_objects,
            object_appear_prob=args.object_appear_prob,
            uncondition_prob=args.uncondition_prob,
            text_only_prob=args.text_only_prob,
            use_multi_embeds=args.use_multi_embeds,
            use_face_mask=args.use_face_mask,
            repeat_token=repeat_token,
        )
    elif args.use_id_loss:
        from idadapter.data_sdxl_multiembed_idloss import get_data_loader, get_concat_full_dataset
        logger.info(f"Using concat full dataset for IDLoss...")
        train_dataset = get_concat_full_dataset(args.dataset_name,
            tokenizer,
            tokenizer_2,
            train_transforms,
            object_transforms,
            object_processor,
            device=accelerator.device,
            max_num_objects=args.max_num_objects,
            object_appear_prob=args.object_appear_prob,
            uncondition_prob=args.uncondition_prob,
            text_only_prob=args.text_only_prob,
            use_multi_embeds=args.use_multi_embeds,
            use_face_mask=args.use_face_mask,
            repeat_token=repeat_token,
        )
    elif args.use_concat_final_chinese_dataset:
        from idadapter.data_sdxl_multiembed import get_data_loader, get_concat_final_chinese_dataset
        logger.info(f"Using concat final+Chinese dataset (FFHQ+IMDB+Wallhaven+Chinese)...")
        train_dataset = get_concat_final_chinese_dataset(args.dataset_name,
            tokenizer,
            tokenizer_2,
            train_transforms,
            object_transforms,
            object_processor,
            device=accelerator.device,
            max_num_objects=args.max_num_objects,
            object_appear_prob=args.object_appear_prob,
            uncondition_prob=args.uncondition_prob,
            text_only_prob=args.text_only_prob,
            use_multi_embeds=args.use_multi_embeds,
            use_face_mask=args.use_face_mask,
            repeat_token=repeat_token,
        )
    else:
        from idadapter.data_sdxl import get_data_loader, FastComposerDataset
        logger.info(f"Using high resolution dataset: {args.use_high_resolution}")
        train_dataset = FastComposerDataset(
            args.dataset_name,
            tokenizer,
            tokenizer_2,
            train_transforms,
            object_transforms,
            object_processor,
            device=accelerator.device,
            max_num_objects=args.max_num_objects,
            num_image_tokens=args.num_image_tokens,
            object_appear_prob=args.object_appear_prob,
            uncondition_prob=args.uncondition_prob,
            text_only_prob=args.text_only_prob,
            object_types=object_types,
            split="train",
            min_num_objects=args.min_num_objects,
            balance_num_objects=args.balance_num_objects,
            use_high_resolution=args.use_high_resolution,
        )

    train_dataloader = get_data_loader(train_dataset, args.train_batch_size)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        accelerator.register_for_checkpointing(model.module.ema_param)
        model.module.ema_param.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("FastComposer", config=vars(args))

    # Train!
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"Trainable parameter: {name} with shape {param.shape}")

    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path), map_location="cpu")
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps
            )

            # move all the state to the correct device
            model.to(accelerator.device)
            if args.use_ema:
                model.module.ema_param.to(accelerator.device)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        model.train()
        train_loss = 0.0
        denoise_loss = 0.0
        embed_loss = 0.0
        embed_clip_loss = 0.0
        clip_loss = 0.0
        id_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            progress_bar.set_description("Global step: {}".format(global_step))
            with accelerator.accumulate(model), torch.backends.cuda.sdp_kernel(
                enable_flash=not args.disable_flashattention
            ):
                with torch.no_grad():
                    latents = model.module.vae.encode(batch["pixel_values"].to(torch.float32)).latent_dist.sample()
                    latents = latents * model.module.vae.config.scaling_factor
                    # print(latents.shape)
                    prompt_embeds, pooled_prompt_embeds = model.module.encode_prompt(batch["input_ids"], batch["input_ids_2"])
                    batch["latents"], batch["prompt_embeds"], batch["pooled_prompt_embeds"] = latents, prompt_embeds, pooled_prompt_embeds

                # print(batch["pixel_values"].shape, latents.shape)
                # exit()
                # import pdb; pdb.set_trace()
                return_dict = model(batch, noise_scheduler)
                loss = return_dict["loss"]

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                avg_denoise_loss = accelerator.gather(
                    return_dict["denoise_loss"].repeat(args.train_batch_size)
                ).mean()
                denoise_loss += (
                    avg_denoise_loss.item() / args.gradient_accumulation_steps
                )

                if "embed_loss" in return_dict:
                    avg_embed_loss = accelerator.gather(
                        return_dict["embed_loss"].repeat(args.train_batch_size)
                    ).mean()
                    embed_loss += (
                        avg_embed_loss.item() / args.gradient_accumulation_steps
                    )

                if "id_loss" in return_dict:
                    avg_id_loss = accelerator.gather(
                        return_dict["id_loss"].repeat(args.train_batch_size)
                    ).mean()
                    id_loss += (
                        avg_id_loss.item() / args.gradient_accumulation_steps
                    )

                if "clip_loss" in return_dict:
                    avg_clip_loss = accelerator.gather(
                        return_dict["clip_loss"].repeat(args.train_batch_size)
                    ).mean()
                    clip_loss += (
                        avg_clip_loss.item() / args.gradient_accumulation_steps
                    )

                if "embed_clip_loss" in return_dict:
                    avg_embed_clip_loss = accelerator.gather(
                        return_dict["embed_clip_loss"].repeat(args.train_batch_size)
                    ).mean()
                    embed_clip_loss += (
                        avg_embed_clip_loss.item() / args.gradient_accumulation_steps
                    )

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(parameters, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    model.module.ema_param.step(model.module.unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log(
                    {
                        "train_loss": train_loss,
                        "denoise_loss": denoise_loss,
                        "embed_loss": embed_loss,
                        "embed_clip_loss": embed_clip_loss,
                        "clip_loss": clip_loss,
                        "id_loss": id_loss,
                    },
                    step=global_step,
                )
                train_loss = 0.0
                denoise_loss = 0.0
                embed_loss = 0.0
                embed_clip_loss = 0.0
                clip_loss = 0.0
                id_loss = 0.0

                if (
                    global_step % args.checkpointing_steps == 0
                    and accelerator.is_local_main_process
                ):
                    log_str = f"global_step: {global_step} denoise loss: {return_dict['denoise_loss'].detach().item()}"
                    if args.use_embed_loss:
                        log_str += f" embed loss: {return_dict['embed_loss'].detach().item()}"
                    if args.use_embed_clip_loss:
                        log_str += f" embed clip loss: {return_dict['embed_clip_loss'].detach().item()}"
                    if args.use_id_loss:
                        log_str += f" id loss: {return_dict['id_loss'].detach().item()}"
                    if args.use_clip_loss:
                        log_str += f" clip loss: {return_dict['clip_loss'].detach().item()}"
                    logger.info(log_str)
                    save_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}"
                    )
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
                    if args.keep_only_last_checkpoint:
                        # Remove all other checkpoints
                        for file in os.listdir(args.output_dir):
                            if file.startswith(
                                "checkpoint"
                            ) and file != os.path.basename(save_path):
                                ckpt_num = int(file.split("-")[1])
                                if (
                                    args.keep_interval is None
                                    or ckpt_num % args.keep_interval != 0
                                ):
                                    logger.info(f"Removing {file}")
                                    shutil.rmtree(os.path.join(args.output_dir, file))

            logs = {
                "l_noise": return_dict["denoise_loss"].detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }

            if "embed_loss" in return_dict:
                logs["l_embed"] = return_dict["embed_loss"].detach().item()

            if "embed_clip_loss" in return_dict:
                logs["l_embed_clip"] = return_dict["embed_clip_loss"].detach().item()

            if "clip_loss" in return_dict:
                logs["l_clip"] = return_dict["clip_loss"].detach().item()

            if "id_loss" in return_dict:
                logs["l_id"] = return_dict["id_loss"].detach().item()
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        if args.use_ema:
            model.ema_param.copy_to(model.unet.parameters())

        pipeline = model.to_pipeline()
        pipeline.save_pretrained(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    train()
