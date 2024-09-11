accelerate launch --config_file 'bash/accelerate_config_8npu.yaml' \
  train_latent.py \
  --pretrained_model_name_or_path="./pretrain_model/stable-diffusion-v1-5" \
  --train_data_dir "datasets/CeleV-Text" \
  --checkpointing_steps=5000 \
  --resolution 512 \
  --output_dir "sh_checkpoints/debug_sd15_latent_snr_lr_1e-6" \
  --checkpoints_total_limit=10 \
  --num_train_epochs=100 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate=1e-6 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --resume_from_checkpoint="latest" \
  --snr_gamma 5.0 \
  --adam_weight_decay 0.05


# accelerate launch --config_file 'bash/accelerate_config_8npu.yaml' \
#   train_npu.py \
#   --pretrained_model_name_or_path="./pretrain_model/RealVisXL_V4.0" \
#   --train_data_dir "datasets/CeleV-Text" \
#   --checkpointing_steps=200 \
#   --unet_inject_txt "block.txt" \
#   --output_dir "sh_checkpoints/train_snr_lr1e5_npu_drop_long_time_weight_0.08" \
#   --checkpoints_total_limit=10 \
#   --max_train_steps=1000 \
#   --train_batch_size=2 \
#   --gradient_accumulation_steps=2 \
#   --gradient_checkpointing \
#   --learning_rate=1e-05 \
#   --max_grad_norm=1 \
#   --lr_scheduler="constant" --lr_warmup_steps=0 \
#   --resume_from_checkpoint="latest" \
#   --snr_gamma 5.0 \
#   --adam_weight_decay 0.08  


# # ASCEND_LAUNCH_BLOCKING=1 accelerate launch --config_file 'bash/accelerate_config_8npu.yaml' \
# #   train_npu.py \
# #   --pretrained_model_name_or_path="./pretrain_model/stable-diffusion-xl-base-1.0" \
# #   --train_data_dir "datasets/CeleV-Text" \
# #   --checkpointing_steps=1000 \
# #   --output_dir "sh_checkpoints/train_snr_lr1e5_npu_sdxl" \
# #   --checkpoints_total_limit=10 \
# #   --train_batch_size=2 \
# #   --num_train_epochs=21 \
# #   --gradient_accumulation_steps=2 \
# #   --gradient_checkpointing \
# #   --learning_rate=1e-05 \
# #   --max_grad_norm=1 \
# #   --lr_scheduler="constant" --lr_warmup_steps=0 \
# #   --resume_from_checkpoint="latest" \
# #   --snr_gamma 5.0

# # ASCEND_LAUNCH_BLOCKING=1 accelerate launch --config_file 'bash/accelerate_config_8npu.yaml' \
# #   train_npu.py \
# #   --pretrained_model_name_or_path="./pretrain_model/RealVisXL_V4.0" \
# #   --train_data_dir "datasets/CeleV-Text" \
# #   --checkpointing_steps=100 \
# #   --output_dir "sh_checkpoints/train_snr_lr1e5_npu_drop_only_last_1024" \
# #   --resolution 1024 \
# #   --unet_inject_txt "block_1024.txt" \
# #   --checkpoints_total_limit=10 \
# #   --max_train_steps=1000 \
# #   --train_batch_size=1 \
# #   --gradient_accumulation_steps=2 \
# #   --gradient_checkpointing \
# #   --learning_rate=1e-05 \
# #   --max_grad_norm=1 \
# #   --lr_scheduler="constant" --lr_warmup_steps=0 \
# #   --resume_from_checkpoint="latest" \
# #   --snr_gamma 5.0

# # ASCEND_LAUNCH_BLOCKING=1 accelerate launch --config_file 'bash/accelerate_config_8npu.yaml' \
# #   train_npu.py \
# #   --pretrained_model_name_or_path="./pretrain_model/stable-diffusion-xl-base-1.0" \
# #   --train_data_dir "datasets/CeleV-Text" \
# #   --checkpointing_steps=1000 \
# #   --output_dir "sh_checkpoints/train_snr_lr1e5_npu_sdxl_only_linear" \
# #   --checkpoints_total_limit=10 \
# #   --train_batch_size=2 \
# #   --num_train_epochs=21 \
# #   --gradient_accumulation_steps=2 \
# #   --gradient_checkpointing \
# #   --learning_rate=1e-05 \
# #   --max_grad_norm=1 \
# #   --lr_scheduler="constant" --lr_warmup_steps=0 \
# #   --resume_from_checkpoint="latest" \
# #   --snr_gamma 5.0
# accelerate launch --config_file 'bash/accelerate_config_8npu.yaml' \
#   train_npu.py \
#   --pretrained_model_name_or_path="./pretrain_model/RealVisXL_V4.0" \
#   --train_data_dir "datasets/CeleV-Text" \
#   --checkpointing_steps=1000 \
#   --unet_inject_txt "block.txt" \
#   --output_dir "checkpoints/train_snr_lr1e5_npu_drop_long_time" \
#   --checkpoints_total_limit=10 \
#   --num_train_epochs=100 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=1 \
#   --gradient_checkpointing \
#   --learning_rate=1e-05 \
#   --max_grad_norm=1 \
#   --lr_scheduler="constant" --lr_warmup_steps=0 \
#   --resume_from_checkpoint="latest" \
#   --snr_gamma 5.0
