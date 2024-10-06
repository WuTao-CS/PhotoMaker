# accelerate launch --config_file 'bash/accelerate_config_8npu.yaml' \
#   train_latent.py \
#   --pretrained_model_name_or_path="./pretrain_model/stable-diffusion-v1-5" \
#   --train_data_dir "datasets/sh_CeleV-Text" \
#   --checkpointing_steps=1000 \
#   --resolution 512 \
#   --output_dir "sh_checkpoints/debug_sd15_latent_scale_lr_3e-6_adam_1e-2_same_a100" \
#   --checkpoints_total_limit=10 \
#   --num_train_epochs=100 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=1 \
#   --gradient_checkpointing \
#   --learning_rate=3e-6 \
#   --max_grad_norm=1 \
#   --lr_scheduler="constant" --lr_warmup_steps=0 \
#   --resume_from_checkpoint="latest" \
#   --snr_gamma 5.0


accelerate launch --config_file 'bash/accelerate_config_4a100.yaml' \
  train_sdxl_lora.py \
  --pretrained_model_name_or_path="./pretrain_model/stable-diffusion-xl-base-1.0" \
  --train_data_dir "datasets/CeleV-Text" \
  --checkpointing_steps=5000 \
  --output_dir "checkpoints/train_sdxl_lora_new_without_motion_adam" \
  --checkpoints_total_limit=10 \
  --num_train_epochs=20 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate=1e-05 \
  --adam_epsilon=1e-06 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --resume_from_checkpoint="latest" \
  --snr_gamma 5.0
