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

accelerate launch --config_file 'bash/accelerate_config_8npu.yaml' \
  train_npu.py \
  --pretrained_model_name_or_path="./pretrain_model/stable-diffusion-xl-base-1.0" \
  --train_data_dir "datasets/CeleV-Text" \
  --checkpointing_steps=100 \
  --output_dir "sh_checkpoints/train_snr_lr1e5_npu_drop_long_time_eps_1e-5" \
  --checkpoints_total_limit=10 \
  --max_train_steps=1000 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --resume_from_checkpoint="latest" \
  --snr_gamma 5.0 \
  --adam_epsilon 1e-5

