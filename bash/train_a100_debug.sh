# accelerate launch --config_file 'bash/accelerate_config_4a100.yaml' \
#   train.py \
#   --pretrained_model_name_or_path="./pretrain_model/RealVisXL_V4.0" \
#   --train_data_dir "datasets/CeleV-Text" \
#   --checkpointing_steps=6000 \
#   --unet_inject_txt "block_1024.txt" \
#   --resolution 1024 \
#   --output_dir "checkpoints/train_snr_lr1e5_a100_drop_only_linear_long_time_1024" \
#   --checkpoints_total_limit=10 \
#   --num_train_epochs=100 \
#   --train_batch_size=2 \
#   --gradient_accumulation_steps=2 \
#   --gradient_checkpointing \
#   --learning_rate=1e-05 \
#   --learning_rate=1e-05 \
#   --max_grad_norm=1 \
#   --lr_scheduler="constant" --lr_warmup_steps=0 \
#   --resume_from_checkpoint="latest" \
#   --snr_gamma 5.0
accelerate launch --config_file 'bash/accelerate_config_4a100.yaml' \
  train_latent_gate_skipt_new.py \
  --pretrained_model_name_or_path="./pretrain_model/stable-diffusion-v1-5" \
  --train_data_dir "datasets/CeleV-Text" \
  --checkpointing_steps=5000 \
  --resolution 512 \
  --output_dir "checkpoints/sd15_gate_latent_fix_lr_1e-5_a100-without-motion" \
  --checkpoints_total_limit=10 \
  --num_train_epochs=100 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate=1e-5 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --resume_from_checkpoint="latest" \
  --snr_gamma 5.0
