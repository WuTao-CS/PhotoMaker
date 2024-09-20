bash# pip install -r requirements.txt
accelerate launch --config_file 'bash/accelerate_config_4a100.yaml' \
  train_latent_sdxl.py \
  --pretrained_model_name_or_path="./pretrain_model/stable-diffusion-xl-base-1.0" \
  --train_data_dir "datasets/CeleV-Text" \
  --checkpointing_steps=3000 \
  --resolution 512 \
  --output_dir "checkpoints/sdxl_latent_fix_lr_1e-5_4a100" \
  --checkpoints_total_limit=10 \
  --num_train_epochs=100 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate=1e-5 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --resume_from_checkpoint="latest" \
  --snr_gamma 5.0 \
  --unet_inject_txt 'block_1024.txt'