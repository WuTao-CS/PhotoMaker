accelerate launch --config-file 'default_config.yaml' \
  train.py \
  --pretrained_model_name_or_path="./pretrain_model/RealVisXL_V4.0" \
  --train_data_dir "datasets/CeleV-Text" \
  --checkpointing_steps=3000 \
  --output_dir "checkpoints/train_debug" \
  --checkpoints_total_limit=50 \
  --train_batch_size=1 \
  --num_train_epochs=100 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --max_train_steps=30000 \
  --learning_rate=1e-06 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --resume_from_checkpoint="latest" 

  #  \
  # --with_vae
