echo 'Begin to install python packages...'
nvidia-smi
conda init
source ~/.bashrc
echo "conda activate env-novelai"
conda activate env-novelai 
cd /group/40034/jackeywu/code/PhotoMaker/
accelerate launch --config_file 'bash/accelerate_config_2v100.yaml' \
  train_sdxl_lora.py \
  --pretrained_model_name_or_path="./pretrain_model/stable-diffusion-xl-base-1.0" \
  --train_data_dir "datasets/CeleV-Text" \
  --checkpointing_steps=5000 \
  --output_dir "checkpoints/train_sdxl_lora_new_without_motion" \
  --checkpoints_total_limit=10 \
  --num_train_epochs=20 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --resume_from_checkpoint="latest" \
  --snr_gamma 5.0

