echo 'Begin to install python packages...'
nvidia-smi
conda init
source ~/.bashrc
echo "conda activate env-novelai"
conda activate env-novelai 
cd /group/40034/jackeywu/code/PhotoMaker/
accelerate launch --config_file 'bash/accelerate_config_8a100.yaml' \
  train_latent_skipt.py \
  --pretrained_model_name_or_path="./pretrain_model/stable-diffusion-v1-5" \
  --train_data_dir "datasets/CeleV-Text" \
  --checkpointing_steps=1000 \
  --resolution 512 \
  --output_dir "checkpoints/debug_sd15_latent_scale_lr_1e-5_skipt-a100" \
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
  --scale_lr


