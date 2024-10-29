echo 'Begin to install python packages...'
nvidia-smi
conda init
source ~/.bashrc
echo "conda activate env-novelai"
conda activate env-novelai 
cd /group/40034/jackeywu/code/PhotoMaker/


accelerate launch --config_file 'bash/accelerate_config_4a100_zero1.yaml' \
  train_latent_new.py \
  --pretrained_model_name_or_path="./pretrain_model/Realistic_Vision_V5.1_noVAE" \
  --train_data_dir "datasets/CeleV-Text" \
  --checkpointing_steps=5000 \
  --resolution 512 \
  --output_dir "checkpoints/sd15_latent_new_lr_1e-5_4a100-with-motion-update-1016-with-ref-noisy" \
  --checkpoints_total_limit=50 \
  --max_train_steps 200000 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate=1e-5 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --resume_from_checkpoint="latest" \
  --snr_gamma 5.0 \
  --enable_update_motion \
  --enable_reference_noisy \
  --ref_noisy_ratio 0.01 \
  --ref_loss_weight 0.1 \
  --refer_noisy_type "random" \
  --with_vae \
  --frame_stride 4