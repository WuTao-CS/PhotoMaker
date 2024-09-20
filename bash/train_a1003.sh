echo 'Begin to install python packages...'
nvidia-smi
conda init
source ~/.bashrc
echo "conda activate env-novelai"
conda activate env-novelai 
cd /group/40034/jackeywu/code/PhotoMaker/


accelerate launch --config_file 'bash/accelerate_config_4a100.yaml' \
  train.py \
  --pretrained_model_name_or_path="./pretrain_model/stable-diffusion-xl-base-1.0" \
  --train_data_dir "datasets/CeleV-Text" \
  --resolution 512 \
  --checkpointing_steps=1000 \
  --output_dir "checkpoints/train_snr_lr1e5_4a100_drop_long_time_new_ip-adapter_xformer_high" \
  --unet_inject_txt 'block_up_down_xformer.txt' \
  --checkpoints_total_limit=10 \
  --num_train_epochs=100 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --resume_from_checkpoint="latest" \
  --enable_new_ip_adapter \
  --enable_xformers_memory_efficient_attention \
  --snr_gamma 5.0 
