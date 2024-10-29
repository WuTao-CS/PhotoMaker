#   --unet_inject_txt 'block_1024.txt'
# echo 'Begin to install python packages...'
# nvidia-smi
# conda init
# source ~/.bashrc
# echo "conda activate env-novelai"
# conda activate env-novelai 
# cd /group/40034/jackeywu/code/PhotoMaker/

accelerate launch --config_file 'bash/accelerate_config_4a100_zero1.yaml' \
  train_sd15_lora_textual.py \
  --pretrained_model_name_or_path="./pretrain_model/Realistic_Vision_V5.1_noVAE" \
  --train_data_dir "datasets/CeleV-Text" \
  --checkpointing_steps=5000 \
  --resolution 512 \
  --output_dir "checkpoints/sd15_lora-textual-1007-cross" \
  --checkpoints_total_limit=50 \
  --num_train_epochs=20 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate=3e-5 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --resume_from_checkpoint="latest" \
  --snr_gamma 5.0
