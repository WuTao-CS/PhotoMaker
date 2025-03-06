echo 'Begin to install python packages...'
nvidia-smi
conda init
source ~/.bashrc
echo "conda activate env-novelai"
conda activate env-novelai 
cd /group/40034/jackeywu/code/PhotoMaker/


accelerate launch --config_file 'bash/accelerate_config_2a100_zero3.yaml' \
  train_latent_new_whitebg_head_sdxl.py \
  --pretrained_model_name_or_path="./pretrain_model/RealVisXL_V4.0" \
  --train_data_dir "datasets/sh_CeleV-Text" \
  --checkpointing_steps=10000 \
  --resolution 512 \
  --output_dir "checkpoints/sdxl_latent_new_lr_1e-5_4a100-motion-update-1202-with-ref-noisy-cross-attn-whitebg-head-frame_stride-stride-4-4card" \
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
  --enable_reference_noisy \
  --ref_noisy_ratio 0.01 \
  --ref_loss_weight 0.1 \
  --enable_origin_cross_attn \
  --refer_noisy_type "random" \
  --frame_stride 4


#   --enable_new_prompt \   --enable_update_motion \