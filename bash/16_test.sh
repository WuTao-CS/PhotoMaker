#!/bin/bash
# echo 'Begin to install python packages...'
# nvidia-smi
# conda init
# source ~/.bashrc
# echo "conda activate env-novelai"
# conda activate env-novelai 
# cd /group/40034/jackeywu/code/PhotoMaker/
# CUDA_VISIBLE_DEVICES=7 python process_reg_data.py --phase 15
CUDA_VISIBLE_DEVICES=1 python latent_gudience_infer.py \
    -i 'datasets/lecun/yann-lecun.jpg' \
    --unet_path "checkpoints/sd15_latent_new_lr_1e-5_4a100-with-motion-1004/checkpoint-60000/pytorch_model.bin" \
    --output "outputs/sd15_latent_new_lr_1e-5_4a100-with-motion-1004/checkpoint-60000-crop/" \
    --enable_crop_face \
    --enable_update_motion