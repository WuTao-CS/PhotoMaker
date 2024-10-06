# echo 'Begin to install python packages...'
# nvidia-smi
# conda init
# source ~/.bashrc
# echo "conda activate env-novelai"
# conda activate env-novelai 
# cd /group/40034/jackeywu/code/PhotoMaker/

# python preprocess_sdxl.py --phase 2
# python latent_gudience_infer.py \
#     -i 'datasets/lecun/yann-lecun.jpg' \
#     --unet_path "checkpoints/sd15_latent_new_lr_1e-5_4a100-with-motion-no-update-1004/checkpoint-100000/pytorch_model.bin" \
#     --output "outputs/sd15_latent_new_lr_1e-5_4a100-with-motion-no-update-1004/checkpoint-100000-crop/" \
#     --enable_crop_face \
#     --enable_update_motion



    # --enable_update_motion

echo 'Begin to install python packages...'
nvidia-smi
conda init
source ~/.bashrc
echo "conda activate env-novelai"
conda activate env-novelai 
cd /group/40034/jackeywu/code/PhotoMaker/
CUDA_VISIBLE_DEVICES=3 python process_reg_data.py --phase 3