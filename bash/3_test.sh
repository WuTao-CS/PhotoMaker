# echo 'Begin to install python packages...'
# nvidia-smi
# conda init
# source ~/.bashrc
# echo "conda activate env-novelai"
# conda activate env-novelai 
# cd /group/40034/jackeywu/code/PhotoMaker/
# python preprocess_sd15.py --phase 0

# CUDA_VISIBLE_DEVICES=0 python photomaker_fusion_infer_crop_face.py \
#     -i 'datasets/lecun/yann-lecun.jpg' \
#     --size 1024 \
#     --inject_block_txt "checkpoints/train_snr_lr1e5_4a100_drop_long_time_new_ip-adapter_xformer_high/block.txt" \
#     --unet_path "checkpoints/train_snr_lr1e5_4a100_drop_long_time_new_ip-adapter_xformer_high/checkpoint-38000/pytorch_model.bin" \
#     --output "outputs/train_snr_lr1e5_4a100_drop_long_time_new_ip-adapter_xformer_high/checkpoint-38000-512-crop-face/" \
#     --enable_new_ip_adapter 
echo 'Begin to install python packages...'
nvidia-smi
conda init
source ~/.bashrc
echo "conda activate env-novelai"
conda activate env-novelai 
cd /group/40034/jackeywu/code/PhotoMaker/
CUDA_VISIBLE_DEVICES=2 python process_reg_data.py --phase 2