# nvidia-smi
# conda init
# source ~/.bashrc
# echo "conda activate env-novelai"
# conda activate env-novelai 
# cd /group/40034/jackeywu/code/PhotoMaker/

# python photomaker_fusion_infer_npu.py \
#     -i 'datasets/lecun/yann-lecun.jpg' \
#     --unet_path "sh_checkpoints/train_snr_lr1e5_npu_drop/checkpoint-1000/pytorch_model.bin" \
#     --output "outputs/train_snr_lr1e5_npu_drop-1000/"

# python photomaker_fusion_infer.py \
#     -i 'datasets/lecun/yann-lecun.jpg' \
#     --name "photomaker_mix_512_test" \
#     --output "outputs/"
    
# CUDA_VISIBLE_DEVICES=1 python photomaker_fusion_infer.py \
#     -i 'datasets/lecun/yann-lecun.jpg' \
#     --size 1024 \
#     --inject_block_txt "checkpoints/train_snr_lr1e5_npu_drop_long_time_ddim/block.txt" \
#     --unet_path "checkpoints/train_snr_lr1e5_npu_drop_long_time_ddim/checkpoint-16000/pytorch_model.bin" \
#     --output "outputs/train_snr_lr1e5_npu_drop_long_time_ddim/checkpoint-16000-1024/"

# CUDA_VISIBLE_DEVICES=1 python photomaker_fusion_infer_crop_face.py \
#     -i 'datasets/lecun/yann-lecun.jpg' \
#     --size 512 \
#     --inject_block_txt "checkpoints/train_snr_lr1e5_4a100_drop_long_time_new_ip-adapter_xformer_high/block.txt" \
#     --unet_path "checkpoints/train_snr_lr1e5_4a100_drop_long_time_new_ip-adapter_xformer_high/checkpoint-38000/pytorch_model.bin" \
#     --output "outputs/train_snr_lr1e5_4a100_drop_long_time_new_ip-adapter_xformer_high/checkpoint-38000-512-crop-face/" \
#     --enable_new_ip_adapter 


# CUDA_VISIBLE_DEVICES=1 python photomaker_fusion_infer_crop_face.py \
#     -i 'examples/newton_man/newton_0.jpg' \
#     --size 1024 \
#     --inject_block_txt "checkpoints/train_snr_lr1e5_4a100_drop_long_time_new_ip-adapter_xformer_high/block.txt" \
#     --unet_path "checkpoints/train_snr_lr1e5_4a100_drop_long_time_new_ip-adapter_xformer_high/checkpoint-38000/pytorch_model.bin" \
#     --output "outputs/train_snr_lr1e5_4a100_drop_long_time_new_ip-adapter_xformer_high/checkpoint-38000-1024-crop-face/" \
# #     --enable_new_ip_adapter 

# CUDA_VISIBLE_DEVICES=1 python latent_gudience_gated_infer_motion_sdxl.py \
#     -i 'examples/newton_man/newton_0.jpg' \
#     --unet_path "checkpoints/sdxl_gated_latent_fix_lr_1e-5_8v100/checkpoint-36000/pytorch_model.bin" \
#     --output "outputs/sdxl_gated_latent_fix_lr_1e-5_8v100/checkpoint-36000-1024" \
#     --inject_block_txt "checkpoints/sdxl_gated_latent_fix_lr_1e-5_8v100/block.txt"

echo 'Begin to install python packages...'
nvidia-smi
conda init
source ~/.bashrc
echo "conda activate env-novelai"
conda activate env-novelai 
cd /group/40034/jackeywu/code/PhotoMaker/
CUDA_VISIBLE_DEVICES=1 python process_reg_data.py --phase 1