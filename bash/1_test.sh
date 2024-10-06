# nvidia-smi
# conda init
# source ~/.bashrc
# echo "conda activate env-novelai"
# conda activate env-novelai 
# cd /group/40034/jackeywu/code/PhotoMaker/

# CUDA_VISIBLE_DEVICES=0 python photomaker_fusion_infer.py \
#     -i 'datasets/lecun/yann-lecun.jpg' \
#     --size 512 \
#     --pretrain_path "./pretrain_model/stable-diffusion-xl-base-1.0" \
#     --inject_block_txt "checkpoints/train_snr_lr1e5_4a100_drop_long_time_new_ip-adapter_xformer_high/block.txt" \
#     --unet_path "checkpoints/train_snr_lr1e5_4a100_drop_long_time_new_ip-adapter_xformer_high/checkpoint-38000/pytorch_model.bin" \
#     --output "outputs/train_snr_lr1e5_4a100_drop_long_time_new_ip-adapter_xformer_high/checkpoint-38000-512-sdxl/" \
#     --enable_new_ip_adapter 

# python latent_gudience_gated_infer_motion_sdxl.py \
#     -i 'datasets/lecun/yann-lecun.jpg' \
#     --size 512 \
#     --unet_path "checkpoints/sdxl_gated_latent_fix_lr_1e-5_8v100/checkpoint-36000/pytorch_model.bin" \
#     --output "outputs/sdxl_gated_latent_fix_lr_1e-5_8v100/checkpoint-36000" \
#     --inject_block_txt "checkpoints/sdxl_gated_latent_fix_lr_1e-5_8v100/block.txt"


    
# CUDA_VISIBLE_DEVICES=0 python photomaker_fusion_infer.py \
#     -i 'datasets/lecun/yann-lecun.jpg' \
#     --size 512 \
#     --inject_block_txt "checkpoints/train_snr_lr1e5_4a100_drop_long_time_new_ip-adapter_xformer_realv/block.txt" \
#     --unet_path "checkpoints/train_snr_lr1e5_4a100_drop_long_time_new_ip-adapter_xformer_realv/checkpoint-7000/pytorch_model.bin" \
#     --output "outputs/train_snr_lr1e5_4a100_drop_long_time_new_ip-adapter_xformer_realv/checkpoint-7000-512/" \
#     --enable_new_ip_adapter 

# CUDA_VISIBLE_DEVICES=0 python photomaker_fusion_infer.py \
#     -i 'datasets/lecun/yann-lecun.jpg' \
#     --size 1024 \
#     --inject_block_txt "checkpoints/train_snr_lr1e5_4a100_drop_long_time_new_ip-adapter_xformer_realv/block.txt" \
#     --unet_path "checkpoints/train_snr_lr1e5_4a100_drop_long_time_new_ip-adapter_xformer_realv/checkpoint-7000/pytorch_model.bin" \
#     --output "outputs/train_snr_lr1e5_4a100_drop_long_time_new_ip-adapter_xformer_realv/checkpoint-7000-1024/" \
#     --enable_new_ip_adapter 

# CUDA_VISIBLE_DEVICES=0 python photomaker_fusion_infer.py \
#     -i 'datasets/lecun/yann-lecun.jpg' \
#     --size 1024 \
#     --inject_block_txt "checkpoints/train_snr_lr1e5_4a100_drop_long_time_new_ip-adapter_xformer_realv/block.txt" \
#     --unet_path "checkpoints/train_snr_lr1e5_4a100_drop_long_time_new_ip-adapter_xformer_realv/checkpoint-15000/pytorch_model.bin" \
#     --output "outputs/train_snr_lr1e5_4a100_drop_long_time_new_ip-adapter_xformer_realv/checkpoint-15000-1024/" \
#     --enable_new_ip_adapter 

# python photomaker_fusion_infer.py \
#     -i 'examples/newton_man/newton_0.jpg' \
#     --size 1024 \
#     --unet_path "checkpoints/train_snr_lr1e5_npu_drop_long_time_ddim/checkpoint-18000/pytorch_model.bin" \
#     --output "outputs/train_snr_lr1e5_npu_drop_long_time_ddim/checkpoint-18000-1024/" 

# python photomaker_fusion_infer.py \
#     -i 'datasets/lecun/yann-lecun.jpg' \
#     --size 512 \
#     --inject_block_txt "block_512_a100.txt" \
#     --output "outputs/train_snr_lr1e5_npu_drop_long_time_ddim/checkpoint-9000-512/" 
# # python photomaker_fusion_infer.py \
#     -i 'datasets/lecun/yann-lecun.jpg' \
#     --unet_path "checkpoints/train_snr_lr1e5_drop_debug/checkpoint-1000/pytorch_model.bin" \
#     --output "outputs/2v100_train_snr_lr1e5_drop_debug-1000/"
# CUDA_VISIBLE_DEVICES=0 python photomaker_fusion_infer.py -i 'examples/scarletthead_woman/scarlett_1.jpg' --prompt 'woman.txt'
# CUDA_VISIBLE_DEVICES=1 python photomaker_fusion_infer.py -i 'datasets/3.jpeg' --prompt 'person.txt'
# CUDA_VISIBLE_DEVICES=2 python photomaker_fusion_infer.py -i 'examples/yangmi_woman/yangmi_6.jpg'  --prompt 'woman.txt'
# CUDA_VISIBLE_DEVICES=3 python photomaker_fusion_infer.py -i 'datasets/hp.jpg'  --prompt 'person.txt'
# python photomaker_fusion_infer.py -i 'examples/yangmi_woman/yangmi_6.jpg'

# python latent_gudience_gated_infer_motion_sdxl.py \
#     -i 'datasets/lecun/yann-lecun.jpg' \
#     --size 512 \
#     --unet_path "checkpoints/sdxl_gated_latent_fix_lr_1e-5_4a100/checkpoint-12000/pytorch_model.bin" \
#     --output "outputs/sdxl_gated_latent_fix_lr_1e-5_4a100/checkpoint-12000" \
#     --inject_block_txt "checkpoints/sdxl_gated_latent_fix_lr_1e-5_4a100/block.txt"

echo 'Begin to install python packages...'
nvidia-smi
conda init
source ~/.bashrc
echo "conda activate env-novelai"
conda activate env-novelai 
cd /group/40034/jackeywu/code/PhotoMaker/
CUDA_VISIBLE_DEVICES=0 python process_reg_data.py --phase 0