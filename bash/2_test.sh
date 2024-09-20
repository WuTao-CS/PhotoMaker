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
    
CUDA_VISIBLE_DEVICES=1 python photomaker_fusion_infer.py \
    -i 'datasets/lecun/yann-lecun.jpg' \
    --size 1024 \
    --inject_block_txt "checkpoints/train_snr_lr1e5_npu_drop_long_time_ddim/block.txt" \
    --unet_path "checkpoints/train_snr_lr1e5_npu_drop_long_time_ddim/checkpoint-16000/pytorch_model.bin" \
    --output "outputs/train_snr_lr1e5_npu_drop_long_time_ddim/checkpoint-16000-1024/"