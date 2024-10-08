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

python photomaker_animate.py \
    -i 'datasets/lecun/yann-lecun.jpg' \
    --output "test_512_photomaker"