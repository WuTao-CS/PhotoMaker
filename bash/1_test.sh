# nvidia-smi
# conda init
# source ~/.bashrc
# echo "conda activate env-novelai"
# conda activate env-novelai 
# cd /group/40034/jackeywu/code/PhotoMaker/

# python photomaker_fusion_infer.py \
#     -i 'datasets/lecun/yann-lecun.jpg' \
#     --unet_path "checkpoints/train_snr_lr1e5_a100/checkpoint-3000/pytorch_model.bin" \
#     --output "outputs/train_snr_lr1e5_npu_drop-1000/"

python photomaker_fusion_infer.py \
    -i 'datasets/lecun/yann-lecun.jpg' \
    --size 512 \
    --unet_path "checkpoints/train_snr_lr1e5_npu_drop_long_time_beta_1e-5_ddim/checkpoint-6000/pytorch_model.bin" \
    --output "outputs/train_snr_lr1e5_npu_drop_long_time_beta_1e-5_ddim/checkpoint-6000-512/" 
    
# python photomaker_fusion_infer.py \
#     -i 'datasets/lecun/yann-lecun.jpg' \
#     --unet_path "checkpoints/train_snr_lr1e5_drop_debug/checkpoint-1000/pytorch_model.bin" \
#     --output "outputs/2v100_train_snr_lr1e5_drop_debug-1000/"
# CUDA_VISIBLE_DEVICES=0 python photomaker_fusion_infer.py -i 'examples/scarletthead_woman/scarlett_1.jpg' --prompt 'woman.txt'
# CUDA_VISIBLE_DEVICES=1 python photomaker_fusion_infer.py -i 'datasets/3.jpeg' --prompt 'person.txt'
# CUDA_VISIBLE_DEVICES=2 python photomaker_fusion_infer.py -i 'examples/yangmi_woman/yangmi_6.jpg'  --prompt 'woman.txt'
# CUDA_VISIBLE_DEVICES=3 python photomaker_fusion_infer.py -i 'datasets/hp.jpg'  --prompt 'person.txt'
# python photomaker_fusion_infer.py -i 'examples/yangmi_woman/yangmi_6.jpg'

# python latent_gudience_gated_infer.py \
#     -i 'datasets/lecun/yann-lecun.jpg' \
#     --output "outputs/debug_sd15_gate_latent_lr_1e-5_skipt-a100-80gtest/checkpoint-0" 