# echo 'Begin to install python packages...'
# nvidia-smi
# conda init
# source ~/.bashrc
# echo "conda activate env-novelai"
# conda activate env-novelai 
# cd /group/40034/jackeywu/code/PhotoMaker/
# python preprocess_sd15.py --phase 0
CUDA_VISIBLE_DEVICES=2 python latent_gudience_gated_infer.py \
    -i 'datasets/lecun/yann-lecun.jpg' \
    --unet_path "checkpoints/sd15_gate_latent_fix_lr_1e-5_skipt-a100/checkpoint-81000/pytorch_model.bin" \
    --output "outputs/sd15_gate_latent_fix_lr_1e-5_skipt-a100/checkpoint-81000/" 