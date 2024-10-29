
cd /group/40034/jackeywu/code/PhotoMaker/

CUDA_VISIBLE_DEVICES=1 python latent_gudience_infer.py \
    -i 'datasets/lecun/yann-lecun.jpg' \
    --unet_path 'checkpoints/sd15_latent_new_lr_1e-5_4a100-motion-noupdate-1026-ref-noisy/checkpoint-140000/pytorch_model.bin' \
    --output 'outputs/sd15_latent_new_lr_1e-5_4a100-motion-noupdate-1026-ref-noisy/checkpoint-140000' \
    --enable_origin_cross_attn \
    --enable_update_motion \
    --enable_crop_face
