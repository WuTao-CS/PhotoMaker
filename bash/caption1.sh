
cd /group/40034/jackeywu/code/PhotoMaker/

CUDA_VISIBLE_DEVICES=0 python latent_gudience_infer.py \
    -i 'datasets/lecun/yann-lecun.jpg' \
    --unet_path 'checkpoints/sd15_latent_new_lr_1e-5_4a100-motion-update-1022-with-ref-noisy-cross-attn-only-face/checkpoint-95000/pytorch_model.bin' \
    --output 'outputs/sd15_latent_new_lr_1e-5_4a100-motion-update-1022-with-ref-noisy-cross-attn-only-face/checkpoint-95000/' \
    --enable_origin_cross_attn \
    --enable_update_motion \
    --enable_crop_face




