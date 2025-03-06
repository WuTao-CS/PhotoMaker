
#!/bin/bash
echo 'Begin to install python packages...'
nvidia-smi
conda init
source ~/.bashrc
echo "conda activate env-novelai"
conda activate env-novelai 
cd /group/40034/jackeywu/code/PhotoMaker/
PHASE=0
TOTOAL=8


CUDA_VISIBLE_DEVICES=0 python inference_canny_many_batch.py \
    --output 'outputs_final_test/sd15_select_4_canny_self-attn_1218-4card-motion/checkpoint-100000/' \
    --unet_path "checkpoints/sd15_select_4_canny_self-attn_1218-4card-motion/checkpoint-100000/pytorch_model.bin" \
    --num_reference_frame 4


CUDA_VISIBLE_DEVICES=0 python inference_canny_many_batch.py \
    --output 'outputs_final_test/sd15_select_4_canny_self-attn_1218-4card-motion/checkpoint-100000/' \
    --unet_path "checkpoints/sd15_select_4_canny_self-attn_1218-4card-motion/checkpoint-100000/pytorch_model.bin" \
    --num_reference_frame 4