echo 'Begin to install python packages...'
nvidia-smi
conda init
source ~/.bashrc
echo "conda activate env-novelai"
conda activate env-novelai 
cd /group/40034/jackeywu/code/PhotoMaker/
PHASE=5
TOTOAL=8

epochs_name=("checkpoint-100000" "checkpoint-150000")
checkpoint_dir="sd15_latent_new_lr_1e-5_4a100-motion-update-1112-with-ref-noisy-cross-attn-whitebg-head-frame_stride-stride-4-8card"
for name in ${epochs_name[@]}
    do
    dir_path="checkpoints/$checkpoint_dir/$name/"
    output_dir="outputs_final/common_person_seed/$checkpoint_dir/$name/"
    ckpt="checkpoints/$checkpoint_dir/$name/pytorch_model.bin"
    CUDA_VISIBLE_DEVICES=$PHASE python eval_latent_gudience_infer_many.py \
        --unet_path $ckpt \
        --output $output_dir \
        --enable_update_motion \
        --enable_origin_cross_attn \
        --image_dir 'datasets/common_person_seg' \
        --phase $PHASE \
        --seed 2048 9999 412 \
        --prompt 'new_person_prompt.txt' \
        -n 30 \
        --total $TOTOAL 
    done