echo 'Begin to install python packages...'
nvidia-smi
conda init
source ~/.bashrc
echo "conda activate env-novelai"
conda activate env-novelai 
cd /group/40034/jackeywu/code/PhotoMaker/
PHASE=0
TOTOAL=4


epochs_name=("checkpoint-130000")
checkpoint_dir="sd15_latent_new_lr_1e-5_4a100-motion-update-0124-story-with-ref-noisy-cross-attn-whitebg-head-frame_stride-stride-4-4card"
for name in ${epochs_name[@]}
    do
    dir_path="checkpoints/$checkpoint_dir/$name/"
    output_dir="/group/40075/jackeywu/outputs_final/$checkpoint_dir/$name/"
    ckpt="checkpoints/$checkpoint_dir/$name/pytorch_model.bin"
    CUDA_VISIBLE_DEVICES=$PHASE python eval_latent_gudience_infer_many_story.py \
        --unet_path $ckpt \
        --output $output_dir \
        --enable_update_motion \
        --enable_origin_cross_attn \
        --image_dir 'datasets/Famous_people_new_hand_recap' \
        --phase $PHASE \
        --seed 42 \
        --prompt 'benchmark.txt' \
        -n 30 \
        --total $TOTOAL 
    done