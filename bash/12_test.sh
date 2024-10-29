
#!/bin/bash
echo 'Begin to install python packages...'
nvidia-smi
conda init
source ~/.bashrc
echo "conda activate env-novelai"
conda activate env-novelai 
cd /group/40034/jackeywu/code/PhotoMaker/
PHASE=3

epochs_name=("checkpoint-100000" "checkpoint-150000" "checkpoint-200000")
checkpoint_dir="sd15_latent_new_lr_1e-5_4a100-with-motion-no-update-1004"
for name in ${epochs_name[@]}
    do
    dir_path="checkpoints/$checkpoint_dir/$name/"
    output_dir="outputs/$checkpoint_dir/$name/"
    cd $dir_path
    python zero_to_fp32.py . pytorch_model.bin
    ckpt="checkpoints/$checkpoint_dir/$name/pytorch_model.bin"
    cd /group/40034/jackeywu/code/PhotoMaker/
    CUDA_VISIBLE_DEVICES=$PHASE python eval_latent_gudience_infer_many.py \
        --unet_path $ckpt \
        --output $output_dir \
        --enable_update_motion \
        --enable_crop_face \
        --phase $PHASE
    done

checkpoint_dir="sd15_latent_new_lr_1e-5_4a100-with-motion-no-update-1008-with-ref-noisy"
for name in ${epochs_name[@]}
    do
    dir_path="checkpoints/$checkpoint_dir/$name/"
    output_dir="outputs/$checkpoint_dir/$name/"
    cd $dir_path
    python zero_to_fp32.py . pytorch_model.bin
    ckpt="checkpoints/$checkpoint_dir/$name/pytorch_model.bin"
    cd /group/40034/jackeywu/code/PhotoMaker/
    CUDA_VISIBLE_DEVICES=$PHASE python eval_latent_gudience_infer_many.py \
        --unet_path $ckpt \
        --output $output_dir \
        --enable_update_motion \
        --enable_crop_face \
        --phase $PHASE
    done

checkpoint_dir="sd15_latent_new_lr_1e-5_4a100-with-motion-no-update-1011-cross"
for name in ${epochs_name[@]}
    do
    dir_path="checkpoints/$checkpoint_dir/$name/"
    output_dir="outputs/$checkpoint_dir/$name/"
    cd $dir_path
    python zero_to_fp32.py . pytorch_model.bin
    ckpt="checkpoints/$checkpoint_dir/$name/pytorch_model.bin"
    cd /group/40034/jackeywu/code/PhotoMaker/
    CUDA_VISIBLE_DEVICES=$PHASE python eval_latent_gudience_infer_many.py \
        --unet_path $ckpt \
        --output $output_dir \
        --enable_update_motion \
        --enable_crop_face \
        --enable_origin_cross_attn \
        --phase $PHASE
    done
