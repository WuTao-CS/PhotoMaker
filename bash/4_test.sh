echo 'Begin to install python packages...'
nvidia-smi
conda init
source ~/.bashrc
echo "conda activate env-novelai"
conda activate env-novelai 
cd /group/40034/jackeywu/code/PhotoMaker/
PHASE=3



epochs_name=("checkpoint-150000" "checkpoint-200000")
checkpoint_dir="sd15_latent_new_lr_1e-5_4a100-with-motion-update-1016-with-ref-noisy-cross-attn"
for name in ${epochs_name[@]}
    do
    dir_path="checkpoints/$checkpoint_dir/$name/"
    output_dir="outputs_seed/$checkpoint_dir/$name/"
    ckpt="checkpoints/$checkpoint_dir/$name/pytorch_model.bin"
    python eval_latent_gudience_infer_many.py \
        --unet_path $ckpt \
        --output $output_dir \
        --enable_update_motion \
        --enable_crop_face \
        --enable_origin_cross_attn \
        --seed 128 512 1024 \
        --phase $PHASE
        
    done

epochs_name=("checkpoint-150000" "checkpoint-200000")
checkpoint_dir="sd15_latent_new_lr_1e-5_4a100-with-motion-no-update-1016-with-ref-noisy-cross-attn"
for name in ${epochs_name[@]}
    do
    dir_path="checkpoints/$checkpoint_dir/$name/"
    output_dir="outputs_seed/time_infer/$checkpoint_dir/$name/"
    ckpt="checkpoints/$checkpoint_dir/$name/pytorch_model.bin"
    python python eval_latent_gudience_infer_many.py \
        --unet_path $ckpt \
        --output $output_dir \
        --enable_update_motion \
        --enable_crop_face \
        --enable_origin_cross_attn \
        --seed 128 512 1024 \
        --phase $PHASE

    done

epochs_name=("checkpoint-150000" "checkpoint-200000")
checkpoint_dir="sd15_latent_new_lr_1e-5_4a100-with-motion-1004"
for name in ${epochs_name[@]}
    do
    dir_path="checkpoints/$checkpoint_dir/$name/"
    output_dir="outputs_seed/time_infer/$checkpoint_dir/$name/"
    ckpt="checkpoints/$checkpoint_dir/$name/pytorch_model.bin"
    python python eval_latent_gudience_infer_many.py \
        --unet_path $ckpt \
        --output $output_dir \
        --enable_update_motion \
        --enable_crop_face \
        --enable_origin_cross_attn \
        --seed 128 512 1024 \
        --phase $PHASE

    done

epochs_name=("checkpoint-150000" "checkpoint-200000")
checkpoint_dir="sd15_latent_new_lr_1e-5_4a100-with-motion-1004"
for name in ${epochs_name[@]}
    do
    dir_path="checkpoints/$checkpoint_dir/$name/"
    output_dir="outputs_seed/time_infer/$checkpoint_dir/$name/"
    ckpt="checkpoints/$checkpoint_dir/$name/pytorch_model.bin"
    python python eval_latent_gudience_infer_many.py \
        --unet_path $ckpt \
        --output $output_dir \
        --enable_update_motion \
        --enable_crop_face \
        --enable_origin_cross_attn \
        --seed 128 512 1024 \
        --phase $PHASE

    done