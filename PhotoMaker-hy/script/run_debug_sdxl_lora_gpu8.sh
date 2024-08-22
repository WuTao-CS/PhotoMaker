EXP_NAME=006-sdxl-lora-highres-loraproj1e-4-r64-finaldata+chinese-multiembed-append-1e-5-bs6-n8
export WANDB_NAME=${EXP_NAME}
export WANDB_DISABLE_SERVICE=true
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export HF_HOME="./cache/huggingface"

echo "${EXP_NAME}"

if [ ! -e config_log/${EXP_NAME}.sh ]; then
    echo "File does not exist. Performing the operation..."
    # 在这里添加您要执行的操作，例如：
    cp script/run_debug_sdxl_lora_gpu8.sh config_log/${EXP_NAME}.sh
    echo "Operation completed."
else
    echo "File exists. Skipping the operation."
fi


DATASET_PATH="./projects/IDAdapter-diffusers/data"

DATASET_NAME="ffhq"
FAMILY=stabilityai
MODEL=stable-diffusion-xl-base-1.0
IMAGE_ENCODER=openai/clip-vit-large-patch14

accelerate launch \
    --mixed_precision=bf16 \
    --machine_rank 0 \
    --num_machines 1 \
    --main_process_port 11136 \
    --num_processes 8 \
    --multi_gpu \
    idadapter/train_sdxl_lora.py \
    --pretrained_model_name_or_path ${FAMILY}/${MODEL} \
    --dataset_name ${DATASET_PATH} \
    --logging_dir logs/${MODEL}/${DATASET_NAME}/${WANDB_NAME} \
    --output_dir models/${MODEL}/${DATASET_NAME}/${WANDB_NAME} \
    --max_train_steps 300000 \
    --num_train_epochs 300000 \
    --train_batch_size 7 \
    --learning_rate 1e-5 \
    --unet_lr_scale 10.0 \
    --checkpointing_steps 1000 \
    --mixed_precision bf16 \
    --allow_tf32 \
    --keep_only_last_checkpoint \
    --keep_interval 10000 \
    --seed 42 \
    --image_encoder_type clip \
    --image_encoder_name_or_path ${IMAGE_ENCODER} \
    --num_image_tokens 1 \
    --max_num_objects 4 \
    --train_resolution 1024 \
    --object_resolution 224 \
    --text_image_linking postfuse \
    --object_appear_prob 0.9 \
    --uncondition_prob 0.1 \
    --object_background_processor random \
    --disable_flashattention \
    --train_image_encoder \
    --image_encoder_trainable_layers 2 \
    --object_types person \
    --mask_loss \
    --mask_loss_prob 0.5 \
    --resume_from_checkpoint latest \
    --report_to tensorboard \
    --gradient_checkpointing \
    --rank 64 \
    --use_multi_embeds \
    --fuse_type append \
    --use_concat_final_chinese_dataset \
    --load_model "./projects/IDAdapter-diffusers/models/stable-diffusion-xl-base-1.0/ffhq/006-sdxl-lora-highres-loraproj1e-4-r64-finaldata-multiembed-append-1e-5-bs6-n8/checkpoint-150000" \
    # --gradient_accumulation_steps 2 \
    # --enable_xformers_memory_efficient_attention \
    # --object_localization \
    # --object_localization_weight 1e-3 \
    # --object_localization_loss balanced_l1 \
    # --enable_xformers_memory_efficient_attention