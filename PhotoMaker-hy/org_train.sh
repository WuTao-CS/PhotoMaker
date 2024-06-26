EXP_NAME=006-sdxl-lora-highres-loraproj1e-4-r64-finaldata+chinese-multiembed-append-1e-5-bs6-n8
export WANDB_NAME=${EXP_NAME}
export WANDB_DISABLE_SERVICE=true
export CUDA_VISIBLE_DEVICES="0,1"
# export HF_HOME="./cache/huggingface"
gpu_num=2


echo "${EXP_NAME}"

# if [ ! -e config_log/${EXP_NAME}.sh ]; then
#     echo "File does not exist. Performing the operation..."
#     # 在这里添加您要执行的操作，例如：
#     cp script/run_debug_sdxl_lora_gpu8.sh config_log/${EXP_NAME}.sh
#     echo "Operation completed."
# else
#     echo "File exists. Skipping the operation."
# fi


DATASET_PATH="/apdcephfs_cq2/share_1367250/nicehuang/data/photomaker/arc_org_data"

DATASET_NAME="ffhq"
FAMILY=stabilityai
MODEL=stable-diffusion-xl-base-1.0
IMAGE_ENCODER=openai/clip-vit-large-
pretrained_model_name_or_path="/apdcephfs/share_1367250/rongweiquan/ControlNet_SR/models/controlnet_sr_zh"
unet_path="/apdcephfs/share_1367250/rongweiquan/hunyuan_latent_zh/unet_v1.4_human"
pooling_path="/apdcephfs/share_1367250/rongweiquan/hunyuan_latent_zh/unet_v1.4_human/pooling_weight.pt"
out_dir="/apdcephfs_data_cq5_2/share_300167803/nicehuang/train_data/photomaker"
train_batch_size=10


accelerate launch \
    --mixed_precision=bf16 \
    --machine_rank 0 \
    --num_machines 1 \
    --main_process_port 11136 \
    --num_processes ${gpu_num} \
    --multi_gpu \
    idadapter/train_sdxl_lora.py \
    --pretrained_model_name_or_path ${pretrained_model_name_or_path} \
    --dataset_name ${DATASET_PATH} \
    --logging_dir ${out_dir}/logs/${MODEL}/${DATASET_NAME}/${WANDB_NAME} \
    --output_dir ${out_dir}/models/${MODEL}/${DATASET_NAME}/${WANDB_NAME} \
    --max_train_steps 300000 \
    --num_train_epochs 300000 \
    --train_batch_size ${train_batch_size} \
    --learning_rate 1e-5 \
    --unet_lr_scale 10.0 \
    --checkpointing_steps 5 \
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
    --report_to tensorboard \
    --gradient_checkpointing \
    --rank 64 \
    --use_multi_embeds \
    --fuse_type append \
    --use_concat_final_chinese_dataset \
    --use_hunyuan_image_encoder \
    --unet_path ${unet_path} \
    --pooling_path ${pooling_path} \
    --one_machine
    # --resume_from_checkpoint latest \
    # --load_model "./projects/IDAdapter-diffusers/models/stable-diffusion-xl-base-1.0/ffhq/006-sdxl-lora-highres-loraproj1e-4-r64-finaldata-multiembed-append-1e-5-bs6-n8/checkpoint-150000" \
    # --gradient_accumulation_steps 2 \
    # --enable_xformers_memory_efficient_attention \
    # --object_localization \
    # --object_localization_weight 1e-3 \
    # --object_localization_loss balanced_l1 \
    # --enable_xformers_memory_efficient_attention