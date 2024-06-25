DEVICES=3

CUDA_VISIBLE_DEVICES=$DEVICES python photomaker_adapter_single.py \
    -o "photomaker_faceid" \
    --image 'datasets/Face_data/00002.png' \
    --skip


# CUDA_VISIBLE_DEVICES=3 python photomaker_multi_adapter2.py \
#     -o "photomaker_multi_cliph" \
#     --multi_ip_adapter \
#     --clip_h \
#     --index 4