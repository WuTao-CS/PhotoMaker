DEVICES=1

CUDA_VISIBLE_DEVICES=$DEVICES python test_animate.py \
    -o "photomaker_clip" \
    --image 'datasets/Face_data/00002.png' \
    --name '00002'

# CUDA_VISIBLE_DEVICES=$DEVICES python photomaker_multi_adapter2.py \
#     -o "photomaker_multi" \
#     --multi_ip_adapter \
#     --index 2

# CUDA_VISIBLE_DEVICES=1 python photomaker_multi_adapter2.py \
#     -o "photomaker_multi_cliph" \
#     --multi_ip_adapter \
#     --clip_h \
#     --index 2