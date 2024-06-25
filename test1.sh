DEVICES=0

CUDA_VISIBLE_DEVICES=0 python photomaker_multi_adapter_single.py \
    -o "photomaker_multi_cliph" \
    --multi_ip_adapter \
    --clip_h \
    --image 'datasets/Face_data/00002.png' \
    --skip
