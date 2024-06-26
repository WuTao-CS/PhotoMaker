DEVICES=0

CUDA_VISIBLE_DEVICES=0 python ip_adapter_sdxlanimatediff.py \
    -o "ip_adapter_cliph" \
    --multi_ip_adapter \
    --clip_h \
    --skip
