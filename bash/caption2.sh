CUDA_VISIBLE_DEVICES=1 python eval_ip_adapter_animatediff_object.py \
    --image 'datasets/cat.png' \
    --output 'outputs/ip-adapter-object' \
    --prompt 'cat_prompts.txt'

CUDA_VISIBLE_DEVICES=1 python eval_ip_adapter_animatediff_object.py \
    --image 'datasets/cat.png' \
    --version 'ip-adapter-plus_sd15.bin' \
    --output 'outputs/ip-adapter-plus-object' \
    --prompt 'cat_prompts.txt'