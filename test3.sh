DEVICES=2

CUDA_VISIBLE_DEVICES=$DEVICES python photomaker_adapter_single.py \
    -o "photomaker_clip_h" \
    --clip_h \
    --image 'datasets/Face_data/00002.png' \
    --skip


# CUDA_VISIBLE_DEVICES=$DEVICES python test_animate_multi.py \
#     -i "examples/yangmi_woman" \
#     -p "woman.txt" \
#     --name "yangmi" \
#     -o "photomaker_faceid_yangmi_ratio_40"\
#     -r 40 \
#     --multi_scale

# CUDA_VISIBLE_DEVICES=$DEVICES python test_animate_multi.py \
#     -i "examples/scarletthead_woman" \
#     -p "woman.txt" \
#     --name "scarletthead" \
#     -o "photomaker_faceid_scarletthead_ratio_40" \
#     -r 40 \
#     --multi_scale

# CUDA_VISIBLE_DEVICES=$DEVICES python test_animate_multi.py \
#     -i "examples/newton_man" \
#     --name "newton" \
#     -o "photomaker_faceid_newton_ratio_40"\
#     -r 40 \
#     --multi_scale

# CUDA_VISIBLE_DEVICES=$DEVICES python test_animate_multi.py \
#     -i "examples/yangmi_woman" \
#     -p "woman.txt" \
#     --name "yangmi" \
#     -o "photomaker_faceid_yangmi_ratio_80"\
#     -r 80 \
#     --multi_scale

# CUDA_VISIBLE_DEVICES=$DEVICES python test_animate_multi.py \
#     -i "examples/scarletthead_woman" \
#     -p "woman.txt" \
#     --name "scarletthead" \
#     -o "photomaker_faceid_scarletthead_ratio_80" \
#     -r 80 \
#     --multi_scale

# CUDA_VISIBLE_DEVICES=$DEVICES python test_animate_multi.py \
#     -i "examples/newton_man" \
#     --name "newton" \
#     -o "photomaker_faceid_newton_ratio_80"\
#     -r 80 \
#     --multi_scale