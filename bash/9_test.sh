
#!/bin/bash
echo 'Begin to install python packages...'
nvidia-smi
conda init
source ~/.bashrc
echo "conda activate env-novelai"
conda activate env-novelai 
cd /group/40034/jackeywu/code/PhotoMaker/
PHASE=0
TOTOAL=8


CUDA_VISIBLE_DEVICES=$PHASE python eval_ip_adapter_animatediff.py \
    --output 'outputs_final/common_person_bench/ip-adapter_plus_new/' \
    --phase $PHASE \
    --seed 42 \
    --image_dir 'datasets/common_person' \
    --prompt 'new_person_prompt.txt' \
    --version 'ip-adapter-plus_sd15.bin' \
    --total $TOTOAL \
    -n 30