# echo 'Begin to install python packages...'
# nvidia-smi
# conda init
# source ~/.bashrc
# echo "conda activate env-novelai"
# conda activate env-novelai 
# cd /group/40034/jackeywu/code/PhotoMaker/

CUDA_VISIBLE_DEVICES=0 python photomaker_multi_adapter_single.py \
    -i 'datasets/lecun/yann-lecun.jpg' \
    --output "test_new" \
    --multi_ip_adapter \
    --clip_h \
    --size 512
