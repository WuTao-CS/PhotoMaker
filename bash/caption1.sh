echo 'Begin to install python packages...'
nvidia-smi
conda init
source ~/.bashrc
echo "conda activate env-novelai"
conda activate env-novelai 
cd /group/40034/jackeywu/code/PhotoMaker/
PHASE=0


CUDA_VISIBLE_DEVICES=$PHASE python eval_ip_adapter_animatediff_object.py \
        --image_dir 'datasets/object_benchmark_seg/bear' \
        --version 'ip-adapter-plus_sd15.bin' \
        --output 'outputs/object_bench/ip-adapter-object' \

CUDA_VISIBLE_DEVICES=$PHASE python eval_ip_adapter_animatediff_object.py \
        --image_dir 'datasets/object_benchmark_seg/bear' \
        --version 'ip-adapter-plus_sd15.bin' \
        --output 'outputs/object_bench/ip-adapter-plus-object' \
