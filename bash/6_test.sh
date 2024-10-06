#!/bin/bash
echo 'Begin to install python packages...'
nvidia-smi
conda init
source ~/.bashrc
echo "conda activate env-novelai"
conda activate env-novelai 
cd /group/40034/jackeywu/code/PhotoMaker/
CUDA_VISIBLE_DEVICES=5 python process_reg_data.py --phase 5
