echo 'Begin to install python packages...'
nvidia-smi
conda init
source ~/.bashrc
echo "conda activate env-novelai"
conda activate env-novelai 
cd /group/40034/jackeywu/code/PhotoMaker/

# python eval_animatediff.py

python eval_dd_animatediff.py --input 'outputs/animatediff_V3'