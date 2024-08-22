# nvidia-smi
# conda init
# source ~/.bashrc
# echo "conda activate env-novelai"
# conda activate env-novelai 
# cd /group/40034/jackeywu/code/PhotoMaker/

CUDA_VISIBLE_DEVICES=0 python photomaker_fusion_infer.py \
    -i 'datasets/lecun/yann-lecun.jpg' \
    --unet_path "checkpoints/checkpoint-6000/pytorch_model_final.bin" \
    --output "outputs/4_card_checkpoint-6000/"


CUDA_VISIBLE_DEVICES=1 python photomaker_fusion_infer.py \
    -i 'datasets/lecun/yann-lecun.jpg' \
    --unet_path "checkpoints/checkpoint-9000/pytorch_model_final.bin" \
    --output "outputs/4_card_checkpoint-9000/"
CUDA_VISIBLE_DEVICES=0 python photomaker_fusion_infer.py -i 'examples/scarletthead_woman/scarlett_1.jpg' --prompt 'woman.txt'
CUDA_VISIBLE_DEVICES=1 python photomaker_fusion_infer.py -i 'datasets/3.jpeg' --prompt 'person.txt'
CUDA_VISIBLE_DEVICES=2 python photomaker_fusion_infer.py -i 'examples/yangmi_woman/yangmi_6.jpg'  --prompt 'woman.txt'
CUDA_VISIBLE_DEVICES=3 python photomaker_fusion_infer.py -i 'datasets/hp.jpg'  --prompt 'person.txt'
python photomaker_fusion_infer.py -i 'examples/yangmi_woman/yangmi_6.jpg'