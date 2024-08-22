from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download
import os

os.environ['HF_ENDPOINT'] = 'hf-mirror.com'
snapshot_download(repo_id="h94/IP-Adapter-FaceID",local_dir="./pretrain_model/IP-Adapter-FaceID")
snapshot_download(repo_id="h94/IP-Adapter",local_dir="./pretrain_model/IP-Adapter")
snapshot_download(repo_id="guoyww/animatediff-motion-adapter-sdxl-beta",local_dir="./pretrain_model/animatediff-motion-adapter-sdxl-beta")
# snapshot_download(repo_id="h94/IP-Adapter",local_dir="./pretrain_model/IP-Adapter")
