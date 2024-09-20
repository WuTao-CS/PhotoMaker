import torch
import json
import os
from tqdm import tqdm
with open("datasets/CeleV-Text/processed_sdxl_512_final.json", 'r') as file:
    all_data = json.load(file)

all_success_data = []
for data in tqdm(all_data):
    process_data_path = data['path']
    try:
        process_data = torch.load(process_data_path, map_location='cpu')
    except:
        print(process_data_path)
        continue