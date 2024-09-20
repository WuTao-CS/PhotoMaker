import torch
import json
import os
from tqdm import tqdm
with open("datasets/CeleV-Text/processed_sdxl_512_final.json", 'r') as file:
    all_data = json.load(file)

all_success_data = []
for data in tqdm(all_data):
    if data['path'] == "/group/40033/public_datasets/CeleV-Text//processed_sdxl_512/9lFRpV1C_WY_32_0.pt":
        print("get")
        continue
    all_success_data.append(data)

with open('datasets/CeleV-Text/processed_sdxl_512_final_new.json', 'w') as file:
    json.dump(all_success_data, file, indent=4)