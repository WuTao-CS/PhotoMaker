import torch
import json
import os
from tqdm import tqdm
with open("datasets/CeleV-Text/processed_sdxl_512.json", 'r') as file:
    all_data = json.load(file)

all_success_data = []
for data in tqdm(all_data):
    process_data_path = data['path']
    try:
        process_data = torch.load(process_data_path, map_location='cpu')
    except:
        print(process_data_path)
        continue
    if process_data["latent"].shape[0]==1:
        video_ref = torch.cat([process_data["latent"].unsqueeze(dim=1),process_data["ref_images_latent"]],dim=1)
        process_data["latent"]=video_ref[:,:,:16,:,:].squeeze(0)
        process_data["ref_images_latent"]=video_ref[:,:,16:,:,:]
    all_success_data.append(data)
    torch.save(process_data, process_data_path)

with open('datasets/CeleV-Text/processed_sdxl_512_final.json', 'w') as file:
    json.dump(all_success_data, file, indent=4)