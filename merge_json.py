import json
import os
import itertools

all_data = []
for i in range(4):
    path = '/group/40007/public_datasets/CeleV-Text/processed_sdxl_512_{}.json'.format(i)
    with open(path, 'r') as file:
        loaded_data = json.load(file)
    all_data.append(loaded_data)

all_data = itertools.chain(*all_data)
all_data= list(all_data)
with open('datasets/CeleV-Text/processed_sdxl_512.json', 'w') as file:
    json.dump(all_data, file, indent=4)