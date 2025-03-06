import json
import os
import itertools

# all_data = []
# for i in range(4):
#     path = '/group/40034/jackeywu/code/PhotoMaker/datasets/CeleV-Text/qwen7b_caption_{}.json'.format(i)
#     with open(path, 'r') as file:
#         loaded_data = json.load(file)
#     all_data.append(loaded_data)

# all_data = itertools.chain(*all_data)
# all_data= list(all_data)
# with open('datasets/CeleV-Text/qwen7b_caption.json', 'w') as file:
#     json.dump(all_data, file, indent=4)

with open('datasets/CeleV-Text/qwen7b_caption.json', 'r') as file:
   all_data = json.load(file)

# new_data=[]
for data in all_data:
    data['prompt'] = data['prompt'][0]
with open('datasets/CeleV-Text/qwen7b_caption.json', 'w') as file:
    json.dump(all_data, file, indent=4)