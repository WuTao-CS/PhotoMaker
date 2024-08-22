import torch

model = torch.load("checkpoints/checkpoint-3000/pytorch_model.bin",map_location='cpu')
new_model = {}
for key in model.keys():
    name = key[5:]
    new_model[name]=model[key]
torch.save(new_model,"checkpoints/checkpoint-3000/pytorch_model.bin")