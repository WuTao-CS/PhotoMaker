import torch

model = torch.load("sh_checkpoints/train_npu/pytorch_model.bin",map_location='cpu')
new_model = {}
for key in model.keys():
    name = key[5:]
    new_model[name]=model[key]
torch.save(new_model,"sh_checkpoints/train_npu/pytorch_model_final.bin")