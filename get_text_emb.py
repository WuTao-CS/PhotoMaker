from transformers import CLIPTextModel, CLIPTokenizer
tokenizer = CLIPTokenizer.from_pretrained(
    './pretrain_model/stable-diffusion-v1-5', subfolder="tokenizer"
)
text_encoder = CLIPTextModel.from_pretrained(
    './pretrain_model/stable-diffusion-v1-5', subfolder="text_encoder"
)

input_ids = tokenizer('bust shot', max_length=77, padding="max_length", truncation=True, return_tensors="pt").input_ids
print(input_ids)