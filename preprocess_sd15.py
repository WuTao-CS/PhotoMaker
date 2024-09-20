import os, sys
import numpy as np
import torch
from torchvision.transforms import Compose
import cv2
from PIL import Image
from glob import glob
from einops import rearrange
from tqdm import tqdm
from torch import nn
from typing import Optional
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection,CLIPImageProcessor
import argparse
from photomaker.model import PhotoMakerIDEncoder
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from photomaker.datasets.celebv_text import CelebVTextProcessDataset
import json
# import torch_npu
# from torch_npu.contrib import transfer_to_npu

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--root", type=str, default='/group/40033/public_datasets/CeleV-Text/', help="data path")
    parser.add_argument("--save_path", type=str, default='/group/40033/public_datasets/CeleV-Text', help="data path")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default='./pretrain_model/stable-diffusion-v1-5', help="pretrained model path")
    parser.add_argument("--phase",type=int,default=0)
    parser.add_argument("--total",type=int,default=4)
    return parser



class Annotator(nn.Module):
    def __init__(self, pretrained_model_name_or_path="./pretrain_model/stable-diffusion-v1-5"):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            use_safetensors=True, 
        )
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae",
            use_safetensors=True, 
        )
        self.app = FaceAnalysis(name="buffalo_l",
                        root="./pretrain_model",
                        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.vae.enable_tiling()
        self.vae.enable_slicing()
        self.device = "cuda"

    def get_face_image(self, video):
        # Extract Face features using insightface
        ref_images = []
        bbox = []
        for i in range(video.shape[0]):
            img = video[i]
            img = np.array(img)
            face_info = self.app.get(img)
            
            if len(face_info)==0:
                continue
            face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
            norm_face = face_align.norm_crop(img, landmark=face_info.kps, image_size=512)
            bbox.append(face_info.bbox)
            ref_images.append(torch.tensor(norm_face))
        return ref_images,bbox

    def encode_video_with_vae(self, video):
        video_length = video.shape[1]
        pixel_values = rearrange(video, "b f c h w -> (b f) c h w")
        latents = self.vae.encode(pixel_values).latent_dist
        latents = latents.sample()
        latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
        return latents

    def encode_prompt(
        self,
        prompt,
        clip_skip: Optional[int] = None,
    ):
        device = self.device
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            )
            print(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        if clip_skip is None:
            prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
            prompt_embeds = prompt_embeds[0]
        else:
            prompt_embeds = self.text_encoder(
                text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
            )
            # Access the `hidden_states` first, that contains a tuple of
            # all the hidden states from the encoder layers. Then index into
            # the tuple to access the hidden states from the desired layer.
            prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
            # We also need to apply the final LayerNorm here to not mess with the
            # representations. The `last_hidden_states` that we typically use for
            # obtaining the final prompt representations passes through the LayerNorm
            # layer.
            prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)


        prompt_embeds_dtype = self.text_encoder.dtype
        
        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        return prompt_embeds


    def forward(self, batch, ref_data=None):
        video = batch["video"].to(self.device)
        ref_frames = batch["ref_frames"][0]
        prompt = batch["prompt"][0]
        # Extract Face features using insightface
        ref_images, bbox = self.get_face_image(ref_frames)
        if len(ref_images) == 0:
            return {}
        ref_images = torch.stack(ref_images, dim=0)
        ref_images = ref_images.permute(0, 3, 1, 2).unsqueeze(dim=0).to(device=self.device, dtype=video.dtype)

        ref_images = (ref_images / 255. - 0.5) * 2 
        ref_images_latent = self.encode_video_with_vae(ref_images)
        latent = self.encode_video_with_vae(video)
        # Encode the image

        prompt_embeds = self.encode_prompt(prompt)
        
        return {"latent": latent.squeeze(dim=0), "prompt_embeds": prompt_embeds.squeeze(dim=0), "ref_images_latent": ref_images_latent.squeeze(dim=0), "bbox":bbox}


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    annotator = Annotator()
    annotator.eval()
    annotator.to("cuda")
    print(args.phase)
    data = CelebVTextProcessDataset(root=args.root,resolution=[512,512],load_all_frames=False,phase=args.phase,total=args.total)
    print(len(data))
    data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=6, pin_memory=True)
    all_data = []
    for batch in tqdm(data_loader):
        if batch == {}:
            continue
        ref_data = None
        if os.path.exists(f"{args.save_path}/processed_sd15/{batch['name'][0]}.pt"):
            save_path = f"{args.save_path}/processed_sd15/{batch['name'][0]}.pt"
            # data = torch.load(save_path)
            all_data.append({"path":save_path, "prompt":batch['prompt'][0]})
            continue
        with torch.no_grad():
            output = annotator(batch,ref_data)
        if output == {}:
            continue
        output['prompt'] = batch['prompt'][0]
        save_path = f"{args.root}/processed_sd15/{batch['name'][0]}.pt"
        all_data.append({"path":save_path, "prompt":batch['prompt'][0]})
        torch.save(output, f"{args.save_path}/processed_sd15/{batch['name'][0]}.pt")

    json_file_name = os.path.join(args.save_path, "processed_sd15_{}.json".format(args.phase))
    with open(json_file_name, 'w') as  f:
        json.dump(all_data, f, indent=4)
    print("ok")

