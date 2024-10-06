import numpy as np
import torch
import torch.nn as nn
from glob import glob
import torch.utils
import torch.utils.data
from tqdm import tqdm
import logging
from decord import VideoReader, cpu
from typing import List, Optional, Tuple, Union, Any
from time import time
from glob import glob
import random
import json
from torchvision import transforms
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from typing import Optional
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection,CLIPImageProcessor
import argparse
from photomaker.model import PhotoMakerIDEncoder
from einops import rearrange
import os
MATCHED_WORDS = ["person", "male", "female", "man", "woman", "men", "women", "girl", "boy", "girls", "boys", "lady", "ladies", "teen", "teens", "student", "students"]
def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--save_path", type=str, default='/group/40075/public_datasets/CeleV-Text/', help="data path")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default='./pretrain_model/RealVisXL_V4.0', help="pretrained model path")
    parser.add_argument("--phase",type=int,default=0)
    parser.add_argument("--total",type=int,default=8)
    return parser

class PhotoMakerIDEncoderForlabel(PhotoMakerIDEncoder):
    def __init__(self):
        super().__init__()
    def forward(self, id_pixel_values, prompt_embeds, class_tokens_mask):
        b, num_inputs, c, h, w = id_pixel_values.shape
        id_pixel_values = id_pixel_values.view(b * num_inputs, c, h, w)

        vision_outputs= self.vision_model(id_pixel_values)
        shared_id_embeds = vision_outputs[1] # 768
        id_embeds = self.visual_projection(shared_id_embeds)
        id_embeds_2 = self.visual_projection_2(shared_id_embeds)

        id_embeds = id_embeds.view(b, num_inputs, 1, -1)
        id_embeds_2 = id_embeds_2.view(b, num_inputs, 1, -1)    

        id_embeds = torch.cat((id_embeds, id_embeds_2), dim=-1) #[b, num_inputs, 1, 2048]
        updated_prompt_embeds=[]
        for i in range(num_inputs):
            updated_prompt_embed = self.fuse_module(prompt_embeds, id_embeds[:,i,:,:], class_tokens_mask)
            updated_prompt_embeds.append(updated_prompt_embed.squeeze(0))
        updated_prompt_embeds = torch.stack(updated_prompt_embeds, dim=0)
        return updated_prompt_embeds, id_embeds_2

class Annotator(nn.Module):
    def __init__(self, pretrained_model_name_or_path="./pretrain_model/RealVisXL_V4.0"):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            use_safetensors=True, 
        )
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            use_safetensors=True, 
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_2")
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae",
            use_safetensors=True, 
        )
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained("./pretrain_model/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.id_encoder = PhotoMakerIDEncoderForlabel()
        self.id_encoder.load_from_pretrained('./pretrain_model/PhotoMaker/photomaker-v1.bin')
        self.trigger_word = "img"
        self.app = FaceAnalysis(name="buffalo_l",
                        root="./pretrain_model",
                        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.feature_extractor = CLIPImageProcessor()
        self.vae.enable_tiling()
        self.vae.enable_slicing()
        self.device = "cuda"

    def encode_faceid(self, video):
        # Extract Face features using insightface
        ref_images = []
        ref_images_emb = []
        bbox = []
        for i in range(video.shape[0]):
            img = video[i]
            img = np.array(img)
            face_info = self.app.get(img)
            
            if len(face_info)==0:
                continue
            if len(face_info)>1:
                return None,None,None
            face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
            norm_face = face_align.norm_crop(img, landmark=face_info.kps, image_size=512)
            bbox.append(face_info.bbox)
            ref_images.append(torch.tensor(norm_face))
            emb = torch.from_numpy(face_info.normed_embedding)
            ref_images_emb.append(emb)

        if len(ref_images)==0:
            return None,None,None
        else:
            ref_images_emb = torch.stack(ref_images_emb, dim=0)
        return ref_images,bbox,ref_images_emb

    def encode_image_adpater(self, video , device, output_hidden_states=True):
        dtype = next(self.image_encoder.parameters()).dtype
        image = video
        image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        if output_hidden_states:
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(1, dim=0)
            ref_images=image_enc_hidden_states
            
        else:
            image_embeds = self.image_encoder(image).image_embeds
            image_embeds = image_embeds.repeat_interleave(1, dim=0)
            ref_images=image_embeds

        return ref_images
        
    def encode_video_with_vae(self, video):
        video_length = video.shape[1]
        pixel_values = rearrange(video, "b f c h w -> (b f) c h w")
        latents = self.vae.encode(pixel_values).latent_dist
        latents = latents.sample()
        latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
        return latents

    def encode_prompt(self, prompt, device, prompt_2=None, clip_skip=None):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )
        prompt_2 = prompt_2 or prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

        # textual inversion: process multi-vector tokens if necessary
        prompt_embeds_list = []
        prompts = [prompt, prompt_2]
        for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            if clip_skip is None:
                prompt_embeds = prompt_embeds.hidden_states[-2]
            else:
                # "2" because SDXL always indexes from the penultimate layer.
                prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        return prompt_embeds, pooled_prompt_embeds
    
    def encode_prompt_with_trigger_word(
            self,
            prompt: str,
            prompt_2: Optional[str] = None,
            num_id_images: int = 1,
            device: Optional[torch.device] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            class_tokens_mask: Optional[torch.LongTensor] = None,
        ):
            device = self.device

            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            # Find the token id of the trigger word
            image_token_id = self.tokenizer_2.convert_tokens_to_ids(self.trigger_word)

            # Define tokenizers and text encoders
            tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
            text_encoders = (
                [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
            )

            if prompt_embeds is None:
                prompt_2 = prompt_2 or prompt
                prompt_embeds_list = []
                prompts = [prompt, prompt_2]
                for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
                    input_ids = tokenizer.encode(prompt) # TODO: batch encode
                    clean_index = 0
                    clean_input_ids = []
                    class_token_index = []
                    # Find out the corresponding class word token based on the newly added trigger word token
                    for i, token_id in enumerate(input_ids):
                        if token_id == image_token_id:
                            class_token_index.append(clean_index - 1)
                        else:
                            clean_input_ids.append(token_id)
                            clean_index += 1

                    class_token_index = class_token_index[0]

                    # Expand the class word token and corresponding mask
                    class_token = clean_input_ids[class_token_index]
                    clean_input_ids = clean_input_ids[:class_token_index] + [class_token] * num_id_images + \
                        clean_input_ids[class_token_index+1:]                
                        
                    # Truncation or padding
                    max_len = tokenizer.model_max_length
                    if len(clean_input_ids) > max_len:
                        clean_input_ids = clean_input_ids[:max_len]
                    else:
                        clean_input_ids = clean_input_ids + [tokenizer.pad_token_id] * (
                            max_len - len(clean_input_ids)
                        )

                    class_tokens_mask = [True if class_token_index <= i < class_token_index+num_id_images else False \
                        for i in range(len(clean_input_ids))]
                    
                    clean_input_ids = torch.tensor(clean_input_ids, dtype=torch.long).unsqueeze(0).to(device)
                    class_tokens_mask = torch.tensor(class_tokens_mask, dtype=torch.bool).unsqueeze(0).to(device)
                    
                    prompt_embeds = text_encoder(
                        input_ids=clean_input_ids,
                        output_hidden_states=True,
                    )

                    # We are only ALWAYS interested in the pooled output of the final text encoder
                    pooled_prompt_embeds = prompt_embeds[0]
                    prompt_embeds = prompt_embeds.hidden_states[-2]
                    prompt_embeds_list.append(prompt_embeds)

                prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
            class_tokens_mask = class_tokens_mask.to(device=device) # TODO: ignoring two-prompt case

            return prompt_embeds, pooled_prompt_embeds, class_tokens_mask

    def encode_photomaker_prompt(self, video, prompt_embeds, class_tokens_mask, device):
        # 5. Prepare the input ID images
        dtype = next(self.id_encoder.parameters()).dtype
        input_id_images = video
        id_pixel_values = self.feature_extractor(input_id_images, return_tensors="pt").pixel_values

        id_pixel_values = id_pixel_values.unsqueeze(0).to(device=device, dtype=dtype) # TODO: multiple prompts

        # 6. Get the update text embedding with the stacked ID embedding
        prompt_embeds, _ip_adapter_image_embeds = self.id_encoder(id_pixel_values, prompt_embeds, class_tokens_mask)

        return prompt_embeds, _ip_adapter_image_embeds

    def forward(self, batch, ref_data=None):
        video = batch["video"].to(self.device)
        ref_frames = batch["ref_frames"][0]
        prompt = batch["prompt"][0]
        prompt_trigger = batch["prompt_trigger"][0]
        ref_images, bbox, face_ids = self.encode_faceid(ref_frames)
        if face_ids is None:
            return {}
        ref_images = torch.stack(ref_images, dim=0)
        # Encode the image
        image_embeds = self.encode_image_adpater(ref_images, self.device)
        # Encode the prompt
        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(prompt, self.device)
        # Encode the prompt with the trigger word
        prompt_embeds_trigger, pooled_prompt_embeds_trigger, class_tokens_mask = self.encode_prompt_with_trigger_word(prompt_trigger, self.device)
        prompt_embeds_trigger, _ip_adapter_image_embeds = self.encode_photomaker_prompt(ref_images, prompt_embeds_trigger, class_tokens_mask, self.device)
        # save all in a dict
        ref_images = ref_images.permute(0, 3, 1, 2).unsqueeze(dim=0).to(device=self.device, dtype=video.dtype)
        ref_images = (ref_images / 255. - 0.5) * 2 
        # Encode the video
        vae_input = torch.cat([video,ref_images],dim=1)
        vae_output = self.encode_video_with_vae(vae_input)
        latent = vae_output[:,:,:video.shape[1],:,:]
        ref_latent = vae_output[:,:,video.shape[1]:,:,:]
        return {"latent": latent.squeeze(dim=0),"ref_images_latent": ref_latent, "face_ids": face_ids, "image_embeds": image_embeds, "prompt_embeds": prompt_embeds.squeeze(dim=0), "pooled_prompt_embeds": pooled_prompt_embeds.squeeze(dim=0), "prompt_embeds_trigger": prompt_embeds_trigger.squeeze(dim=0), "pooled_prompt_embeds_trigger": pooled_prompt_embeds_trigger.squeeze(dim=0)}



def insert_img_after_keyword(text):
    for word in MATCHED_WORDS:
        index = text.find(word)
        if index != -1:
            # Find the end index of the matched word
            end_index = index + len(word)
            # Insert "img" after the matched word
            new_text = text[:end_index] + " img" + text[end_index:]
            return new_text, True
    return text, False


def exists(val: Any) -> bool:
    return val is not None

def make_spatial_transformations(resolution, type, ori_resolution=None):
    """ 
    resolution: target resolution, a list of int, [h, w]
    """
    if type == "random_crop":
        transformations = transforms.RandomCropss(resolution)
    elif type == "resize_center_crop":
        is_square = (resolution[0] == resolution[1])
        if is_square:
            transformations = transforms.Compose([
                transforms.Resize(resolution[0]),
                transforms.CenterCrop(resolution[0]),
                ])
        else:
            if ori_resolution is not None:
                # resize while keeping original aspect ratio,
                # then centercrop to target resolution
                resize_ratio = max(resolution[0] / ori_resolution[0], resolution[1] / ori_resolution[1])
                resolution_after_resize = [int(ori_resolution[0] * resize_ratio), int(ori_resolution[1] * resize_ratio)]
                transformations = transforms.Compose([
                    transforms.Resize(resolution_after_resize),
                    transforms.CenterCrop(resolution),
                    ])
            else:
                # directly resize to target resolution
                transformations = transforms.Compose([
                    transforms.Resize(resolution),
                    ])
    elif type == "resize":
        transformations = transforms.Compose([
            transforms.Resize(resolution),
            ])
    else:
        raise NotImplementedError
    return transformations


class PexelsHumanDataset(torch.utils.data.Dataset):
    def __init__(self, save_path, root = './datasets/Pexels_Human/', video_length = 16, resolution = [512,512], frame_stride=8, spatial_transform_type="resize_center_crop", fixed_fps=None, phase=1, total=4):
        self.root = root
        self.video_length = video_length
        self.resolution = resolution
        with open(os.path.join(self.root, "video_caption.json"), 'r') as file:
            self.all_data = json.load(file)
        self.spatial_transform_type = spatial_transform_type
        self.spatial_transform = make_spatial_transformations(self.resolution, type=self.spatial_transform_type) \
            if self.spatial_transform_type is not None else None
        self.fixed_fps = fixed_fps
        self.frame_stride = frame_stride
        per_devie_num = len(self.all_data)/total
        start = int(phase*per_devie_num)
        end = int((phase+1)*per_devie_num)
        if end >= len(self.all_data):
            self.all_data = self.all_data[start:]
        else:
            self.all_data = self.all_data[start:end]
        self.save_path = save_path

    def __len__(self) -> int:
        return len(self.all_data)
    
    def __getitem__(self, index):
        index = index % len(self.all_data)
        data = self.all_data[index]
        prompt = data["prompt"]
        process_data_path = data["path"]
        name = os.path.basename(process_data_path).split(".")[0]
        save_path = os.path.join(self.save_path, f"{name}.pt")
        prompt_trigger, is_have_word = insert_img_after_keyword(prompt)
        if is_have_word is False:
            # print("skip prompt")
            return {}
        if os.path.exists(save_path):
            return {"prompt": prompt, "path": process_data_path, "prompt_trigger":prompt_trigger}
        
        video_path = process_data_path
        video_reader = VideoReader(video_path, ctx=cpu(), width=self.resolution[1], height=self.resolution[0])
        fps_ori = video_reader.get_avg_fps()
        if self.fixed_fps is not None:
            frame_stride = int(self.frame_stride * (1.0 * fps_ori / self.fixed_fps))
        else:
            frame_stride = self.frame_stride
        frame_stride = max(frame_stride, 1)
        required_frame_num = frame_stride * (self.video_length-1) + 1
        frame_num = len(video_reader)
        if frame_num < required_frame_num:
            frame_stride = frame_num // self.video_length
            required_frame_num = frame_stride * (self.video_length-1) + 1
        random_range = frame_num - required_frame_num
        start_idx = random.randint(0, random_range) if random_range > 0 else 0
        frame_indices = [start_idx + frame_stride*i for i in range(self.video_length)]
        frames = video_reader.get_batch(frame_indices)
        ref_frames = frames.asnumpy()
        assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
        frames = torch.tensor(frames.asnumpy()).permute(0,3,1,2).float() # [t,h,w,c] -> [t,c,h,w]
        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
        frames = (frames / 255 - 0.5) * 2
        # no normalize frames
        return {"video":frames, "prompt": prompt, "path": process_data_path, "prompt_trigger":prompt_trigger, "ref_frames":ref_frames}


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    annotator = Annotator()
    annotator.eval()
    annotator.to("cuda")
    print(args.phase)
    save_dir = os.path.join(args.save_path, "reg_data_sdxl_512_face1")
    data = PexelsHumanDataset(phase=args.phase,total=args.total,resolution=[512,512],save_path=save_dir)
    data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    all_data = []
    for batch in tqdm(data_loader):
        if batch == {}:
            continue
        name = os.path.basename(batch['path'][0]).split(".")[0]
        save_path = os.path.join(args.save_path, "reg_data_sdxl_512_face1", f"{name}.pt")
        if "video" not in batch.keys():
            all_data.append({"path":save_path, "prompt":batch['prompt'][0], "prompt_trigger":batch['prompt_trigger'][0]})
            continue
        ref_data = None
        with torch.no_grad():
            output = annotator(batch,ref_data)
        if output == {}:
            continue
        output['prompt'] = batch['prompt'][0]
        output['prompt_trigger'] = batch['prompt_trigger'][0]
        
        
        all_data.append({"path":batch['path'][0], "prompt":batch['prompt'][0], "prompt_trigger":batch['prompt_trigger'][0]})
        torch.save(output, save_path)

    json_file_name = os.path.join(args.save_path, "processed_sdxl_512_reg_data_{}.json".format(args.phase))
    with open(json_file_name, 'w') as  f:
        json.dump(all_data, f, indent=4)
    print("ok")
