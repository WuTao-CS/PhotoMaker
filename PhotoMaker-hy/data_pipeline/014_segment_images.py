import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import os
from idadapter.transforms import tensor_to_image
from tqdm import tqdm
import glob
import json
"""
TODO: count IOU part for imdb faces 
"""
def select_seg_id(segmentation, seg_id_list, face_bbox):
    """
    select mask with the highest iou
    """
    x1, y1, x2, y2 = face_bbox
    segmentation_face = segmentation[y1:y2, x1:x2]
    print(segmentation_face.shape)

    ###
    # select the maximum area 
    max_sum = 0
    matched_id = None
    for seg_id in seg_id_list:
        cur_sum = (segmentation_face == seg_id).sum().item()
        if  cur_sum > max_sum:
            matched_id = seg_id
            max_sum = cur_sum
    return matched_id

# load Mask2Former fine-tuned on COCO panoptic segmentation
processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-coco-panoptic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-coco-panoptic")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
root_path = './projects/IDAdapter-diffusers/data/poco_celeb_images_cropped'
# root_path = './projects/IDAdapter-diffusers/data/imdb_celeb_images_toy_before_crop_cropped'
root_name = os.path.basename(root_path)
folder_list = os.listdir(root_path)

for folder_name in tqdm(sorted(folder_list)):
    folder = os.path.join(root_path, folder_name)   

    img_and_mask_list = set(glob.glob(os.path.join(folder, '*.png')))
    mask_list = set(glob.glob(os.path.join(folder, '*.mask.png')))
    img_list = img_and_mask_list - mask_list

    for idx, img_path in enumerate(sorted(list(img_list))):
        if os.path.exists(img_path.replace('.png', '.mask.png')):
            continue
        json_path = img_path.replace('.png', '.json')
        print(img_path)
        with open(json_path) as f:
            meta_dict = json.load(f)
        face_bbox = meta_dict["bbox_after_cropped"]
        face_bbox = [max(0, bb) for bb in face_bbox]
    # img_path = "./projects/IDAdapter-diffusers/data/ffhq_wild_files/00000/000000002.jpg"
        image = Image.open(img_path)
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # model predicts class_queries_logits of shape `(batch_size, num_queries)`
        # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        class_queries_logits = outputs.class_queries_logits
        masks_queries_logits = outputs.masks_queries_logits

        # you can pass them to processor for postprocessing
        result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        seg_id_list = []
        for seg_info in result["segments_info"]:
            if seg_info['label_id'] == 0:
                seg_id_list.append(seg_info['id'])
        
        ###
        # select the maximum area 
        # max_sum = 0
        # for seg_id in seg_id_list:
        #     cur_sum = (result["segmentation"] == seg_id).sum().item()
        #     if  cur_sum > max_sum:
        #         matched_id = seg_id
        #         max_sum = cur_sum

        ###
        # select with the highest IoU
        matched_id = select_seg_id(result["segmentation"], seg_id_list, face_bbox)
        if matched_id is None:
            print(f'{img_path} did not find any segment')
            with open('data-process/poco/014_failed_segment.txt', 'a+') as f:
                f.write(f"{img_path}\n")    
            continue
        
        mask = result["segmentation"] == matched_id
        filename = img_path.replace(".png", ".mask.png")
        tensor_to_image(mask*255, filename=filename, norm=False)
    # import pdb; pdb.set_trace()
    # we refer to the demo notebooks for visualization (see "Resources" section in the Mask2Former docs)
