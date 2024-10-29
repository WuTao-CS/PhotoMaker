import json
import os
import argparse
import math

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--input_dir", type=str, default='outputs/sd15_latent_new_lr_1e-5_4a100-with-motion-no-update-1004/checkpoint-100000', help="prompt file path")
    return parser

args = get_parser().parse_args()

person_name = [f for f in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, f))]

all_data = []
total_clip_t = 0
total_frame_c = 0
total_clip_i = 0
total_dino_i = 0
total_face_similarity = 0
face_cnt = 0
for name in person_name:
    json_path = os.path.join(args.input_dir, name, 'result.json')
    data = json.load(open(json_path))
    all_data.append({name: data})
    total_clip_t += data['clip_t']
    total_frame_c += data['frame_c']
    total_clip_i += data['clip_i']
    total_dino_i += data['dino_i']

    
    if isinstance(data['face_similarity'], float) and math.isnan(data['face_similarity']):
        continue
    else:
        face_cnt += 1
        total_face_similarity += data['face_similarity']

with open(os.path.join(args.input_dir, 'all_result.json'), 'w') as file:
    json.dump(all_data, file, indent=4)


mean_face_similarity = total_face_similarity / face_cnt
mean_clip_t = total_clip_t / len(person_name)
mean_frame_c = total_frame_c / len(person_name)
mean_clip_i = total_clip_i / len(person_name)
mean_dino_i = total_dino_i / len(person_name)

# save to json
result = {
    'mean_face_similarity': mean_face_similarity,
    'mean_clip_t': mean_clip_t,
    'mean_frame_c': mean_frame_c,
    'mean_clip_i': mean_clip_i,
    'mean_dino_i': mean_dino_i
}
with open(os.path.join(args.input_dir, 'result_final.json'), 'w') as file:
    json.dump(result, file, indent=4)