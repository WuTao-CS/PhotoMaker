import cv2
import json
import os
from tqdm import tqdm
import argparse
import decord
from decord import VideoReader, cpu
import argparse


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--phase",type=int,default=0)
    parser.add_argument("--total",type=int,default=4)
    return parser

def process_video_with_canny(input_video_path, output_video_path):
    # Initialize the video reader
    decord.bridge.set_bridge('native')
    vr = decord.VideoReader(input_video_path, ctx=cpu())
    
    # Get video properties
    frame_width = vr[0].shape[1]
    frame_height = vr[0].shape[0]
    fps = vr.get_avg_fps()
    
    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height), isColor=False)
    
    for frame in vr:
        # Convert frame to numpy array
        frame = frame.asnumpy()
        
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray_frame, 100, 200)
        
        # Write the processed frame to the output video
        out.write(edges)
    
    # Release the video writer
    out.release()


# process_video_with_canny(input_video_path, output_video_path)
with open("datasets/sh_CeleV-Text/all_segment_results_head_path.json", 'r') as file:
    all_data = json.load(file)

parser = get_parser()
args = parser.parse_args()
# 计算每个阶段的处理数量
per_device_num = len(all_data) / args.total
start = int(args.phase * per_device_num)
end = int((args.phase + 1) * per_device_num)

# 如果 end 超出范围，则处理剩余的所有数据
if end >= len(all_data):
    all_data = all_data[start:]
else:
    all_data = all_data[start:end]


for data in tqdm(all_data):
    base_name = os.path.basename(data['path']).split('.')[0]
    video_path = os.path.join("datasets/CeleV-Text/celebvtext_6", base_name + '.mp4')
    output_video_path = os.path.join("datasets/CeleV-Text/celebvtext_6_canny", base_name + '.mp4')
    if os.path.exists(output_video_path):
        continue
    process_video_with_canny(video_path, output_video_path)