import os
import cv2

def extract_frames(video_path, output_folder):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
    
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    frame_count = 0
    
    while True:
        # 读取一帧
        ret, frame = cap.read()
        
        # 如果读取失败，退出循环
        if not ret:
            break
        
        # 保存帧为图像文件
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        
        frame_count += 1
    
    # 释放视频对象
    cap.release()
    print(f"已提取 {frame_count} 帧到 {output_folder}")

def process_folder(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp4"):
            # 获取视频文件的完整路径
            video_path = os.path.join(folder_path, filename)
            
            # 创建同名文件夹
            output_folder = os.path.join(folder_path, os.path.splitext(filename)[0])
            
            # 提取视频帧
            extract_frames(video_path, output_folder)

if __name__ == "__main__":
    # 指定包含MP4文件的文件夹路径
    folder_path = "datasets/test_canny"
    
    # 处理文件夹中的MP4文件
    process_folder(folder_path)