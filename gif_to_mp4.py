from moviepy.editor import VideoFileClip
import os

def gif_to_mp4(gif_path):
    # 读取 GIF 文件
    video = VideoFileClip(gif_path)
    
    # 获取 GIF 文件的文件名（不包括扩展名）
    base_name = os.path.splitext(os.path.basename(gif_path))[0]
    
    # 构建 MP4 文件的保存路径
    mp4_path = os.path.join(os.path.dirname(gif_path), f"{base_name}.mp4")
    
    # 将 GIF 文件写入 MP4 文件
    video.write_videofile(mp4_path, codec='libx264')
    
    # 关闭视频文件
    video.close()
    
    print(f"GIF 文件已成功转换为 MP4 文件并保存为: {mp4_path}")

# 示例用法
gif_path = 'animation_0_adin_1024.gif'
gif_to_mp4(gif_path)