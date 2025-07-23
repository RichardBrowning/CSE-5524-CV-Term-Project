import cv2
import os
import random
import argparse

def extract_random_frames(video_path, output_dir, num_frames, output_format='jpg', verbose=False):
    """
    从视频中随机提取N帧并保存
    
    参数:
        video_path (str): 输入视频文件路径
        output_dir (str): 输出目录
        num_frames (int): 要提取的帧数
        output_format (str): 输出图像格式(如'jpg','png')
        verbose (bool): 是否显示详细信息
    """
    # 检查输出目录是否存在，不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print(f"创建输出目录: {output_dir}")
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return
    
    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print("错误: 视频中没有帧")
        return
    
    if verbose:
        print(f"视频总帧数: {total_frames}")
        print(f"提取帧数: {num_frames}")
    
    # 如果请求的帧数大于总帧数，则调整
    if num_frames > total_frames:
        print(f"警告: 请求的帧数({num_frames})大于总帧数({total_frames})，将提取所有帧")
        num_frames = total_frames
    
    # 随机选择帧索引
    frame_indices = sorted(random.sample(range(total_frames), num_frames))
    if verbose:
        print(f"随机选择的帧索引: {frame_indices}")
    
    # 提取并保存帧
    saved_count = 0
    for idx in frame_indices:
        # 设置到指定帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            output_path = os.path.join(output_dir, f"frame_{idx}.{output_format}")
            cv2.imwrite(output_path, frame)
            saved_count += 1
            if verbose:
                print(f"已保存: {output_path}")
        else:
            print(f"警告: 无法读取帧 {idx}")
    
    cap.release()
    print(f"完成! 成功保存 {saved_count}/{num_frames} 帧到 {output_dir}")

if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='从MOV视频中随机提取N帧')
    parser.add_argument('video_path', help='输入视频文件路径')
    parser.add_argument('-n', '--num_frames', type=int, default=10, 
                        help='要提取的帧数 (默认: 10)')
    parser.add_argument('-o', '--output_dir', default='output_frames',
                        help='输出目录 (默认: output_frames)')
    parser.add_argument('-f', '--format', default='jpg',
                        help='输出图像格式 (默认: jpg)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='显示详细信息')
    
    args = parser.parse_args()
    
    # 调用函数
    extract_random_frames(
        video_path=args.video_path,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        output_format=args.format,
        verbose=args.verbose
    )

"""
Example command: python ./helper/getRandomFrames.py ./input/watson00024168.mov -n 10 -o ./output/frames_output -f png -v
"""