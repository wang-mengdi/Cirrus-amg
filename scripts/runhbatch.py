'''
用法：python runhbatch.py path/to/hipnc_file 0:401（假设渲0~400帧）
1、脚本开头的OUTPUT_DIR是输出和hipnc文件的相对路径，这个用来检查是否渲染成功，一般默认好像是smoke，你按需修改
2、render_frame开头的渲染节点你需要自己设置，现在这个是/stage/usdrender_rop1，之前还有一个mantra_ipr
3、argparse里面的output_dir无效，是一个失败的尝试，不要理它，你可以自己删掉
4、在cc-westcity上设置，好像用cmder跑的效果优于powershell，原因不明
'''

import subprocess
import argparse
import os
import time
import psutil
from threading import Timer
#from datetime import datetime  # 引入 datetime 模块
import sys
from datetime import timedelta


env = os.environ.copy()

#OUTPUT_DIR = "render"
# 超时时间（秒）
RETRY_DELAY = 10      # 每次重试间隔秒数
RENDER_TIMEOUT = 10 * 60

def kill_process_tree(pid):
    parent = psutil.Process(pid)
    for child in parent.children(recursive=True):
        child.kill()
    parent.kill()

def is_render_successful(output_dir, frame):
    """检查渲染是否成功"""
    frame_pattern = f"{frame:04d}"
    has_image_file = False
    has_checkpoint_file = False

    for filename in os.listdir(output_dir):
        # 检查是否存在以 .png 或 .jpg 结尾的文件
        if frame_pattern in filename and (filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".exr")):
            has_image_file = True

        # 检查是否存在以 checkpoint 结尾的文件
        if frame_pattern in filename and (filename.endswith("checkpoint")):
            has_checkpoint_file = True

    # 成功的条件：有图像文件，且没有 checkpoint 文件
    return has_image_file and not has_checkpoint_file

def render_frame(hip_file, frame, output_path):
    #render_command = f"render -V -f {frame} {frame} /stage/usdrender_rop1 -o {output_path}/frame_{frame:04d}.png"
    #render_command = f"render -V -f {frame} {frame} /out/mantra_ipr -o {output_path}/frame_{frame}.exr"
    #render_command = f"render -V -f {frame} {frame} /out/mantra_ipr"
    render_command = f"render -V -f {frame} {frame} /out/mantra_ipr -o {output_path}/frame_{frame:04d}.png"
    print(f"Rendering frame {frame}...")

    # 启动 hbatch 进程，直接将 stdout 和 stderr 转发到父进程的终端
    process = subprocess.Popen(
        ['hbatch', hip_file],
        stdin=subprocess.PIPE,
        stdout=sys.stdout,  # 将 stdout 转发到父进程的 stdout
        stderr=sys.stderr,  # 将 stderr 转发到父进程的 stderr
        text=True,
        bufsize=1  # 实时输出，逐行读取
    )

    # 启动定时器
    timer = Timer(RENDER_TIMEOUT, lambda: kill_process_tree(process.pid))
    timer.start()

    # 输入渲染命令并执行
    process.stdin.write(f"{render_command}\nquit\n")
    process.stdin.close()  # 关闭输入流

    process.wait()  # 等待进程结束

    timer.cancel()  # 取消定时器


    return process.returncode  # 返回进程的退出码


def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Render frames using Houdini hbatch")
    parser.add_argument("hip_file", help="Path to the .hipnc file")
    parser.add_argument("frame_range", help="Slice of frames to render (e.g., '0:10:2' or '5:20:1')")
    parser.add_argument("--output_dir", default="render", help="Output directory for rendered frames")
    args = parser.parse_args()

    # 解析帧范围，支持步长
    frame_range_parts = args.frame_range.split(":")
    if len(frame_range_parts) not in {2, 3}:
        print("Error: frame_range must be in the format 'start:end' or 'start:end:stride'.")
        return

    start = int(frame_range_parts[0])
    end = int(frame_range_parts[1])
    stride = int(frame_range_parts[2]) if len(frame_range_parts) == 3 else 1

    if stride <= 0:
        print("Error: stride must be a positive integer.")
        return

    frames = list(range(start, end, stride))
    total_frames = len(frames)

    # 检查 .hipnc 文件的路径是否存在
    if not os.path.isfile(args.hip_file):
        print(f"Error: {args.hip_file} does not exist.")
        return

    #output_dir = os.path.join(os.path.dirname(args.hip_file), args.output_dir)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 记录开始时间
    start_time = time.time()

    # 渲染每一帧
    for i, frame in enumerate(frames, start=1):
        success = False
        frame_start_time = time.time()

        while not success:
            return_code = render_frame(args.hip_file, frame, output_dir)
            if return_code == 0 and is_render_successful(output_dir, frame):
                frame_end_time = time.time()
                elapsed = frame_end_time - frame_start_time
                total_elapsed = frame_end_time - start_time
                avg_time_per_frame = total_elapsed / i
                remaining_frames = total_frames - i
                eta = timedelta(seconds=remaining_frames * avg_time_per_frame)

                print(f"Frame {i}/{total_frames} (Frame {frame}) rendered successfully."
                      f" Time taken: {elapsed:.2f} seconds. ETA: {eta}.")
                success = True
            else:
                print(f"Error occurred while rendering frame {frame}. Retrying...")
                time.sleep(RETRY_DELAY)

if __name__ == "__main__":
    main()