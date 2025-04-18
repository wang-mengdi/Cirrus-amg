import subprocess
import argparse
import os
import time
import psutil
from threading import Timer
import sys
from datetime import timedelta

RETRY_DELAY = 10      # 每次重试间隔秒数
RENDER_TIMEOUT = 10 * 60  # 渲染超时秒数

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
        if frame_pattern in filename and filename.endswith((".png", ".jpg", ".exr")):
            has_image_file = True
        if frame_pattern in filename and filename.endswith("checkpoint"):
            has_checkpoint_file = True

    return has_image_file and not has_checkpoint_file

def render_frame_with_husk(hip_file, frame, output_path, render_node):
    render_command = [
        "husk",
        "-f", str(frame), str(frame),
        "-o", f"{output_path}/{frame:04d}.png",
        hip_file,
        render_node
    ]
    print(f"Rendering frame {frame} with husk...")
    process = subprocess.Popen(render_command, stdout=sys.stdout, stderr=sys.stderr)
    process.wait()
    return process.returncode


def render_frame_with_command(hip_file, frame, output_path):
    render_command = f"render -V -f {frame} {frame} /stage/usdrender_rop1 -o {output_path}/{frame:04d}.png"
    #render_command = f"render -V -f {frame} {frame} /out/mantra_ipr"
    print(f"Rendering frame {frame}...")

    process = subprocess.Popen(
        ['hbatch', hip_file],
        stdin=subprocess.PIPE,
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
        bufsize=1
    )

    timer = Timer(RENDER_TIMEOUT, lambda: kill_process_tree(process.pid))
    timer.start()

    process.stdin.write(f"{render_command}\nquit\n")
    process.stdin.close()
    process.wait()
    timer.cancel()

    return process.returncode

def main():
    parser = argparse.ArgumentParser(description="Render frames using Houdini hbatch")
    parser.add_argument("hip_file", help="Path to the .hipnc file")
    parser.add_argument("frame_range", help="Slice of frames to render (e.g., '0:10:2' or '10:0:-1')")
    parser.add_argument("--output_dir", help="Output directory for rendered frames (optional)")
    args = parser.parse_args()

    frame_range_parts = args.frame_range.split(":")
    if len(frame_range_parts) not in {2, 3}:
        print("Error: frame_range must be in the format 'start:end' or 'start:end:stride'.")
        return

    start = int(frame_range_parts[0])
    end = int(frame_range_parts[1])
    stride = int(frame_range_parts[2]) if len(frame_range_parts) == 3 else 1

    if stride == 0:
        print("Error: stride cannot be zero.")
        return
    if (stride > 0 and start >= end) or (stride < 0 and start <= end):
        print("Error: frame_range and stride are incompatible.")
        return

    frames = list(range(start, end, stride))
    total_frames = len(frames)
    print(f"Rendering {total_frames} frames: {frames}")

    if not os.path.isfile(args.hip_file):
        print(f"Error: {args.hip_file} does not exist.")
        return

    # 自动设置 output_dir（若未指定）
    if args.output_dir:
        output_dir = args.output_dir
    else:
        hip_basename = os.path.basename(args.hip_file)
        output_name = hip_basename.rsplit('.', 1)[0]  # 去掉后缀
        output_dir = os.path.join(os.path.dirname(args.hip_file), output_name)
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()

    for i, frame in enumerate(frames, start=1):
        success = False
        frame_start_time = time.time()

        while not success:
            return_code = render_frame_with_husk(args.hip_file, frame, output_dir, "/stage/usdrender_rop1")
            if return_code == 0 and is_render_successful(output_dir, frame):
                frame_end_time = time.time()
                elapsed = frame_end_time - frame_start_time
                total_elapsed = frame_end_time - start_time
                avg_time_per_frame = total_elapsed / i
                eta = timedelta(seconds=(total_frames - i) * avg_time_per_frame)

                print(f"Frame {i}/{total_frames} (Frame {frame}) rendered successfully."
                      f" Time taken: {elapsed:.2f} seconds. ETA: {eta}.")
                success = True
            else:
                print(f"Error occurred while rendering frame {frame}. Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)

if __name__ == "__main__":
    main()
