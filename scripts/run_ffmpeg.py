import os
import re
import sys
import glob
import argparse

def detect_image_pattern(folder):
    for ext in ('png', 'jpg'):
        files = sorted(glob.glob(os.path.join(folder, f'*.{ext}')))
        if not files:
            continue

        for file in files:
            filename = os.path.basename(file)
            match = re.match(r'(.*?)(\d+)\.' + ext, filename)
            if match:
                prefix = match.group(1)
                digits = match.group(2)
                num_digits = len(digits)
                pattern = f'{prefix}%0{num_digits}d.{ext}'
                full_pattern = os.path.join(folder, pattern)
                return full_pattern, prefix.rstrip('.'), ext
    return None, None, None

def fallback_output_name_from_folder(folder):
    parts = os.path.normpath(folder).split(os.sep)
    return parts[-1] if parts else "output"

def main():
    parser = argparse.ArgumentParser(description="Render a video from image sequence using ffmpeg.")
    parser.add_argument('folder', type=str, help='Path to the folder containing image frames')
    parser.add_argument('--output', '-o', type=str, default=None, help='Optional output file name (e.g., result.mp4)')
    parser.add_argument('--framerate', '-f', type=int, default=25, help='Framerate for the output video (default: 25)')

    args = parser.parse_args()
    folder = args.folder
    framerate = args.framerate

    image_pattern, prefix, ext = detect_image_pattern(folder)
    if not image_pattern:
        print("❌ No valid image sequence files found (expected numbered .png or .jpg).")
        sys.exit(1)

    if args.output:
        output_name = args.output
    else:
        fallback = fallback_output_name_from_folder(folder)
        output_name = f"{prefix or fallback}.mp4"

    output_path = os.path.join(folder, output_name)

    cmd = f'ffmpeg -framerate {framerate} -i "{image_pattern}" -c:v libx264 -pix_fmt yuv420p "{output_path}"'
    print(f"▶️ Running command: {cmd}")
    os.system(cmd)

if __name__ == '__main__':
    main()
