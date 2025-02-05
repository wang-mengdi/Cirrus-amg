from parautils import *
import argparse
import os

if __name__ == '__main__':
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Render VTK Image Data (.vti) files to PNG images.')
    parser.add_argument('input_path', type=str, help='Path to the directory containing .vti files.')
    parser.add_argument('--name', type=str, default='dye_density', help='Name of the data array to visualize (default: dye_density).')
    parser.add_argument('--slice', type=str, default=None, help='Slice notation to select files (e.g., "0:5" or "1:10:2").')
    parser.add_argument('--noinfo', action='store_true', help='If set, disable display of additional information (default: enabled).')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Determine the last folder name in input_path
    last_folder = os.path.basename(os.path.normpath(args.input_path))
    
    # default camera
    cam_pos = [-0.5206968888777139, 1.185731530874039, 3.2695103418850517]
    cam_focal = [0.965593201751042, 0.42536166799965003, 0.5020286511739026]
    cam_up = [0.10359017000732675, 0.9718954402396329, -0.2113961445231755]
    if "smokesphere" in last_folder:
        cam_pos = [0, -1.0, 0.4]
        cam_focal = [0, 0, 0]
        cam_up = [0, 0, 1]

    
    # Call the render function
    render_all_vti_files(args.input_path, args.name, slice_spec=args.slice, info=not args.noinfo, cam_pos=cam_pos, cam_focal=cam_focal, cam_up=cam_up)
