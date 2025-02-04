import os
import glob
import sys
from paraview.simple import *
from paraview.servermanager import Fetch
import argparse
import numpy as np
import time

def format_count(count):
    """
    Format a count value to use 'K' for thousands and 'M' for millions.

    Parameters:
    count (int): The count value to format.

    Returns:
    str: Formatted string with appropriate unit.
    """
    if count >= 1024 ** 2:
        return f"{count / (1024 ** 2):.2f}M"
    elif count >= 1024:
        return f"{count / 1024:.2f}K"
    else:
        return str(count)

def read_simulator_stats(frame_number, stats_dir):
    """Read particle and cell count from simulator_stat_{frame_number}.txt"""
    particles = 0
    cells = 0
    stats_file = os.path.join(stats_dir, f"simulator_stat_{frame_number}.txt")
    try:
        with open(stats_file, 'r') as file:
            for line in file:
                if 'total particles' in line:
                    particles = int(line.split()[-1])
                elif 'total leaf cells' in line:
                    cells = int(line.split()[-1])
    except FileNotFoundError:
        print(f"Warning: {stats_file} not found.")
    return particles, cells

def read_driver_stats(frame_number, logs_dir):
    """Read the runtime from frame_driver_info{frame_number}.txt"""
    runtime = 0
    iters = 0
    runtime_file = os.path.join(logs_dir, f"frame_driver_info{frame_number}.txt")
    try:
        with open(runtime_file, 'r') as file:
            for line in file:
                if 'Frame time (seconds)' in line:
                    runtime = float(line.split()[-1])
                elif 'Iterations' in line:
                    iters = int(line.split()[-1])
    except FileNotFoundError:
        print(f"Warning: {runtime_file} not found.")
    return runtime, iters

def get_color_range_from_vti(vti_file_path, array_name):
    """
    Get the color range (min and max values) from a VTI file for a specified data array.

    Parameters:
    vti_file_path (str): Path to the .vti file.
    array_name (str): Name of the data array to visualize.

    Returns:
    tuple: (min_value, max_value) of the data array.
    """
    # Open the VTI file
    data = OpenDataFile(vti_file_path)
    data.UpdatePipeline()

    # Fetch the actual VTK data object
    vtk_data = Fetch(data)

    # Ensure the data is a valid VTI object
    if vtk_data.IsA("vtkImageData"):
        point_data = vtk_data.GetPointData()
        array = point_data.GetArray(array_name)
        if array:
            return array.GetRange()
        else:
            raise ValueError(f"Array '{array_name}' not found in {vti_file_path}.")
    else:
        raise TypeError(f"The file '{vti_file_path}' is not a valid VTI file.")
        
def render_vti_to_png(frame_number, vti_file_path, png_file_path, array_name, color_range, info=True):
    paraview.simple._DisableFirstRenderCameraReset()

    try:
        # 读取新的 VTI 数据
        vti_data = OpenDataFile(vti_file_path)

        # 显示涡量场（每帧刷新）
        vti_representation = Show(vti_data)
        vti_representation.Representation = 'Volume'

        # 应用颜色映射
        ColorBy(vti_representation, ('POINT_DATA', array_name))
        dye_density_lut = GetColorTransferFunction(array_name)
        dye_density_lut.AutomaticRescaleRangeMode = 'Never'
        dye_density_lut.ApplyPreset('Cool to Warm', True)
        dye_density_lut.RescaleTransferFunction(color_range[0], color_range[1])
        #vti_representation.RescaleTransferFunctionToDataRange(True)

        dye_density_pwf = GetOpacityTransferFunction(array_name)
        dye_density_pwf.RescaleTransferFunction(color_range[0], color_range[1])

        # 手动固定透明度映射
        dye_density_pwf.Points = [
            color_range[0], 0.0, 0.5, 0.0,  # 起始透明度为 0
            color_range[1], 1.0, 0.5, 0.0   # 结束透明度为 1
        ]



        # 调整相机位置（只需调整一次，但为了安全每帧设置）
        render_view = GetActiveViewOrCreate('RenderView')
        camera = render_view.GetActiveCamera()
        camera.SetPosition(-0.5206968888777139, 1.185731530874039, 3.2695103418850517)
        camera.SetFocalPoint(0.965593201751042, 0.42536166799965003, 0.5020286511739026)
        camera.SetViewUp(0.10359017000732675, 0.9718954402396329, -0.2113961445231755)

        # 添加注释信息（可选）
        if info:
            logs_dir = os.path.join(os.path.dirname(vti_file_path), 'logs')

            runtime, iters = read_driver_stats(frame_number, logs_dir)
            annotation_text = f"Frame: {frame_number} ({iters} iters)"
            annotation_text += f"\nRuntime: {runtime:.2f} s"

            particle_count, cell_count = read_simulator_stats(frame_number, logs_dir)
            annotation_text += f"\nParticles: {format_count(particle_count)}"
            annotation_text += f"\nLeaf cells: {format_count(cell_count)}"

            text = Text()
            text.Text = annotation_text
            text_representation = Show(text, render_view)
            text_representation.Color = [1.0, 0.0, 0.0]  # 红色
            text_representation.FontSize = 12
            text_representation.WindowLocation = 'Upper Left Corner'
            text_representation.Justification = 'Left'  # Ensure left alignment
            text_representation.VerticalJustification = 'Top'  # Ensure alignment from the top

        # 渲染并保存图片
        Render()
        print(f"Saving PNG image to {png_file_path}...")
        SaveScreenshot(png_file_path, render_view, ImageResolution=[1920, 1080])

        # 隐藏当前帧的 VTI 数据，准备下一帧
        Hide(vti_data)
        Delete(vti_data)
        Hide(text)
        Delete(text)
        #Delete(obj_representation)  # 删除上一个帧的对象

    except Exception as e:
        print(f"Error rendering frame {frame_number}: {e}")




def parse_slice(slice_spec):
    """
    Parse a slice notation string (e.g., '0:5' or '1:10:2') into a slice object.
    """
    parts = [int(part) if part else None for part in slice_spec.split(':')]
    return slice(*parts)

def render_all_vti_files(input_path, array_name, slice_spec=None, particles=False, info=True):
    """
    Render selected fluid*.vti files in the specified directory and save them as PNG images,
    optionally with particles rendered from corresponding .vtu files.

    Parameters:
    input_path (str): Path to the directory containing .vti files.
    array_name (str): Name of the data array to visualize.
    slice_spec (str): Slice notation to select files, e.g., '0:5' or '1:10:2'.
    particles (bool): Whether to render particles from .vtu files.
    info (bool): Whether to display additional information (particle and cell counts).
    """
    output_dir = os.path.join(input_path, f'render_{array_name}')
    os.makedirs(output_dir, exist_ok=True)

    vti_files = sorted(glob.glob(os.path.join(input_path, 'fluid*.vti')))
    
    # Get color range from the first frame
    if not vti_files:
        print("No .vti files found in the specified directory.")
        return

    first_frame_color_range = get_color_range_from_vti(vti_files[0], array_name)
    print(f"Using color range {first_frame_color_range} from the first frame.")

    # Apply slice to select a subset of files
    if slice_spec:
        slice_indices = parse_slice(slice_spec)
        vti_files = vti_files[slice_indices]

    for vti_file in vti_files:
        base_name = os.path.basename(vti_file)
        frame_number = base_name[5:9]
        png_file_name = f'frame.{frame_number}.png'
        png_file_path = os.path.join(output_dir, png_file_name)

        render_vti_to_png(frame_number, vti_file, png_file_path, array_name, first_frame_color_range, info)

if __name__ == '__main__':
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Render VTK Image Data (.vti) files to PNG images.')
    parser.add_argument('input_path', type=str, help='Path to the directory containing .vti files.')
    parser.add_argument('--name', type=str, default='dye_density', help='Name of the data array to visualize (default: dye_density).')
    parser.add_argument('--slice', type=str, default=None, help='Slice notation to select files (e.g., "0:5" or "1:10:2").')
    parser.add_argument('--particles', action='store_true', help='If set, render particles from .vtu files.')
    parser.add_argument('--noinfo', action='store_true', help='If set, disable display of additional information (default: enabled).')

    # Parse the arguments
    args = parser.parse_args()
    
    # Call the render function
    render_all_vti_files(args.input_path, args.name, slice_spec=args.slice, particles=args.particles, info=not args.noinfo)
