import os
import glob
import argparse

from paraview.simple import *
from paraview.servermanager import Fetch


def format_count(count):
    """
    Format a count value to use 'K' for thousands and 'M' for millions.
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
        with open(stats_file, "r") as file:
            for line in file:
                if "total particles" in line:
                    particles = int(line.split()[-1])
                elif "total leaf cells" in line:
                    cells = int(line.split()[-1])
    except FileNotFoundError:
        print(f"Warning: {stats_file} not found.")
    return particles, cells


def read_driver_stats(frame_number, logs_dir):
    """Read the runtime from frame_driver_info{frame_number}.txt"""
    runtime = 0.0
    iters = 0
    runtime_file = os.path.join(logs_dir, f"frame_driver_info{frame_number}.txt")
    try:
        with open(runtime_file, "r") as file:
            for line in file:
                if "Frame time (seconds)" in line:
                    runtime = float(line.split()[-1])
                elif "Iterations" in line:
                    iters = int(line.split()[-1])
    except FileNotFoundError:
        print(f"Warning: {runtime_file} not found.")
    return runtime, iters


def get_color_range_from_vti(vti_file_path, array_name):
    """
    Get the color range (min and max values) from a VTI file for a specified data array.
    """
    data = OpenDataFile(vti_file_path)
    data.UpdatePipeline()
    vtk_data = Fetch(data)

    if vtk_data.IsA("vtkImageData"):
        point_data = vtk_data.GetPointData()
        array = point_data.GetArray(array_name)
        if array:
            value_range = array.GetRange()
            Delete(data)
            return value_range
        else:
            Delete(data)
            raise ValueError(f"Array '{array_name}' not found in {vti_file_path}.")
    else:
        Delete(data)
        raise TypeError(f"The file '{vti_file_path}' is not a valid VTI file.")


def extract_frame_number_from_path(vti_file_path):
    """
    Extract frame number from file name like fluid0000.vti -> 0000.
    """
    base_name = os.path.basename(vti_file_path)
    stem = os.path.splitext(base_name)[0]
    if stem.startswith("fluid"):
        return stem[len("fluid"):]
    return stem

def get_point_array_range_from_vti(vti_file_path, array_name):
    """
    Get range of a point-data array from a VTI file.
    Returns (min_value, max_value), or None if the array does not exist.
    """
    data = OpenDataFile(vti_file_path)
    data.UpdatePipeline()
    vtk_data = Fetch(data)

    result = None
    if vtk_data.IsA("vtkImageData"):
        point_data = vtk_data.GetPointData()
        array = point_data.GetArray(array_name)
        if array:
            result = array.GetRange()

    Delete(data)
    return result


def build_masked_source_if_needed(vti_data, vti_file_path, array_name, mask_non_finest):
    """
    If mask_non_finest is enabled and 'level' exists, create a Calculator
    that masks out points whose level is not the maximum level.
    Returns:
        source_to_show, array_name_to_render
    Otherwise returns:
        vti_data, array_name
    """
    if not mask_non_finest:
        return vti_data, array_name

    level_range = get_point_array_range_from_vti(vti_file_path, "level")
    if level_range is None:
        print(f"Warning: 'level' array not found in {vti_file_path}, skip masking.")
        return vti_data, array_name

    max_level = int(level_range[1])
    masked_array_name = f"{array_name}_finest_only"

    calc = Calculator(Input=vti_data)
    calc.ResultArrayName = masked_array_name
    calc.Function = f"{array_name} * (level == {max_level})"

    return calc, masked_array_name

def render_vti_to_png(frame_number, vti_file_path, png_file_path,
                      array_name, color_range, info,
                      cam_pos, cam_focal, cam_up,
                      mask_non_finest):
    _DisableFirstRenderCameraReset()

    vti_data = None
    display_source = None
    text = None
    masked_array_name = array_name

    try:
        vti_data = OpenDataFile(vti_file_path)
        render_view = GetActiveViewOrCreate("RenderView")

        display_source, masked_array_name = build_masked_source_if_needed(
            vti_data, vti_file_path, array_name, mask_non_finest
        )

        vti_representation = Show(display_source, render_view)
        vti_representation.Representation = "Volume"

        ColorBy(vti_representation, ("POINT_DATA", masked_array_name))

        lut = GetColorTransferFunction(masked_array_name)
        lut.AutomaticRescaleRangeMode = "Never"
        lut.ApplyPreset("Cool to Warm", True)
        lut.RescaleTransferFunction(color_range[0], color_range[1])

        pwf = GetOpacityTransferFunction(masked_array_name)
        pwf.RescaleTransferFunction(color_range[0], color_range[1])
        pwf.Points = [
            color_range[0], 0.0, 0.5, 0.0,
            color_range[1], 1.0, 0.5, 0.0
        ]

        camera = render_view.GetActiveCamera()
        camera.SetPosition(cam_pos)
        camera.SetFocalPoint(cam_focal)
        camera.SetViewUp(cam_up)

        if info:
            logs_dir = os.path.join(os.path.dirname(vti_file_path), "logs")
            runtime, iters = read_driver_stats(frame_number, logs_dir)
            particle_count, cell_count = read_simulator_stats(frame_number, logs_dir)

            annotation_text = f"Frame: {frame_number} ({iters} iters)"
            annotation_text += f"\nRuntime: {runtime:.2f} s"
            annotation_text += f"\nParticles: {format_count(particle_count)}"
            annotation_text += f"\nLeaf cells: {format_count(cell_count)}"

            text = Text()
            text.Text = annotation_text
            text_representation = Show(text, render_view)
            text_representation.Color = [1.0, 0.0, 0.0]
            text_representation.FontSize = 12
            text_representation.WindowLocation = "Upper Left Corner"
            text_representation.Justification = "Left"
            text_representation.VerticalJustification = "Top"

        Render()
        print(f"Saving PNG image to {png_file_path} ...")
        SaveScreenshot(png_file_path, render_view, ImageResolution=[1920, 1080])

    except Exception as e:
        print(f"Error rendering frame {frame_number}: {e}")

    finally:
        if text is not None:
            Hide(text, render_view)
            Delete(text)
        if display_source is not None:
            Hide(display_source, render_view)
            if display_source is not vti_data:
                Delete(display_source)
        if vti_data is not None:
            Hide(vti_data, render_view)
            Delete(vti_data)


def parse_slice(slice_spec):
    """
    Parse a slice notation string (e.g., '0:5' or '1:10:2') into a slice object.
    """
    parts = [int(part) if part else None for part in slice_spec.split(":")]
    return slice(*parts)


def render_all_vti_files(input_path, array_name, slice_spec, info,
                         cam_pos, cam_focal, cam_up, mask_non_finest):
    """
    Render selected fluid*.vti files in the specified directory and save them as PNG images.
    """
    output_dir = os.path.join(input_path, f"render_{array_name}")
    os.makedirs(output_dir, exist_ok=True)

    vti_files = sorted(glob.glob(os.path.join(input_path, "fluid*.vti")))
    print(f"Found {len(vti_files)} vti files")

    if not vti_files:
        print("No .vti files found in the specified directory.")
        return

    first_frame_color_range = get_color_range_from_vti(vti_files[0], array_name)
    print(f"Using color range {first_frame_color_range} from {vti_files[0]}.")

    if slice_spec:
        slice_indices = parse_slice(slice_spec)
        vti_files = vti_files[slice_indices]
        print(f"Selected {len(vti_files)} files after slice {slice_spec}")

    for vti_file in vti_files:
        frame_number = extract_frame_number_from_path(vti_file)
        png_file_name = f"frame.{frame_number}.png"
        png_file_path = os.path.join(output_dir, png_file_name)
        render_vti_to_png(
            frame_number, vti_file, png_file_path,
            array_name, first_frame_color_range, info,
            cam_pos, cam_focal, cam_up, mask_non_finest
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render VTK Image Data (.vti) files to PNG images."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to the directory containing .vti files."
    )
    parser.add_argument(
        "--name",
        type=str,
        default="dye_density",
        help="Name of the data array to visualize (default: dye_density)."
    )
    parser.add_argument(
        "--slice",
        type=str,
        default=None,
        help='Slice notation to select files (e.g., "0:5" or "1:10:2").'
    )
    parser.add_argument(
        "--noinfo",
        action="store_true",
        help="If set, disable display of additional information."
    )
    parser.add_argument(
        "--mask-non-finest",
        action="store_true",
        help="If set, and if point-data array 'level' exists, set the render field to 0 for points whose level is not the maximum."
    )

    args = parser.parse_args()

    last_folder = os.path.basename(os.path.normpath(args.input_path))

    # cam_pos = [-0.5206968888777139, 1.185731530874039, 3.2695103418850517]
    # cam_focal = [0.965593201751042, 0.42536166799965003, 0.5020286511739026]
    # cam_up = [0.10359017000732675, 0.9718954402396329, -0.2113961445231755]
    cam_pos = [-1.47342, 1.2625, 1.62391]
    cam_focal = [0.526026, 0.492487, 1.11681]
    cam_up = [0.348582, 0.936099, -0.0470125]

    # if "smokesphere" in last_folder:
    #     cam_pos = [-1, 0.5, 0.5]
    #     cam_focal = [0.5, 0.5, 0.5]
    #     cam_up = [0, 0, 1]

    render_all_vti_files(
        args.input_path,
        args.name,
        slice_spec=args.slice,
        info=not args.noinfo,
        cam_pos=cam_pos,
        cam_focal=cam_focal,
        cam_up=cam_up,
        mask_non_finest=args.mask_non_finest
    )