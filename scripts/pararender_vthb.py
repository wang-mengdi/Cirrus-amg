import os
import glob
import argparse

from paraview.simple import *


def format_count(count):
    if count >= 1024 ** 2:
        return f"{count / (1024 ** 2):.2f}M"
    elif count >= 1024:
        return f"{count / 1024:.2f}K"
    else:
        return str(count)


def read_simulator_stats(frame_number, stats_dir):
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


def extract_frame_number_from_path(vthb_file_path):
    base_name = os.path.basename(vthb_file_path)
    stem = os.path.splitext(base_name)[0]
    if stem.startswith("fluid"):
        return stem[len("fluid"):]
    return stem


def build_point_data_source_for_amr(amr_source):
    c2p = CellDatatoPointData(Input=amr_source)
    return c2p


def build_masked_source_if_needed(display_source, array_name, mask_non_finest):
    if not mask_non_finest:
        return display_source, array_name

    masked_array_name = f"{array_name}_finest_only"
    calc = Calculator(Input=display_source)
    calc.ResultArrayName = masked_array_name
    calc.Function = f"{array_name} * (level == max(level))"
    return calc, masked_array_name


def render_vthb_to_png(frame_number, vthb_file_path, png_file_path,
                       array_name, color_range, auto_rescale_each_frame, info,
                       cam_pos, cam_focal, cam_up,
                       mask_non_finest):
    #_DisableFirstRenderCameraReset()

    amr_data = None
    point_data_source = None
    display_source = None
    text = None
    render_view = GetActiveViewOrCreate("RenderView")

    try:
        print(f"[Frame {frame_number}] Opening {vthb_file_path}")
        amr_data = OpenDataFile(vthb_file_path)
        if amr_data is None:
            raise RuntimeError(f"OpenDataFile failed for {vthb_file_path}")

        amr_data.UpdatePipeline()
        print(f"[Frame {frame_number}] Opened source type: {type(amr_data)}")

        point_data_source = build_point_data_source_for_amr(amr_data)
        point_data_source.UpdatePipeline()

        display_source, masked_array_name = build_masked_source_if_needed(
            point_data_source, array_name, mask_non_finest
        )
        display_source.UpdatePipeline()

        rep = Show(display_source, render_view)
        rep.Representation = "Volume"

        print(f"[Frame {frame_number}] Coloring by POINT_DATA/{masked_array_name}")
        ColorBy(rep, ("POINT_DATA", masked_array_name))

        lut = GetColorTransferFunction(masked_array_name)
        lut.ApplyPreset("Cool to Warm", True)

        pwf = GetOpacityTransferFunction(masked_array_name)

        if color_range is not None:
            print(f"[Frame {frame_number}] Using fixed color range {color_range}")
            lut.AutomaticRescaleRangeMode = "Never"
            lut.RescaleTransferFunction(color_range[0], color_range[1])
            pwf.RescaleTransferFunction(color_range[0], color_range[1])
            pwf.Points = [
                color_range[0], 0.0, 0.5, 0.0,
                color_range[1], 1.0, 0.5, 0.0
            ]
        elif auto_rescale_each_frame:
            print(f"[Frame {frame_number}] Auto-rescaling to current frame")
            rep.RescaleTransferFunctionToDataRange(True, False)
        else:
            print(f"[Frame {frame_number}] Auto-rescaling once")
            rep.RescaleTransferFunctionToDataRange(True, False)

        camera = render_view.GetActiveCamera()
        camera.SetPosition(cam_pos)
        camera.SetFocalPoint(cam_focal)
        camera.SetViewUp(cam_up)

        if info:
            logs_dir = os.path.join(os.path.dirname(vthb_file_path), "logs")
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
        print(f"[Frame {frame_number}] Saving PNG image to {png_file_path} ...")
        SaveScreenshot(png_file_path, render_view, ImageResolution=[1920, 1080])

    except Exception as e:
        print(f"Error rendering frame {frame_number}: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if text is not None:
            Hide(text, render_view)
            Delete(text)
        if display_source is not None:
            Hide(display_source, render_view)
            if display_source is not point_data_source:
                Delete(display_source)
        if point_data_source is not None:
            Hide(point_data_source, render_view)
            if point_data_source is not amr_data:
                Delete(point_data_source)
        if amr_data is not None:
            Hide(amr_data, render_view)
            Delete(amr_data)


def parse_slice(slice_spec):
    parts = [int(part) if part else None for part in slice_spec.split(":")]
    return slice(*parts)


def render_all_vthb_files(input_path, array_name, slice_spec, info,
                          cam_pos, cam_focal, cam_up, mask_non_finest,
                          color_range, auto_rescale_each_frame):
    output_dir = os.path.join(input_path, f"render_{array_name}")
    os.makedirs(output_dir, exist_ok=True)

    vthb_files = sorted(glob.glob(os.path.join(input_path, "fluid*.vthb")))
    print(f"Found {len(vthb_files)} vthb files")

    if not vthb_files:
        print("No .vthb files found in the specified directory.")
        return

    if slice_spec:
        slice_indices = parse_slice(slice_spec)
        vthb_files = vthb_files[slice_indices]
        print(f"Selected {len(vthb_files)} files after slice {slice_spec}")

    for vthb_file in vthb_files:
        frame_number = extract_frame_number_from_path(vthb_file)
        png_file_name = f"frame.{frame_number}.png"
        png_file_path = os.path.join(output_dir, png_file_name)
        render_vthb_to_png(
            frame_number, vthb_file, png_file_path,
            array_name, color_range, auto_rescale_each_frame, info,
            cam_pos, cam_focal, cam_up, mask_non_finest
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render VTK Overlapping AMR (.vthb) files to PNG images."
    )
    parser.add_argument("input_path", type=str)
    parser.add_argument("--name", type=str, default="dye_density")
    parser.add_argument("--slice", type=str, default=None)
    parser.add_argument("--noinfo", action="store_true")
    parser.add_argument("--mask-non-finest", action="store_true")
    parser.add_argument("--range", type=float, nargs=2, default=None,
                        help="Fixed color range: --range min max")
    parser.add_argument("--auto-rescale-each-frame", action="store_true",
                        help="Rescale transfer function independently for each frame")

    args = parser.parse_args()

    cam_pos = [-1.47342, 1.2625, 1.62391]
    cam_focal = [0.526026, 0.492487, 1.11681]
    cam_up = [0.348582, 0.936099, -0.0470125]

    render_all_vthb_files(
        args.input_path,
        args.name,
        slice_spec=args.slice,
        info=not args.noinfo,
        cam_pos=cam_pos,
        cam_focal=cam_focal,
        cam_up=cam_up,
        mask_non_finest=args.mask_non_finest,
        color_range=args.range,
        auto_rescale_each_frame=args.auto_rescale_each_frame
    )