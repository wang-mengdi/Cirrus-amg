import os
import glob
import argparse
import numpy as np

from paraview.simple import *
from paraview.servermanager import Fetch

try:
    from paraview.simple import _DisableFirstRenderCameraReset
except ImportError:
    def _DisableFirstRenderCameraReset():
        pass


YELLOW = "\033[93m"
RESET = "\033[0m"


def warn_yellow(msg):
    print(f"{YELLOW}{msg}{RESET}")


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


def get_color_range_from_vti(vti_file_path, array_name):
    data = OpenDataFile(vti_file_path)
    data.UpdatePipeline()
    vtk_data = Fetch(data)

    try:
        if not vtk_data.IsA("vtkImageData"):
            raise TypeError(f"The file '{vti_file_path}' is not a valid VTI file.")

        point_data = vtk_data.GetPointData()
        cell_data = vtk_data.GetCellData()

        point_arrays = []
        for i in range(point_data.GetNumberOfArrays()):
            arr = point_data.GetArray(i)
            if arr:
                point_arrays.append(arr.GetName())

        cell_arrays = []
        for i in range(cell_data.GetNumberOfArrays()):
            arr = cell_data.GetArray(i)
            if arr:
                cell_arrays.append(arr.GetName())

        print(f"[DEBUG] Arrays in {vti_file_path}")
        print(f"[DEBUG]   Point arrays ({len(point_arrays)}): {point_arrays}")
        print(f"[DEBUG]   Cell arrays  ({len(cell_arrays)}): {cell_arrays}")

        array = cell_data.GetArray(array_name)
        if array:
            return array.GetRange()

        raise ValueError(f"Cell array '{array_name}' not found in {vti_file_path}.")
    finally:
        Delete(data)


def extract_frame_number_from_path(vti_file_path):
    base_name = os.path.basename(vti_file_path)
    stem = os.path.splitext(base_name)[0]
    if stem.startswith("fluid"):
        return stem[len("fluid"):]
    return stem


def get_cell_array_range_from_vti(vti_file_path, array_name):
    data = OpenDataFile(vti_file_path)
    data.UpdatePipeline()
    vtk_data = Fetch(data)

    result = None
    try:
        if vtk_data.IsA("vtkImageData"):
            cell_data = vtk_data.GetCellData()
            array = cell_data.GetArray(array_name)
            if array:
                result = array.GetRange()
    finally:
        Delete(data)

    return result


def build_masked_source_if_needed(vti_data, vti_file_path, array_name, mask_non_finest):
    if not mask_non_finest:
        return vti_data, array_name

    level_range = get_cell_array_range_from_vti(vti_file_path, "level")
    if level_range is None:
        print(f"Warning: CELL array 'level' not found in {vti_file_path}, skip masking.")
        return vti_data, array_name

    max_level = int(level_range[1])
    masked_array_name = f"{array_name}_finest_only"

    calc = Calculator(Input=vti_data)
    calc.AttributeType = "Cell Data"
    calc.ResultArrayName = masked_array_name
    calc.Function = f"{array_name} * (level == {max_level})"

    return calc, masked_array_name


def parse_slice(slice_spec):
    parts = [int(part) if part else None for part in slice_spec.split(":")]
    return slice(*parts)


def read_transform_matrix(xform_file_path):
    M = np.loadtxt(xform_file_path)
    if M.shape != (4, 4):
        raise ValueError(f"{xform_file_path} does not contain a 4x4 matrix")
    return M


def matrix_to_translate_rotate_xyz(M):
    tx, ty, tz = M[0, 3], M[1, 3], M[2, 3]
    R = M[:3, :3]

    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-8

    if not singular:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0.0

    rot_deg = [np.degrees(rx), np.degrees(ry), np.degrees(rz)]
    trans = [float(tx), float(ty), float(tz)]
    return trans, rot_deg


def read_mesh_source(mesh_path):
    ext = os.path.splitext(mesh_path)[1].lower()

    if ext == ".ply":
        print(f"[Mesh] Using PLYReader for {mesh_path}")
        return PLYReader(FileNames=[mesh_path])
    else:
        print(f"[Mesh] Using OpenDataFile for {mesh_path}")
        return OpenDataFile(mesh_path)

def build_mesh_pipeline(mesh_path, render_view):
    mesh = read_mesh_source(mesh_path)

    mesh_xform = ProgrammableFilter(Input=mesh)
    mesh_xform.OutputDataSetType = "vtkPolyData"
    mesh_xform.CopyArrays = 1

    mesh_display = Show(mesh_xform, render_view)
    mesh_display.Representation = "Surface"

    return mesh, mesh_xform, mesh_display


def apply_transform_to_mesh(mesh_xform, xform_file_path):
    M = read_transform_matrix(xform_file_path)

    flat = []
    for i in range(4):
        for j in range(4):
            flat.append(float(M[i, j]))

    mesh_xform.Script = f"""
import vtk

inp = self.GetInputDataObject(0, 0)
output = self.GetOutputDataObject(0)

mat = vtk.vtkMatrix4x4()
vals = {flat}
k = 0
for i in range(4):
    for j in range(4):
        mat.SetElement(i, j, vals[k])
        k += 1

transform = vtk.vtkTransform()
transform.SetMatrix(mat)

tf = vtk.vtkTransformPolyDataFilter()
tf.SetTransform(transform)
tf.SetInputData(inp)
tf.Update()

output.ShallowCopy(tf.GetOutput())
"""
    mesh_xform.Modified()
    mesh_xform.UpdatePipeline()



def build_render_jobs(vti_files, slice_spec):
    """
    Returns a list of (target_frame_number_str, vti_file_path).

    Rules:
    1. If slice has explicit stop, force frame list from range(stop)[slice].
       If vti_files is shorter than required, pad with the last vti.
    2. If slice has no explicit stop, apply slice directly on vti_files.
    3. If slice is None, render all vti_files normally.
    """
    if not vti_files:
        return []

    if slice_spec is None:
        return [(extract_frame_number_from_path(v), v) for v in vti_files]

    s = parse_slice(slice_spec)

    # Explicit stop: force target frame ids from slice itself
    if s.stop is not None:
        target_ids = list(range(s.stop))[s]
        jobs = []

        for idx, frame_id in enumerate(target_ids):
            if frame_id < len(vti_files):
                vti_file = vti_files[frame_id]
            else:
                vti_file = vti_files[-1]

            jobs.append((f"{frame_id:04d}", vti_file))

        return jobs

    # No explicit stop: slice within available VTI list
    sliced_vti_files = vti_files[s]
    return [(extract_frame_number_from_path(v), v) for v in sliced_vti_files]


def render_vti_to_png(frame_number, vti_file_path, png_file_path,
                      array_name, color_range, info,
                      cam_pos, cam_focal, cam_up,
                      mask_non_finest, render_outline,
                      mesh_path=None, xform_dir=None):
    _DisableFirstRenderCameraReset()

    render_view = GetActiveViewOrCreate("RenderView")
    vti_data = None
    display_source = None
    text = None
    outline = None
    mesh = None
    mesh_xform = None
    masked_array_name = array_name

    try:
        if array_name is not None or render_outline:
            vti_data = OpenDataFile(vti_file_path)

        if array_name is not None:
            display_source, masked_array_name = build_masked_source_if_needed(
                vti_data, vti_file_path, array_name, mask_non_finest
            )

            vti_representation = Show(display_source, render_view)
            vti_representation.Representation = "Volume"

            ColorBy(vti_representation, ("CELL_DATA", masked_array_name))

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

        if render_outline:
            outline = Outline(Input=vti_data)
            outline_display = Show(outline, render_view)
            outline_display.Representation = "Wireframe"
            outline_display.LineWidth = 2.0
            outline_display.DiffuseColor = [1.0, 1.0, 1.0]

        if mesh_path is not None:
            mesh, mesh_xform, _ = build_mesh_pipeline(
                mesh_path=mesh_path,
                render_view=render_view
            )

            if xform_dir is not None:
                xform_file_path = os.path.join(xform_dir, f"transform_{frame_number}.txt")
                if not os.path.exists(xform_file_path):
                    raise FileNotFoundError(f"Transform file not found: {xform_file_path}")
                apply_transform_to_mesh(mesh_xform, xform_file_path)

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
        print(f"[Frame {frame_number}] Saving PNG image to {png_file_path} ...")
        SaveScreenshot(png_file_path, render_view, ImageResolution=[1920, 1080])
        print(f"[Frame {frame_number}] Saved.")

    except Exception as e:
        print(f"Error rendering frame {frame_number}: {e}")

    finally:
        if text is not None:
            Hide(text, render_view)
            Delete(text)

        if outline is not None:
            Hide(outline, render_view)
            Delete(outline)

        if mesh_xform is not None:
            Hide(mesh_xform, render_view)
            Delete(mesh_xform)


        if mesh is not None:
            Delete(mesh)

        if display_source is not None:
            Hide(display_source, render_view)
            if display_source is not vti_data:
                Delete(display_source)

        if vti_data is not None:
            Hide(vti_data, render_view)
            Delete(vti_data)


def render_all_vti_files(args):
    array_name = None if args.name.lower() == "none" else args.name
    input_path = args.input_path
    slice_spec = args.slice
    info = not args.noinfo
    mask_non_finest = args.mask_non_finest
    render_outline = args.outline

    mesh_path = args.mesh
    xform_dir = args.xform_dir

    cam_pos = [-1.47342, 1.2625, 1.62391]
    cam_focal = [0.526026, 0.492487, 1.11681]
    cam_up = [0.348582, 0.936099, -0.0470125]

    output_dir = os.path.join(input_path, "render_mesh" if array_name is None else f"render_{array_name}")
    os.makedirs(output_dir, exist_ok=True)

    vti_files = sorted(glob.glob(os.path.join(input_path, "fluid*.vti")))
    print(f"Found {len(vti_files)} vti files")

    if not vti_files and (array_name is not None or render_outline):
        print("No .vti files found in the specified directory.")
        return

    jobs = build_render_jobs(vti_files, slice_spec)

    if slice_spec:
        print(f"Built {len(jobs)} render jobs after slice {slice_spec}")

    if not jobs:
        print("No frames remain after slicing.")
        return

    if array_name is not None:
        first_frame_color_range = get_color_range_from_vti(jobs[0][1], array_name)
        print(f"Using color range {first_frame_color_range} from {jobs[0][1]}.")
    else:
        first_frame_color_range = None
        print("Volume rendering disabled because --name none was specified.")

    total_frames = len(jobs)
    print(f"Total frames to render: {total_frames}")

    if mesh_path is not None:
        print(f"Using mesh: {mesh_path}")
    if xform_dir is not None:
        print(f"Using transform dir: {xform_dir}")

    for i, (frame_number, vti_file) in enumerate(jobs, start=1):
        vti_frame = extract_frame_number_from_path(vti_file)
        if vti_frame != frame_number:
            warn_yellow(
                f"[Warning] target frame {frame_number} uses VTI {os.path.basename(vti_file)} "
                f"(VTI frame {vti_frame}); continuing."
            )

        png_file_name = f"frame.{frame_number}.png"
        png_file_path = os.path.join(output_dir, png_file_name)

        print(f"[{i}/{total_frames}] Rendering frame {frame_number}: {vti_file}")

        render_vti_to_png(
            frame_number=frame_number,
            vti_file_path=vti_file,
            png_file_path=png_file_path,
            array_name=array_name,
            color_range=first_frame_color_range,
            info=info,
            cam_pos=cam_pos,
            cam_focal=cam_focal,
            cam_up=cam_up,
            mask_non_finest=mask_non_finest,
            render_outline=render_outline,
            mesh_path=mesh_path,
            xform_dir=xform_dir
        )

        print(f"[{i}/{total_frames}] Done frame {frame_number}")


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
        help='Name of the CELL data array to visualize, or "none" to disable volume rendering.'
    )
    parser.add_argument(
        "--slice",
        type=str,
        default=None,
        help='Slice notation to select frames/files (e.g., "0:5" or "1:10:2"). '
             'If stop is explicitly given, frame ids are forced by the slice.'
    )
    parser.add_argument(
        "--noinfo",
        action="store_true",
        help="If set, disable display of additional information."
    )
    parser.add_argument(
        "--mask-non-finest",
        action="store_true",
        help="If set, and if CELL-data array 'level' exists, set the render field to 0 for cells whose level is not the maximum."
    )
    parser.add_argument(
        "--outline",
        action="store_true",
        help="If set, render the outline of the VTI domain."
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default=None,
        help="Optional mesh file to overlay."
    )
    parser.add_argument(
        "--xform-dir",
        type=str,
        default=None,
        help="Directory containing per-frame transform files like transform_0000.txt."
    )

    args = parser.parse_args()
    render_all_vti_files(args)
