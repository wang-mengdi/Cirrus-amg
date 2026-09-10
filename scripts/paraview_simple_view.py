r"""Create an editable ParaView state and PNG for the validated SIMPLE cases.

Run with ParaView 5.13 pvpython, for example (PowerShell)::

  & 'C:\Program Files\ParaView 5.13.0\bin\pvpython.exe' `
      --force-offscreen-rendering scripts/paraview_simple_view.py

Input files, produced by the CSV-to-VTK exporter, are
``<root>/adaptive16/solution.vtu`` and ``<root>/duct16/solution.vtu``.
This script does not interpolate the cell values or triangulate the slices:
the displayed edges are cross sections of the actual octree leaf cells.

The two periodic, body-force-driven cases store pressure perturbations.
Their Pressure field is not the full physical streamwise pressure drop.
The independent color scales are deliberate: a parallel-plate channel and a
four-wall square duct have different velocity profiles and peak speeds.
"""

import argparse
import json
from pathlib import Path

import paraview.simple as pvs


def text_label(view, title, location="Upper Left Corner", font_size=17):
    source = pvs.Text(registrationName=title, Text=title)
    display = pvs.Show(source, view, "TextSourceRepresentation")
    display.WindowLocation = location
    display.Color = [0.08, 0.08, 0.08]
    display.FontSize = font_size
    return source


def render_view():
    view = pvs.CreateView("RenderView")
    view.UseColorPaletteForBackground = 0
    view.Background = [1.0, 1.0, 1.0]
    view.CameraParallelProjection = 1
    view.OrientationAxesVisibility = 1
    return view


def add_case(case_directory, view, axis, title, view_aspect):
    reader = pvs.XMLUnstructuredGridReader(
        registrationName=f"{case_directory.name}: full octree volume (hidden)",
        FileName=[str(case_directory / "solution.vtu")],
    )
    reader.UpdatePipeline()
    bounds = list(reader.GetDataInformation().GetBounds())
    center = [(bounds[2 * k] + bounds[2 * k + 1]) / 2 for k in range(3)]
    spans = [bounds[2 * k + 1] - bounds[2 * k] for k in range(3)]
    if min(spans) <= 0:
        raise ValueError(f"Invalid volume bounds for {case_directory}: {bounds}")

    # Keep the full volume ready to toggle on in the saved Pipeline Browser.
    volume_display = pvs.Show(reader, view, "UnstructuredGridRepresentation")
    volume_display.Representation = "Surface With Edges"
    pvs.ColorBy(volume_display, ("CELLS", "Speed"), separate=True)
    volume_display.SetScalarBarVisibility(view, False)
    pvs.Hide(reader, view)

    plane_center = list(center)
    # Avoid cutting exactly along a shared cell face at the domain midpoint.
    plane_center[axis] += spans[axis] * 1e-8
    normal = [0.0, 0.0, 0.0]
    normal[axis] = 1.0
    section = pvs.Slice(registrationName=f"{case_directory.name}: interior slice", Input=reader)
    section.SliceType = "Plane"
    section.SliceType.Origin = plane_center
    section.SliceType.Normal = normal
    section.Triangulatetheslice = 0
    section.UpdatePipeline()
    if section.GetDataInformation().GetNumberOfCells() == 0:
        raise RuntimeError(f"Slice is empty for {case_directory}")

    display = pvs.Show(section, view, "GeometryRepresentation")
    display.Representation = "Surface With Edges"
    display.EdgeColor = [0.20, 0.22, 0.25]
    display.LineWidth = 0.5
    display.Ambient = 1.0
    display.Diffuse = 0.0
    pvs.ColorBy(display, ("CELLS", "Speed"), separate=True)
    lut = pvs.GetColorTransferFunction("Speed", representation=display, separate=True)
    lut.ApplyPreset("Viridis (matplotlib)", True)
    speed_range = list(section.CellData["Speed"].GetRange())
    # Include zero, even though cell centers do not lie directly on the walls.
    lut.RescaleTransferFunction(0.0, speed_range[1])
    lut.AutomaticRescaleRangeMode = "Never"
    display.SetScalarBarVisibility(view, True)
    bar = pvs.GetScalarBar(lut, view)
    bar.Title = "Speed (m/s)"
    bar.ComponentTitle = ""
    bar.TitleColor = [0.08, 0.08, 0.08]
    bar.LabelColor = [0.08, 0.08, 0.08]
    bar.TitleFontSize = 15
    bar.LabelFontSize = 13
    bar.Orientation = "Horizontal"
    bar.WindowLocation = "Any Location"
    bar.Position = [0.32, 0.08]
    bar.ScalarBarLength = 0.36
    bar.RangeLabelFormat = "%.4g"
    bar.LabelFormat = "%.3g"

    camera_position = list(center)
    camera_position[axis] += 2.0 * max(spans)
    view.CameraPosition = camera_position
    view.CameraFocalPoint = plane_center
    view.CameraViewUp = [0.0, 1.0, 0.0] if axis == 2 else [0.0, 0.0, 1.0]
    # Reserve room above the slice for its title and below for the scale.
    horizontal_span, vertical_span = (spans[0], spans[1]) if axis == 2 else (spans[1], spans[2])
    view.CameraParallelScale = max(vertical_span / 0.62, horizontal_span / (view_aspect * 0.88)) / 2
    text_label(view, title)
    text_label(
        view,
        "Cell-centered values | actual leaf-cell edges | Pressure = periodic perturbation",
        "Lower Left Corner",
        11,
    )

    faces = case_directory / "faces.vtp"
    if faces.is_file():
        faces_reader = pvs.XMLPolyDataReader(
            registrationName=f"{case_directory.name}: conservative faces (hidden)",
            FileName=[str(faces)],
        )
        # Register a hidden representation without obscuring the interior slice.
        pvs.Show(faces_reader, view, "GeometryRepresentation")
        pvs.Hide(faces_reader, view)

    return section, {
        "source": str(case_directory / "solution.vtu"),
        "volume_cells": reader.GetDataInformation().GetNumberOfCells(),
        "slice_cells": section.GetDataInformation().GetNumberOfCells(),
        "slice_origin": plane_center,
        "slice_normal": normal,
        "speed_range_on_slice": speed_range,
        "color_range": [0.0, speed_range[1]],
        "pressure_interpretation": "Periodic perturbation pressure; not total streamwise pressure drop.",
    }


def main():
    default_root = Path(__file__).resolve().parents[1] / "output" / "simple_validation_full" / "paraview"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=1000)
    args = parser.parse_args()
    if args.width < 600 or args.height < 600:
        parser.error("Image width and height must each be at least 600 pixels.")
    root = args.root.resolve()
    for case in ("adaptive16", "duct16"):
        if not (root / case / "solution.vtu").is_file():
            parser.error(f"Missing input: {root / case / 'solution.vtu'}; run the VTK exporter first.")

    pvs._DisableFirstRenderCameraReset()
    layout = pvs.CreateLayout(name="SIMPLE: native octree channel and square duct")
    upper = render_view()
    lower = render_view()
    split = 0.42
    pvs.AssignViewToLayout(view=upper, layout=layout, hint=0)
    layout.SplitVertical(0, split)
    pvs.AssignViewToLayout(view=lower, layout=layout, hint=2)
    layout.SetSize(args.width, args.height)

    upper_slice, adaptive_info = add_case(
        root / "adaptive16", upper, 2,
        "Adaptive narrow channel | longitudinal mid-plane | Speed",
        args.width / (args.height * split),
    )
    _, duct_info = add_case(
        root / "duct16", lower, 0,
        "Four-wall square duct | transverse mid-section | Speed",
        args.width / (args.height * (1 - split)),
    )
    pvs.SetActiveView(upper)
    pvs.SetActiveSource(upper_slice)
    pvs.RenderAllViews()
    png = root / "overview.png"
    state = root / "overview.pvsm"
    pvs.SaveScreenshot(str(png), layout, ImageResolution=[args.width, args.height])
    pvs.SaveState(str(state))
    metadata = {
        "paraview_version": str(pvs.GetParaViewVersion()),
        "state": str(state),
        "preview": str(png),
        "display": "Uninterpolated cell data; untriangulated slices; separate speed scales.",
        "adaptive16": adaptive_info,
        "duct16": duct_info,
    }
    (root / "overview_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
