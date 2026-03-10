from paraview.simple import *

# =========================
# Hardcoded file paths
# =========================
vti_path = r"C:\Code\Cirrus-dev\output\laplacian_test_sphere_empty_levels_3_5.vti"
sphere_path = r"C:\Code\Cirrus-dev\output\sphere.vtp"
output_path = r"C:\Code\Cirrus-dev\output\render.png"

# =========================
# Read data
# =========================
vti = XMLImageDataReader(FileName=[vti_path])
sphere = XMLPolyDataReader(FileName=[sphere_path])

vti.UpdatePipeline()
sphere.UpdatePipeline()

print("=== DEBUG INFO ===")
print("vti bounds   :", vti.GetDataInformation().GetBounds())
print("sphere bounds:", sphere.GetDataInformation().GetBounds())

# =========================
# Slice
# =========================
slice1 = Slice(Input=vti)
slice1.SliceType = 'Plane'
slice1.SliceType.Origin = [0.5, 0.5, 0.5]
slice1.SliceType.Normal = [1.0, 0.0, 0.0]
slice1.UpdatePipeline()

print("slice bounds :", slice1.GetDataInformation().GetBounds())

# =========================
# Render view
# =========================
view = CreateView('RenderView')
view.ViewSize = [1024, 1024]
view.OrientationAxesVisibility = 0
view.UseColorPaletteForBackground = 0
view.Background = [1.0, 1.0, 1.0]
view.CameraParallelProjection = 1

# =========================
# Show slice
# =========================
slice_display = Show(slice1, view)

# level is on POINTS
ColorBy(slice_display, ('POINTS', 'level'))
slice_display.SetScalarBarVisibility(view, False)
slice_display.RescaleTransferFunctionToDataRange(True, False)

# =========================
# Show sphere
# =========================
sphere_display = Show(sphere, view)
sphere_display.ColorArrayName = [None, '']
sphere_display.DiffuseColor = [0.0, 1.0, 0.0]
sphere_display.AmbientColor = [0.0, 1.0, 0.0]
sphere_display.Opacity = 0.75

# =========================
# First render (this lets ParaView do its auto camera reset)
# =========================
Render(view)

# =========================
# Now override camera
# =========================
bounds = slice1.GetDataInformation().GetBounds()
xmin, xmax, ymin, ymax, zmin, zmax = bounds

yc = 0.5 * (ymin + ymax)
zc = 0.5 * (zmin + zmax)

half_y = 0.5 * (ymax - ymin)
half_z = 0.5 * (zmax - zmin)

parallel_scale = max(half_y, half_z)

print("camera debug:")
print("  slice bounds =", bounds)
print("  parallel_scale =", parallel_scale)

view.CameraPosition = [xmin - 1.0, yc, zc]
view.CameraFocalPoint = [xmax, yc, zc]
view.CameraViewUp = [0.0, 0.0, 1.0]
view.CameraParallelScale = parallel_scale

# =========================
# Final render
# =========================
Render(view)

SaveScreenshot(output_path, view, ImageResolution=[1024, 1024])

print("saved screenshot to:", output_path)
print("=== END ===")