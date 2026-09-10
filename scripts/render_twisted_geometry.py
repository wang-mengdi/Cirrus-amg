"""Render actual exported cut cells and wall polygons for implementation review."""
import argparse
from pathlib import Path
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    r=vtk.vtkXMLUnstructuredGridReader();r.SetFileName(str(args.run/'solution.vtu'));r.Update()
    wall=vtk.vtkXMLPolyDataReader();wall.SetFileName(str(args.run/'walls.vtp'));wall.Update()
    plane=vtk.vtkPlane();plane.SetOrigin(0,0,.0625);plane.SetNormal(0,0,1)
    clip=vtk.vtkClipDataSet();clip.SetInputConnection(r.GetOutputPort());clip.SetClipFunction(plane);clip.InsideOutOn()
    centers=vtk_to_numpy(r.GetOutput().GetCellData().GetArray('UnknownLocation'))
    iscut=vtk_to_numpy(r.GetOutput().GetCellData().GetArray('CutCell'))
    ids=vtk.vtkIdList()
    for i,(x,y,z) in enumerate(centers):
        if iscut[i] and .075<x<.14 and z<.0625:ids.InsertNextId(i)
    patch=vtk.vtkExtractCells();patch.SetInputConnection(r.GetOutputPort());patch.SetCellList(ids)
    window=vtk.vtkRenderWindow();window.SetOffScreenRendering(1);window.SetSize(1600,800);window.SetMultiSamples(4)
    for side in range(2):
        renderer=vtk.vtkRenderer();renderer.SetViewport(side*.5,0,(side+1)*.5,1)
        renderer.SetBackground(.96,.97,.99);window.AddRenderer(renderer)
        mapper=vtk.vtkDataSetMapper();mapper.SetInputConnection(clip.GetOutputPort() if side==0 else patch.GetOutputPort())
        mapper.SetScalarModeToUseCellFieldData();mapper.SelectColorArray('Speed' if side==0 else 'FluidFraction')
        mapper.SetScalarRange(r.GetOutput().GetCellData().GetArray('Speed').GetRange() if side==0 else (0,1))
        lut=vtk.vtkLookupTable();lut.SetHueRange(.64,0);lut.Build();mapper.SetLookupTable(lut)
        actor=vtk.vtkActor();actor.SetMapper(mapper);actor.GetProperty().SetEdgeVisibility(side==1)
        actor.GetProperty().SetEdgeColor(.12,.15,.20);actor.GetProperty().SetLineWidth(.5)
        renderer.AddActor(actor)
        if side==0:
            wm=vtk.vtkPolyDataMapper();wm.SetInputConnection(wall.GetOutputPort());wm.ScalarVisibilityOff()
            wa=vtk.vtkActor();wa.SetMapper(wm);wa.GetProperty().SetColor(.5,.55,.6);wa.GetProperty().SetOpacity(.12)
            renderer.AddActor(wa)
        title=vtk.vtkTextActor();title.SetInput('3D twisted tube | speed (m/s)' if side==0 else 'Wall cut-cell patch | fluid fraction')
        title.SetPosition(24,748);title.GetTextProperty().SetFontSize(22);title.GetTextProperty().SetColor(.12,.15,.2)
        renderer.AddActor2D(title)
        scalar=vtk.vtkScalarBarActor();scalar.SetLookupTable(lut);scalar.SetOrientationToHorizontal()
        scalar.SetPosition(.15,.07);scalar.SetWidth(.7);scalar.SetHeight(.09);scalar.SetNumberOfLabels(4)
        scalar.SetLabelFormat('%.2g');scalar.SetUnconstrainedFontSize(True)
        scalar.GetLabelTextProperty().SetColor(.12,.15,.2);scalar.GetLabelTextProperty().SetFontSize(16)
        scalar.GetLabelTextProperty().BoldOff();scalar.GetLabelTextProperty().ItalicOff();scalar.GetLabelTextProperty().ShadowOff()
        renderer.AddActor2D(scalar)
        camera=renderer.GetActiveCamera();camera.SetPosition(.4,-.32,.40);camera.SetFocalPoint(.125,.0625,.0625)
        camera.SetViewUp(0,0,1);renderer.ResetCamera();camera.ParallelProjectionOn()
        if side==1:
            # Fit the actual patch, including translated reference geometries,
            # with room for the color bar below the mesh.
            camera.SetParallelScale(camera.GetParallelScale()*1.2)
        renderer.ResetCameraClippingRange()
    window.Render()
    capture=vtk.vtkWindowToImageFilter();capture.SetInput(window);capture.SetInputBufferTypeToRGB();capture.ReadFrontBufferOff();capture.Update()
    writer=vtk.vtkPNGWriter();writer.SetFileName(str(args.run/'cut_geometry.png'));writer.SetInputConnection(capture.GetOutputPort());writer.Write()
    print(args.run/'cut_geometry.png')


if __name__=='__main__':main()
