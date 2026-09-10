"""Create a ParaView state and a two-panel scientific view of accepted absolute field differences."""
import argparse
import json
from pathlib import Path
import numpy as np
from paraview.simple import (XMLUnstructuredGridReader,XMLPolyDataReader,CreateLayout,CreateView,
    AssignViewToLayout,Show,ColorBy,GetColorTransferFunction,GetOpacityTransferFunction,
    GetScalarBar,Text,SaveScreenshot,SaveState)
from run_twisted_solver import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--visualization',type=Path,required=True)
    p.add_argument('--output',type=Path,help='New directory for a revised render, preserving earlier attempts')
    a=p.parse_args();root=a.visualization.resolve();out=root if a.output is None else a.output.resolve()
    if out.drive.lower()!='d:':raise ValueError('Keep rendered artifacts on D')
    for name in ('comparison.png','comparison.pvsm','comparison_view.json'):
        if (out/name).exists():raise ValueError('Preserve previous rendered comparison')
    proof=json.loads((root/'paraview_difference_readback.json').read_text())
    if not proof['passed']:raise ValueError('Actual ParaView readback must pass before rendering')
    metadata=json.loads((root/'difference_export.json').read_text())
    case_path=Path(metadata['source_step'])/'case.json'
    case=json.loads(case_path.read_text())
    if metadata['source_sha256'].get(str(case_path.resolve()))!=sha(case_path):
        raise ValueError('Native grid label is not covered by the verified export')
    ny=case['ny']
    if metadata.get('native_finest_ny',ny)!=ny:
        raise ValueError('Export resolution differs from its source configuration')
    reference_kind='complete reference' if metadata['reference_configured_trajectory_complete'] else 'step snapshot'
    reference_label=f"Aphros t={metadata['reference_time']:.3g} s; {reference_kind}"
    fields=np.load(root/'expected_fields.npz')
    layout=CreateLayout(name='Cirrus and Aphros steady differences')
    layout.SplitHorizontal(0,.5)
    sources=[]
    for index,reader_type,filename,field,prefix,title,unit in (
        (0,XMLUnstructuredGridReader,'solution.vtu','VelocityDifferenceMagnitude','cells',
         'Velocity difference magnitude','m/s'),
        (1,XMLPolyDataReader,'walls.vtp','WallShearDifferenceMagnitude','walls',
         'Wall shear difference magnitude','Pa')):
        view=CreateView('RenderView');AssignViewToLayout(view,layout,index+1)
        view.UseColorPaletteForBackground=0;view.BackgroundColorMode='Single Color'
        view.Background=[1,1,1];view.OrientationAxesVisibility=0
        reader=reader_type(FileName=[str(root/filename)]);reader.UpdatePipeline();sources.append(reader)
        display=Show(reader,view);display.Representation='Surface';ColorBy(display,('CELLS',field))
        display.Ambient=1;display.Diffuse=0
        lut=GetColorTransferFunction(field);lut.ApplyPreset('Viridis (matplotlib)',True)
        maximum=float(np.max(fields[prefix+'__'+field]));assert maximum>0
        lut.RescaleTransferFunction(0,maximum);GetOpacityTransferFunction(field).RescaleTransferFunction(0,maximum)
        display.SetScalarBarVisibility(view,True);bar=GetScalarBar(lut,view)
        bar.Title=unit;bar.ComponentTitle='';bar.TitleColor=[.1,.1,.1];bar.LabelColor=[.1,.1,.1]
        bar.TitleFontSize=16;bar.LabelFontSize=13;bar.RangeLabelFormat='%.2e';bar.LabelFormat='%.2e'
        bar.WindowLocation='Any Location';bar.Position=[.18,.07]
        bar.Orientation='Horizontal';bar.ScalarBarLength=.64
        bounds=reader.GetDataInformation().GetBounds();center=np.array([(bounds[2*d]+bounds[2*d+1])/2 for d in range(3)])
        view.CameraFocalPoint=center.tolist();view.CameraPosition=(center+np.array([.35,-.65,.45])).tolist()
        view.CameraViewUp=[0,0,1];view.CameraParallelProjection=1;view.ResetCamera()
        label=Text();label.Text=title+f'\nCirrus - Aphros; steady {ny} grid\n'+reference_label
        annotation=Show(label,view);annotation.WindowLocation='Upper Center';annotation.Color=[.1,.1,.1];annotation.FontSize=16
    layout.SetSize(1800,760)
    out.mkdir(parents=True,exist_ok=True)
    SaveScreenshot(str(out/'comparison.png'),layout,ImageResolution=[1800,760])
    SaveState(str(out/'comparison.pvsm'))
    report={'passed':True,'scope':__doc__,'range_policy':'Zero to actual maximum absolute difference; no pointwise relative normalization',
            'native_finest_ny':ny,'reference_label':reference_label,
            'source_sha256':{str(p.resolve()):sha(p) for p in (Path(__file__),root/'difference_export.json',root/'paraview_difference_readback.json',
                             root/'solution.vtu',root/'walls.vtp',root/'expected_fields.npz',case_path)},
            'output_sha256':{name:sha(out/name) for name in ('comparison.png','comparison.pvsm')}}
    (out/'comparison_view.json').write_text(json.dumps(report,indent=2)+'\n');print(out/'comparison.png')


if __name__=='__main__':main()
