"""Export a verified steady Aphros comparison on the actual native cut-cell geometry.

The reference is sampled exactly as in the accepted comparison: identity at
fine/cut centers, tensor cubic only at coarse centers. Wall locations coincide.
Difference fields are absolute differences, not unstable pointwise percentages.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from run_twisted_solver import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ours',type=Path,required=True)
    p.add_argument('--aphros',type=Path,required=True)
    p.add_argument('--comparison',type=Path,required=True)
    p.add_argument('--native-visualization',type=Path,
                   help='Verified native geometry directory when exported separately from the solved fields')
    p.add_argument('--completed-reference',action='store_true',
                   help='Require a verified complete reference trajectory instead of a completed-step snapshot')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    ours,reference,out=a.ours.resolve(),a.aphros.resolve(),a.output.resolve()
    visual=ours if a.native_visualization is None else a.native_visualization.resolve()
    if out.drive.lower()!='d:':raise ValueError('Keep visualization artifacts on D')
    if out.exists():raise ValueError('Preserve earlier visualization artifacts')
    pair=json.loads(a.comparison.read_text())
    if not pair['passed'] or not all(pair['checks'].values()) or pair['flow_regime']!='steady Navier-Stokes':
        raise ValueError('Require an actually accepted steady field comparison')
    if a.completed_reference:
        if not pair['complete_reference_run_checked'] or pair.get('reference_checkpoint') is not None:
            raise ValueError('Completed-reference export requires a verified full reference trajectory')
        config=json.loads((reference/'case_manifest.json').read_text())
        provenance=pair.get('extended_reference_provenance',{})
        if not provenance.get('passed') or provenance.get('complete_physical_steps')!=config['time_steps']:
            raise ValueError('Completed reference provenance does not cover its configured duration')
        reference_records=('run_completion.json','tube_b0_time.csv','a.conf','case_manifest.json')
    else:
        if pair['complete_reference_run_checked'] or pair.get('source_reference_run_complete_proven'):
            raise ValueError('Full reference trajectories require --completed-reference')
        reference_records=('extended_step_checkpoint.json',)
    hashes=dict(pair['source_sha256'])
    for root,names in ((ours,('solution.csv','walls.csv','case.json')),
                       (reference,('proj_final_b0_cells.csv','tube_final_b0_walls.csv',*reference_records))):
        for name in names:
            path=str(root/name)
            if path not in hashes or sha(Path(path))!=hashes[path]:raise ValueError('Comparison does not cover requested field: '+path)
    for path,digest in hashes.items():
        if sha(Path(path))!=digest:raise ValueError('Compared input changed: '+path)
    for path in (a.comparison,Path(__file__),Path(__file__).with_name('export_twisted_paraview.py'),
                 visual/'solution.vtu',visual/'walls.vtp',visual/'paraview_export.json'):
        hashes[str(path.resolve())]=sha(path)
    native_geometry=json.loads((visual/'paraview_export.json').read_text())
    if not native_geometry['geometry_verified']:
        raise ValueError('Original visualization geometry has not passed its volume checks')
    geometry_source=native_geometry.get('source_step')
    if geometry_source is None and visual!=ours:
        raise ValueError('Separate native visualization must identify its solved field')
    if geometry_source is not None and Path(geometry_source).resolve()!=ours:
        raise ValueError('Native visualization belongs to a different solved field')
    native_case=json.loads((ours/'case.json').read_text())

    if a.completed_reference:
        from compare_twisted import read,ordered,vector,transfer,error
    else:
        from compare_twisted_aphros_step import read,ordered,vector,transfer,error
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    from export_twisted_paraview import add,write

    cells=ordered(read(ours/'solution.csv'));walls=ordered(read(ours/'walls.csv'))
    config=json.loads((reference/'case_manifest.json').read_text())
    shift=np.array(pair['reference_translation']);period=config['spec']['extent'][0]

    def translated(data):
        for d,key in enumerate(('x','y','z')):data[key]+=shift[d]
        data['x']%=period
        return ordered(data)

    sampled,transfer_report=transfer(cells,translated(read(reference/'proj_final_b0_cells.csv')),
                                    config['shape'],config['spec']['extent'][1]/config['ny'])
    wall_reference=translated(read(reference/'tube_final_b0_walls.csv'))
    if not np.allclose(vector(walls,'xyz'),vector(wall_reference,'xyz'),rtol=0,atol=1e-13):
        raise ValueError('Wall sampling locations differ')
    u=vector(cells,('u','v','w'));pressure=cells['p']-np.average(cells['p'],weights=cells['volume'])
    reference_pressure=sampled[:,3]-np.average(sampled[:,3],weights=cells['volume'])
    shear=vector(walls,('tau_x','tau_y','tau_z'));reference_shear=vector(wall_reference,('tau_x','tau_y','tau_z'))
    cut=np.isin(cells['id'],walls['owner'])
    measured={'velocity':error(u,sampled[:,:3],cells['volume']),
              'pressure':error(pressure,reference_pressure,cells['volume']),
              'cut_cell_velocity':error(u[cut],sampled[cut,:3],cells['volume'][cut]),
              'wall_shear':error(shear,reference_shear,walls['area'])}
    for key,value in measured.items():
        if not np.isclose(value['relative_l2'],pair[key]['relative_l2'],rtol=1e-12,atol=1e-16):
            raise ValueError('Export differs from accepted field comparison: '+key)
    index=np.argsort(cells['id']);wall_index=np.argsort(walls['face_id'])
    if not np.array_equal(cells['id'][index],np.arange(len(cells))):raise ValueError('Native IDs are not dense')
    cell_fields={'AphrosVelocity':sampled[:,:3], 'VelocityDifference':u-sampled[:,:3],
                 'VelocityDifferenceMagnitude':np.linalg.norm(u-sampled[:,:3],axis=1),
                 'CirrusPressureMeanZero':pressure,'AphrosPressureMeanZero':reference_pressure,
                 'PressureDifference':pressure-reference_pressure,
                 'ReferenceInterpolated':(cells['h']>min(cells['h'])).astype(np.uint8)}
    wall_fields={'AphrosWallShear':reference_shear,'WallShearDifference':shear-reference_shear,
                 'WallShearDifferenceMagnitude':np.linalg.norm(shear-reference_shear,axis=1)}
    reader=vtk.vtkXMLUnstructuredGridReader();reader.SetFileName(str(visual/'solution.vtu'));reader.Update()
    volume=reader.GetOutput();wall_reader=vtk.vtkXMLPolyDataReader();wall_reader.SetFileName(str(visual/'walls.vtp'));wall_reader.Update()
    surface=wall_reader.GetOutput()
    expected={}
    for data,fields,order,count,primary,prefix in (
            (volume,cell_fields,index,len(cells),{'Velocity':u,'Pressure':cells['p'],'FluidVolume':cells['volume']},'cells'),
            (surface,wall_fields,wall_index,len(walls),{'WallShear':shear},'walls')):
        if data.GetNumberOfCells()!=count:raise ValueError('Native geometry has different cell count')
        for name,values in primary.items():
            if not np.array_equal(vtk_to_numpy(data.GetCellData().GetArray(name)),values[order]):
                raise ValueError('Original VTK differs from native CSV: '+name)
            expected[prefix+'__'+name]=values[order]
        for name,values in fields.items():
            add(data.GetCellData(),name,values[order]);expected[prefix+'__'+name]=values[order]
    expected['cells__CutMask']=cut[index]
    expected['walls__Area']=walls['area'][wall_index]
    out.mkdir(parents=True,exist_ok=False)
    write(out/'solution.vtu',volume,vtk.vtkXMLUnstructuredGridWriter)
    write(out/'walls.vtp',surface,vtk.vtkXMLPolyDataWriter)
    np.savez_compressed(out/'expected_fields.npz',**expected)
    if any(sha(Path(path))!=digest for path,digest in hashes.items()):raise ValueError('Visualization sources changed during export')
    report={'passed':True,'scope':__doc__,'source_step':str(ours),
            'native_visualization':str(visual),'native_finest_ny':native_case['ny'],
            'native_iteration_mode':pair.get('native_iteration_mode','physical_time_step'),
            'aphros_snapshot':None if a.completed_reference else str(reference),'aphros_reference':str(reference),
            'comparison':str(a.comparison.resolve()),'geometry_unchanged':True,'cells':len(cells),'walls':len(walls),
            'reference_interpolation':transfer_report,'field_errors':measured,'reference_time':pair['aphros_temporal_diagnostic']['time'],
            'reference_configured_trajectory_complete':a.completed_reference,'spatial_convergence_claimed':False,
            'source_sha256':hashes,'output_sha256':{name:sha(out/name) for name in ('solution.vtu','walls.vtp','expected_fields.npz')}}
    (out/'difference_export.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':True,'output':str(out),'cells':len(cells),'walls':len(walls)}))


if __name__=='__main__':main()
