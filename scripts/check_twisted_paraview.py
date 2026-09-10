"""Run with pvpython: read every PVD time and compare the exported fields to CSV."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from paraview.simple import PVDReader, Delete
from paraview import servermanager
from vtkmodules.util.numpy_support import vtk_to_numpy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();root=args.run.resolve()
    output=root/'paraview_time_readback.json'
    if output.exists():raise ValueError('Preserve previous readback report')
    with (root/'time_history.csv').open() as stream:history=list(csv.DictReader(stream))
    if not len(history) or any(r['inner_converged']!='true' for r in history):
        raise ValueError('Incomplete physical steps')
    summary=json.loads((root/'transient_summary.json').read_text())
    if not summary['converged'] or summary['steps_completed']!=len(history):
        raise ValueError('Incomplete configured run')
    stride=summary.get('output_stride',1)
    if stride<1:raise ValueError('Invalid output stride')
    selected=[r for r in history if int(r['step'])==1 or int(r['step'])==len(history) or int(r['step'])%stride==0]
    field_steps=[int(r['step']) for r in selected]
    if summary.get('field_output_steps',field_steps)!=field_steps:
        raise ValueError('Declared output schedule differs')
    expected=np.array([float(r['time']) for r in selected])
    hashes={}
    def record(path):hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    record(root/'time_history.csv')
    record(root/'transient_summary.json')
    for r in history:
        path=root/f'step_{int(r["step"]):04d}'/'metrics.json';record(path)
        metrics=json.loads(path.read_text())
        if not metrics['converged'] or metrics.get('field_output_written',True)!=(int(r['step']) in field_steps):
            raise ValueError('Per-step fields do not match the declared schedule')
    reports=[]
    for name,ext,columns in [('solution','vtu',{'Velocity':['u','v','w'],'Pressure':['p']}),
                             ('walls','vtp',{'WallShear':['tau_x','tau_y','tau_z']})]:
        collection=root/(name+'.pvd');record(collection)
        reader=PVDReader(FileName=str(collection))
        times=np.array(reader.TimestepValues)
        if times.shape!=expected.shape or not np.allclose(times,expected,rtol=1e-13,atol=0):
            raise ValueError('PVD physical times differ from solver history')
        for item,t in zip(selected,expected):
            reader.UpdatePipeline(time=float(t));data=servermanager.Fetch(reader)
            while data.IsA('vtkMultiBlockDataSet'):
                blocks=[data.GetBlock(i) for i in range(data.GetNumberOfBlocks()) if data.GetBlock(i) is not None]
                if len(blocks)!=1:raise ValueError('Expected a single exported dataset')
                data=blocks[0]
            folder=root/f'step_{int(item["step"]):04d}'
            record(folder/(name+'.'+ext));reference=folder/(name+'.csv');record(reference)
            rows=np.genfromtxt(reference,names=True,delimiter=',')
            arrays=data.GetCellData();differences={}
            for i in range(arrays.GetNumberOfArrays()):
                if not np.isfinite(vtk_to_numpy(arrays.GetArray(i))).all():raise ValueError('Nonfinite exported array')
            for label,names in columns.items():
                actual=vtk_to_numpy(arrays.GetArray(label))
                values=np.column_stack([rows[k] for k in names]) if len(names)>1 else rows[names[0]]
                if actual.shape!=values.shape or not np.array_equal(actual,values):
                    raise ValueError('ParaView values differ from CSV: '+label)
                differences[label]=float(np.max(abs(actual-values)))
            reports.append({'collection':name,'step':int(item['step']),'time':float(t),
                            'cells':data.GetNumberOfCells(),'csv_absolute_max_difference':differences})
        Delete(reader)
    result={'passed':True,'scope':__doc__,'physical_steps':len(history),'time_steps':len(selected),
            'field_output_steps':field_steps,'reads':reports,'source_sha256':hashes}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'physical_steps':len(history),'time_steps':len(selected),'dataset_reads':len(reports)}))


if __name__=='__main__':main()
