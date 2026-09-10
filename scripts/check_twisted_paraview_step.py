"""Read one completed cut-cell step through ParaView and compare all primary fields."""
import argparse
import json
from pathlib import Path
import numpy as np
from paraview.simple import XMLUnstructuredGridReader,XMLPolyDataReader,Delete
from paraview import servermanager
from vtkmodules.util.numpy_support import vtk_to_numpy
from run_twisted_solver import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step',type=Path,required=True)
    parser.add_argument('--visualization',type=Path,help='Directory containing exported files when separate from the solved step')
    args=parser.parse_args();root=args.step.resolve()
    visual=root if args.visualization is None else args.visualization.resolve()
    output=visual/'paraview_step_readback.json'
    if output.exists():raise ValueError('Preserve earlier readback evidence')
    if not json.loads((root/'metrics.json').read_text())['converged']:raise ValueError('Unconverged step')
    checks=[];hashes={}
    for name,ext,reader_type,fields in [('solution','vtu',XMLUnstructuredGridReader,{'Velocity':['u','v','w'],'Pressure':['p']}),
                                      ('walls','vtp',XMLPolyDataReader,{'WallShear':['tau_x','tau_y','tau_z']})]:
        path=visual/(name+'.'+ext);source=root/(name+'.csv');reader=reader_type(FileName=[str(path)])
        reader.UpdatePipeline();data=servermanager.Fetch(reader);arrays=data.GetCellData()
        rows=np.genfromtxt(source,names=True,delimiter=',',ndmin=1)
        if data.GetNumberOfCells()!=len(rows):raise ValueError('Different cell count')
        for i in range(arrays.GetNumberOfArrays()):
            if not np.isfinite(vtk_to_numpy(arrays.GetArray(i))).all():raise ValueError('Nonfinite field')
        differences={}
        for label,columns in fields.items():
            actual=vtk_to_numpy(arrays.GetArray(label))
            expected=np.column_stack([rows[k] for k in columns]) if len(columns)>1 else rows[columns[0]]
            if not np.array_equal(actual,expected):raise ValueError('ParaView field differs from solver CSV: '+label)
            differences[label]=float(np.max(abs(actual-expected)))
        checks.append({'file':str(path),'cells':len(rows),'absolute_max_differences':differences})
        for p in (path,source):hashes[str(p)]=sha(p)
        Delete(reader)
    output.write_text(json.dumps({'passed':True,'scope':__doc__,'source_step':str(root),
                                 'visualization_directory':str(visual),'checks':checks,'source_sha256':hashes},indent=2)+'\n')
    print(output)


if __name__=='__main__':main()
