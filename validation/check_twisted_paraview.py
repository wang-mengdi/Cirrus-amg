"""Use pvpython to read every PVD time and compare actual arrays with solver CSV."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from paraview.simple import PVDReader, Delete
from paraview import servermanager
from vtkmodules.util.numpy_support import vtk_to_numpy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    case = json.loads((root/'case.json').read_text(encoding='utf-8-sig'))
    summary = json.loads((root/'transient_summary.json').read_text())
    if not summary['converged'] or summary['steps_completed'] != case['time_steps']:
        raise ValueError('Incomplete time sequence')
    expected_times = np.arange(1, case['time_steps']+1)*case['time_step']
    checks = {}
    for name, field, filename, columns, count_key in (
            ('solution', 'Velocity', 'solution.csv', ['u', 'v', 'w'], 'cells'),
            ('walls', 'WallShear', 'walls.csv', ['tau_x', 'tau_y', 'tau_z'], 'cut_cells')):
        reader = PVDReader(FileName=str(root/(name+'.pvd')))
        times = list(reader.TimestepValues)
        if len(times) != len(expected_times) or not np.allclose(times, expected_times, rtol=1e-13, atol=0):
            raise ValueError('Unexpected PVD times')
        rows = []
        for step, time in enumerate(times, 1):
            reader.UpdatePipeline(time=time)
            dataset = servermanager.Fetch(reader)
            if dataset.IsA('vtkCompositeDataSet'):
                iterator = dataset.NewIterator(); iterator.InitTraversal()
                leaves = []
                while not iterator.IsDoneWithTraversal():
                    leaf = iterator.GetCurrentDataObject()
                    if leaf is not None and leaf.GetNumberOfCells():
                        leaves.append(leaf)
                    iterator.GoToNextItem()
            else:
                leaves = [dataset]
            if len(leaves) != 1:
                raise ValueError('Expected one exported domain per time')
            data = leaves[0]
            folder = root/f'step_{step:04d}'
            metrics = json.loads((folder/'metrics.json').read_text())
            if data.GetNumberOfCells() != metrics[count_key]:
                raise ValueError('Cell count differs from solver')
            arrays = data.GetCellData()
            for index in range(arrays.GetNumberOfArrays()):
                if not np.isfinite(vtk_to_numpy(arrays.GetArray(index))).all():
                    raise ValueError('Nonfinite values in an exported array')
            values = vtk_to_numpy(arrays.GetArray(field))
            source = np.genfromtxt(folder/filename, delimiter=',', names=True, ndmin=1)
            reference = np.column_stack([source[key] for key in columns])
            if values.shape != reference.shape or not np.allclose(values, reference, rtol=1e-13, atol=1e-15):
                raise ValueError('PVD values differ from solver CSV')
            magnitude = np.linalg.norm(values, axis=1)
            rows.append({'time': time, 'cells': len(values),
                         'magnitude_range': [float(magnitude.min()), float(magnitude.max())],
                         'csv_absolute_max_difference': float(np.max(np.abs(values-reference)))})
        Delete(reader)
        checks[name] = rows
    result = {'passed': True, 'reader': 'ParaView PVDReader + fetched VTK arrays',
              'scope': 'All times, all numeric cell arrays finite; velocity and wall shear compared with solver CSV',
              'collections': checks,
              'pvd_sha256': {name: hashlib.sha256((root/(name+'.pvd')).read_bytes()).hexdigest()
                             for name in checks}}
    (root/'paraview_time_readback.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'passed': True, 'times': len(expected_times), 'run': str(root)}))


if __name__ == '__main__':
    main()
