"""Export converged SIMPLE CSV results as double-precision native-cell VTK.

Requires numpy and vtk. Run from the repository with:
    python scripts/export_simple_paraview.py

Source CSV, solver outputs, and validation manifests are never modified.
No point interpolation or uniform-grid resampling is performed. Hanging nodes
at octree coarse/fine interfaces remain in the visualization mesh.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy


CELL_COLUMNS = 'id,level,x,y,z,h,volume,u,v,w,p,u_exact,p_exact'
FACE_COLUMNS = 'id,owner,neighbor,axis,sign,boundary,x,y,z,area,distance,flux,predicted_flux'


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_csv(path, expected):
    with path.open(encoding='utf-8') as stream:
        header = stream.readline().strip()
    if header != expected:
        raise ValueError(f'Unexpected columns in {path}: {header}')
    values = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError(f'Empty or nonfinite data in {path}')
    return {name: values[:, i] for i, name in enumerate(header.split(','))}


def geometry(corners):
    count, width, _ = corners.shape
    coordinates, connectivity = np.unique(corners.reshape(-1, 3), axis=0, return_inverse=True)
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(coordinates, deep=True))
    cells = vtk.vtkCellArray()
    cells.SetData(
        numpy_to_vtkIdTypeArray(np.arange(count + 1, dtype=np.int64) * width, deep=True),
        numpy_to_vtkIdTypeArray(connectivity.astype(np.int64), deep=True),
    )
    return points, cells


def add_arrays(dataset, arrays):
    for name, values in arrays.items():
        array = numpy_to_vtk(np.ascontiguousarray(values), deep=True)
        array.SetName(name)
        dataset.GetCellData().AddArray(array)


def add_metadata(dataset, metadata):
    array = vtk.vtkStringArray()
    array.SetName('SIMPLE_Metadata_JSON')
    array.InsertNextValue(json.dumps(metadata, sort_keys=True))
    dataset.GetFieldData().AddArray(array)


def write_verify(dataset, destination, arrays, metadata, volume=False):
    add_arrays(dataset, arrays)
    add_metadata(dataset, metadata)
    if volume:
        dataset.GetCellData().SetActiveScalars('Speed')
        dataset.GetCellData().SetActiveVectors('Velocity')
        writer, reader = vtk.vtkXMLUnstructuredGridWriter(), vtk.vtkXMLUnstructuredGridReader()
    else:
        dataset.GetCellData().SetActiveScalars('PositiveAxisVolumeFlux')
        writer, reader = vtk.vtkXMLPolyDataWriter(), vtk.vtkXMLPolyDataReader()
    writer.SetFileName(str(destination))
    writer.SetInputData(dataset)
    writer.SetDataModeToAppended()
    writer.SetCompressorTypeToZLib()
    if writer.Write() != 1:
        raise RuntimeError(f'Failed to write {destination}')
    reader.SetFileName(str(destination))
    reader.Update()
    restored = reader.GetOutput()
    if restored.GetNumberOfCells() != dataset.GetNumberOfCells():
        raise ValueError(f'Cell count changed in {destination}')
    if not np.array_equal(vtk_to_numpy(restored.GetPoints().GetData()),
                          vtk_to_numpy(dataset.GetPoints().GetData())):
        raise ValueError(f'Point coordinates changed in {destination}')
    source_connectivity = dataset.GetCells() if volume else dataset.GetPolys()
    restored_connectivity = restored.GetCells() if volume else restored.GetPolys()
    if not np.array_equal(vtk_to_numpy(source_connectivity.GetConnectivityArray()),
                          vtk_to_numpy(restored_connectivity.GetConnectivityArray())):
        raise ValueError(f'Connectivity changed in {destination}')
    for name, values in arrays.items():
        array = restored.GetCellData().GetArray(name)
        if array is None or not np.array_equal(vtk_to_numpy(array), values):
            raise ValueError(f'{name} changed during VTK round trip: {destination}')
    result = {'file': str(destination), 'sha256': sha256(destination),
              'cells': restored.GetNumberOfCells(), 'points': restored.GetNumberOfPoints(),
              'bounds': list(restored.GetBounds()), 'arrays': list(arrays),
              'round_trip_exact': True}
    if volume:
        quality = vtk.vtkMeshQuality()
        quality.SetInputData(restored)
        quality.SetHexQualityMeasureToVolume()
        quality.Update()
        volumes = vtk_to_numpy(quality.GetOutput().GetCellData().GetArray('Quality'))
        if not (volumes > 0).all() or not np.allclose(volumes, arrays['CellVolume'], rtol=1e-12, atol=0):
            raise ValueError(f'Invalid cell volume/orientation in {destination}')
        result['positive_volumes_match_source'] = True
        result['total_volume'] = float(volumes.sum())
    return result


def export_case(source, output):
    metrics_path, config_path = source / 'metrics.json', source / 'case.json'
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if not metrics.get('converged'):
        raise ValueError(f'Refusing to label unconverged output as final: {source}')
    cell_path = source / 'solution.csv'
    face_path = source / f"iter_{metrics['iterations']}" / 'faces.csv'
    files = [cell_path, face_path, metrics_path, config_path]
    hashes = {str(path): sha256(path) for path in files}
    cell = read_csv(cell_path, CELL_COLUMNS)
    face = read_csv(face_path, FACE_COLUMNS)
    nc, nf = len(cell['id']), len(face['id'])
    if nc != metrics['cells'] or nf != metrics['faces']:
        raise ValueError(f'Counts disagree with solver metrics: {source}')
    if not np.array_equal(cell['id'], np.arange(nc)) or not np.array_equal(face['id'], np.arange(nf)):
        raise ValueError(f'Expected contiguous source IDs: {source}')
    if not (cell['h'] > 0).all() or not (face['area'] > 0).all():
        raise ValueError(f'Nonpositive mesh size: {source}')
    destination = output / source.name
    destination.mkdir(parents=True, exist_ok=True)
    metadata = {
        'case': source.name, 'iteration': metrics['iterations'], 'rho': metrics['rho'],
        'nu': metrics['nu'], 'body_acceleration': config.get('force', [0, 0, 0]),
        'periodic_x': metrics['periodic_x'], 'periodic_z': metrics['periodic_z'],
        'pressure': metrics['pressure_units'], 'velocity_units': 'm/s',
        'length_units': 'm', 'volume_flux_units': 'm^3/s',
        'mesh': 'Original octree leaf cubes, including hanging nodes; no resampling.',
        'cell_velocity': 'Full cell-centered velocity; distinct from conservative face-normal flux.',
        'face_flux': 'VolumeFlux is signed outwards from Owner; PositiveAxisVolumeFlux = sign * VolumeFlux.',
        'face_normal_velocity': 'NormalVelocity contains only the face-normal component, not full velocity.',
        'boundary_types': {'0': 'internal or periodic', '1': 'wall', '2': 'pressure inlet', '3': 'pressure outlet'},
    }
    offsets = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float) / 2
    centers = np.column_stack([cell[k] for k in ('x', 'y', 'z')])
    points, connectivity = geometry(centers[:, None, :] + cell['h'][:, None, None] * offsets)
    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(points)
    grid.SetCells(vtk.VTK_HEXAHEDRON, connectivity)
    velocity = np.column_stack([cell[k] for k in ('u', 'v', 'w')])
    exact = np.zeros_like(velocity)
    exact[:, 0] = cell['u_exact']
    arrays = {
        'CellId': cell['id'].astype(np.int64), 'RefinementLevel': cell['level'].astype(np.int32),
        'CellSize': cell['h'], 'CellVolume': cell['volume'],
        'Velocity': velocity, 'Speed': np.linalg.norm(velocity, axis=1), 'Pressure': cell['p'],
        'VelocityExact': exact, 'VelocityErrorMagnitude': np.linalg.norm(velocity - exact, axis=1),
        'PressureExact': cell['p_exact'], 'PressureError': cell['p'] - cell['p_exact'],
    }
    volume_result = write_verify(grid, destination / 'solution.vtu', arrays, metadata, volume=True)

    axis, sign = face['axis'].astype(np.int32), face['sign'].astype(np.int32)
    owner, neighbor = face['owner'].astype(np.int64), face['neighbor'].astype(np.int64)
    if not np.isin(axis, [0, 1, 2]).all() or not np.isin(sign, [-1, 1]).all():
        raise ValueError(f'Unexpected face direction in {source}')
    if not ((owner >= 0) & (owner < nc) & (neighbor >= -1) & (neighbor < nc)).all():
        raise ValueError(f'Invalid face adjacency in {source}')
    centers = np.column_stack([face[k] for k in ('x', 'y', 'z')])
    corners = np.repeat(centers[:, None, :], 4, axis=1)
    half_width = np.sqrt(face['area']) / 2
    for corner, (a, b) in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
        corners[np.arange(nf), corner, (axis + 1) % 3] += a * half_width
        corners[np.arange(nf), corner, (axis + 2) % 3] += b * half_width * sign
    points, polygons = geometry(corners)
    surface = vtk.vtkPolyData()
    surface.SetPoints(points)
    surface.SetPolys(polygons)
    normal_velocity = np.zeros((nf, 3))
    normal_velocity[np.arange(nf), axis] = sign * face['flux'] / face['area']
    coarse_fine = np.zeros(nf, dtype=np.int32)
    internal = neighbor >= 0
    coarse_fine[internal] = (cell['level'][owner[internal]] != cell['level'][neighbor[internal]])
    face_arrays = {
        'FaceId': face['id'].astype(np.int64), 'Owner': owner, 'Neighbor': neighbor,
        'Axis': axis, 'NormalSign': sign, 'BoundaryType': face['boundary'].astype(np.int32),
        'Area': face['area'], 'VolumeFlux': face['flux'], 'PredictedVolumeFlux': face['predicted_flux'],
        'PositiveAxisVolumeFlux': sign * face['flux'], 'NormalVelocity': normal_velocity,
        'NormalSpeed': face['flux'] / face['area'], 'CoarseFineInterface': coarse_fine,
    }
    face_result = write_verify(surface, destination / 'faces.vtp', face_arrays, metadata)
    if {str(path): sha256(path) for path in files} != hashes:
        raise RuntimeError(f'Source changed during export: {source}')
    result = {'case': source.name, 'source_hashes': hashes, 'metadata': metadata,
              'volume': volume_result, 'faces': face_result}
    (destination / 'export_metadata.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1] / 'output' / 'simple_validation_full')
    parser.add_argument('--output', type=Path, help='Default: <root>/paraview')
    parser.add_argument('--cases', help='Comma-separated case names; default: all cases with solution.csv')
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve() if args.output else root / 'paraview'
    cases = [root / name.strip() for name in args.cases.split(',')] if args.cases else sorted(
        path for path in root.iterdir() if path.is_dir() and (path / 'solution.csv').is_file())
    if not cases:
        parser.error(f'No SIMPLE result cases found under {root}')
    results = []
    for case in cases:
        result = export_case(case, output)
        results.append(result)
        print(f"{case.name}: {result['volume']['cells']} volume cells, {result['faces']['cells']} faces; exact round trip passed", flush=True)
    manifest = {'source_root': str(root), 'exporter_sha256': sha256(Path(__file__)),
                'vtk_version': vtk.vtkVersion.GetVTKVersion(), 'cases': results}
    (output / 'export_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Exported {len(results)} cases to {output}', flush=True)


if __name__ == '__main__':
    main()
