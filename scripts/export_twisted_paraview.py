"""Export solved tube fields on local planar cut cells, plus wall shear polygons.

The volume shape uses the same local plane that defines the Aphros cut volume.
Its polygon openings can differ slightly from the independently reconstructed
and limited apertures used by the numerical flux operator. No flow interpolation
or invented analytic solution is used. Unknown locations are exported explicitly.
"""
import argparse
import itertools
import json
from twisted_geometry import load_geometry
from twisted_polygons import load_wall_polygons
from pathlib import Path
import numpy as np
import vtk
from scipy.spatial import ConvexHull
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


def clip_cube(center,h,normal,alpha):
    """Intersect a cube with one half-space, preserving its polygon faces."""
    cube=np.array(list(itertools.product((-.5,.5),repeat=3)))*h
    signed=cube@normal-alpha
    vertices=[];original={};crossings={}
    for i in range(8):
        if signed[i]<=0:
            original[i]=len(vertices);vertices.append(cube[i])
    faces=[]
    for quad in ((0,1,3,2),(4,6,7,5),(0,4,5,1),(2,3,7,6),(0,2,6,4),(1,5,7,3)):
        polygon=[]
        for a,b in zip(quad,quad[1:]+quad[:1]):
            if a in original:polygon.append(original[a])
            if signed[a]*signed[b]<0:
                edge=tuple(sorted((a,b)))
                if edge not in crossings:
                    lo,hi=edge;crossings[edge]=len(vertices)
                    vertices.append(cube[lo]+(cube[hi]-cube[lo])*signed[lo]/(signed[lo]-signed[hi]))
                polygon.append(crossings[edge])
        if len(polygon)>=3:faces.append(polygon)
    wall=list(crossings.values())+[original[i] for i in original if signed[i]==0]
    if len(wall)>=3:
        xyz=np.array(vertices)[wall];relative=xyz-xyz.mean(axis=0)
        tangent=np.cross(normal,np.eye(3)[np.argmin(abs(normal))]);tangent/=np.linalg.norm(tangent)
        bitangent=np.cross(normal,tangent)
        order=np.argsort(np.arctan2(relative@bitangent,relative@tangent))
        faces.append([wall[i] for i in order])
    return center+np.array(vertices),faces


def insert_polyhedron(grid,ids,faces):
    stream=vtk.vtkIdList();stream.InsertNextId(len(faces))
    for face in faces:
        stream.InsertNextId(len(face))
        for i in face:stream.InsertNextId(ids[i])
    grid.InsertNextCell(vtk.VTK_POLYHEDRON,stream)


def boundary_volume(vertices,faces):
    """Integrate the exported closed, oriented surface about an interior point."""
    edges={};triangles=[]
    for face in faces:
        for a,b in zip(face,face[1:]+face[:1]):
            key=tuple(sorted((a,b)));count,direction=edges.get(key,(0,0))
            edges[key]=(count+1,direction+(1 if a<b else -1))
        for j in range(1,len(face)-1):triangles.append([face[0],face[j],face[j+1]])
    if any(value!=(2,0) for value in edges.values()):raise ValueError('Exported polyhedron is not closed and consistently oriented')
    local=vertices-vertices.mean(axis=0);a,b,c=local[np.array(triangles)].transpose(1,0,2)
    volume=float(np.einsum('ij,ij->i',a,np.cross(b,c)).sum()/6)
    if volume<=0:raise ValueError('Exported polyhedron has nonpositive volume')
    return volume


def read(path):
    a = np.genfromtxt(path, delimiter=',', names=True, ndmin=1)
    if not len(a) or any(not np.isfinite(a[n]).all() for n in a.dtype.names):
        raise ValueError(f'Invalid data: {path}')
    return a


def field(data, names):
    return np.column_stack([data[n] for n in names])


def add(arrays, name, data):
    a = numpy_to_vtk(np.ascontiguousarray(data), deep=True)
    a.SetName(name)
    arrays.AddArray(a)


def write(path, data, kind):
    writer = kind()
    writer.SetFileName(str(path))
    writer.SetInputData(data)
    writer.SetDataModeToAppended()
    if not writer.Write():
        raise RuntimeError(f'Cannot write {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='Write visualization files to a fresh directory, keeping the solved trajectory in place')
    args = parser.parse_args()
    output = args.run if args.output is None else args.output.resolve()
    if args.output is not None:
        output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.run/'case.json').read_text(encoding='utf-8-sig'))
    geometry_path=Path(config['embedded_geometry'])
    meta=geometry_path.with_suffix('.meta.json')
    geometry = json.loads((meta if meta.exists() else geometry_path).read_text(encoding='utf-8'))
    if geometry['format']=='aphros_cut_geometry_v2':
        geometry=load_geometry(geometry_path,tables=('walls',))
    shift=np.array(geometry.get('reference_translation',[0,0,0]))
    shift_cells=np.array(geometry.get('reference_translation_cells',[0,0,0]),dtype=int)
    nx=int(round(geometry['extent'][0]/geometry['finest_h']))
    if 'walls' not in geometry:
        source=next(Path(p) for p in geometry['source_sha256'] if p.endswith('_geometry_walls.csv'))
        rows=np.loadtxt(source,delimiter=',',skiprows=1,ndmin=2)
        rows[:,:3]+=shift_cells
        wrap=np.floor_divide(rows[:,0].astype(int),nx)
        rows[:,0]-=wrap*nx;rows[:,3:6]+=shift
        rows[:,3]-=wrap*geometry['extent'][0]
        geometry['walls']=rows.tolist()
    solution, shear = read(args.run/'solution.csv'), read(args.run/'walls.csv')
    wall_lookup = {tuple(int(q) for q in row[:3]): row for row in geometry['walls']}
    hmin = geometry['finest_h']
    centers = field(solution, ['x', 'y', 'z'])
    cube = np.array(list(itertools.product((-.5, .5), repeat=3)))
    points, cells = vtk.vtkPoints(), vtk.vtkUnstructuredGrid()
    points.SetDataTypeToDouble()
    cells.Allocate(len(solution))
    cut = np.zeros(len(solution), dtype=np.int32)
    hull_volumes=solution['h']**3
    boundary_volumes=hull_volumes.copy()
    for i, (center, row) in enumerate(zip(centers, solution)):
        vertices = center + cube*row['h']
        key = tuple(np.rint(center/hmin-.5).astype(int))
        wall = wall_lookup.get(key) if abs(row['h']-hmin) < hmin*1e-12 else None
        if wall is not None:
            cut[i] = 1
            normal, alpha = np.array(wall[6:9]), wall[10]
            vertices,polygons=clip_cube(center,row['h'],normal,alpha)
        if len(vertices) < 4:
            raise ValueError('Degenerate reconstructed volume')
        ids = [points.InsertNextPoint(vertex) for vertex in vertices]
        if wall is None:
            cells.InsertNextCell(vtk.VTK_HEXAHEDRON, 8, [ids[j] for j in (0,4,6,2,1,5,7,3)])
        else:
            local = (vertices-center)/row['h']
            hull_volumes[i]=ConvexHull(local).volume*row['h']**3
            boundary_volumes[i]=boundary_volume(vertices,polygons)
            insert_polyhedron(cells,ids,polygons)
    cells.SetPoints(points)
    arrays = cells.GetCellData()
    add(arrays, 'Velocity', field(solution, ['u', 'v', 'w']))
    add(arrays, 'Speed', np.linalg.norm(field(solution, ['u', 'v', 'w']), axis=1))
    for name, key in [('Pressure', 'p'), ('Level', 'level'), ('CellSize', 'h'), ('FluidVolume', 'volume')]:
        add(arrays, name, solution[key])
    add(arrays, 'CutCell', cut)
    add(arrays, 'UnknownLocation', centers)
    add(arrays, 'FluidFraction', solution['volume']/solution['h']**3)
    arrays.SetActiveVectors('Velocity')
    arrays.SetActiveScalars('Speed')
    sizes = vtk.vtkCellSizeFilter()
    sizes.SetInputData(cells)
    sizes.SetComputeArea(False)
    sizes.SetComputeLength(False)
    sizes.SetComputeVertexCount(False)
    sizes.Update()
    volumes = vtk_to_numpy(sizes.GetOutput().GetCellData().GetArray('Volume'))
    # Geometric volume integration becomes relative ill-conditioned in slivers;
    # normalize by full cube volume for the maximum, and by fluid total globally.
    local_error = float(np.max(np.abs(boundary_volumes-solution['volume'])/solution['h']**3))
    global_error = float(abs(boundary_volumes.sum()-solution['volume'].sum())/solution['volume'].sum())
    hull_error=float(np.max(np.abs(hull_volumes-solution['volume'])/solution['h']**3))
    if hull_error>1e-10:raise ValueError(f'Independent convex hull differs from solver volume: {hull_error}')
    if local_error>1e-10 or global_error>1e-10:
        raise ValueError(f'Exported closed boundary differs from solver volume: {local_error}, {global_error}')
    vtk_local_error=float(np.max(np.abs(volumes-solution['volume'])/solution['h']**3))
    vtk_global_error=float(abs(volumes.sum()-solution['volume'].sum())/solution['volume'].sum())
    vtk_agrees=vtk_local_error<=1e-8 and vtk_global_error<=1e-10
    # VTK 9.3.1's generic polyhedron volume filter retriangulates vertices with
    # vtkOrderedTriangulator. On the n128 near-corner fixture it omits a sliver;
    # its result also changes with vertex IDs despite identical surface faces.
    # Keep this diagnostic visible. Geometry acceptance uses TWO independent
    # volume constructions above, plus exact closed/oriented edge incidence.
    if not vtk_agrees:
        bad=np.argsort(np.abs(volumes-solution['volume'])/solution['h']**3)[-16:]
        np.savetxt(output/'vtk_volume_filter_discrepancies.csv',np.column_stack((bad,centers[bad],solution['h'][bad],
                   solution['volume'][bad],hull_volumes[bad],volumes[bad])),delimiter=',',
                   header='id,x,y,z,h,solver_volume,convex_hull_volume,vtk_volume',comments='')
    write(output/'solution.vtu', cells, vtk.vtkXMLUnstructuredGridWriter)

    groups, polygon_source = load_wall_polygons(geometry, geometry_path)
    wp, polygons = vtk.vtkPoints(), vtk.vtkCellArray()
    wp.SetDataTypeToDouble()
    for row in shear:
        owner = int(row['owner'])
        key = tuple(np.rint(centers[owner]/hmin-.5).astype(int))
        vertices = groups[key]
        polygons.InsertNextCell(len(vertices))
        for vertex in vertices:
            polygons.InsertCellPoint(wp.InsertNextPoint(vertex))
    wall = vtk.vtkPolyData()
    wall.SetPoints(wp)
    wall.SetPolys(polygons)
    tau = field(shear, ['tau_x', 'tau_y', 'tau_z'])
    add(wall.GetCellData(), 'WallShear', tau)
    add(wall.GetCellData(), 'WallShearMagnitude', np.linalg.norm(tau, axis=1))
    add(wall.GetCellData(), 'Normal', field(shear, ['nx', 'ny', 'nz']))
    add(wall.GetCellData(), 'OperatorWallArea', shear['area'])
    wall.GetCellData().SetActiveScalars('WallShearMagnitude')
    write(output/'walls.vtp', wall, vtk.vtkXMLPolyDataWriter)
    report = {'cells': len(solution), 'cut_cells': int(cut.sum()), 'wall_polygons': len(shear),
              'max_volume_difference_per_full_cube': local_error,
              'total_volume_relative_difference': global_error,
              'max_convex_hull_volume_difference_per_full_cube':hull_error,
              'geometry_verified':True,'vtk_version':vtk.vtkVersion.GetVTKVersion(),
              'vtk_cell_size_filter_agrees':vtk_agrees,
              'vtk_cell_size_max_difference_per_full_cube':vtk_local_error,
              'vtk_cell_size_total_relative_difference':vtk_global_error,
              'volume_verification':'Closed oriented polygon boundary integration AND independent convex hull; VTK generic volume filter reported separately',
              'geometry': 'Local plane cut volumes; flux apertures may differ slightly; see module docstring',
              'wall_polygon_source': polygon_source,
              'source_step': str(args.run.resolve()), 'visualization_directory': str(output.resolve())}
    (output/'paraview_export.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
