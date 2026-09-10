"""Check exported polygon volumes against analytic slabs and corner tetrahedra."""
from pathlib import Path
import sys
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from export_twisted_paraview import clip_cube,insert_polyhedron,boundary_volume

center=np.array([.2,.08,.06]);h=1/1024
points=vtk.vtkPoints();points.SetDataTypeToDouble()
grid=vtk.vtkUnstructuredGrid();expected=[]
for normal,alpha,volume in (
    *[(np.array([1.,0,0]),h*(fraction-.5),fraction*h**3) for fraction in (.001,.01,.3,.99)],
    *[(np.ones(3)/np.sqrt(3),h*(-np.sqrt(3)/2+e),(e*h*np.sqrt(3))**3/6) for e in (.001,.01,.1)]):
    vertices,faces=clip_cube(center,h,normal,alpha)
    origin=vertices.mean(axis=0);surface_volume=0.
    for face in faces:
        a=vertices[face[0]]-origin
        for j in range(1,len(face)-1):
            b=vertices[face[j]]-origin;c=vertices[face[j+1]]-origin
            surface_volume+=np.dot(a,np.cross(b,c))/6
    assert abs(surface_volume-volume)/h**3<1e-12,(surface_volume,volume)
    ids=[points.InsertNextPoint(v) for v in vertices]
    insert_polyhedron(grid,ids,faces);expected.append(volume)
grid.SetPoints(points)
sizes=vtk.vtkCellSizeFilter();sizes.SetInputData(grid);sizes.Update()
actual=vtk_to_numpy(sizes.GetOutput().GetCellData().GetArray('Volume'))
error=max(abs(actual-expected))/h**3
assert error<1e-10,(error,actual,expected)
print(f'PASS: analytic clipped slabs/corner tetrahedra; VTK volume error/h^3={error:.3e}')

# Real n128 near-corner fixture. The closed surface has the correct volume;
# vtkOrderedTriangulator in VTK 9.3.1 omits a thin tetrahedron from its interior.
from scipy.spatial import ConvexHull
normal=np.array([.3393075567884618,-.42956318176212327,.8368666887746493])
vertices,faces=clip_cube(center,h,normal,.00036456226359597716)
volume=boundary_volume(vertices,faces)
hull=ConvexHull((vertices-center)/h).volume*h**3
assert abs(volume-hull)/h**3<1e-12
assert abs(volume-8.313944330959088e-10)/h**3<1e-12
broken=[list(f) for f in faces];broken[0].reverse()
try:boundary_volume(vertices,broken)
except ValueError as e:assert 'oriented' in str(e)
else:raise AssertionError('Reversed polygon silently accepted')
print('PASS: actual near-corner fixture; independently integrated surface; reversed-face guard')
