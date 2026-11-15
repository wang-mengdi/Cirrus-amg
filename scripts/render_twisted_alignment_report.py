"""Render saved Cirrus/Aphros fields without running either flow solver.

The 64 comparison is a completed steady pair; the 128 pair is explicitly a
partial, seeded Aphros transient. All native cells and wall faces are exported.
Local planar cut geometry is checked against solver volumes and wall areas.
"""
import argparse
import gc
import html
import json
import os
from pathlib import Path
import sys


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('all', '64', '128', '256', 'report'), default='all')
    parser.add_argument('--scientific-packages', type=Path)
    return parser.parse_args()


ARGS = arguments()
DLL_HANDLES = []
if ARGS.scientific_packages:
    sys.path.insert(0, str(ARGS.scientific_packages))
    library = ARGS.scientific_packages.parent.parent / 'Library' / 'bin'
    if os.name == 'nt' and library.is_dir():
        DLL_HANDLES.append(os.add_dll_directory(str(library)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from scipy.spatial import ConvexHull

from compare_twisted import transfer, error, ordered, vector
from twisted_geometry import load_geometry
from twisted_polygons import load_wall_polygons
from export_twisted_paraview import clip_cube, boundary_volume, insert_polyhedron, add, write

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('D:/CirrusExperiments/cirrus-amg')
OUT = ARGS.output.resolve()
if not OUT.is_relative_to(ROOT) or 'OneDrive' in str(OUT):
    raise ValueError('Write local visualization artifacts under D:/CirrusExperiments/cirrus-amg')
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                     'axes.titlesize': 14, 'axes.labelsize': 12})


def log(message):
    print(message, flush=True)


def read(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        names = stream.readline().strip().split(',')
    result = np.loadtxt(path, delimiter=',', skiprows=1,
                        dtype=[(name, np.float64) for name in names], ndmin=1)
    if not len(result) or any(not np.isfinite(result[n]).all() for n in names):
        raise ValueError('Nonfinite or empty field: ' + str(path))
    return result


def json_read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def json_write(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


def dataset_read(path):
    reader = vtk.vtkXMLPolyDataReader() if Path(path).suffix == '.vtp' else vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    return reader.GetOutput()


def make_geometry(cells, walls, geometry_path, reference):
    geometry = load_geometry(geometry_path, tables=('walls',))
    wall_lookup = {tuple(int(x) for x in row[:3]): row for row in geometry['walls']}
    hmin = geometry['finest_h']
    centers = vector(cells, 'xyz')
    if not np.array_equal(cells['id'], np.arange(len(cells))):
        raise ValueError('Expected dense cell IDs in original file order')
    points = vtk.vtkPoints()
    points.SetDataTypeToDouble()
    grid = vtk.vtkUnstructuredGrid()
    grid.Allocate(len(cells))
    cube = np.array([(x,y,z) for x in (-.5,.5) for y in (-.5,.5) for z in (-.5,.5)])
    caps = {}
    max_volume = 0.0
    max_hull = 0.0
    for index, row in enumerate(cells):
        center, size = centers[index], row['h']
        key = tuple(np.rint(center/hmin-.5).astype(int))
        plane = wall_lookup.get(key) if np.isclose(size, hmin, rtol=1e-12, atol=0) else None
        if plane is None:
            vertices = center + cube*size
            ids = [points.InsertNextPoint(p) for p in vertices]
            grid.InsertNextCell(vtk.VTK_HEXAHEDRON, 8, [ids[j] for j in (0,4,6,2,1,5,7,3)])
            if not np.isclose(size**3, row['volume'], rtol=1e-12, atol=0):
                raise ValueError('Uncut volume mismatch')
        else:
            vertices, polygons = clip_cube(center, size, np.array(plane[6:9]), plane[10])
            ids = [points.InsertNextPoint(p) for p in vertices]
            insert_polyhedron(grid, ids, polygons)
            volume = boundary_volume(vertices, polygons)
            hull = ConvexHull((vertices-center)/size).volume*size**3
            max_volume = max(max_volume, abs(volume-row['volume'])/size**3)
            max_hull = max(max_hull, abs(hull-row['volume'])/size**3)
            cap = vertices[polygons[-1]]
            if np.max(np.abs((cap-center) @ np.asarray(plane[6:9])-plane[10])) > size*1e-10:
                raise ValueError('Cap is not on the stored cut plane')
            caps[index] = cap
        if index and index % 100000 == 0:
            log(f'  geometry {index:,}/{len(cells):,} cells')
    if max(max_volume,max_hull) > 1e-10:
        raise ValueError('Cut volume validation failed')
    grid.SetPoints(points)
    for name in cells.dtype.names:
        add(grid.GetCellData(), name, cells[name])
    add(grid.GetCellData(), 'Velocity', vector(cells, 'uvw'))
    add(grid.GetCellData(), 'Speed', np.linalg.norm(vector(cells,'uvw'),axis=1))
    add(grid.GetCellData(), 'UnknownLocation', centers)
    add(grid.GetCellData(), 'FluidVolume', cells['volume'])
    add(grid.GetCellData(), 'CutCell', np.isin(cells['id'], walls['owner']).astype(np.uint8))
    grid.GetCellData().SetActiveVectors('Velocity')
    grid.GetCellData().SetActiveScalars('Speed')
    wp = vtk.vtkPoints()
    wp.SetDataTypeToDouble()
    polygons = vtk.vtkCellArray()
    max_area = 0.0
    max_vertex_difference = 0.0
    geometry['baseline_polygons'] = str(reference/'tube_b0_geometry_polygons.csv')
    groups, polygon_report = load_wall_polygons(geometry, geometry_path)
    for row in walls:
        owner = int(row['owner'])
        key = tuple(np.rint(centers[owner]/hmin-.5).astype(int))
        cap = groups[key]
        plane_cap = caps[owner]
        distances = np.linalg.norm(cap[:,None,:]-plane_cap[None,:,:],axis=2)
        max_vertex_difference = max(max_vertex_difference,
                                    float(max(distances.min(axis=0).max(),distances.min(axis=1).max())/hmin))
        local = cap-cap.mean(axis=0)
        area = np.linalg.norm(np.cross(local,np.roll(local,-1,axis=0)).sum(axis=0))/2
        max_area = max(max_area, abs(area-row['area'])/hmin**2)
        polygons.InsertNextCell(len(cap))
        for p in cap:
            polygons.InsertCellPoint(wp.InsertNextPoint(p))
    if max_vertex_difference > 1e-10:
        raise ValueError('Original polygon vertices differ from the stored cut plane')
    surface = vtk.vtkPolyData()
    surface.SetPoints(wp)
    surface.SetPolys(polygons)
    for name in walls.dtype.names:
        add(surface.GetCellData(), name, walls[name])
    add(surface.GetCellData(), 'WallShear', vector(walls, ('tau_x','tau_y','tau_z')))
    add(surface.GetCellData(), 'WallShearMagnitude', np.linalg.norm(vector(walls, ('tau_x','tau_y','tau_z')),axis=1))
    surface.GetCellData().SetActiveScalars('WallShearMagnitude')
    return grid, surface, {
        'all_native_cells':len(cells), 'all_wall_faces':len(walls),
        'geometry':'Local planar cut volumes and original Aphros wall vertices; no decimation',
        'original_polygon_source_available':True,
        'original_polygon_source':polygon_report,
        'max_original_vs_cut_plane_vertex_distance_per_h':max_vertex_difference,
        'max_volume_error_per_full_cube':max_volume,
        'max_independent_convex_hull_error_per_full_cube':max_hull,
        'max_polygon_vs_operator_wall_area_difference_per_full_square':max_area,
        'area_policy':'Original polygons are displayed; original operator wall areas are retained and used in norms. Their area difference is reported, not treated as a new field-accuracy gate.',
    }


def slice_grid(grid, z):
    plane = vtk.vtkPlane()
    plane.SetOrigin(0,0,z)
    plane.SetNormal(0,0,1)
    cutter = vtk.vtkCutter()
    cutter.SetInputData(grid)
    cutter.SetCutFunction(plane)
    cutter.Update()
    result = vtk.vtkPolyData()
    result.DeepCopy(cutter.GetOutput())
    return result


def panel(data, name, cmap, limits, context=None):
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1,1,1)
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetSize(960,560)
    window.SetMultiSamples(4)
    window.AddRenderer(renderer)
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.SetTableRange(*limits)
    lut.Build()
    for i, rgba in enumerate(plt.get_cmap(cmap)(np.linspace(0,1,256))):
        lut.SetTableValue(i,*rgba)
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputData(data)
    mapper.SetScalarModeToUseCellFieldData()
    mapper.SelectColorArray(name)
    mapper.SetLookupTable(lut)
    mapper.SetScalarRange(*limits)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetAmbient(1)
    actor.GetProperty().SetDiffuse(0)
    renderer.AddActor(actor)
    if context is not None:
        ghost_mapper = vtk.vtkPolyDataMapper()
        ghost_mapper.SetInputData(context)
        ghost_mapper.ScalarVisibilityOff()
        ghost = vtk.vtkActor()
        ghost.SetMapper(ghost_mapper)
        ghost.GetProperty().SetColor(.32,.38,.45)
        ghost.GetProperty().SetOpacity(.08)
        renderer.AddActor(ghost)
    bounds = (context or data).GetBounds()
    center = np.array([(bounds[2*i]+bounds[2*i+1])/2 for i in range(3)])
    camera = renderer.GetActiveCamera()
    camera.SetFocalPoint(*center)
    camera.SetPosition(*(center + np.array([.09,-.38,.36])))
    camera.SetViewUp(0,0,1)
    camera.ParallelProjectionOn()
    renderer.ResetCamera(*bounds)
    camera.SetParallelScale(camera.GetParallelScale()*.83)
    renderer.ResetCameraClippingRange()
    window.Render()
    capture = vtk.vtkWindowToImageFilter()
    capture.SetInput(window)
    capture.SetInputBufferTypeToRGB()
    capture.ReadFrontBufferOff()
    capture.Update()
    result = vtk_to_numpy(capture.GetOutput().GetPointData().GetScalars()).reshape(560,960,3)[::-1].copy()
    window.Finalize()
    return result


def add_bar(fig, ax, cmap, limits, unit):
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(*limits),cmap=cmap),
                      ax=ax, orientation='horizontal', fraction=.08, pad=.025, shrink=.84)
    bar.set_label(unit, fontsize=11)
    bar.ax.tick_params(labelsize=10)
    bar.formatter.set_powerlimits((-2,3))
    bar.update_ticks()


def compare_figure(ny, grid, walls, metadata):
    z = .0625+metadata['translation'][2]
    section = slice_grid(grid,z)
    specs = [
        ('Velocity magnitude / mid-plane','Speed','AphrosSpeed','VelocityDifferenceMagnitude','viridis','m/s','velocity',section,walls),
        ('Pressure / mid-plane, mean removed','CirrusPressureMeanZero','AphrosPressureMeanZero','PressureDifferenceAbsolute','RdBu_r','Pa','pressure',section,walls),
        ('Wall shear magnitude / full wall','WallShearMagnitude','AphrosWallShearMagnitude','WallShearDifferenceMagnitude','viridis','Pa','wall_shear',walls,None)]
    fig, axes = plt.subplots(3,3,figsize=(20,15),facecolor='white')
    fig.subplots_adjust(left=.025,right=.98,top=.88,bottom=.10,wspace=.08,hspace=.34)
    for row, (title, native, ref, diff, cmap, unit, metric, data, ghost) in enumerate(specs):
        array_source = grid if row < 2 else walls
        a = vtk_to_numpy(array_source.GetCellData().GetArray(native))
        b = vtk_to_numpy(array_source.GetCellData().GetArray(ref))
        delta = vtk_to_numpy(array_source.GetCellData().GetArray(diff))
        bounds = (0,float(max(a.max(),b.max()))) if row != 1 else (-float(max(abs(a).max(),abs(b).max())),float(max(abs(a).max(),abs(b).max())))
        for col, field in enumerate((native,ref,diff)):
            limits = bounds if col < 2 else (0,float(delta.max()))
            palette = cmap if col < 2 else 'inferno'
            axes[row,col].imshow(panel(data,field,palette,limits,ghost))
            axes[row,col].axis('off')
            label = ('Cirrus','Aphros','Absolute vector difference' if row != 1 else 'Absolute pressure difference')[col]
            heading = label+'\n'+title
            if col == 2:
                heading += f'\nFull-field relative L2 = {metadata["measured"][metric]["relative_l2"]*100:.4f}%'
            axes[row,col].set_title(heading,fontsize=13,pad=6)
            add_bar(fig, axes[row,col],palette,limits,unit)
    if ny == 64:
        heading = '64  |  Cirrus vs Aphros: completed steady comparison'
        subtitle = 'Stored acceptance report: PASS  |  Current field errors reproduced  |  Same scale and camera in each pair'
        color = '#147d64'
    else:
        heading = '128  |  Partial Aphros transient vs Cirrus steady field'
        subtitle = 'DIAGNOSTIC ONLY — Aphros 5 / 128 steps, t = 0.025 s; seeded from Cirrus. Steady alignment is NOT established.'
        color = '#a75a12'
    fig.suptitle(heading,x=.03,y=.97,ha='left',fontsize=23,fontweight='bold',color='#172b40')
    fig.text(.03,.928,subtitle,fontsize=13,color=color)
    fig.text(.03,.047,f'{metadata["cells"]:,} Cirrus cells; {metadata["wall_faces"]:,} wall faces. All cells retained in the VTK files. '
             f'Interior view: z = {z:.8f} m; tube length = 0.25 m.',fontsize=11)
    fig.text(.03,.022,'Aphros: exact fine/cut-center values, tensor cubic at coarse centers. Pressure gauges aligned with volume weights. '
             'Difference colors use the full-field maximum; no pointwise relative division.',fontsize=10)
    filename = OUT / ('comparison_64.png' if ny == 64 else 'comparison_128_partial.png')
    fig.savefig(filename,dpi=160,facecolor='white')
    plt.close(fig)
    log('Rendered '+str(filename))


def comparison(ny):
    directory = OUT / str(ny)
    directory.mkdir(exist_ok=True)
    ours_root = ROOT/'runs'/('steady_outer64_v1/iterate_0016' if ny == 64 else 'reproduction_native128_v1/iterate_0023')
    reference = ROOT/'runs'/('aphros64_initial_native_v1' if ny == 64 else 'reproduction_aphros128_v2')
    saved_report = ROOT/'checks'/('handoff64_pair_v1.json' if ny == 64 else 'aphros128_partial24h_vs_cirrus128_20260924.json')
    pair, case, cfg = json_read(saved_report), json_read(ours_root/'case.json'), json_read(reference/'case_manifest.json')
    if ny == 64 and not (pair['passed'] and all(pair['checks'].values())):
        raise ValueError('64 report was not accepted')
    if ny == 128 and pair['steady_alignment_established']:
        raise ValueError('Unexpected diagnostic scope')
    if (directory/'metadata.json').exists():
        log(f'Reusing locally generated {ny} visualization')
        metadata = json_read(directory/'metadata.json')
        compare_figure(ny,dataset_read(directory/'solution.vtu'),dataset_read(directory/'walls.vtp'),metadata)
        return
    log(f'Loading all {ny} cells and wall fields')
    cells, walls = read(ours_root/'solution.csv'), read(ours_root/'walls.csv')
    geometry_path = Path(case['embedded_geometry'])
    if not geometry_path.is_absolute():
        geometry_path = REPO/geometry_path
    geometry = json_read(geometry_path.with_suffix('.meta.json') if geometry_path.with_suffix('.meta.json').exists() else geometry_path)
    shift = np.asarray(geometry.get('reference_translation',[0,0,0]))
    def translate(data):
        for i,key in enumerate('xyz'):
            data[key] += shift[i]
        data['x'] %= cfg['spec']['extent'][0]
        return ordered(data)
    ref = translate(read(reference/'proj_final_b0_cells.csv'))
    sampled, transfer_report = transfer(cells,ref,cfg['shape'],cfg['spec']['extent'][1]/ny)
    refwall = translate(read(reference/'tube_final_b0_walls.csv'))
    order = np.lexsort((walls['z'],walls['y'],walls['x']))
    if len(refwall) != len(walls) or not np.allclose(vector(walls[order],'xyz'),vector(refwall,'xyz'),rtol=0,atol=1e-13):
        raise ValueError('Wall coordinate mismatch')
    if not np.allclose(walls['area'][order],refwall['area'],rtol=1e-12,atol=0):
        raise ValueError('Wall area mismatch')
    reference_tau = np.empty((len(walls),3))
    reference_tau[order] = vector(refwall,('tau_x','tau_y','tau_z'))
    velocity, tau = vector(cells,'uvw'), vector(walls,('tau_x','tau_y','tau_z'))
    pressure = cells['p']-np.average(cells['p'],weights=cells['volume'])
    refpressure = sampled[:,3]-np.average(sampled[:,3],weights=cells['volume'])
    cut = np.isin(cells['id'],walls['owner'])
    measured = {'velocity':error(velocity,sampled[:,:3],cells['volume']),
                'pressure':error(pressure,refpressure,cells['volume']),
                'wall_shear':error(tau,reference_tau,walls['area']),
                'cut_cell_velocity':error(velocity[cut],sampled[cut,:3],cells['volume'][cut])}
    recorded = pair if ny == 64 else pair['fields']
    for name in measured:
        for stat in ('relative_l2','absolute_l2','absolute_max'):
            if not np.isclose(measured[name][stat],recorded[name][stat],rtol=1e-8,atol=1e-14):
                raise ValueError('Current data differs from stored comparison: '+name+'/'+stat)
    log(f'{ny} field metrics match stored report: '+str({k:v['relative_l2'] for k,v in measured.items()}))
    del ref,refwall
    grid, surface, geom = make_geometry(cells,walls,geometry_path,reference)
    cell_fields = {'AphrosVelocity':sampled[:,:3],'AphrosSpeed':np.linalg.norm(sampled[:,:3],axis=1),
                   'VelocityDifference':velocity-sampled[:,:3],
                   'VelocityDifferenceMagnitude':np.linalg.norm(velocity-sampled[:,:3],axis=1),
                   'CirrusPressureMeanZero':pressure,'AphrosPressureMeanZero':refpressure,
                   'PressureDifference':pressure-refpressure,'PressureDifferenceAbsolute':abs(pressure-refpressure),
                   'ReferenceInterpolated':(cells['h']>cells['h'].min()).astype(np.uint8)}
    wall_fields = {'AphrosWallShear':reference_tau,'AphrosWallShearMagnitude':np.linalg.norm(reference_tau,axis=1),
                   'WallShearDifference':tau-reference_tau,'WallShearDifferenceMagnitude':np.linalg.norm(tau-reference_tau,axis=1)}
    for data,fields in ((grid,cell_fields),(surface,wall_fields)):
        for name,values in fields.items():
            add(data.GetCellData(),name,values)
    log(f'Writing complete {ny} VTK datasets')
    write(directory/'solution.vtu',grid,vtk.vtkXMLUnstructuredGridWriter)
    write(directory/'walls.vtp',surface,vtk.vtkXMLPolyDataWriter)
    for path, fields, count in ((directory/'solution.vtu',cell_fields,len(cells)),(directory/'walls.vtp',wall_fields,len(walls))):
        check=dataset_read(path)
        if check.GetNumberOfCells() != count:
            raise ValueError('VTK readback cell count mismatch')
        for name,values in fields.items():
            if not np.array_equal(vtk_to_numpy(check.GetCellData().GetArray(name)),values):
                raise ValueError('VTK readback differs: '+name)
        del check
    metadata={'resolution':ny,'cells':len(cells),'wall_faces':len(walls),'measured':measured,
              'stored_section_flux':recorded['section_flux'],'translation':shift.tolist(),
              'transfer':transfer_report,'geometry':geom,'vtk_readback_exact':True,
              'saved_acceptance_report':str(saved_report),'source_cirrus':str(ours_root),'source_aphros':str(reference),
              'stored_steady_alignment_passed':ny==64,'revalidated_full_run_provenance':False,
              'aphros_physical_time':pair['aphros_temporal_diagnostic']['time'],
              'aphros_complete_trajectory':ny==64,'aphros_seeded_from_cirrus':True,
              'scope':'Existing field comparison and visualization; no flow solve or new full provenance certification.'}
    json_write(directory/'metadata.json',metadata)
    compare_figure(ny,grid,surface,metadata)
    del grid,surface,cells,walls,sampled,cell_fields,wall_fields
    gc.collect()


def native256():
    log('Loading complete existing 256 geometry: 3,995,168 cells')
    source = ROOT/'runs/reproduction_native256_v2_viz_full'
    grid, wall = dataset_read(source/'solution.vtu'), dataset_read(source/'walls.vtp')
    metrics = json_read(ROOT/'runs/reproduction_native256_v2/iterate_0043/metrics.json')
    summary = json_read(ROOT/'runs/reproduction_native256_v2/steady_summary.json')
    if grid.GetNumberOfCells()!=3995168 or wall.GetNumberOfCells()!=349008 or not summary['steady_converged']:
        raise ValueError('Unexpected 256 state')
    cut = slice_grid(grid,.0625)
    fig = plt.figure(figsize=(20,12),facecolor='white')
    spec = fig.add_gridspec(2,2,height_ratios=[1.45,1],left=.055,right=.97,top=.86,bottom=.10,hspace=.5,wspace=.19)
    for col,data,name,title,unit,context in [(0,cut,'Speed','Velocity on the mid-plane z = 0.0625 m','m/s',wall),
                                          (1,wall,'WallShearMagnitude','Wall shear on the complete tube wall','Pa',None)]:
        ax=fig.add_subplot(spec[0,col])
        maximum=float((grid if col==0 else wall).GetCellData().GetArray(name).GetRange()[1])
        ax.imshow(panel(data,name,'viridis',(0,maximum),context));ax.axis('off');ax.set_title(title)
        add_bar(fig,ax,'viridis',(0,maximum),unit)
    history=np.genfromtxt(ROOT/'runs/reproduction_native256_v2/steady_history.csv',delimiter=',',names=True,dtype=None,encoding='utf-8')
    ax=fig.add_subplot(spec[1,0])
    ax.semilogy(history['iteration'],history['steady_momentum_relative_l2'],'o-',ms=3,color='#177e89',label='Momentum residual')
    ax.semilogy(history['iteration'],history['steady_map_velocity_defect_relative_l2'],'o-',ms=3,color='#6a4c93',label='Velocity map defect')
    ax.axhline(1e-8,color='#b74c34',ls='--',label='Required threshold: 1e-8')
    ax.set(xlabel='Steady outer iteration (not physical time)',ylabel='Relative residual',title='Recorded convergence history')
    ax.grid(alpha=.2);ax.legend(fontsize=10)
    ax=fig.add_subplot(spec[1,1]);ax.axis('off')
    lines=[('Cells / wall faces','3,995,168 / 349,008'),('Steady iteration','43 / 64 maximum'),
           ('Momentum residual',f'{metrics["steady_momentum_relative_l2"]:.3e}'),
           ('Velocity map defect',f'{metrics["steady_map_velocity_defect_relative_l2"]:.3e}'),
           ('Complete fixed-point residual',f'{metrics["steady_complete_fixed_point_residual"]:.3e}'),
           ('Aphros 256 comparison','NOT AVAILABLE')]
    for i,(key,value) in enumerate(lines):
        ax.text(.01,.94-i*.15,key,fontsize=12,color='#3d4e5e')
        ax.text(.98,.94-i*.15,value,fontsize=13,ha='right',fontweight='bold',color='#172b40')
    fig.suptitle('256  |  Cirrus steady solution and convergence',x=.04,y=.97,ha='left',fontsize=24,fontweight='bold',color='#172b40')
    fig.text(.04,.92,'Solver summary: converged. No Aphros 256 alignment result; direct trial lacks the formal run hash manifest.',fontsize=13,color='#a75a12')
    fig.text(.055,.035,'Full original 256 VTK geometry used; no mesh decimation. Images expose one interior plane; the source volume retains every cell.',fontsize=11)
    fig.savefig(OUT/'native256_overview.png',dpi=160,facecolor='white');plt.close(fig)
    json_write(OUT/'native256_metadata.json',{'cells':grid.GetNumberOfCells(),'wall_faces':wall.GetNumberOfCells(),
               'source_visualization':str(source),'metrics':metrics,'summary':summary,'aphros_alignment_established':False,
               'formal_run_hash_manifest_available':False})
    log('Rendered native256_overview.png')
    del grid,wall,cut
    gc.collect()


def report():
    rows=[]
    for ny in (64,128):
        data=json_read(OUT/str(ny)/'metadata.json')
        state='已通过既定门槛（历史验收报告；本次重算场差一致）' if ny==64 else '尚未对齐：Aphros 仅完成 5/128 步，t=0.025 s'
        rows.append('<tr><td>'+str(ny)+'</td><td>'+state+'</td>'+''.join(f'<td>{data["measured"][k]["relative_l2"]*100:.4f}%</td>' for k in ('velocity','pressure','wall_shear'))+'</tr>')
    rows.append('<tr><td>256</td><td>Cirrus 第 43 轮定常；无 Aphros 256 对照</td><td colspan="3">不适用</td></tr>')
    content='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Cirrus / Aphros 对比 · 2026-11-06</title>
<style>body{max-width:1250px;margin:40px auto;font:17px/1.65 system-ui;color:#172b40;padding:0 25px}h1{font-size:32px}h2{margin-top:40px}img{width:100%;border:1px solid #dde3e9}table{border-collapse:collapse;width:100%}td,th{border:1px solid #dde3e9;padding:12px;text-align:left}th{background:#edf3f6}.note{background:#fff4e6;padding:15px;border-left:4px solid #bd7529}code{background:#edf3f6}a{color:#0a6a8c}</style>
<h1>Cirrus 与 Aphros：当前结果及完整场可视化</h1><p>核查日期：2026-11-06。读取现有文件并制作本地渲染，没有运行流动求解器，也没有提交、上传或修改云端内容。</p>
<table><tr><th>最细横向分辨率</th><th>结论</th><th>速度 L2</th><th>压力 L2</th><th>壁面剪切 L2</th></tr>'''+''.join(rows)+'''</table>
<p>64 的既定门槛：速度和截面流量 0.5%，压力、切割单元速度和壁面剪切 1%。速度/压力按体积加权，壁面剪切按面积加权。压力先去除体积加权均值；差值图为绝对向量差模长（压力为绝对标量差），不对近零值做逐点相对除法。</p>
<div class="note">128 的 Aphros 从 Cirrus 速度场作初猜，仅完成 5 步后停止，退出码 15。小场差不能证明独立定常对齐。256 的 steady_summary.json 报告收敛，但它是直接启动的试跑，缺少正式运行哈希清单，未完成 Aphros 256 验收。迭代收敛、跨程序对齐与空间收敛是不同问题；历史 64→128 空间门槛仍未通过。</div>
<h2>64：完成的定常对比</h2><a href="comparison_64.png"><img src="comparison_64.png"></a>
<h2>128：未完成参考的诊断对比</h2><a href="comparison_128_partial.png"><img src="comparison_128_partial.png"></a>
<h2>256：完整网格及自身收敛</h2><a href="native256_overview.png"><img src="native256_overview.png"></a>
<h2>用 ParaView 查看完整场</h2><ol><li>打开本目录 <code>64/solution.vtu</code> 或 <code>128/solution.vtu</code>，点击 Apply。所有原生流体单元均保留；没有抽样或删减网格。</li>
<li>Coloring 选 Cell Data：<code>Speed</code>、<code>AphrosSpeed</code>、<code>VelocityDifferenceMagnitude</code>，或两组 <code>*PressureMeanZero</code> 与 <code>PressureDifference</code>。对比两解时使用同一色标范围。</li>
<li>查看内部用 Slice，法向量 (0,0,1)。64 的中部平面 z=0.05859375 m；128 的中部平面 z=0.0625 m。剖面只改变显示，VTU 内完整体网格仍保留。</li>
<li>打开相同目录的 <code>walls.vtp</code>；选择 <code>WallShearMagnitude</code>、<code>AphrosWallShearMagnitude</code> 或 <code>WallShearDifferenceMagnitude</code>。全部壁面片保留。</li>
<li>256 完整体数据位于 <code>D:\\CirrusExperiments\\cirrus-amg\\runs\\reproduction_native256_v2_viz_full</code>，包含 3,995,168 个流体单元和 349,008 个壁面片。</li></ol>
<h2>数据与几何说明</h2><p>Aphros 的细网格/切割单元中心值精确对应；Cirrus 粗中心使用原比较器的 64 点张量三次插值。64 有明确坐标平移，两个解在渲染前已对齐。原始 Aphros 均匀网格数据保留在来源目录；输出的参考字段是在 Cirrus 网格上的对应值。</p>
<p>完整壁面使用各 Aphros 运行目录中找到的原始 polygon CSV 顶点；体网格依据保存的求解器切割平面重建，并用闭合面体积分与独立凸包核对体积。原始壁面顶点也与切割平面顶点逐面核对。未使用光滑拟合曲面替代网格。原始多边形几何面积与求解器的算子壁面面积并不完全一致；图保留原始顶点，误差范数继续使用原始算子面积，差异作为诊断记录在 <code>metadata.json</code> 中。</p>
<p>本次重算四类场差并与保存报告一致，且 VTK 写入/读回的附加场逐值一致；未重新执行原实验的全部来源哈希与守恒验收，不能作为新的正式实验认证。</p>
<p>64 验收来源：<code>checks/handoff64_pair_v1.json</code>；128 诊断来源：<code>checks/aphros128_partial24h_vs_cirrus128_20260924.json</code>；256 来源：<code>runs/reproduction_native256_v2/steady_summary.json</code>。以上相对路径以 <code>D:\\CirrusExperiments\\cirrus-amg</code> 为根。</p></html>'''
    (OUT/'report.html').write_text(content,encoding='utf-8')
    log('Report: '+str(OUT/'report.html'))


if __name__ == '__main__':
    if ARGS.stage in ('all','64'):
        comparison(64)
    if ARGS.stage in ('all','128'):
        comparison(128)
    if ARGS.stage in ('all','256'):
        native256()
    if ARGS.stage in ('all','report'):
        report()
