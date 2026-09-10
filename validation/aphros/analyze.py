import csv
import json
import math
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
H, NU, G = .125, .01, 1.

@lru_cache(None)
def rectangular_velocity(y, z, height=H, width=H, max_odd=255):
    """Fourier solution of nu*(u_yy+u_zz)=-G with four no-slip walls.

    The parabolic particular solution is evaluated in closed form. Only
    exponentially decaying side-wall corrections are summed, avoiding cosh
    overflow and slow cancellation near the first/last cell center.
    """
    if y == 0 or y == height or z == 0 or z == width:
        return 0.
    correction = math.fsum(
        math.sin(n*math.pi*y/height)/n**3 *
        (math.exp(-n*math.pi*z/height)+math.exp(-n*math.pi*(width-z)/height)) /
        (1+math.exp(-n*math.pi*width/height))
        for n in range(1,max_odd+1,2))
    return G*y*(height-y)/(2*NU) - 4*G*height**2/(NU*math.pi**3)*correction

def rectangular_flow(height=H, width=H, max_odd=4095):
    correction = math.fsum(math.tanh(n*math.pi*width/(2*height))/n**5
                           for n in range(1,max_odd+1,2))
    return G*width*height**3/(12*NU)*(1-192*height/(math.pi**5*width)*correction)

def check_rectangular_reference():
    symmetry=[]
    residual=[]
    epsilon=H*.001
    for y,z in ((H*.2,H*.3),(H*.45,H*.5),(H*.7,H*.6)):
        u=rectangular_velocity(y,z)
        symmetry.append(abs(u-rectangular_velocity(z,y)))
        curvature=0.
        for axis in (0,1):
            def shifted(offset):
                point=[y,z]
                point[axis]+=offset*epsilon
                return rectangular_velocity(*point)
            curvature+=(-shifted(2)+16*shifted(1)-30*u+16*shifted(-1)-shifted(-2))/(12*epsilon**2)
        residual.append(abs(NU*curvature+G))
    assert max(symmetry)<1e-12, symmetry
    assert max(residual)<1e-7, residual
    return dict(square_symmetry_linf=max(symmetry),
        pde_fourth_order_difference_check_linf=max(residual),
        flow_series_truncation_absolute=abs(rectangular_flow()-rectangular_flow(max_odd=2047)),
        exact_square_max_velocity=rectangular_velocity(H/2,H/2),
        exact_square_mean_velocity=rectangular_flow()/H**2,
        exact_square_volume_flow=rectangular_flow())

reference_check=check_rectangular_reference()
(ROOT/'rectangular_reference_check.json').write_text(json.dumps(reference_check,indent=2))

def rows(path):
    with path.open() as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]

reports = []
for case,ny,dim in [(f'periodic_n{n}',n,2) for n in (8,16,32)] + [('periodic_3d_n8',8,3),('perturb_3d_n8',8,3),('duct_3d_n8',8,3),('duct_3d_n16',16,3)]:
    folder = ROOT / case
    path = folder / 'simple_final_b0_cells.csv'
    if not path.exists():
        continue
    cells = rows(path)
    h = H/ny
    is_duct = case.startswith('duct_')
    for r in cells:
        assert all(math.isfinite(v) for v in r.values()), path
        r['u_exact'] = rectangular_velocity(r['y'],r['z']) if is_duct else G/(2*NU)*r['y']*(H-r['y'])
    profile = [r for r in cells if r['x'] == cells[0]['x'] and r['z'] == cells[0]['z']]
    profile.sort(key=lambda r: r['y'])
    l2 = math.sqrt(sum((r['u']-r['u_exact'])**2 for r in cells)/sum(r['u_exact']**2 for r in cells))
    mean = sum(r['u'] for r in cells)/len(cells)
    section=[r for r in cells if r['x']==cells[0]['x']]
    section_values={(round(r['y']/h-.5),round(r['z']/h-.5) if dim==3 else 0):r['u'] for r in section}
    residual = []
    for (j,k), u in section_values.items():
        curvature=0
        for axis in range(2 if is_duct else 1):
            coords=[j,k]
            index=coords[axis]
            def value(offset):
                q=coords.copy(); q[axis]+=offset
                return section_values[tuple(q)]
            grad_left=(u-value(-1))/h if index else (9*u-value(1))/(3*h)
            grad_right=(value(1)-u)/h if index+1<ny else (-9*u+value(-1))/(3*h)
            curvature+=(grad_right-grad_left)/h
        residual.append(NU*curvature+G)
    faces = rows(folder/'simple_final_b0_faces.csv')
    def facekey(r):
        return tuple(round(r[k]/h-(0 if int(r['axis'])==d else .5)) for d,k in enumerate(('x','y','z')[:dim]))
    face_maps=[{facekey(r):r['flux'] for r in faces if r['axis']==d} for d in range(dim)]
    div = []
    for r in cells:
        ijk=tuple(round(r[k]/h-.5) for k in ('x','y','z')[:dim])
        divergence=0
        for d in range(dim):
            right=list(ijk)
            right[d]+=1
            divergence+=face_maps[d][tuple(right)]-face_maps[d][ijk]
        div.append(divergence/r['volume'])
    firstcells=rows(folder/'simple_0_b0_cells.csv')
    firstfaces=rows(folder/'simple_0_b0_faces.csv')
    first_maps=[{facekey(r):r['corrected_flux'] for r in firstfaces if r['axis']==d} for d in range(dim)]
    first_div=[]
    for r in firstcells:
        ijk=tuple(round(r[k]/h-.5) for k in ('x','y','z')[:dim])
        divergence=0
        for d in range(dim):
            right=list(ijk); right[d]+=1
            divergence+=first_maps[d][tuple(right)]-first_maps[d][ijk]
        first_div.append(divergence/r['volume'])
    mean_exact=rectangular_flow()/H**2 if is_duct else G*H*H/(12*NU)
    midplane=min(abs(r['x']-.5) for r in faces if r['axis']==0)
    xface=min(r['x'] for r in faces if r['axis']==0 and abs(abs(r['x']-.5)-midplane)<1e-12)
    flux_through_section=sum(r['flux'] for r in faces if r['axis']==0 and r['x']==xface)
    report = dict(case=case, nx=8*ny, ny=ny, dimension=dim, four_wall_duct=is_duct, u_l2_relative=l2,
        u_linf_absolute=max(abs(r['u']-r['u_exact']) for r in cells),
        umean=mean, umean_exact=mean_exact,
        flow_per_depth=mean*H, flow_per_depth_exact=mean_exact*H,
        section_flux=flux_through_section,
        section_flux_exact=mean_exact*H*(H if dim==3 else 1),
        flow_relative_error=mean/mean_exact-1,
        momentum_residual_linf_per_volume=max(map(abs,residual)),
        divergence_linf=max(map(abs,div)), v_linf=max(abs(r['v']) for r in cells),
        pressure_range=max(r['p'] for r in cells)-min(r['p'] for r in cells),
        first_iteration_predicted_divergence_linf=max(abs(r['pcorr_rhs']/r['volume']) for r in firstcells),
        first_iteration_corrected_divergence_linf=max(map(abs,first_div)),
        first_iteration_pcorr_linf=max(abs(r['pcorr']) for r in firstcells))
    if is_duct:
        report.update(
            analytic_max_odd=255,
            analytic_velocity_truncation_linf=max(abs(r['u_exact']-rectangular_velocity(r['y'],r['z'],max_odd=511)) for r in section),
            sampled_analytic_flow_relative_error=sum(r['u_exact'] for r in section)/len(section)/mean_exact-1,
            axial_u_variation_linf=max(abs(r['u']-section_values[round(r['y']/h-.5),round(r['z']/h-.5)]) for r in cells),
            w_linf=max(abs(r['w']) for r in cells))
        with (folder/'section.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['y','z','u','u_exact'])
            writer.writeheader()
            writer.writerows({key:r[key] for key in writer.fieldnames} for r in section)
    reports.append(report)
    with (folder/'profile.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=['y','u','u_exact'])
        writer.writeheader()
        writer.writerows({k:r[k] for k in writer.fieldnames} for r in profile)
print(json.dumps(reports, indent=2))
(ROOT/'baseline_results.json').write_text(json.dumps(reports,indent=2))
