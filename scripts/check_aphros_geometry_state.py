"""Verify captured and loaded geometry against original Aphros geometry dumps."""
import argparse
import json
from pathlib import Path
import numpy as np
from check_twisted_time_pair import read
from run_twisted_solver import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',type=Path,required=True);p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--reload',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-derived-wall-roundoff',action='store_true',help='Only allow up to four double epsilons in recomputed wall display vertices; raw state must match exactly')
    args=p.parse_args();roots=[args.original.resolve(),args.capture.resolve()]
    if args.reload:roots.append(args.reload.resolve())
    if args.allow_derived_wall_roundoff and not args.reload:raise ValueError('Derived-vertex check requires a raw-state round trip')
    if args.output.exists():raise ValueError('Preserve previous geometry checks')
    hashes={};checks={}
    for root in roots[1:]:
        completion=json.loads((root/'run_completion.json').read_text());manifest=json.loads((root/'run_manifest.json').read_text())
        if completion['exit_code'] or not completion['source_unchanged'] or not completion['no_flow_trajectory']:
            raise ValueError('Geometry process did not complete as a geometry-only run')
        if 'End of geometry-only snapshot' not in (root/'run.log').read_text():raise ValueError('Missing geometry-only completion marker')
        for name,value in manifest['source_sha256'].items():
            if sha(Path(name))!=value:raise ValueError('Geometry run input changed')
        if sha(root/'geometry.bin')!=completion['geometry_sha256']:raise ValueError('Geometry payload changed')
        for name in ('run_completion.json','run_manifest.json','case_manifest.json','geometry.bin'):
            hashes[str(root/name)]=sha(root/name)
    for name in ('tube_b0_geometry_cells.csv','tube_b0_geometry_faces.csv','tube_b0_geometry_walls.csv','tube_b0_geometry_polygons.csv'):
        arrays=[read(root/name) for root in roots]
        for root in roots:hashes[str(root/name)]=sha(root/name)
        same=all(a.dtype.names==arrays[0].dtype.names and a.shape==arrays[0].shape for a in arrays)
        if same:same=all(np.array_equal(arrays[0][field],a[field]) for a in arrays[1:] for field in a.dtype.names)
        item={'all_numeric_values_equal':same,'all_csv_bytes_equal':len({sha(root/name) for root in roots})==1,'passed':same}
        if name=='tube_b0_geometry_polygons.csv' and args.allow_derived_wall_roundoff and not same:
            a=arrays[0];h=float(read(roots[0]/'tube_b0_geometry_cells.csv')['h'][0]);maximum=0.;allowed=True
            for b in arrays[1:]:
                if b.shape!=a.shape or b.dtype.names!=a.dtype.names:allowed=False;break
                for field in ('axis','i','j','k','vertex'):
                    allowed=allowed and np.array_equal(a[field],b[field])
                wall=a['axis']==3
                for field in ('x','y','z'):
                    delta=abs(a[field]-b[field]);maximum=max(maximum,float(delta.max()))
                    allowed=allowed and np.all(delta[~wall]==0) and np.all(delta[wall]<=4*np.finfo(float).eps*np.maximum(abs(a[field][wall]),h))
            item.update(passed=bool(allowed),derived_wall_vertex_roundoff_only=bool(allowed),maximum_coordinate_difference=maximum)
        checks[name]=item
    if args.reload:checks['complete_raw_state_bytes_equal']=sha(roots[1]/'geometry.bin')==sha(roots[2]/'geometry.bin')
    passed=all(c if isinstance(c,bool) else c['passed'] for c in checks.values())
    result={'passed':passed,'scope':__doc__+' No flow solution claim.', 'checks':checks,'source_sha256':hashes,
            'checker_sha256':sha(Path(__file__))}
    if any(sha(Path(name))!=value for name,value in hashes.items()):raise ValueError('Geometry inputs changed during inspection')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
