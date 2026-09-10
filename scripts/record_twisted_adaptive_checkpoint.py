"""Archive the actual adaptive evidence, including failed refinement checks."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import numpy
import scipy
import vtk


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    repo=Path(__file__).resolve().parents[1];out=repo/'validation/twisted/results/adaptive_checkpoint'
    out.mkdir(parents=True,exist_ok=True)
    paths={
        'adaptive64_alignment':('compare_adaptive64_aphros.json',True),
        'uniform64_alignment':('compare_uniform64_aphros.json',True),
        'operators128':('operators128/embedded_operator_checks.json',True),
        'aphros_interpolation128':('interpolation_policy128.json',True),
        'amg_backend_n16':('backend_n16_amgv5_vs_conjugate.json',True),
        'direct_backend_n32':('backend_n32_direct_vs_conjugate.json',None),
        'grid64_to_adaptive128':('refinement64_128/refinement.json',False),
        'adaptive128_solution':('ours_adaptive128_amg98p03_v2/metrics.json',None),
        'adaptive128_visualization':('ours_adaptive128_amg98p03_v2/paraview_export.json',None),
        'straight_quick':('straight_regression_amg/suite_results.json',None)}
    evidence={}
    for name,(relative,expected) in paths.items():
        path=repo/'output/twisted'/relative;result=json.loads(path.read_text(encoding='utf-8'))
        if expected is not None and result['passed']!=expected:raise ValueError(f'Unexpected result for {name}')
        if name=='straight_quick' and not (result['quick_pass'] and result['strict_current_binary_pass']):
            raise ValueError('Current-binary straight regression failed')
        if name=='adaptive128_solution' and not result['converged']:raise ValueError('Adaptive128 not converged')
        if name=='adaptive128_visualization' and not result['geometry_verified']:raise ValueError('Visualization geometry not verified')
        (out/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        evidence[name]={'source':str(path),'sha256':sha(path),'result':expected}
    exe=repo/'build/windows/x64/release/simple_channel.exe'
    quick=json.loads((out/'straight_quick.json').read_text())
    if quick['executable_sha256']!=sha(exe):raise ValueError('Executable changed after quick regression')
    files=list((repo/'simple').glob('*'))+list((repo/'scripts').glob('*twisted*.py'))
    files+=list((repo/'validation').glob('test_twisted*.py'))+list((repo/'validation/aphros').glob('twisted*'))
    files+=[repo/'xmake.lua',repo/'validation/aphros/prepare_twisted.py',repo/'validation/aphros/build_baseline.ps1']
    base=Path('D:/Dropbox/Agent-simulation/twisted-baseline/aphros')
    record={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'scope':'Actual native-octree cut-cell alignment checkpoint; fine-grid and near-wall accuracy goal remains open',
            'completed_goal':False,
            'parent_commit_at_capture':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
            'cirrus_executable_sha256':sha(exe),
            'source_sha256':{str(p.relative_to(repo)):sha(p) for p in sorted(set(files)) if p.is_file()},
            'aphros_revision':'b60ce3da52c19935fa24c778f62f02141eaf7f80',
            'amgcl_revision':'f4614a7e9ccfe716c4c96df75dc349157229609a',
            'aphros_executable_hashes_observed_at_capture':{name:sha(base/'src'/name) for name in ('main.exe','main_direct.exe','main_amg_v3.exe','main_amg_v5.exe')},
            'aphros_changes':['prescribed geometry and read-only diagnostic dumps','explicit uninitialized wall-flux fix',
                              'optional linear backend with original-equation residual checks; v5 roundoff nullspace compatibility projection'],
            'versions':{'numpy':numpy.__version__,'scipy':scipy.__version__,'vtk':vtk.vtkVersion.GetVTKVersion()},
            'evidence':evidence,
            'remaining':['complete independent Aphros n128 reference and compare actual adaptive result',
                         'separate uniform refinement error from octree interior coarsening error',
                         'resolve near-wall grid convergence and surface sampling sensitivity',
                         'implement and independently validate inertial curved flow']}
    (out/'checkpoint.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(out/'checkpoint.json')


if __name__=='__main__':main()
