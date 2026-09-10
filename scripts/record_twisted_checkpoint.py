"""Record tested curved-tube checkpoint evidence without claiming full acceptance."""
from pathlib import Path
import datetime
import hashlib
import json
import numpy as np
from compare_twisted import ordered, read, vector


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def main():
    repo=Path(__file__).resolve().parents[1]
    out=repo/'validation/twisted/results';out.mkdir(parents=True,exist_ok=True)
    base=Path('D:/Dropbox/Agent-simulation/twisted-baseline')
    evidence={}
    for ny in (16,32):
        path=repo/f'output/twisted/compare_n{ny}_checked.json'
        report=json.loads(path.read_text(encoding='utf-8'))
        if not report['passed']:raise ValueError(f'Failed comparison {ny}')
        (out/f'compare_n{ny}.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        evidence[f'comparison_n{ny}']={'path':str(path),'sha256':sha(path)}
    quick=repo/'output/twisted/straight_regression/suite_results.json'
    checks=json.loads(quick.read_text(encoding='utf-8'))
    if not checks['quick_pass'] or not checks['strict_current_binary_pass']:raise ValueError('Quick regression failed')
    executable=repo/'build/windows/x64/release/simple_channel.exe'
    if checks['executable_sha256']!=sha(executable):raise ValueError('Quick regression executable changed')
    evidence['straight_quick']={'path':str(quick),'sha256':sha(quick),'cases':checks['requested_cases'],
                                'quick_pass':True,'full_pass':False}
    paths=[base/f'stokes_n16_{name}/simple_final_b0_cells.csv' for name in ('initialized','direct')]
    a,b=[ordered(read(path)) for path in paths]
    p=a['p']-b['p'];p-=np.average(p,weights=a['volume'])
    backend={'velocity_max_abs':float(np.max(np.abs(vector(a,['u','v','w'])-vector(b,['u','v','w'])))),
             'pressure_gauge_aligned_max_abs':float(np.max(np.abs(p))),
             'sources':{str(path):sha(path) for path in paths}}
    if backend['velocity_max_abs']>1e-10 or backend['pressure_gauge_aligned_max_abs']>1e-10:
        raise ValueError('Direct backend differs from conjugate reference')
    sources=list((repo/'simple').glob('*.cpp'))+list((repo/'simple').glob('*.h'))+list((repo/'simple').glob('*.cu'))
    sources+=list((repo/'validation/aphros').glob('twisted*'))+[repo/'validation/aphros/prepare_twisted.py',repo/'validation/aphros/build_baseline.ps1']
    report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'scope':'Same-grid shared-geometry 3D Stokes consistency checkpoint; full curved-tube goal remains open',
            'completed_goal':False,'cirrus_executable_sha256':sha(executable),
            'source_sha256':{str(path.relative_to(repo)):sha(path) for path in sorted(sources)},
            'aphros_revision':'b60ce3da52c19935fa24c778f62f02141eaf7f80',
            'aphros_changes':['diagnostic dumps','analytic tube initialization','explicit embedded wall flux initialization fix',
                              'optional direct linear backend using assembled matrices and checking original residuals'],
            'aphros_executables':{name:sha(base/'aphros/src'/name) for name in ('main.exe','main_direct.exe')},
            'evidence':evidence,'direct_backend_vs_conjugate_n16':backend,
            'remaining':['wall-normal and grid-refinement accuracy','actual fluid coarse/fine interface validation',
                         'finer Aphros reference completion','inertial curved flow validation']}
    (out/'checkpoint.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(out/'checkpoint.json')


if __name__=='__main__':main()
