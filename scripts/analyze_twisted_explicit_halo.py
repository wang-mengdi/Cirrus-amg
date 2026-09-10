"""Attribute the second SIMPLE predictor discrepancy to the upstream residual halo.

Requires the original run, a numerically unchanged diagnostic run, its opt-in
halo-fixed control, and the independent native exp-mode run. This is a discrete
implementation audit, not evidence of steady flow or physical grid convergence.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from run_twisted_solver import sha
from check_twisted_mass import read


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('original','diagnostic','fixed','ours','override-source','override-build','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve prior audit; use a fresh output')
    inputs={}
    def record(path):
        path=path.resolve();inputs[str(path)]=sha(path);return path
    def load(path):return json.loads(record(path).read_text(encoding='utf-8-sig'))
    runs=[]
    for root in (args.original,args.diagnostic,args.fixed):
        manifest=load(root/'run_manifest.json');completion=load(root/'run_completion.json')
        if completion['exit_code']!=0:raise ValueError('Incomplete baseline')
        if sha(record(root/'a.conf'))!=manifest['config_sha256']:raise ValueError('Changed config')
        if sha(record(Path(manifest['executable'])))!=manifest['executable_sha256']:raise ValueError('Changed executable')
        runs.append(manifest)
    if len({r['config_sha256'] for r in runs})!=1:raise ValueError('Different baseline settings')
    key='APHROS_TWISTED_FIX_EXPLICIT_RESIDUAL_HALO'
    if any(r['environment'].get(key) is not None for r in runs[:2]) or runs[2]['environment'].get(key)!='1':
        raise ValueError('Expected original/unfixed diagnostic/fixed controls')
    if runs[1]['executable_sha256']!=runs[2]['executable_sha256']:raise ValueError('Control binary differs')
    if runs[1]['environment'].get('APHROS_TWISTED_EXPLICIT_RESIDUAL_DUMP')!='explicit_residual':
        raise ValueError('Missing diagnostic selection')
    build=load(args.override_build/'build_manifest.json')
    override=load(args.override_source/'override_manifest.json')
    for paths in (override['inputs'],override['generated']):
        for path,expected in paths.items():
            if sha(record(Path(path)))!=expected:raise ValueError('Changed diagnostic source')
    for path,digest in ((build['source'],build['source_sha256']),(build['library'],build['library_sha256'])):
        if sha(record(Path(path)))!=digest:raise ValueError('Changed baseline build input')
    if build['executable_sha256']!=runs[1]['executable_sha256']:raise ValueError('Wrong diagnostic build')
    if load(args.ours/'run_completion.json')['exit_code']!=0:raise ValueError('Incomplete native run')
    native_manifest=load(args.ours/'run_manifest.json')
    config=load(args.ours/'case.json')
    baseline_case=load(args.original/'case_manifest.json')
    for name in ('executable','config'):
        if sha(record(Path(native_manifest[name])))!=native_manifest[name+'_sha256']:
            raise ValueError('Changed native executable/configuration')
    for path,expected in native_manifest['geometry_input_sha256'].items():
        if sha(record(Path(path)))!=expected:raise ValueError('Changed native geometry')
    if config.get('momentum_mode')!='exp' or config.get('adaptive'):raise ValueError('Expected uniform exp-mode audit')
    def numerical_csv(root):
        return {p.name:p for p in root.glob('*.csv') if p.name!='driver_memory.csv' and not p.name.startswith('explicit_residual_')}
    csv=[numerical_csv(p) for p in (args.original,args.diagnostic)]
    if not csv[0] or csv[0].keys()!=csv[1].keys():raise ValueError('Diagnostic CSV coverage differs')
    identical=all(sha(record(csv[0][k]))==sha(record(csv[1][k])) for k in csv[0])
    key_of=lambda r:tuple(float(r[d]) for d in ('x','y','z'))
    def index(path):
        rows=read(record(path));result={key_of(r):r for r in rows}
        if len(result)!=len(rows):raise ValueError('Duplicate cell location')
        return result
    stage=args.ours/'step_0001'
    ours=index(stage/'iter_2/cells.csv');old=index(stage/'iter_1/cells.csv')
    base=index(args.original/'simple_1_b0_cells.csv')
    if not set(ours)==set(old)==set(base):raise ValueError('Cell coverage mismatch')
    matrix=read(record(stage/'iter_2/momentum_matrix.csv'))
    if any(r['row']!=r['column'] for r in matrix) or len(matrix)!=len(ours):
        raise ValueError('Exp momentum matrix should contain only its time diagonal')
    by_id={int(r['id']):r for r in ours.values()}
    if set(int(r['row']) for r in matrix)!=set(by_id):raise ValueError('Matrix coverage mismatch')
    diagonal={int(r['row']):r['value'] for r in matrix}
    # Matrix relaxation and the exported reciprocal-diagonal path can differ
    # by one floating-point rounding. Use the actual matrix for delta RHS.
    diag_error=max(abs(diagonal[i]-r['aP'])/abs(r['aP']) for i,r in by_id.items())
    if diag_error>8*np.finfo(float).eps:raise ValueError('Dumped momentum diagonal differs beyond roundoff')
    components={}
    for d,component in enumerate('uvw'):
        diag=index(args.diagnostic/f'explicit_residual_{3+d}_b0.csv')
        if set(diag)!=set(ours):raise ValueError('Residual diagnostic coverage mismatch')
        keys=sorted(ours)
        observed=np.array([ours[k]['rhs_'+component]-diagonal[int(ours[k]['id'])]*old[k][component]-base[k]['delta_rhs_'+component] for k in keys])
        explained=np.array([diag[k]['redistributed_original']-diag[k]['redistributed_periodic_halo'] for k in keys])
        raw=np.array([diag[k]['raw_residual'] for k in keys])
        after=np.array([diag[k]['redistributed_periodic_halo'] for k in keys])
        before=np.array([diag[k]['redistributed_original'] for k in keys])
        scale=max(float(np.sum(abs(raw))),1e-30)
        changed=[k for k,v in zip(keys,explained) if abs(v)>1e-20]
        h=float(next(iter(ours.values()))['h']);length=baseline_case['spec']['extent'][0]
        source_max=float(np.max(abs(observed)));unexplained=float(np.max(abs(observed-explained)))
        balance=float(abs(np.sum(after)-np.sum(raw))/scale)
        components[component]={'observed_delta_rhs_max':source_max,'unexplained_max':unexplained,
            'affected_cells':len(changed),'affected_x':sorted(set(k[0] for k in changed)),
            'only_periodic_adjacent_cells':all(min(k[0],length-k[0])<h for k in changed),
            'raw_sum':float(raw.sum()),'unfilled_redistribution_sum':float(before.sum()),
            'halo_filled_redistribution_sum':float(after.sum()),'filled_global_balance_relative':balance,
            'passed':bool(source_max>1e-12 and unexplained<1e-18 and balance<1e-12 and changed and
                          all(min(k[0],length-k[0])<h for k in changed))}
    checks={'diagnostic_numerical_csv_identical':identical,**{d:r['passed'] for d,r in components.items()}}
    result={'passed':all(checks.values()),'scope':__doc__,'checks':checks,'components':components,
            'diagnostic_csv_files':len(csv[0]),'baseline_runtime':runs,'native_runtime':native_manifest,
            'momentum_diagonal_max_relative_roundoff':diag_error,
            'override_build':build,'override_source':override,'source_sha256':inputs}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':result['passed'],'checks':checks,'components':components},indent=2))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
