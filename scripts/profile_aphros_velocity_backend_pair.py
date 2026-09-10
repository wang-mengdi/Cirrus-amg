"""Report observed time/memory and actual backend selection for a completed mixed-backend probe.

Different binaries and concurrent workloads preclude an exclusive speedup claim.
The strict outer-history comparison remains separate from numerical field checks.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from profile_aphros_backend_pair import profile
from run_twisted_solver import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('reference','candidate','field-pair','default-control','output'):
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve previous resource reports')
    pair=json.loads(a.field_pair.read_text());control=json.loads(a.default_control.read_text())
    if not control['passed'] or not all(control['bitwise_equal_outputs'].values()):raise ValueError('Default backend regression failed')
    if not pair['checks']['both_mass_checks'] or not pair['checks']['same_discrete_fields']:
        raise ValueError('Completed mixed-backend flow fields or actual mass failed')
    hashes={str(Path(__file__).resolve()):sha(Path(__file__)),str(a.field_pair.resolve()):sha(a.field_pair),
            str(a.default_control.resolve()):sha(a.default_control),
            str(Path(__file__).with_name('profile_aphros_backend_pair.py')):sha(Path(__file__).with_name('profile_aphros_backend_pair.py'))}
    runtimes=[];runs=[]
    for root in (a.reference.resolve(),a.candidate.resolve()):
        runtime,run=profile(root,hashes);runtimes.append(runtime);runs.append(run)
    envs=[r['environment'] for r in runtimes]
    if any(e.get('APHROS_TWISTED_AMG')!='1' for e in envs) or envs[0].get('APHROS_TWISTED_DIRECT_VELOCITY') is not None or envs[1].get('APHROS_TWISTED_DIRECT_VELOCITY')!='1':
        raise ValueError('Expected AMG control and mixed-backend candidate')
    clean=lambda e:{k:v for k,v in e.items() if k!='APHROS_TWISTED_DIRECT_VELOCITY'}
    if clean(envs[0])!=clean(envs[1]):raise ValueError('Other environment controls differ')
    for key in ('config_sha256','geometry_state_sha256'):
        if runtimes[0][key]!=runtimes[1][key]:raise ValueError('Different '+key)
    log=(a.candidate/'run.log').read_text()
    selections=Counter(re.findall(r'factor-cache name=(\w+)[^\r\n]* backend=(\w+)',log))
    expected={('pressure','amg'),('velocity0','direct'),('velocity1','direct'),('velocity2','direct')}
    if set(selections)!=expected:raise ValueError('Actual backend selection differs')
    if sum(selections.values())!=sum(runs[1]['linear_calls_by_system'].values()):raise ValueError('Backend trace call counts differ')
    if any(sha(Path(path))!=digest for path,digest in hashes.items()):raise ValueError('Resource inputs changed')
    result={'scope':__doc__,'actual_backend_selection_verified':True,
            'backend_calls':[{'system':s,'backend':b,'calls':n} for (s,b),n in sorted(selections.items())],
            'strict_field_pair_passed':pair['passed'],'numerical_fields_and_mass_passed':True,
            'default_backend_bitwise_control_passed':True,'runs':runs,
            'source_sha256':hashes,'goal_complete':False}
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}))


if __name__=='__main__':main()
