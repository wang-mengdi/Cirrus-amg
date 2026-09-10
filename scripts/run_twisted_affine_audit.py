"""Run the original Aphros viscous-expression audit without changing its library."""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
from run_twisted_solver import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template',type=Path,required=True)
    parser.add_argument('--build-directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve the existing audit; use a fresh output directory')
    build_path=args.build_directory/'build_manifest.json'
    build=json.loads(build_path.read_text(encoding='utf-8-sig'))
    exe=(args.build_directory/'twisted_affine_audit.exe').resolve(strict=True)
    if build['exit_code'] or sha(exe)!=build['executable_sha256']:
        raise ValueError('Invalid affine-audit build')
    for key in ('source','library'):
        if sha(Path(build[key]))!=build[key+'_sha256']:
            raise ValueError(f'Compiled {key} changed')
    template=args.template.read_text(encoding='utf-8-sig')
    args.output.mkdir(parents=True)
    env=dict(os.environ)
    for key in tuple(env):
        if key.startswith('APHROS_TWISTED_') or key=='APHROS_SIMPLE_DUMP':
            env.pop(key)
    env.update(OMP_NUM_THREADS='1',OMP_WAIT_POLICY='PASSIVE')
    results=[]
    for ny in (16,32,64):
        for offset in (.27,.73):
            case=args.output/f'n{ny}_offset{round(offset*100):02d}'
            case.mkdir()
            config=template
            for key,value in (('bsx',ny*2),('bsy',ny),('bsz',ny),('hypre_periodic_z',1)):
                config,count=re.subn(r'^set int '+key+r' .+$',f'set int {key} {value}',config,flags=re.M)
                if count!=1:
                    raise ValueError(f'Expected exactly one template setting: {key}')
            config+=f'\nset double audit_offset_fraction {offset}\n'
            path=case/'a.conf'
            path.write_text(config)
            manifest={'executable':str(exe),'executable_sha256':sha(exe),
                      'build_manifest_sha256':sha(build_path),'config_sha256':sha(path),
                      'scope':'Read-only affine scalar operator audit; no pipe flow is solved',
                      'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
            (case/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            with (case/'run.log').open('w') as log:
                process=subprocess.run([str(exe),'a.conf'],cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT)
            completion={'exit_code':process.returncode,'executable_unchanged':sha(exe)==manifest['executable_sha256'],
                        'config_unchanged':sha(path)==manifest['config_sha256']}
            (case/'run_completion.json').write_text(json.dumps(completion,indent=2)+'\n')
            if process.returncode or not completion['executable_unchanged'] or not completion['config_unchanged']:
                raise RuntimeError(f'Affine audit did not complete: {case}')
            audit=json.loads((case/'affine_audit.json').read_text())
            if not audit['audit_completed'] or not audit['affine_consistent_before']:
                raise ValueError('Invalid exact affine control')
            results.append({'run':str(case.resolve()),'ny':ny,**audit,
                            'source_sha256':{name:sha(case/name) for name in ('a.conf','affine_audit.json','affine_cells.csv','run_manifest.json','run_completion.json')}})
            print(json.dumps({'ny':ny,'offset':offset,'before':audit['residual_before_max_per_h2'],
                              'after':audit['residual_after_max_per_h2'],
                              'whole':audit['whole_residual_redistribution_max_per_h2']}),flush=True)
    report={'audit_completed':True,'scope':'Original upstream functions on exact affine scalar patches at fixed subcell offsets; not physical grid convergence of the tube',
            'build':build,'template_sha256':sha(args.template),'cases':results,
            'interpretation_limits':'This checks the combined implicit/deferred expression after source redistribution. It neither proves the sole cause of curved-wall error nor changes the Aphros baseline or acceptance gates.'}
    (args.output/'affine_summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
