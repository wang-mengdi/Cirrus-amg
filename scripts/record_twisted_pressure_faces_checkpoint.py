"""Archive read-only reference face capture, failed pressure tightening, and completed GPU128 step evidence."""
import argparse
import datetime
import gzip
import json
import shutil
from pathlib import Path
from run_twisted_solver import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();repo=Path(__file__).resolve().parents[1]
    runs=repo/'output/twisted';base=Path('D:/Dropbox/Agent-simulation/twisted-baseline')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);files=[];retained=[]
    def save(p,dest,scope):
        before=sha(p);compress=p.stat().st_size>2_000_000 and p.suffix in ('.csv','.log','.json')
        target=out/(str(dest)+('.gz' if compress else ''));target.parent.mkdir(parents=True,exist_ok=True)
        with p.open('rb') as src,target.open('xb') as dst:
            if compress:
                with gzip.GzipFile(filename='',fileobj=dst,mode='wb',mtime=0) as z:shutil.copyfileobj(src,z,1024*1024)
            else:shutil.copyfileobj(src,dst,1024*1024)
        if sha(p)!=before:raise ValueError('Archiving source changed')
        files.append({'path':target.relative_to(out).as_posix(),'source':str(p.resolve()),
                      'source_sha256':before,'sha256':sha(target),'size':target.stat().st_size,'gzip':compress,'scope':scope})
    def keep(p,scope):
        retained.append({'source':str(p.resolve()),'source_sha256':sha(p),'size':p.stat().st_size,'scope':scope})
    def tree(root,dest,scope,large=False):
        for p in sorted(root.rglob('*')):
            if not p.is_file():continue
            if p.suffix in ('.exe','.bin','.vtu','.vtp') or (large and p.suffix=='.csv'):
                keep(p,scope+'; retained runtime bytes')
            elif p.suffix in ('.json','.csv','.log','.txt','.py','.ps1','.conf','.cpp','.h','.ipp'):
                save(p,Path(dest)/p.relative_to(root),scope)
    # Current native source integrity, independently of the older GPU128 binary.
    tree(runs/'original_rhs_build_v1','build','current native compiled-source identity')
    tree(base/'projection_faces_build_v1','reference_build','actual immutable reference build and original-source override hashes')
    for suffix in ('on','off'):
        root=base/f'navier_stokes_proj_n16faces_capture_{suffix}_v1'
        if json.loads((root/'run_completion.json').read_text(encoding='utf-8-sig'))['exit_code']:
            raise ValueError('Incomplete reference capture test')
        tree(root,Path('reference16')/suffix,'complete eight physical steps; no grid/steady accuracy claim')
    stems=('aphros_faces_n16_on_off_pair_v1','aphros_faces_n16_original_pair_v1',
           'aphros_faces_n16_diagnostic_v2','original_rhs_n64_steady_step1_pair_v1')
    for stem in stems:
        report=json.loads((runs/(stem+'.json')).read_text())
        if not report.get('passed',report.get('diagnostic_consistent',False)):raise ValueError('Required check failed')
        for ext in ('.json','.log'):
            p=runs/(stem+ext)
            if p.exists():save(p,Path('reports')/p.name,'inspect report scope; original_rhs64 is a first-step prefix only')
    for stem in ('aphros64_volume_diff8_step1_diagnostic_v1','aphros64_volume_diff8_step1_snapshot_v1'):
        for ext in ('.json','.log'):
            p=runs/(stem+ext)
            if p.exists():save(p,Path('reports')/p.name,'reference64 prefix; actual mass failed')
    tree(base/'navier_stokes_proj_n64_steady_volume_pressure_diff8_step1_checkpoint_v1',
         'reference64_step1','verified immutable first-step prefix; not complete steady run',large=True)
    tree(base/'navier_stokes_proj_n64_volume_diff8_pressure2e15_step1_v1',
         'failed_pressure_tightening64','failed at original residual check; no complete physical step',large=True)
    native=runs/'ours_proj128_full_viscosity_gpu_v1'
    completion=json.loads((native/'run_completion.json').read_text())
    if completion['exit_code'] or not all(completion[k] for k in ('executable_unchanged','config_unchanged','geometry_unchanged')):
        raise ValueError('GPU128 configured run incomplete')
    if not json.loads((native/'step_0001/paraview_step_readback.json').read_text())['passed']:
        raise ValueError('GPU128 ParaView readback not complete')
    tree(native,'native128_step1','one complete transient step only; not a physically steady result',large=True)
    tree(runs/'full_viscosity_gpu_build_v1','native128_build','exact binary and compiled-source snapshots used by GPU128')
    save(runs/'config_ours_proj128_full_viscosity_gpu_v1.json','configs/native128.json','completed transient step configuration')
    old=runs/'ours_proj64_steady_full_viscosity_gpu_v1'
    for name in ('run_interruption.json','run_completion.json','run_manifest.json','case.json'):
        save(old/name,Path('retired_native64')/name,'retired after a verified replacement first step; not full steady success')
    for name in ('a.conf','case_manifest.json','run_manifest.json'):
        save(base/'navier_stokes_proj_n64_faces_capture_diff8_v1'/name,Path('pending_capture64')/name,
             'immutable input only; fine-grid capture outcome pending')
    for name in ('check_aphros_pressure_faces.py','check_aphros_capture_pair.py',
                 'run_twisted_baseline.ps1','check_aphros_pressure_snapshot.py','check_twisted_mass.py',
                 'record_twisted_pressure_faces_checkpoint.py','check_twisted_checkpoint.py',
                 'export_twisted_paraview.py','check_twisted_paraview_step.py'):
        save(repo/'scripts'/name,Path('checkers')/name,'reproduction source')
    for name in ('prepare_twisted_linear_cache.py','build_twisted_driver.ps1','twisted_pressure_face_snapshot.h','LICENSE.aphros','LICENSE.amgcl'):
        save(repo/'validation/aphros'/name,Path('reproduction')/name,'build helper, capture header, or upstream license')
    save(runs/'check_original_rhs_steady_prefix_v1.py','checkers/check_original_rhs_steady_prefix_v1.py','actual first-step prefix checker')
    for name in ('build_projection_faces_v1.log','full_viscosity_n128_paraview_export_v1.log','full_viscosity_n128_paraview_readback_v1.log'):
        save(runs/name,Path('logs')/name,'actual completed build or visualization verification log')
    receipt={'scope':__doc__,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'goal_complete':False,'remaining':'Independent adaptive steady matching plus physical grid/time convergence; capture64 is pending.',
             'files':files,'retained_runtime_files':retained}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'files':len(files),'retained':len(retained),'bytes':sum(f['size'] for f in files)}),flush=True)


if __name__=='__main__':main()
