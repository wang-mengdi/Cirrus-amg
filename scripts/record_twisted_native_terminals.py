"""Verify and preserve the completed steady64 trajectory and failed steady128 attempt."""
import argparse,csv,datetime,json,math,shutil
from pathlib import Path
import numpy as np
from check_twisted_mass import read
from run_twisted_solver import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();repo=Path(__file__).resolve().parents[1];runs=repo/'output/twisted'
    native=runs/'ours_proj64_steady_original_rhs_v1';failed=runs/'ours_proj128_steady_original_rhs_v1'
    completion=json.loads((native/'run_completion.json').read_text());summary=json.loads((native/'transient_summary.json').read_text())
    config=json.loads((native/'case.json').read_text());count=config['time_steps']
    if completion['exit_code'] or not all(completion[k] for k in ('executable_unchanged','config_unchanged','geometry_unchanged')):
        raise ValueError('Native64 did not finish with unchanged inputs')
    if not summary['converged'] or not summary['steady_converged'] or summary['steps_completed']!=count:
        raise ValueError('Native64 trajectory is incomplete or not steady')
    with (native/'time_history.csv').open() as stream:times=list(csv.DictReader(stream))
    if len(times)!=count:raise ValueError('Missing physical steps')
    field_steps=[];metric_paths=[]
    for step,row in enumerate(times,1):
        p=native/f'step_{step:04d}'/'metrics.json';m=json.loads(p.read_text());metric_paths.append(p)
        if int(row['step'])!=step or row['inner_converged']!='true' or not m['converged']:
            raise ValueError('Unconverged physical step')
        if not math.isclose(float(row['time']),step*config['time_step'],rel_tol=1e-12,abs_tol=0):
            raise ValueError('Physical time sequence differs')
        if m['iterations']!=int(row['inner_iterations']):raise ValueError('Step diagnostics differ from trajectory')
        if m['field_output_written']:field_steps.append(step)
    if field_steps!=summary['field_output_steps']:raise ValueError('Field output schedule differs')
    last=native/f'step_{count:04d}';m=json.loads((last/'metrics.json').read_text())
    if m['steady_momentum_relative_l2']>=1e-8 or m['temporal_acceleration_relative_l2']>=1e-8:
        raise ValueError('Physical steady criteria failed')
    pv=json.loads((last/'paraview_step_readback.json').read_text())
    if not pv['passed']:raise ValueError('ParaView readback did not pass')
    cells=read(last/'solution.csv');faces=read(last/'mesh_faces.csv');values=read(last/'flux.csv');sections=read(last/'sections.csv')
    if not np.array_equal(cells['id'],np.arange(len(cells))) or not np.array_equal(faces['id'],values['id']):
        raise ValueError('Mismatched native IDs')
    q=values['flux'];inner=faces['neighbor']>=0;net=np.zeros(len(cells))
    if np.any(q[~inner]!=0):raise ValueError('Nonzero stationary-wall flux')
    np.add.at(net,faces['owner'].astype(int),q);np.add.at(net,faces['neighbor'][inner].astype(int),-q[inner])
    # The benchmark's box height comes from the executed geometry, not a typed velocity scale.
    geometry=json.loads(Path(config['embedded_geometry']).read_text());height=geometry['extent'][1]
    speed=np.sqrt(sum(cells[k]**2 for k in 'uvw')).max();through=np.mean(sections['volume_flux'])
    mass={'relative_linf':float(np.max(abs(net)/cells['volume'])/(speed/height)),
          'global_absolute_cell_flux_over_throughflow':float(abs(net).sum()/abs(through)),
          'section_relative_spread':float(np.ptp(sections['volume_flux'])/abs(through))}
    passed=mass['relative_linf']<1e-7 and mass['global_absolute_cell_flux_over_throughflow']<1e-8 and mass['section_relative_spread']<1e-8
    if not passed:raise ValueError('Independent final-field mass check failed')
    failure=json.loads((failed/'run_completion.json').read_text());failure_log=(failed/'run.log').read_text()
    if failure['exit_code']==0 or 'FGMRES failed true residual' not in failure_log:
        raise ValueError('Expected the recorded failed native128 pressure solve')
    report_path=runs/(args.output.name+'_checks.json')
    if report_path.exists():raise ValueError('Preserve prior terminal checks')
    report={'passed':passed,'scope':__doc__,'completed_native64_steps':count,'physical_time':count*config['time_step'],
            'steady_momentum_relative_l2':m['steady_momentum_relative_l2'],
            'temporal_acceleration_relative_l2':m['temporal_acceleration_relative_l2'],
            'independent_mass':mass,'paraview_verified':True,'native128_exit_code':failure['exit_code'],
            'native128_scope':'Only step1 completed; second step failed. No steady128 success.',
            'goal_complete':False,'checker_sha256':sha(Path(__file__))}
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);files=[];retained=[]
    def save(p,dest,scope):
        before=sha(p);target=out/dest;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
        if sha(p)!=before or sha(target)!=before:raise ValueError('Archive source changed')
        files.append({'path':target.relative_to(out).as_posix(),'source':str(p.resolve()),'source_sha256':before,
                      'sha256':before,'size':target.stat().st_size,'gzip':False,'scope':scope})
    def keep(p,scope):retained.append({'source':str(p.resolve()),'source_sha256':sha(p),'size':p.stat().st_size,'scope':scope})
    save(runs/'original_rhs_build_v1/build_manifest.json',Path('build/build_manifest.json'),'same immutable build already archived with all source snapshots in original_rhs_checkpoint')
    keep(runs/'original_rhs_build_v1/simple_channel.exe','actual unchanged native executable')
    for p in metric_paths:save(p,Path('native64')/p.relative_to(native),'every completed physical-step diagnostic')
    for root,label in ((native,'native64'),(failed,'failed_native128')):
        for name in ('case.json','run_manifest.json','run_completion.json','run.log','time_history.csv','transient_summary.json','gpu_linear.csv','projection_pressure.csv'):
            p=root/name
            if p.exists():save(p,Path(label)/name,'terminal trajectory evidence; native128 is a failure')
    for name in ('case.json','metrics.json','paraview_export.json','paraview_step_readback.json'):
        save(last/name,Path('native64/step_0128')/name,'verified steady final step')
    for name in ('solution.csv','flux.csv','walls.csv','mesh_cells.csv','mesh_faces.csv','sections.csv','solution.vtu','walls.vtp'):
        keep(last/name,'complete native64 steady fields retained on disk')
    for p in sorted((failed/'step_0002').iterdir()):
        if p.is_file():save(p,Path('failed_native128/step_0002')/p.name,'partial failed inner step; not a completed state')
    save(report_path,Path('checks.json'),'independent steady trajectory and final mass checks')
    for name in ('record_twisted_native_terminals.py','check_twisted_mass.py','check_twisted_checkpoint.py','check_twisted_paraview_step.py'):
        save(repo/'scripts'/name,Path('checkers')/name,'reproduction source')
    for name in ('original_rhs_n64_steady_paraview_export_v1.log','original_rhs_n64_steady_paraview_readback_v1.log'):
        save(runs/name,Path('logs')/name,'completed visualization verification')
    receipt={'scope':__doc__,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'goal_complete':False,'files':files,'retained_runtime_files':retained}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
