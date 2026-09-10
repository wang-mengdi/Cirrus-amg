"""Verify a completed native GPU continuation from a recursively checked physical prefix and balance its retained face fluxes.

Passing verifies completed time steps and linear/fixed-point/mass tolerances.
It does not establish independent-reference agreement or grid convergence.
"""
import argparse,csv,json,math
from pathlib import Path
import numpy as np
from run_twisted_solver import sha


def columns(path,names):
    with path.open() as f:
        header=f.readline().strip().split(',')
        return np.loadtxt(f,delimiter=',',usecols=[header.index(n) for n in names],ndmin=2)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();root=args.run.resolve()
    if args.output.exists():raise ValueError('Preserve previous terminal checks')
    hashes={}
    def load(path):
        hashes[str(path)]=sha(path);return json.loads(path.read_text())
    cfg=load(root/'case.json');runtime=load(root/'run_manifest.json');completion=load(root/'run_completion.json')
    summary=load(root/'transient_summary.json')
    if completion['exit_code'] or not all(completion[k] for k in ('executable_unchanged','config_unchanged','geometry_unchanged')):
        raise ValueError('Native run did not complete with unchanged inputs')
    if not summary['converged'] or summary['steps_completed']!=cfg['time_steps']:raise ValueError('Incomplete trajectory')
    if cfg['linear_backend']!='native_gpu' or cfg['linear_tolerance']!=1e-13:raise ValueError('Expected the unchanged native GPU tolerance')
    restart_check=None
    if cfg.get('restart_checkpoint'):
        from twisted_chain_restart import verify_resumed_run
        restart_check=verify_resumed_run(root,cfg,runtime,completion,summary)
        hashes.update(restart_check['source_sha256'])
    elif 'restart_step' in summary or 'restart_step' in runtime:
        raise ValueError('Unrecorded resumed trajectory')
    inputs={runtime['executable']:runtime['executable_sha256'],runtime['config']:runtime['config_sha256'],**runtime['geometry_input_sha256']}
    for path,value in inputs.items():
        if sha(Path(path))!=value:raise ValueError('Executed input changed')
    times=list(csv.DictReader((root/'time_history.csv').open()));hashes[str(root/'time_history.csv')]=sha(root/'time_history.csv')
    trace=list(csv.DictReader((root/'gpu_linear.csv').open()));hashes[str(root/'gpu_linear.csv')]=sha(root/'gpu_linear.csv')
    if set(r['operator'] for r in trace)!={'pressure','diffusion'}:raise ValueError('Missing GPU operator trace')
    for r in trace:
        for key in ('true_relative_residual','compatibility_relative_l2'):
            value=float(r[key])
            if not math.isfinite(value) or not 0<=value<=1e-13:raise ValueError('Accepted GPU solve exceeds unchanged tolerance')
    if len(times)!=cfg['time_steps']:raise ValueError('Missing time steps')
    geometry=load(Path(cfg['embedded_geometry']));height=geometry['extent'][1];checks=[];field_steps=[]
    for step,row in enumerate(times,1):
        folder=root/f'step_{step:04d}';metrics=load(folder/'metrics.json')
        if int(row['step'])!=step or row['inner_converged']!='true' or not metrics['converged'] or metrics['iterations']!=int(row['inner_iterations']):
            raise ValueError('Unconverged or inconsistent step')
        if not math.isclose(float(row['time']),step*cfg['time_step'],rel_tol=1e-12,abs_tol=0):raise ValueError('Wrong physical time')
        if metrics['complete_inner_fixed_point_residual']>=min(cfg['tolerance'],1e-8):raise ValueError('Full fixed point failed')
        if metrics['implicit_diffusion_relative_l2']>=min(cfg['tolerance'],1e-8):raise ValueError('Implicit viscosity failed')
        if not metrics['field_output_written']:continue
        field_steps.append(step)
        names=('solution.csv','mesh_faces.csv','flux.csv','sections.csv')
        for name in names:hashes[str(folder/name)]=sha(folder/name)
        cells=columns(folder/'solution.csv',('id','volume','u','v','w'))
        faces=columns(folder/'mesh_faces.csv',('id','owner','neighbor'));flux=columns(folder/'flux.csv',('id','flux'))
        sections=columns(folder/'sections.csv',('volume_flux',))[:,0]
        if not np.array_equal(cells[:,0],np.arange(len(cells))) or not np.array_equal(faces[:,0],flux[:,0]):raise ValueError('Mismatched IDs')
        inside=faces[:,2]>=0;q=flux[:,1];net=np.zeros(len(cells))
        if np.any(q[~inside]!=0):raise ValueError('Nonzero stationary wall flux')
        np.add.at(net,faces[:,1].astype(int),q);np.add.at(net,faces[inside,2].astype(int),-q[inside])
        rate=np.linalg.norm(cells[:,2:],axis=1).max()/height;through=sections.mean()
        result={'step':step,'full_fixed_point':metrics['complete_inner_fixed_point_residual'],
            'divergence_relative_linf':float(np.max(abs(net)/cells[:,1])/rate),
            'global_absolute_cell_flux_over_throughflow':float(abs(net).sum()/abs(through)),
            'section_flux_relative_spread':float(np.ptp(sections)/abs(through))}
        result['passed']=(result['divergence_relative_linf']<1e-7 and result['global_absolute_cell_flux_over_throughflow']<1e-8
                          and result['section_flux_relative_spread']<1e-8)
        checks.append(result)
    if field_steps!=summary['field_output_steps'] or not {1,cfg['time_steps']}<=set(field_steps):raise ValueError('Retained field schedule differs')
    if any(sha(Path(path))!=value for path,value in hashes.items()):raise ValueError('Native outputs changed during check')
    result={'passed':all(c['passed'] for c in checks),'scope':__doc__,'steps_completed':len(times),'retained_field_checks':checks,
        'steady_converged':summary['steady_converged'],'successful_gpu_calls':len(trace),
        'maximum_accepted_linear_residual':max(float(r['true_relative_residual']) for r in trace),
        'source_sha256':hashes,'executed_input_sha256':inputs,'checker_sha256':sha(Path(__file__))}
    if restart_check is not None:
        result['restart_prefix_validation']=restart_check
        result['gpu_trace_scope']='New continuation calls only; the immutable parent trace prefix is verified separately in restart_prefix_validation'
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','executed_input_sha256')}))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
