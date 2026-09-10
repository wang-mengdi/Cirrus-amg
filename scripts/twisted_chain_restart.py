"""Lossless physical-prefix restart with recursively verified continuation ancestry.

Every segment retains its own actual GPU trace and copied prefix. A checkpoint
can advance along the chain only after a complete physical output write barrier;
no segment is relabelled as a zero-start or a completed steady trajectory.
"""
import csv
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import io
import json
import math
from pathlib import Path
import re
import shutil
import numpy as np
import psutil
from check_twisted_native_run import columns
from run_twisted_solver import sha as raw_sha
from twisted_restart import copy_prefix
import twisted_prefix_restart as legacy
from twisted_prefix_restart import load_checkpoint as load_legacy_checkpoint

REPO=Path(__file__).resolve().parents[1]
FORMAT='cirrus_projection_chain_restart_v3'
IGNORED=('output','time_steps','output_stride','dump_iterations','restart_checkpoint','gpu_orthogonalization')
_hash_cache=ContextVar('chain_restart_hash_cache',default=None)


def signature(path):
    s=path.stat()
    return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns


def sha(path):
    """Hash each immutable file once per validation, rejecting intervening changes."""
    path=Path(path).resolve();cache=_hash_cache.get()
    if cache is None:return raw_sha(path)
    current=signature(path)
    if path in cache:
        recorded,digest=cache[path]
        if recorded!=current:raise ValueError('Input changed within validation: '+str(path))
        return digest
    digest=raw_sha(path)
    if signature(path)!=current:raise ValueError('Input changed while hashing: '+str(path))
    cache[path]=(current,digest)
    return digest


@contextmanager
def hash_session():
    if _hash_cache.get() is not None:
        yield
        return
    cache={};token=_hash_cache.set(cache)
    # The isolated validation process may recursively invoke the frozen v2
    # verifier. Its algorithm and proof are unchanged; share only this read
    # cache, and restore the imported function even when validation fails.
    old=legacy.sha;legacy.sha=sha
    try:
        yield
        for path,(recorded,_) in cache.items():
            if signature(path)!=recorded:raise ValueError('Input changed during validation: '+str(path))
    finally:
        legacy.sha=old;_hash_cache.reset(token)


def verified_reads(function):
    @wraps(function)
    def wrapped(*args,**kwargs):
        with hash_session():return function(*args,**kwargs)
    return wrapped

def restored_values(folder):
    """Reconstruct the same seven-state layout without parsing unused columns."""
    metrics=json.loads((folder/'metrics.json').read_text())
    names=('id','x','y','z','u','v','w','p')
    cells=columns(folder/'solution.csv',names)
    final=columns(folder/f'iter_{metrics["iterations"]}'/'cells.csv',names+('u_diff','v_diff','w_diff'))
    if len(cells)!=metrics['cells'] or not np.array_equal(cells[:,0],np.arange(len(cells))):
        raise ValueError('Unordered or incomplete restart cells')
    if not np.isfinite(final).all() or not np.array_equal(cells,final[:,:8]):
        raise ValueError('Final solution and unaccelerated iteration differ')
    values=np.empty(7*metrics['cells']+metrics['faces'],dtype='<f8')
    values[:7*metrics['cells']]=final[:,4:].ravel()
    del cells,final
    flux=columns(folder/'flux.csv',('id','flux'))
    if len(flux)!=metrics['faces'] or not np.array_equal(flux[:,0],np.arange(len(flux))) or not np.isfinite(flux).all():
        raise ValueError('Unordered, nonfinite or incomplete restart face flux')
    values[7*metrics['cells']:]=flux[:,1]
    return values


def physical_config(config):
    if config.get('gpu_orthogonalization','mgs2') not in ('mgs2','cgs2'):
        raise ValueError('Unknown native orthogonalization')
    return {k:v for k,v in config.items() if k not in IGNORED}

def complete_lines(path):
    data=path.read_bytes();return data[:data.rfind(b'\n')+1]

@verified_reads
def check_prefix(run,step,evidence,_seen=frozenset()):
    run=Path(run).resolve();evidence=Path(evidence).resolve();hashes={}
    def keep(path,expected=None):
        path=Path(path).resolve();actual=sha(path)
        if expected is not None and actual!=expected:raise ValueError('Prefix input changed: '+str(path))
        hashes[str(path)]=actual;return path
    def load(path):return json.loads(keep(path).read_text())
    cfg=load(run/'case.json');runtime=load(run/'run_manifest.json')
    ancestry=None
    if cfg.get('restart_checkpoint'):
        checkpoint=Path(cfg['restart_checkpoint'])
        if not checkpoint.is_absolute():checkpoint=REPO/checkpoint
        ancestor,ancestor_inputs=load_checkpoint(checkpoint,cfg,_seen)
        start=ancestor['physical_step']
        if step<=start:raise ValueError('A chained checkpoint must advance beyond its restored ancestor')
        if runtime.get('restart_step')!=start or runtime.get('restart_input_sha256')!=ancestor_inputs:
            raise ValueError('Parent executed checkpoint ancestry differs')
        keep(run/'restart_loaded.bin',ancestor['state_sha256'])
        for relative,row in ancestor['prefix_files'].items():keep(run/relative,row['sha256'])
        for name in ('mesh_cells.csv','mesh_faces.csv'):
            keep(run/name,sha(Path(ancestor['parent_run'])/name))
        for path,digest in ancestor_inputs.items():keep(path,digest)
        ancestry={'checkpoint':str(checkpoint.resolve()),'checkpoint_sha256':sha(checkpoint),
            'restored_step':start,'parent_run':ancestor['parent_run'],
            'depth':ancestor.get('ancestry_depth',0)+1,'exact_state_roundtrip':True,
            'copied_prefix_files':len(ancestor['prefix_files']),'input_sha256':ancestor_inputs}
    elif 'restart_step' in runtime:
        raise ValueError('Unrecorded resumed parent')
    if not 1<=step<=cfg['time_steps']:raise ValueError('Prefix step outside the prescribed trajectory')
    if cfg.get('fluid_solver')!='proj' or cfg.get('linear_backend')!='native_gpu' or cfg.get('linear_tolerance')!=1e-13:
        raise ValueError('Require the original native GPU projection and 1e-13 tolerance')
    for name,value in (('gpu_preconditioner','native_amg'),('gpu_pressure_operator','full'),('gpu_viscosity_operator','full'),('gpu_pressure_gauge','mean_zero')):
        if cfg.get(name)!=value:raise ValueError('Require complete implicit native operators')
    physical_config(cfg)
    inputs={runtime['executable']:runtime['executable_sha256'],runtime['config']:runtime['config_sha256'],**runtime['geometry_input_sha256']}
    for p,h in inputs.items():keep(p,h)
    if json.loads(Path(runtime['config']).read_text())!=cfg:raise ValueError('Actual executed configuration differs')
    captures={}
    for original,captured in (('run.log','parent_run_prefix.log'),('time_history.csv','parent_time_prefix.csv'),('gpu_linear.csv','parent_gpu_prefix.csv')):
        data=keep(evidence/captured).read_bytes()
        if not data.endswith(b'\n') or not (run/original).read_bytes().startswith(data):
            raise ValueError('Captured log is not an unchanged complete-line prefix: '+original)
        captures[original]=data
    starts=[int(s) for s in re.findall(rb'^projection step=(\d+) iter=1(?:\s|$)',captures['run.log'],re.M)]
    later=any(step<s<=cfg['time_steps'] for s in starts)
    terminal=None
    if not later:
        terminal=load(run/'run_completion.json');summary=load(run/'transient_summary.json')
        if terminal['exit_code'] or not all(terminal[k] for k in ('executable_unchanged','config_unchanged','geometry_unchanged')):
            raise ValueError('No later-step write barrier or successful parent completion')
        if ancestry and (not terminal.get('restart_unchanged') or summary.get('restart_step')!=ancestry['restored_step'] or summary.get('steps_computed_this_run')!=cfg['time_steps']-ancestry['restored_step']):
            raise ValueError('Terminal continuation ancestry is incomplete')
        if not summary['converged'] or summary['steps_completed']!=cfg['time_steps']:
            raise ValueError('Parent termination does not establish the requested prefix')
    times=list(csv.DictReader(io.StringIO(captures['time_history.csv'].decode())))
    if len(times)<step:raise ValueError('Missing completed physical times')
    if ancestry:
        inherited=[{k:(v=='true' if k=='inner_converged' else int(v) if k in ('step','inner_iterations') else float(v))
                    for k,v in row.items()} for row in times[:ancestry['restored_step']]]
        if inherited!=ancestor['time_history']:raise ValueError('Inherited physical history differs from ancestor checkpoint')
    trace=list(csv.DictReader(io.StringIO(captures['gpu_linear.csv'].decode())))
    if not trace or set(r['operator'] for r in trace)!={'pressure','diffusion'}:
        raise ValueError('Missing native operator evidence')
    for call,row in enumerate(trace,1):
        checks=int(row['original_rhs_checks']);accepted=int(row['original_rhs_accepted'])
        if int(row['call'])!=call or checks<0 or accepted not in (0,1) or (accepted and checks<1):
            raise ValueError('Invalid GPU call or pressure-specific check counters')
        # These two counters describe Neumann pressure's extra early acceptance
        # against its unmodified RHS. Diffusion does not modify its RHS and
        # leaves the counters zero; its fresh Ax residual is still mandatory.
        # Pressure can also use its final original-RHS check outside the counter.
        if row['operator']=='diffusion' and (checks or accepted or float(row['compatibility_relative_l2'])!=0):
            raise ValueError('Unexpected pressure-specific metadata on diffusion')
        for key in ('true_relative_residual','compatibility_relative_l2'):
            value=float(row[key])
            if not math.isfinite(value) or not 0<=value<=1e-13:raise ValueError('GPU solve violates the original tolerance')
    geometry=Path(cfg['embedded_geometry'])
    if not geometry.is_absolute():geometry=REPO/geometry
    height=load(geometry)['extent'][1];fields=[];checks=[];parsed=[];prefix_files={}
    for i,row in enumerate(times[:step],1):
        folder=run/f'step_{i:04d}';metrics=load(folder/'metrics.json')
        if int(row['step'])!=i or row['inner_converged']!='true' or not metrics['converged'] or metrics['iterations']!=int(row['inner_iterations']):
            raise ValueError('Incomplete or inconsistent physical prefix')
        if float(row['time'])!=i*cfg['time_step']:raise ValueError('Prefix physical time differs')
        for key in ('complete_inner_fixed_point_residual','implicit_diffusion_relative_l2'):
            value=metrics[key]
            if not math.isfinite(value) or not 0<=value<min(cfg['tolerance'],1e-8):raise ValueError('Prefix fixed point or viscosity failed')
        parsed.append({k:(True if k=='inner_converged' else int(v) if k in ('step','inner_iterations') else float(v)) for k,v in row.items()})
        if metrics['field_output_written']:
            fields.append(i)
            c=columns(keep(folder/'solution.csv'),('id','volume','u','v','w'))
            f=columns(keep(folder/'mesh_faces.csv'),('id','owner','neighbor'))
            q=columns(keep(folder/'flux.csv'),('id','flux'))
            sections=columns(keep(folder/'sections.csv'),('volume_flux',))[:,0]
            if not all(np.isfinite(a).all() for a in (c,f,q,sections)) or np.any(c[:,1]<=0):raise ValueError('Invalid prefix fields')
            if not np.array_equal(c[:,0],np.arange(len(c))) or not np.array_equal(f[:,0],q[:,0]):raise ValueError('Prefix IDs differ')
            inside=f[:,2]>=0;net=np.zeros(len(c));flux=q[:,1]
            if np.any(flux[~inside]!=0):raise ValueError('Nonzero wall flux')
            np.add.at(net,f[:,1].astype(int),flux);np.add.at(net,f[inside,2].astype(int),-flux[inside])
            rate=np.linalg.norm(c[:,2:],axis=1).max()/height;through=sections.mean()
            if rate<=0 or through==0:raise ValueError('Degenerate physical flux scale')
            check={'step':i,'divergence_relative_linf':float(np.max(abs(net)/c[:,1])/rate),
                'global_absolute_cell_flux_over_throughflow':float(abs(net).sum()/abs(through)),
                'section_flux_relative_spread':float(np.ptp(sections)/abs(through))}
            check['passed']=check['divergence_relative_linf']<1e-7 and check['global_absolute_cell_flux_over_throughflow']<1e-8 and check['section_flux_relative_spread']<1e-8
            if not check['passed']:raise ValueError('Prefix actual shared-face mass failed')
            checks.append(check)
        for p in sorted(folder.rglob('*')):
            if p.is_file():
                keep(p);prefix_files[p.relative_to(run).as_posix()]={'source':str(p),'sha256':hashes[str(p.resolve())]}
    if not fields or fields[0]!=1 or fields[-1]!=step:raise ValueError('Selected prefix lacks complete retained fields')
    for name in ('mesh_cells.csv','mesh_faces.csv','projection_method.json'):keep(run/name)
    failures=[]
    for folder in sorted(run.glob('linear_failure_*')):
        metadata=folder/'failure.json'
        if not metadata.is_file():continue
        failure=json.loads(metadata.read_text())
        if failure['physical_step']>step:continue
        if failure['failed_solution_accepted'] is not False:raise ValueError('A failed linear solve was accepted')
        for p in folder.iterdir():
            if p.is_file():keep(p)
        failures.append(str(metadata))
    if any(sha(Path(p))!=h for p,h in hashes.items()):raise ValueError('Completed prefix changed during validation')
    return {'passed':True,'scope':__doc__,'steps_completed':step,'parent_configured_steps':cfg['time_steps'],
        'parent_run_complete_claimed':False,'barrier':'later physical step began after completed output writes' if later else 'verified terminal parent',
        'captured_gpu_calls':len(trace),'gpu_trace_scope':'This segment only; captured complete lines cover its selected new steps and may include later calls. Ancestor traces are verified recursively.',
        'ancestry':ancestry,'ancestry_depth':ancestry['depth'] if ancestry else 0,
        'maximum_accepted_linear_residual':max(float(r['true_relative_residual']) for r in trace),
        'retained_field_checks':checks,'field_output_steps':fields,'time_history':parsed,'prefix_files':prefix_files,
        'rejected_linear_failure_records':failures,'source_sha256':hashes,'executed_input_sha256':inputs,
        'checker_sha256':sha(Path(__file__))}

@verified_reads
def create_checkpoint(run,step,output,pid=None,creation_time=None):
    run=Path(run).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('Preserve existing checkpoints')
    runtime=json.loads((run/'run_manifest.json').read_text());observation=None
    if pid is not None:
        p=psutil.Process(pid)
        if creation_time is None or abs(p.create_time()-creation_time)>1e-5 or Path(p.exe()).resolve()!=Path(runtime['executable']).resolve():
            raise ValueError('Live parent identity mismatch')
        if Path(p.cmdline()[-1]).resolve()!=Path(runtime['config']).resolve():raise ValueError('Live process uses a different config')
        observation={'pid':pid,'creation_time':p.create_time(),'executable':p.exe(),'confirmed_live':True}
    elif not (run/'run_completion.json').is_file():raise ValueError('Supply the exact live parent identity')
    output.mkdir(parents=True,exist_ok=False)
    for original,target in (('run.log','parent_run_prefix.log'),('time_history.csv','parent_time_prefix.csv'),('gpu_linear.csv','parent_gpu_prefix.csv')):
        (output/target).write_bytes(complete_lines(run/original))
    proof=check_prefix(run,step,output);config=json.loads((run/'case.json').read_text())
    selected=run/f'step_{step:04d}';metrics=json.loads((selected/'metrics.json').read_text())
    values=restored_values(selected)
    if len(values)!=7*metrics['cells']+metrics['faces'] or not np.isfinite(values).all():raise ValueError('Incomplete restart state')
    values.tofile(output/'state.bin')
    proof_path=output/'parent_prefix_check.json';proof_path.write_text(json.dumps(proof,indent=2)+'\n')
    shutil.copyfile(__file__,output/'producer.py.txt')
    geometry=Path(config['embedded_geometry'])
    if not geometry.is_absolute():geometry=REPO/geometry
    meta={'format':FORMAT,'parent_proof_kind':'completed_physical_prefix_chain','parent_run':str(run),'parent_config':config,
        'physical_step':step,'physical_time':step*config['time_step'],'time_step':config['time_step'],
        'ancestry_depth':proof['ancestry_depth'],
        'cells':metrics['cells'],'faces':metrics['faces'],'extent':json.loads(geometry.read_text())['extent'],
        'state_file':'state.bin','state_sha256':sha(output/'state.bin'),
        'state_layout':'little-endian float64: [u,v,w,p,u_diff,v_diff,w_diff] per cell, then every face flux in ID order',
        'time_history':proof['time_history'],'field_output_steps':proof['field_output_steps'],'prefix_files':proof['prefix_files'],
        'parent_check_file':proof_path.name,'parent_check_sha256':sha(proof_path),'parent_observation':observation,
        'source_sha256':{**proof['source_sha256'],**proof['executed_input_sha256']},'producer_sha256':sha(Path(__file__))}
    (output/'checkpoint.json').write_text(json.dumps(meta,indent=2)+'\n')
    load_checkpoint(output/'checkpoint.json')
    return meta

@verified_reads
def load_checkpoint(path,config=None,_seen=frozenset()):
    path=Path(path).resolve();meta=json.loads(path.read_text());root=path.parent
    if path in _seen:raise ValueError('Cyclic restart checkpoint ancestry')
    if meta['format']=='cirrus_projection_prefix_restart_v2':return load_legacy_checkpoint(path,config)
    _seen=_seen|{path}
    if meta['format']!=FORMAT or meta['parent_proof_kind']!='completed_physical_prefix_chain' or meta['state_file']!='state.bin' or meta['parent_check_file']!='parent_prefix_check.json':
        raise ValueError('Unknown completed-prefix checkpoint format')
    parent=Path(meta['parent_run']);parent_cfg=json.loads((parent/'case.json').read_text());step=meta['physical_step']
    if meta['parent_config']!=parent_cfg:raise ValueError('Changed parent configuration')
    if config is not None and (physical_config(config)!=physical_config(parent_cfg) or config['time_steps']<=step):
        raise ValueError('Restart changes the physical problem or has no tail')
    if meta['physical_time']!=step*meta['time_step'] or meta['time_step']!=parent_cfg['time_step']:raise ValueError('Wrong absolute restart time')
    inputs=dict(meta['source_sha256']);inputs.update({str(path):sha(path),str(root/'state.bin'):meta['state_sha256'],
        str(root/'parent_prefix_check.json'):meta['parent_check_sha256'],str(root/'producer.py.txt'):meta['producer_sha256']})
    for p,h in inputs.items():
        if sha(Path(p))!=h:raise ValueError('Prefix checkpoint input changed: '+p)
    proof=json.loads((root/'parent_prefix_check.json').read_text())
    actual=check_prefix(parent,step,root,_seen)
    if proof!=actual or not actual['passed'] or proof['checker_sha256']!=meta['producer_sha256']:
        raise ValueError('Completed prefix does not reproduce its recorded proof')
    for key in ('time_history','field_output_steps','prefix_files','ancestry_depth'):
        if meta[key]!=proof[key]:raise ValueError('Checkpoint prefix metadata differs: '+key)
    for p,h in {**proof['source_sha256'],**proof['executed_input_sha256']}.items():
        if inputs.get(p)!=h:raise ValueError('Omitted verified prefix input')
    metrics=json.loads((parent/f'step_{step:04d}'/'metrics.json').read_text())
    geometry=Path(parent_cfg['embedded_geometry'])
    if not geometry.is_absolute():geometry=REPO/geometry
    if any(meta[k]!=metrics[k] for k in ('cells','faces')) or meta['extent']!=json.loads(geometry.read_text())['extent']:
        raise ValueError('Restart geometry dimensions differ')
    if (root/'state.bin').stat().st_size!=8*(7*meta['cells']+meta['faces']):raise ValueError('Wrong restart state size')
    state=np.fromfile(root/'state.bin',dtype='<f8');original=restored_values(parent/f'step_{step:04d}')
    if not np.isfinite(state).all() or state.tobytes()!=original.tobytes():raise ValueError('Restart state differs from actual parent values')
    return meta,inputs

def verify_resumed_run(root,config,runtime,completion,summary):
    meta,inputs=load_checkpoint(config['restart_checkpoint'],config);step=meta['physical_step']
    if not completion.get('restart_unchanged') or runtime.get('restart_input_sha256')!=inputs or runtime.get('restart_step')!=step:
        raise ValueError('Executed prefix checkpoint provenance differs')
    if summary.get('restart_step')!=step or summary.get('steps_computed_this_run')!=config['time_steps']-step:
        raise ValueError('Incomplete continuation tail')
    if sha(root/'restart_loaded.bin')!=meta['state_sha256']:raise ValueError('Actual C++ restored state differs')
    inputs[str(root/'restart_loaded.bin')]=sha(root/'restart_loaded.bin')
    for relative,row in meta['prefix_files'].items():
        if sha(root/relative)!=row['sha256']:raise ValueError('Copied prefix changed')
        inputs[str(root/relative)]=row['sha256']
    for name in ('mesh_cells.csv','mesh_faces.csv'):
        if sha(root/name)!=sha(Path(meta['parent_run'])/name):raise ValueError('Rebuilt native grid or face order differs')
    return {'passed':True,'completed_parent_steps_used':step,'new_steps_computed':config['time_steps']-step,
        'parent_run':meta['parent_run'],'exact_state_roundtrip':True,'source_sha256':inputs,
        'parent_prefix_check':json.loads((Path(config['restart_checkpoint']).parent/'parent_prefix_check.json').read_text()),
        'ancestry_depth':meta['ancestry_depth'],
        'scope':'Recursively verified physical segments and separately executed continuation; no full-parent or steady completion claim'}
