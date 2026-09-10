"""Validate an isolated extended-scalar build and its completed flow provenance."""
import json
from pathlib import Path
import numpy as np
from check_twisted_mass import read
from run_twisted_solver import sha


def validate(root):
    root=root.resolve();hashes={}
    def load(path):
        hashes[str(path.resolve())]=sha(path)
        return json.loads(path.read_text(encoding='utf-8-sig'))
    def checked(path,value):
        path=Path(path)
        if sha(path)!=value:raise ValueError('Extended reference input changed: '+str(path))
        hashes[str(path.resolve())]=value
    runtime=load(root/'run_manifest.json');completion=load(root/'run_completion.json')
    cfg=load(root/'case_manifest.json');exe=Path(runtime['executable'])
    build=load(exe.parent/'build_manifest.json')
    if not build['passed'] or not build['source_unchanged'] or build['link_exit_code']!=0:
        raise ValueError('Extended reference build did not complete')
    checked(exe,runtime['executable_sha256']);checked(exe,build['executable_sha256'])
    if completion['exit_code'] or not completion.get('geometry_state_unchanged'):
        raise ValueError('Extended flow did not complete with unchanged geometry')
    if 'End of simulation' not in (root/'run.log').read_text():raise ValueError('Missing final flow marker')
    checked(root/'a.conf',runtime['config_sha256']);checked(root/'a.conf',cfg['config_sha256'])
    env=runtime['environment']
    if any(env.get(name) is not None for name in ('APHROS_TWISTED_GEOMETRY_ONLY','APHROS_TWISTED_GEOMETRY_STATE_ONLY')):
        raise ValueError('Geometry-only execution is not a flow run')
    for name,value in build['source_sha256'].items():checked(name,value)
    drivers=[Path(row['source']) for row in build['results'] if Path(row['source']).name=='driver.cpp']
    if len(drivers)!=1:raise ValueError('Missing unique compiled driver')
    source=drivers[0].parent;prepared=load(source/'prepare_manifest.json')
    if not prepared['extended'] or not prepared['source_unchanged']:raise ValueError('Not an unchanged extended preparation')
    for name,value in prepared['source_sha256'].items():checked(name,value)
    for name,value in prepared['prepared_source_sha256'].items():checked(source/name,value)
    if 'using M=MeshCartesian<long double,3>;' not in drivers[0].read_text():raise ValueError('Driver scalar is not extended')
    for row in build['results']:
        if row['exit_code'] or '-fno-fast-math' not in row['command'] or '-ffp-contract=off' not in row['command']:
            raise ValueError('Unexpected compilation or floating-point options')
        checked(row['object'],row['object_sha256'])
    # Check equation bodies independently of the preparation's declared changes.
    # Only exact zero type disambiguation is permitted in the diffusion bodies.
    core={}
    for relative in ('solver/proj.ipp','solver/proj.h','solver/embed.ipp','solver/approx_eb.ipp',
                     'solver/convdiffi.ipp','solver/convdiffe.ipp'):
        original=[Path(p) for p in prepared['source_sha256'] if Path(p).as_posix().endswith('/src/'+relative)]
        if len(original)!=1:raise ValueError('Missing original algorithm source: '+relative)
        expected=original[0].read_bytes()
        if relative in ('solver/convdiffi.ipp','solver/convdiffe.ipp'):
            expected=expected.replace(b'GetGradCoeffs(0.,',b'GetGradCoeffs(Scal(0),')
        if (source/'src'/relative).read_bytes()!=expected:
            raise ValueError('Original equation body changed: '+relative)
        core[relative]={'original_sha256':sha(original[0]),'prepared_sha256':sha(source/'src'/relative)}
    geometry=Path(runtime['geometry_state']);checked(geometry,runtime['geometry_state_sha256'])
    if env.get('APHROS_TWISTED_GEOMETRY_STATE_IN')!=str(geometry):raise ValueError('Executed geometry path differs')
    capture=geometry.parent;captured=load(capture/'run_completion.json');capture_run=load(capture/'run_manifest.json')
    if captured['exit_code'] or not captured['source_unchanged'] or not captured['no_flow_trajectory']:
        raise ValueError('Original geometry capture did not complete')
    checked(geometry,captured['geometry_sha256'])
    for name,value in capture_run['source_sha256'].items():checked(name,value)
    # Require the original double snapshot linked against the original library.
    capture_build=load(Path(capture_run['executable']).parent/'build_manifest.json')
    if capture_build['exit_code']:raise ValueError('Original geometry driver did not build')
    checked(capture_run['executable'],capture_build['executable_sha256'])
    for key in ('source','library'):checked(capture_build[key],capture_build[key+'_sha256'])
    if 'using M=MeshCartesian<double,3>;' not in Path(capture_build['source']).read_text():
        raise ValueError('Geometry snapshot was not produced by the original double driver')
    for name in ('cells','faces','walls'):
        filename='tube_b0_geometry_'+name+'.csv';arrays=[read(p/filename) for p in (capture,root)]
        if arrays[0].shape!=arrays[1].shape or arrays[0].dtype.names!=arrays[1].dtype.names or any(
                not np.array_equal(arrays[0][k],arrays[1][k]) for k in arrays[0].dtype.names):
            raise ValueError('Loaded computational geometry differs: '+name)
        for path in (capture/filename,root/filename):hashes[str(path.resolve())]=sha(path)
    times=read(root/'tube_b0_time.csv')
    if len(times)!=cfg['time_steps'] or not np.allclose(times['time'],np.arange(1,len(times)+1)*cfg['time_step'],rtol=1e-12,atol=0):
        raise ValueError('Incomplete physical time sequence')
    for name in ('run.log','tube_b0_time.csv'):hashes[str((root/name).resolve())]=sha(root/name)
    if any(sha(Path(name))!=value for name,value in hashes.items()):raise ValueError('Reference changed during validation')
    return {'passed':True,'scope':__doc__+' This validates provenance, not field agreement or physical accuracy.',
            'core_equation_bodies':core,'compiled_objects':len(build['results']),
            'complete_physical_steps':len(times),'source_sha256':hashes,'checker_sha256':sha(Path(__file__))}
