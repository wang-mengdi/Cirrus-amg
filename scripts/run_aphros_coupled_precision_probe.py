"""Replay a complete captured pressure face graph with extended pressure arithmetic."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import numpy as np
from check_twisted_mass import read,vector
from run_twisted_solver import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--aphros',type=Path,required=True);p.add_argument('--face-report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    repo=Path(__file__).resolve().parents[1];root=args.aphros.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    report=json.loads(args.face_report.read_text())
    if not report['diagnostic_consistent']:raise ValueError('Need a verified original face capture')
    inputs={str(root/name):value for name,value in report['source_sha256'].items()}
    inputs[str(args.face_report.resolve())]=sha(args.face_report)
    if any(sha(Path(name))!=value for name,value in inputs.items()):raise ValueError('Captured source changed')
    case=json.loads((root/'case_manifest.json').read_text());h=case['spec']['extent'][1]/case['ny'];shape=np.array(case['shape'])
    cells=read(root/'proj_final_b0_cells.csv');geometry=read(root/'tube_b0_geometry_faces.csv')
    expressions=read(root/'proj_final_b0_pressure_faces.csv')
    lookup=np.full(shape,-1,dtype=np.int32)
    keys=np.rint(vector(cells,('x','y','z'))/h-.5).astype(int);lookup[tuple(keys.T)]=np.arange(len(cells))
    fkey=vector(geometry,('i','j','k')).astype(int);axis=geometry['axis'].astype(int)
    keep=~((axis==0)&(fkey[:,0]==shape[0]));axis=axis[keep]
    positive=fkey[keep].copy();negative=positive.copy();negative[np.arange(len(axis)),axis]-=1
    positive[:,0]%=shape[0];negative[:,0]%=shape[0]
    owner=lookup[tuple(negative.T)];neighbor=lookup[tuple(positive.T)];expressions=expressions[keep]
    if np.any(owner<0) or np.any(neighbor<0):raise ValueError('Missing adjacent fluid cells')
    if not np.array_equal(expressions['p0'],cells['p'][owner]) or not np.array_equal(expressions['p1'],cells['p'][neighbor]):
        raise ValueError('Captured pressure ordering differs')
    rate=report['actual_mass']['normalization_rate_max_speed_over_box_y']
    data=out/'input.bin'
    with data.open('wb') as f:
        f.write(struct.pack('<qqd',len(cells),len(expressions),rate))
        np.column_stack((cells['p'],cells['volume'])).astype('<f8').tofile(f)
        faces=np.empty(len(expressions),dtype=[('a','<i4'),('c','<i4'),('e0','<f8'),('e1','<f8'),('b','<f8')])
        faces['a']=owner;faces['c']=neighbor
        for k in ('e0','e1','b'):faces[k]=expressions[k]
        faces.tofile(f)
    source=repo/'validation/aphros/pressure_coupled_precision_probe.cpp';local=out/source.name
    shutil.copyfile(source,local);shutil.copyfile(__file__,out/'runner.py.txt')
    compiler=Path('C:/ProgramData/mingw64/mingw64/bin/g++.exe');exe=out/'pressure_probe.exe'
    flags=['-std=c++17','-O2','-fno-fast-math','-ffp-contract=off','-fopenmp','-static',
           '-ID:/Dropbox/Agent-simulation/twisted-baseline/amgcl']
    command=[str(compiler),*flags,str(local),'-o',str(exe)]
    metadata={'scope':__doc__+' This is not a new Aphros trajectory.',
              'source_sha256':inputs,'input_sha256':sha(data),'probe_source_sha256':sha(source),
              'compiler_sha256':sha(compiler),'compile_command':command,
              'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with (out/'build.log').open('w') as f:code=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT).returncode
    metadata['compile_exit_code']=code
    if not code:
        metadata['executable_sha256']=sha(exe);env=dict(os.environ,OMP_NUM_THREADS='2',OMP_WAIT_POLICY='PASSIVE')
        with (out/'run.log').open('w') as f:
            code=subprocess.run([str(exe),str(data),str(out/'coupled')],stdout=f,stderr=subprocess.STDOUT,env=env).returncode
        metadata['run_exit_code']=code
    metadata['source_unchanged']=all(sha(Path(name))==value for name,value in inputs.items()) and sha(source)==sha(local)
    metadata['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({k:v for k,v in metadata.items() if k!='source_sha256'}),flush=True)
    if code or not metadata['source_unchanged']:raise SystemExit(code or 1)


if __name__=='__main__':main()
