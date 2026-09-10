"""Compile original Mesh/Embed Proj templates with extended scalar precision in isolation.

Explicit template instantiations switch double to long double. Optional type
portability adds long-double NaN and type-matched zero literals in copied headers.
This is not a linked or validated extended-precision solver.
"""
import argparse,datetime,json,subprocess
from pathlib import Path
import hashlib


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aphros',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--portable-types',action='store_true')
    args=parser.parse_args();root=args.aphros.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    (out/'probe_source.py.txt').write_bytes(Path(__file__).read_bytes())
    compiler=Path('C:/ProgramData/mingw64/mingw64/bin/g++.exe')
    inputs={str(p):sha(p) for p in (root/'src').rglob('*') if p.is_file() and p.suffix in ('.cpp','.h','.ipp')}
    include_root=root/'src';modifications={}
    if args.portable_types:
        include_root=out/'src'
        for name in inputs:
            source=Path(name);dest=include_root/source.relative_to(root/'src')
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())
        path=include_root/'geom/vect.h';body=path.read_bytes();ending=b'\r\n' if b'\r\n' in body else b'\n'
        anchor=b'struct GetNanHelper {'
        if body.count(anchor)!=1:raise ValueError('Expected unique NaN helper')
        addition=ending.join((anchor,b'  static auto Get(long double*) {',
                             b'    return std::numeric_limits<long double>::quiet_NaN();',b'  }'))
        path.write_bytes(body.replace(anchor,addition));modifications['geom/vect.h']='Add long-double quiet NaN overload'
        path=include_root/'util/fluid.h';body=path.read_bytes()
        if body.count(b'std::max(0.,')!=4 or body.count(b'std::min(0.,')!=3:
            raise ValueError('Expected three normal-velocity clamps and one inward-flux bound')
        path.write_bytes(body.replace(b'std::max(0.,',b'std::max(Scal(0),').replace(b'std::min(0.,',b'std::min(Scal(0),'))
        modifications['util/fluid.h']='Match exact zero constants to the scalar type in three outlet clamps and one inward-flux bound'
    flags=['-std=c++14','-O0','-fopenmp','-fno-fast-math','-ffp-contract=off',
           '-DM_PI=3.141592653589793','-D_ALIGNBYTES_=16','-D_USE_MPI_=0','-D_USE_OPENMP_=1','-D_USE_DIM1_=0',
           '-D_USE_DIM2_=0','-D_USE_DIM3_=1','-D_USE_DIM4_=0','-D_USE_AVX_=0',
           '-D_USE_HDF_=0','-D_USE_HYPRE_=0','-D_USE_AMGX_=0','-D_USE_FPZIP_=0',
           '-D_USE_BACKEND_CUBISM_=0','-D_USE_BACKEND_LOCAL_=1','-D_USE_BACKEND_NATIVE_=1',
           '-I'+str(include_root),'-I'+str(include_root/'solver'),'-I'+str(include_root/'geom')]
    results=[]
    for relative in ('geom/mesh.cpp','solver/proj_eb.cpp'):
        original=root/'src'/relative;body=original.read_bytes()
        if body.count(b'MeshCartesian<double')!=1:raise ValueError('Expected one original explicit scalar instantiation')
        source=out/original.name;source.write_bytes(body.replace(b'MeshCartesian<double',b'MeshCartesian<long double'))
        (out/(original.name+'.original.txt')).write_bytes(body)
        obj=source.with_suffix('.o');command=[str(compiler),*flags,'-c',str(source),'-o',str(obj)]
        started=datetime.datetime.now(datetime.timezone.utc).isoformat()
        run=subprocess.run(command,capture_output=True,text=True)
        log=source.with_suffix('.log');log.write_text(run.stdout+run.stderr)
        results.append({'source':str(source),'original_sha256':sha(original),'compiled_source_sha256':sha(source),
                        'command':command,'exit_code':run.returncode,'started_utc':started,
                        'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        'object_sha256':sha(obj) if obj.exists() else None,'log_sha256':sha(log)})
        print(json.dumps({'source':str(source),'exit_code':run.returncode}),flush=True)
    unchanged=all(sha(Path(p))==value for p,value in inputs.items())
    result={'passed':unchanged and all(r['exit_code']==0 for r in results),'scope':__doc__,
            'portable_types':args.portable_types,'modifications':modifications,
            'copied_source_sha256':{str(p):sha(p) for p in include_root.rglob('*') if p.is_file()} if args.portable_types else {},
            'source_unchanged':unchanged,'source_sha256':inputs,'compiler':str(compiler),
            'compiler_sha256':sha(compiler),'compiler_version':subprocess.check_output([str(compiler),'--version'],text=True),
            'results':results,'probe_sha256':sha(Path(__file__))}
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
