"""Compile an isolated original Aphros projection driver with MinGW extended scalars."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


SOURCES='''geom/mesh.cpp solver/solver.cpp solver/embed.cpp solver/approx.cpp solver/approx_eb.cpp
solver/convdiffi.cpp solver/convdiffe.cpp solver/convdiffvg.cpp solver/proj_eb.cpp
util/linear.cpp util/convdiff.cpp util/fluid.cpp util/suspender.cpp util/logger.cpp
util/format.cpp util/mpi.cpp util/distr.cpp util/sysinfo.cpp util/system.c util/filesystem.cpp
util/git.cpp util/gitgen.cpp util/histogram.cpp util/timer.cpp util/subcomm_dummy.cpp
parse/vars.cpp parse/parser.cpp parse/codeblocks.cpp parse/evalexpr.cpp parse/argparse.cpp
distr/distr.cpp distr/local.cpp distr/native.cpp distr/comm_manager.cpp distr/distr_particles.cpp
distr/distrsolver.cpp distr/distrbasic.cpp distr/report.cpp
dump/dumper.cpp dump/vtk.cpp dump/raw.cpp dump/xmf.cpp dump/hdf.cpp func/primlist.cpp
linear/linear.cpp parse/template.cpp inside/main.c'''.split()


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--jobs',type=int,default=2);parser.add_argument('--amg',action='store_true')
    parser.add_argument('--optimization',choices=('1','2','3'),default='1',help='Compiler optimization level; fast-math and FP contraction remain disabled')
    args=parser.parse_args()
    src=args.sources.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    if not 1<=args.jobs<=4:raise ValueError('Use one to four compiler processes')
    before={str(p):sha(p) for p in src.rglob('*') if p.is_file()}
    shutil.copyfile(__file__,out/'build_source.py.txt')
    compiler=Path('C:/ProgramData/mingw64/mingw64/bin/g++.exe')
    flags=['-O'+args.optimization,'-ffunction-sections','-fdata-sections','-fno-fast-math','-ffp-contract=off','-fopenmp',
           '-DM_PI=3.141592653589793','-D_ALIGNBYTES_=16','-D_USE_MPI_=0','-D_USE_OPENMP_=1',
           '-D_USE_DIM1_=0','-D_USE_DIM2_=0','-D_USE_DIM3_=1','-D_USE_DIM4_=0','-D_USE_AVX_=0',
           '-D_USE_HDF_=0','-D_USE_HYPRE_=0','-D_USE_AMGX_=0','-D_USE_FPZIP_=0','-D_USE_OPENCL_=0',
           '-D_USE_BACKEND_CUBISM_=0','-D_USE_BACKEND_LOCAL_=1','-D_USE_BACKEND_NATIVE_=1',
           '-I'+str(src),'-I'+str(src/'src'),'-I'+str(src/'src/solver'),'-I'+str(src/'src/geom')]
    paths=[src/'driver.cpp']+[src/'src'/name for name in SOURCES]
    if args.amg:
        flags+=['-DAPHROS_TWISTED_HAVE_AMGCL','-ID:/Dropbox/Agent-simulation/twisted-baseline/amgcl',
                '-IC:/Users/bear/AppData/Local/.xmake/packages/e/eigen/5.0.1/60cb40bd086c4c96b9551ce6ccff0e36/include/eigen3']
    def compile(source):
        name=source.relative_to(src).as_posix().replace('/','__');obj=out/(name+'.o');log=out/(name+'.log')
        cc=compiler.with_name('gcc.exe') if source.suffix=='.c' else compiler
        command=[str(cc),'-std=c11' if source.suffix=='.c' else '-std=c++17',*flags,'-c',str(source),'-o',str(obj)]
        start=now()
        with log.open('w') as f:code=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT).returncode
        row={'source':str(source),'command':command,'started_utc':start,'completed_utc':now(),
             'exit_code':code,'object':str(obj),'object_sha256':sha(obj) if obj.exists() else None,'log_sha256':sha(log)}
        print(json.dumps({'source':name,'exit_code':code}),flush=True);return row
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:results=list(pool.map(compile,paths))
    linked=False;link_code=None;exe=out/'twisted_extended.exe'
    if all(r['exit_code']==0 for r in results):
        command=[str(compiler),'-fopenmp','-static','-Wl,--gc-sections',*[r['object'] for r in results],'-lpsapi','-o',str(exe)]
        with (out/'link.log').open('w') as f:link_code=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT).returncode
        linked=link_code==0
    unchanged=all(sha(Path(name))==value for name,value in before.items())
    result={'scope':__doc__+' A successful build alone does not validate a flow result.',
            'passed':linked and unchanged,'source_unchanged':unchanged,'source_sha256':before,
            'compiler_sha256':sha(compiler),'results':results,'link_exit_code':link_code,
            'executable_sha256':sha(exe) if linked else None,'completed_utc':now()}
    (out/'build_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':result['passed'],'compile_failures':sum(r['exit_code']!=0 for r in results),'link_exit_code':link_code}),flush=True)
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
