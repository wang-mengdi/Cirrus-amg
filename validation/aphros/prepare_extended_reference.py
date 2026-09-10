"""Prepare isolated Aphros sources with optional scalar conversion and complete geometry I/O."""
import argparse
import json
import hashlib
from pathlib import Path
import shutil
import re


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aphros',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--extended',action='store_true')
    parser.add_argument('--portable-platform',action='store_true')
    parser.add_argument('--portable-output',action='store_true')
    parser.add_argument('--extended-amg',action='store_true')
    parser.add_argument('--exact-flow-dump',action='store_true',help='Add read-only hexadecimal dumps without changing the existing outputs or numerical expressions')
    parser.add_argument('--scalar-value-cache',action='store_true',help='Ignore long-double storage padding when checking exact coefficient equality for AMG reuse')
    args=parser.parse_args()
    root=args.aphros.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    here=Path(__file__).resolve().parent;inputs={};changes={}
    for source in (root/'src').rglob('*'):
        if source.is_file() and source.suffix in ('.cpp','.h','.ipp','.c','.inc'):
            dest=out/'src'/source.relative_to(root/'src');dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,dest);inputs[str(source)]=sha(source)
    def replace(relative,old,new,count=1):
        path=out/'src'/relative;body=path.read_bytes()
        if body.count(old)!=count:raise ValueError('Unexpected source anchor: '+relative)
        path.write_bytes(body.replace(old,new));changes[relative]=changes.get(relative,[])+[old.decode()]
    replace('solver/embed.h',b'class Embed {',b'class Embed {\n  friend struct TwistedGeometryState;')
    if args.extended:
        for source in (out/'src').rglob('*.cpp'):
            body=source.read_bytes()
            if b'MeshCartesian<double' in body:
                source.write_bytes(body.replace(b'MeshCartesian<double',b'MeshCartesian<long double'))
                changes[source.relative_to(out/'src').as_posix()]=['Explicit mesh scalar instantiations become long double']
        replace('geom/vect.h',b'struct GetNanHelper {',
                b'struct GetNanHelper {\n  static auto Get(long double*) { return std::numeric_limits<long double>::quiet_NaN(); }')
        replace('util/fluid.h',b'std::max(0.,',b'std::max(Scal(0),',4)
        replace('util/fluid.h',b'std::min(0.,',b'std::min(Scal(0),',3)
        if args.portable_platform:
            replace('kernel/kernelmesh.h',b'using Vect = generic::Vect<double, dim>;',
                    b'using Vect = generic::Vect<long double, dim>;')
            replace('solver/reconst.h',b'std::min(1., (f - ny) / nx)',b'std::min(Scal(1), (f - ny) / nx)')
            for relative in ('solver/convdiffi.ipp','solver/convdiffe.ipp'):
                replace(relative,b'GetGradCoeffs(0.,',b'GetGradCoeffs(Scal(0),')
            replace('solver/approx.cpp',b'using Scal = double;',b'using Scal = long double;')
            replace('distr/distr.ipp',b'<< sched_getcpu();',
                    b'<< -1; // CPU affinity reporting unavailable in this Windows probe.')
            replace('util/system_windows.inc',b'#if !define(S_ISLNK)',b'#if !defined(S_ISLNK)')
        if args.portable_output:
            for relative in ('dump/vtk.cpp','dump/xmf.cpp','func/primlist.cpp'):
                replace(relative,b'generic::Vect<double, dim>',b'generic::Vect<long double, dim>')
        # First establish a linked original extended-CG driver. The existing
        # validation-only Eigen<double> backend cannot retain extended pressure.
        (out/'src/linear/twisted_direct.h').write_text('''#pragma once
template<class M> class TwistedSerialDirect {
 public:
  typename linear::Solver<M>::Info Solve(const FieldCell<typename M::Expr>&,
      FieldCell<typename M::Scal>&,M&,typename M::Scal) {
    throw std::runtime_error("Double validation backend disabled for extended scalar driver");
  }
};
''')
        changes['linear/twisted_direct.h']=['Explicitly reject the double validation backend; original extended CG remains available']
        if args.extended_amg:
            source=here/'twisted_direct.h';body=source.read_text();inputs[str(source)]=sha(source)
            body=re.sub(r'\bdouble\b','Scal',body)
            body=body.replace('class TwistedSerialDirect {','class TwistedSerialDirect {\n  using Scal=typename M::Scal;\n  using Dense=Eigen::Matrix<Scal,Eigen::Dynamic,1>;')
            body=body.replace('Eigen::VectorXd','Dense')
            body=body.replace('p.solver.tol=1e-14;','p.solver.tol=1e-18L;')
            if args.scalar_value_cache:
                anchor='std::memcmp(cached_.valuePtr(),matrix.valuePtr(),matrix.nonZeros()*sizeof(Scal))==0'
                if body.count(anchor)!=1:raise ValueError('Unexpected scalar cache comparison')
                body=body.replace(anchor,'''std::equal(cached_.valuePtr(),cached_.valuePtr()+matrix.nonZeros(),matrix.valuePtr(),
          [](Scal a,Scal b) { return a==b && (a!=Scal(0) || std::signbit(a)==std::signbit(b)); })''')
                body=body.replace('// Least recently used first. A hit requires identical compressed matrix bytes;',
                    '// Least recently used first. Compare exact scalar values and signed zero, excluding long-double padding;')
            (out/'src/linear/twisted_direct.h').write_text(body)
            changes['linear/twisted_direct.h']=['Retain scalar precision in the existing original-row validation backend and AMG; tighten internal Krylov tolerance to 1e-18 without changing requested original-row checks']
            if args.scalar_value_cache:changes['linear/twisted_direct.h'].append('Compare exact scalar coefficient values and zero sign, not indeterminate long-double padding; compressed indices remain byte-identical')
    for source in here.glob('twisted_*.h'):
        shutil.copyfile(source,out/source.name);inputs[str(source)]=sha(source)
    source=here/'twisted_projection_driver.cpp';body=source.read_text();inputs[str(source)]=sha(source)
    body=body.replace('#include "twisted_pressure_snapshot.h"','#include "twisted_pressure_snapshot.h"\n#include "twisted_geometry_state.h"')
    if args.extended:body=body.replace('using M=MeshCartesian<double,3>;','using M=MeshCartesian<long double,3>;')
    if args.exact_flow_dump:
        body=body.replace('#include "twisted_geometry_state.h"','#include "twisted_geometry_state.h"\n#include "twisted_exact_flow.h"')
        anchor='    DumpFlow(m,*t.geometry,*t.solver);'
        if body.count(anchor)!=1:raise ValueError('Unexpected final flow dump anchor')
        body=body.replace(anchor,anchor+'\n    TwistedExactFlowDump(m,*t.geometry,*t.solver);')
        changes['driver.cpp']=['Read-only exact hexadecimal output after the unchanged original flow dump']
    anchor='  if(sem.Nested("embed"))t.geometry->Init(t.levelset);'
    if body.count(anchor)!=1:raise ValueError('Unexpected original Embed initialization')
    body=body.replace(anchor,'''  const char* input_geometry=std::getenv("APHROS_TWISTED_GEOMETRY_STATE_IN");
  if(input_geometry) {
    if(sem("embed-load"))TwistedGeometryState::Transfer(*t.geometry,input_geometry,true);
  } else if(sem.Nested("embed"))t.geometry->Init(t.levelset);
  if(std::getenv("APHROS_TWISTED_GEOMETRY_STATE_ONLY")) {
    if(sem("geometry-state")) {
      if(const char* path=std::getenv("APHROS_TWISTED_GEOMETRY_STATE_OUT"))TwistedGeometryState::Transfer(*t.geometry,path,false);
      TwistedGeometryDump(m,*t.geometry);
      std::cout<<"End of geometry-only snapshot; no flow trajectory"<<std::endl;
    }
    return;
  }''')
    (out/'driver.cpp').write_text(body)
    shutil.copyfile(__file__,out/'prepare_source.py.txt')
    unchanged=all(sha(Path(name))==value for name,value in inputs.items())
    manifest={'scope':__doc__,'extended':args.extended,'source_unchanged':unchanged,'source_sha256':inputs,
              'changes':changes,'prepared_source_sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}}
    (out/'prepare_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if not unchanged:raise ValueError('Original sources changed during preparation')
    print(json.dumps({'prepared':str(out),'extended':args.extended,'source_unchanged':unchanged}))


if __name__=='__main__':main()
