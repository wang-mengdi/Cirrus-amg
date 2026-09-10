"""Add an optional initial-velocity input to an isolated original Aphros driver."""
import argparse
import difflib
import json
from pathlib import Path
import shutil
from build_extended_reference import sha

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-build',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();base=args.reference_build.resolve();out=args.output.resolve()
    build=json.loads((base/'build_manifest.json').read_text())
    if not build['passed'] or not build['source_unchanged']:raise ValueError('Require a verified build')
    for path,digest in build['source_sha256'].items():
        if sha(Path(path))!=digest:raise ValueError('Reference source changed: '+path)
    drivers=[Path(r['source']) for r in build['results'] if Path(r['source']).name=='driver.cpp']
    if len(drivers)!=1:raise ValueError('Require one compiled driver')
    original=drivers[0].parent;shutil.copytree(original,out)
    before=(original/'driver.cpp').read_text();after=before
    changes={
        '#include "twisted_pressure_face_snapshot.h"':'#include "twisted_pressure_face_snapshot.h"\n#include "twisted_initial_velocity.h"',
        'var.String["vel_init"]!="zero"':'(var.String["vel_init"]!="zero" && var.String["vel_init"]!="twisted_seed")',
        '    std::shared_ptr<linear::Solver<M>> linear=ULinear<M>::MakeLinearSolver(var,"symm",m);':
        '''    const char* initial_velocity=std::getenv("APHROS_TWISTED_INITIAL_VELOCITY");
    if((var.String["vel_init"]=="twisted_seed")!=bool(initial_velocity))
      throw std::runtime_error("Initial velocity mode and input file must agree");
    if(initial_velocity)TwistedLoadInitialVelocity(initial_velocity,m,*t.geometry,t.velocity);
    std::shared_ptr<linear::Solver<M>> linear=ULinear<M>::MakeLinearSolver(var,"symm",m);''',
    }
    for a,b in changes.items():
        if after.count(a)!=1:raise ValueError('Unknown driver anchor: '+a)
        after=after.replace(a,b)
    (out/'driver.cpp').write_text(after)
    helper=Path(__file__).with_name('twisted_initial_velocity.h')
    shutil.copyfile(helper,out/helper.name)
    changed=[p.relative_to(original).as_posix() for p in original.rglob('*')
             if p.is_file() and sha(p)!=sha(out/p.relative_to(original))]
    if changed!=['driver.cpp']:raise ValueError('Unexpected existing source changes')
    prepared=json.loads((out/'prepare_manifest.json').read_text())
    for p in (Path(__file__).resolve(),helper.resolve()):prepared['source_sha256'][str(p)]=sha(p)
    prepared['prepared_source_sha256']['driver.cpp']=sha(out/'driver.cpp')
    prepared['prepared_source_sha256'][helper.name]=sha(out/helper.name)
    prepared['initial_velocity_scope']=__doc__+' Original ProjArgs::fcvel only; no pressure, flux, operator, or time-history restoration.'
    prepared['initial_velocity_reference_build_sha256']=sha(base/'build_manifest.json')
    (out/'prepare_manifest.json').write_text(json.dumps(prepared,indent=2)+'\n')
    (out/'initial_velocity.patch').write_text(''.join(difflib.unified_diff(
        before.splitlines(True),after.splitlines(True),fromfile='original/driver.cpp',tofile='candidate/driver.cpp')))
    assert all(sha(Path(p))==h for p,h in build['source_sha256'].items())
    print(json.dumps({'prepared':str(out),'changed_existing_sources':changed,'original_sources_unchanged':True}))

if __name__=='__main__':main()
