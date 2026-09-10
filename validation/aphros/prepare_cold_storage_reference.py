"""Prepare an isolated driver that releases proven-unused construction caches.

Original Proj, Embed initialization, advection, diffusion and gradient bodies
remain byte-identical. Released-cache getters fail explicitly on later use.
"""
import argparse
import difflib
import json
from pathlib import Path
import shutil
from build_extended_reference import sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-build',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();base=a.reference_build.resolve();out=a.output.resolve()
    build=json.loads((base/'build_manifest.json').read_text())
    if not build['passed'] or not build['source_unchanged']:raise ValueError('Require a verified build')
    for path,h in build['source_sha256'].items():
        if sha(Path(path))!=h:raise ValueError('Reference build source changed: '+path)
    drivers=[Path(r['source']) for r in build['results'] if Path(r['source']).name=='driver.cpp']
    if len(drivers)!=1:raise ValueError('Require one actual compiled driver')
    if sha(base/'twisted_extended.exe')!=build['executable_sha256']:raise ValueError('Reference executable changed')
    old=drivers[0].parent;shutil.copytree(old,out);patch=[]
    def edit(relative,replacements):
        before=(old/relative).read_text();after=before
        for anchor,replacement in replacements:
            if after.count(anchor)!=1:raise ValueError('Unknown source anchor: '+anchor)
            after=after.replace(anchor,replacement)
        (out/relative).write_text(after)
        patch.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),
            fromfile='original/'+relative,tofile='candidate/'+relative))
    edit('driver.cpp',[
        ('#include "twisted_initial_velocity.h"','#include "twisted_initial_velocity.h"\n#include "twisted_cold_storage.h"'),
        ('    TwistedTimeDump(m,*t.geometry,t.velocity,0.,true);',
         '''    TwistedTimeDump(m,*t.geometry,t.velocity,0.,true);
    if(std::getenv("APHROS_TWISTED_RELEASE_COLD_STORAGE"))
      TwistedColdStorage::Release(*t.geometry,t.levelset,t.velocity);''')])
    guards=[]
    for name,field,return_type in [('GetCellCenter','fc_cell_center_','Vect'),
        ('GetSignedDistance','fc_sdf_','Scal'),('GetDisplacementToCutFace','fc_cutdx_','Vect')]:
        anchor=f'  {return_type} {name}(IdxCell c) const {{\n    return {field}[c];'
        guards.append((anchor,f'  {return_type} {name}(IdxCell c) const {{\n    if({field}.empty())throw std::runtime_error("Released geometry cache: {field}");\n    return {field}[c];'))
    edit('src/solver/embed.h',[
        ('  friend struct TwistedGeometryState;','  friend struct TwistedGeometryState;\n  friend struct TwistedColdStorage;'),
        ('  std::vector<Vect> GetFacePoly(IdxFace f) const {',
         '  std::vector<Vect> GetFacePoly(IdxFace f) const {\n    if(ffpoly_.empty())throw std::runtime_error("Released geometry cache: face polygons");'),
        ('  void DumpPoly(std::string filename, bool vtkbin, bool vtkmerge) const {',
         '  void DumpPoly(std::string filename, bool vtkbin, bool vtkmerge) const {\n    if(ffpoly_.empty())throw std::runtime_error("Released geometry cache: face polygons");'),
        *guards])
    helper=Path(__file__).with_name('twisted_cold_storage.h');shutil.copyfile(helper,out/helper.name)
    changed=sorted(p.relative_to(old).as_posix() for p in old.rglob('*') if p.is_file() and sha(p)!=sha(out/p.relative_to(old)))
    if changed!=['driver.cpp','src/solver/embed.h']:raise ValueError('Unexpected existing source changes')
    for relative in ('solver/proj.ipp','solver/proj.h','solver/embed.ipp','solver/approx_eb.ipp','solver/convdiffi.ipp','solver/convdiffe.ipp'):
        if (old/'src'/relative).read_bytes()!=(out/'src'/relative).read_bytes():raise ValueError('Numerical algorithm changed')
    prepared=json.loads((out/'prepare_manifest.json').read_text())
    for path in (Path(__file__).resolve(),helper.resolve()):prepared['source_sha256'][str(path)]=sha(path)
    for relative in (*changed,helper.name):
        keys=[k for k in prepared['prepared_source_sha256'] if Path(k).as_posix()==relative]
        if len(keys)>1:raise ValueError('Duplicate prepared source')
        prepared['prepared_source_sha256'][keys[0] if keys else relative]=sha(out/relative)
    prepared['cold_storage_scope']=__doc__
    prepared['cold_storage_reference_build_sha256']=sha(base/'build_manifest.json')
    (out/'prepare_manifest.json').write_text(json.dumps(prepared,indent=2)+'\n')
    (out/'cold_storage.patch').write_text(''.join(patch))
    assert all(sha(Path(path))==h for path,h in build['source_sha256'].items())
    print(json.dumps({'prepared':str(out),'changed_existing_sources':changed,'original_algorithm_bodies_unchanged':True}))

if __name__=='__main__':main()
