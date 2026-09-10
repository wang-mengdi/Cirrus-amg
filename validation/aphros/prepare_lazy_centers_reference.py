"""Compute Cartesian centers using their original formulas instead of storing arrays.

This optional validation-build storage change preserves all original CFD
equation bodies. The default constructor and access path retain stored centers.
"""
import argparse
import difflib
import json
from pathlib import Path
import shutil
from build_extended_reference import sha

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference-build',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();base=a.reference_build.resolve();out=a.output.resolve()
    build=json.loads((base/'build_manifest.json').read_text())
    if not build['passed'] or not build['source_unchanged']:raise ValueError('Require an unchanged verified build')
    for name,h in build['source_sha256'].items():
        if sha(Path(name))!=h:raise ValueError('Compiled source changed: '+name)
    drivers=[Path(row['source']) for row in build['results'] if Path(row['source']).name=='driver.cpp']
    if len(drivers)!=1:raise ValueError('Require one compiled driver')
    if sha(base/'twisted_extended.exe')!=build['executable_sha256']:raise ValueError('Reference executable changed')
    old=drivers[0].parent;shutil.copytree(old,out);patch=[]
    def edit(relative,changes):
        before=(old/relative).read_text();after=before
        for anchor,value in changes:
            if after.count(anchor)!=1:raise ValueError('Unknown center-storage anchor: '+anchor)
            after=after.replace(anchor,value)
        (out/relative).write_text(after)
        patch.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='original/'+relative,tofile='candidate/'+relative))
    edit('src/geom/mesh.h',[
        ('#include <cassert>','#include <cassert>\n#include <cstdlib>'),
        ('  Vect GetCenter(IdxCell c) const {\n    return fc_center_[c];',
         '''  Vect GetCenter(IdxCell c) const {
    if(twisted_lazy_centers_) {
      return (domain_.low + half_cell_size_) +
             Vect(indexc_.GetMIdx(c) - incells_begin_) * cell_size_;
    }
    return fc_center_[c];'''),
        ('  Vect GetCenter(IdxFace f) const {\n    return ff_center_[f];',
         '''  Vect GetCenter(IdxFace f) const {
    if(twisted_lazy_centers_) {
      auto p = indexf_.GetMIdxDir(f);
      const MIdx& w = p.first;
      size_t d(p.second);
      Vect r = (domain_.low + half_cell_size_) +
               Vect(w - incells_begin_) * cell_size_;
      r[d] -= half_cell_size_[d];
      return r;
    }
    return ff_center_[f];'''),
        ('  FieldCell<Vect> fc_center_;',
         '  const bool twisted_lazy_centers_ = std::getenv("APHROS_TWISTED_LAZY_CENTERS") != nullptr;\n  FieldCell<Vect> fc_center_;'),
        ('  Vect GetCenter(IdxCell c) const {',
         '''  // Read-only validation metadata; no coordinate array is exposed.
  size_t TwistedCenterStorageBytes() const {
    return (fc_center_.size()+ff_center_.size())*sizeof(Vect);
  }
  bool TwistedUsesLazyCenters() const {return twisted_lazy_centers_;}
  Vect GetCenter(IdxCell c) const {''')])
    edit('src/geom/mesh.ipp',[
        ('  { // cell centers\n    fc_center_.Reinit(*this);',
         '  if(!twisted_lazy_centers_) { // cell centers\n    fc_center_.Reinit(*this);'),
        ('  { // face centers\n    ff_center_.Reinit(*this);',
         '  if(!twisted_lazy_centers_) { // face centers\n    ff_center_.Reinit(*this);')])
    edit('driver.cpp',[
        ('    TwistedTimeDump(m,*t.geometry,t.velocity,0.,true);',
         '''    TwistedTimeDump(m,*t.geometry,t.velocity,0.,true);
    {
      std::ofstream storage("mesh_center_storage.json");
      storage<<"{\\\"lazy_centers\\\":"<<(m.TwistedUsesLazyCenters()?"true":"false")
             <<",\\\"coordinate_array_bytes\\\":"<<m.TwistedCenterStorageBytes()<<"}\\n";
      storage.flush();if(!storage.good())throw std::runtime_error("Center-storage metadata failed");
    }''')])
    changed=sorted(path.relative_to(old).as_posix() for path in old.rglob('*') if path.is_file() and sha(path)!=sha(out/path.relative_to(old)))
    if changed!=['driver.cpp','src/geom/mesh.h','src/geom/mesh.ipp']:raise ValueError('Unexpected source changes')
    for relative in ('solver/proj.ipp','solver/proj.h','solver/embed.ipp','solver/approx_eb.ipp','solver/convdiffi.ipp','solver/convdiffe.ipp'):
        if (old/'src'/relative).read_bytes()!=(out/'src'/relative).read_bytes():raise ValueError('Original CFD equation body changed')
    prepared=json.loads((out/'prepare_manifest.json').read_text())
    prepared['source_sha256'][str(Path(__file__).resolve())]=sha(Path(__file__))
    for relative in changed:
        keys=[key for key in prepared['prepared_source_sha256'] if Path(key).as_posix()==relative]
        if len(keys)!=1:raise ValueError('Require one prepared source key')
        prepared['prepared_source_sha256'][keys[0]]=sha(out/relative)
    prepared['lazy_centers_scope']=__doc__
    prepared['lazy_centers_reference_build_sha256']=sha(base/'build_manifest.json')
    (out/'prepare_manifest.json').write_text(json.dumps(prepared,indent=2)+'\n')
    (out/'lazy_centers.patch').write_text(''.join(patch))
    assert all(sha(Path(path))==h for path,h in build['source_sha256'].items())
    print(json.dumps({'prepared':str(out),'changed_existing_sources':changed,'original_CFD_equation_bodies_unchanged':True}))

if __name__=='__main__':main()
