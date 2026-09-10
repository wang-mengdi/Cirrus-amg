"""Prepare an isolated reference with demand-zero allocation, copy, and same-address reset.

Change only Vector/GField storage, preserving full geometry, original equation
bodies, scalar types, array indexing, physical configuration, and tolerances.
"""
import argparse
import difflib
import json
from pathlib import Path
import shutil

from build_extended_reference import sha
import prepare_zero_page_reference as base_preparer
from prepare_zero_page_reference import transform as transform_base


def transform(body):
    body=transform_base(body)
    body=body.replace('"twisted_zero_pages.h"','"twisted_zero_pages_reset.h"')
    anchor='  bool virtual_allocation() const noexcept {return virtual_allocation_;}'
    if body.count(anchor)!=1:raise ValueError('Unknown Vector anchor')
    body=body.replace(anchor,anchor+"""
  bool ResetToZero() {
    return owning_ && twisted_zero_pages::Reset(data_,size_,virtual_allocation_,pristine_zero_);
  }""")
    anchor='    if(data_.pristine_zero() && twisted_zero_pages::PositiveZero(value))return;'
    if body.count(anchor)!=1:raise ValueError('Unknown field zero-fill anchor')
    body=body.replace(anchor,anchor+"""
    if(size_t(*range_.begin())==0 && size_t(*range_.end())==data_.size() &&
       twisted_zero_pages::PositiveZero(value) && data_.ResetToZero())return;""")
    return body


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-build',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    base=args.reference_build.resolve();out=args.output.resolve()
    build=json.loads((base/'build_manifest.json').read_text())
    if not build['passed'] or not build['source_unchanged']:raise ValueError('Require an unchanged verified build')
    for path,digest in build['source_sha256'].items():
        if sha(Path(path))!=digest:raise ValueError('Reference source changed: '+path)
    drivers=[r for r in build['results'] if Path(r['source']).name=='driver.cpp']
    if len(drivers)!=1:raise ValueError('Require one compiled driver')
    original=Path(drivers[0]['source']).parent
    shutil.copytree(original,out)
    relative=Path('src/geom/field.h')
    before=(original/relative).read_text();after=transform(before)
    (out/relative).write_text(after)
    helper=Path(__file__).with_name('twisted_zero_pages_reset.h')
    shutil.copyfile(helper,out/helper.name)
    changes=[p.relative_to(original).as_posix() for p in original.rglob('*')
             if p.is_file() and sha(p)!=sha(out/p.relative_to(original))]
    if changes!=[relative.as_posix()]:raise ValueError('Unexpected existing source changes')
    prepared=json.loads((out/'prepare_manifest.json').read_text())
    for path in (Path(__file__).resolve(),helper.resolve(),Path(base_preparer.__file__).resolve()):
        prepared['source_sha256'][str(path)]=sha(path)
    keys=[k for k in prepared['prepared_source_sha256'] if Path(k).as_posix()==relative.as_posix()]
    if len(keys)!=1:raise ValueError('Require one original field source hash')
    prepared['prepared_source_sha256'][keys[0]]=sha(out/relative)
    prepared['prepared_source_sha256'][helper.name]=sha(out/helper.name)
    prepared['zero_page_scope']=__doc__
    prepared['reference_build_manifest_sha256']=sha(base/'build_manifest.json')
    (out/'prepare_manifest.json').write_text(json.dumps(prepared,indent=2)+'\n')
    (out/'field_storage.patch').write_text(''.join(difflib.unified_diff(
        before.splitlines(True),after.splitlines(True),fromfile='original/'+relative.as_posix(),tofile='candidate/'+relative.as_posix())))
    if any(sha(Path(p))!=h for p,h in build['source_sha256'].items()):raise ValueError('Original source changed')
    print(json.dumps({'prepared':str(out),'changed_existing_sources':changes,'new_header':helper.name,
                      'original_sources_unchanged':True,'scope':__doc__}),flush=True)


if __name__=='__main__':main()
