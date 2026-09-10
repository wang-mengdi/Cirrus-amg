"""Prepare an isolated reference with demand-zero storage and zero-page-aware copies.

Change only Vector/GField storage, preserving full geometry, original equation
bodies, scalar types, array indexing, physical configuration, and tolerances.
"""
import argparse
import difflib
import json
from pathlib import Path
import shutil

from build_extended_reference import sha


def transform(body):
    def replace(old,new):
        nonlocal body
        if body.count(old)!=1:raise ValueError('Unknown field storage anchor: '+old)
        body=body.replace(old,new)
    replace('#include "util/logger.h"','#include "util/logger.h"\n#include "twisted_zero_pages.h"')
    replace('''      delete[] data_;
    }
  }
  explicit Vector(size_t size)
      : size_(size), data_(new T[size]), owning_(true) {}''','''      twisted_zero_pages::Release(data_,virtual_allocation_);
    }
  }
  explicit Vector(size_t size) : size_(size) {
    data_=twisted_zero_pages::Allocate<T>(size_,virtual_allocation_,pristine_zero_);
  }''')
    replace('''      : size_(size), data_(new T[size]), owning_(true) {
    std::fill(data_, data_ + size_, value);
  }''','''      : Vector(size) {
    if(pristine_zero_ && twisted_zero_pages::PositiveZero(value))return;
    std::fill(data_, data_ + size_, value);pristine_zero_=false;
  }''')
    replace('''      : size_(other.size_), data_(new T[other.size_]), owning_(true) {
    std::copy(other.data_, other.data_ + size_, data_);
  }''','''      : Vector(other.size_) {
    twisted_zero_pages::Copy(other.data_,data_,size_,virtual_allocation_,pristine_zero_);
  }''')
    replace('''      : size_(other.size_), data_(other.data_), owning_(other.owning_) {
    other.size_ = 0;''','''      : size_(other.size_), data_(other.data_), owning_(other.owning_),
        virtual_allocation_(other.virtual_allocation_),pristine_zero_(other.pristine_zero_.load(std::memory_order_relaxed)) {
    other.virtual_allocation_=false;other.pristine_zero_=false;
    other.size_ = 0;''')
    replace('''  Vector& operator=(Vector&& other) {
    if (owning_) {
      delete[] data_;
    }
    size_ = other.size_;''','''  Vector& operator=(Vector&& other) {
    if (owning_) {
      twisted_zero_pages::Release(data_,virtual_allocation_);
    }
    virtual_allocation_=other.virtual_allocation_;pristine_zero_=other.pristine_zero_.load(std::memory_order_relaxed);
    other.virtual_allocation_=false;other.pristine_zero_=false;
    size_ = other.size_;''')
    replace('''  T* data() noexcept {
    return data_;
  }''','''  T* data() noexcept {
    if(pristine_zero_.load(std::memory_order_relaxed))pristine_zero_.store(false,std::memory_order_relaxed);
    return data_;
  }''')
    replace('''  T& operator[](size_t i) noexcept {
    return data_[i];
  }''','''  T& operator[](size_t i) noexcept {
    if(pristine_zero_.load(std::memory_order_relaxed))pristine_zero_.store(false,std::memory_order_relaxed);
    return data_[i];
  }''')
    replace('''  bool owning() const noexcept {
    return owning_;
  }''','''  bool owning() const noexcept {
    return owning_;
  }
  bool pristine_zero() const noexcept {return pristine_zero_.load(std::memory_order_relaxed);}
  bool virtual_allocation() const noexcept {return virtual_allocation_;}''')
    replace('''  bool owning_ = true; // true if memory is managed by the object
};''','''  bool owning_ = true; // true if memory is managed by the object
  bool virtual_allocation_ = false;
  std::atomic<bool> pristine_zero_{false};
};''')
    replace('''  void Reinit(const Range& range, const Value& value) {
    Reinit(range);
    for (auto i : range_) {''','''  void Reinit(const Range& range, const Value& value) {
    Reinit(range);
    if(data_.pristine_zero() && twisted_zero_pages::PositiveZero(value))return;
    for (auto i : range_) {''')
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
    helper=Path(__file__).with_name('twisted_zero_pages.h')
    shutil.copyfile(helper,out/helper.name)
    changes=[p.relative_to(original).as_posix() for p in original.rglob('*')
             if p.is_file() and sha(p)!=sha(out/p.relative_to(original))]
    if changes!=[relative.as_posix()]:raise ValueError('Unexpected existing source changes')
    prepared=json.loads((out/'prepare_manifest.json').read_text())
    for path in (Path(__file__).resolve(),helper.resolve()):
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
