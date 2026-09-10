"""Build verified native executables in an explicit external cache, protecting ongoing experiments from memory pressure."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import time
import psutil
from run_twisted_solver import sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--build-cache',type=Path,required=True)
    p.add_argument('--source-inventory',type=Path,required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];out=a.output.resolve();cache=a.build_cache.resolve()
    if cache.drive.lower()!='d:' or out.drive.lower()!='d:':raise ValueError('Use D drive for this experimental build')
    if out==cache or out in cache.parents or cache in out.parents:raise ValueError('Keep preserved build and mutable cache separate')
    out.mkdir(parents=True,exist_ok=False)
    inventory=json.loads(a.source_inventory.read_text())
    names=set(inventory['source_sha256']);names.add(Path(__file__).resolve().relative_to(repo).as_posix())
    before={name:sha(repo/name) for name in sorted(names)}
    for name in before:
        target=out/'sources'/(name+'.txt');target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(repo/name,target)
    shutil.copyfile(__file__,out/'build_script.py.txt')
    xmake='C:/Users/bear/xmake/xmake.exe';targets=('simple_channel','native_compact_gpu_audit')
    commands=[[xmake,'f','-o',str(cache)],*[[xmake,'build','-j','1',target] for target in targets]]
    now=lambda:datetime.now(timezone.utc).isoformat()
    report={'scope':__doc__,'source_sha256':before,'build_started_utc':now(),'commands':commands,
        'build_cache':str(cache),'source_inventory_sha256':sha(a.source_inventory),
        'minimum_available_bytes':int(2.5*1024**3),'resource_stop_policy':'Immediate stop of only this build process tree'}
    manifest=out/'build_manifest.json';manifest.write_text(json.dumps(report,indent=2)+'\n')
    rows=[];code=0
    with (out/'build.log').open('w') as log:
        for command in commands:
            process=subprocess.Popen(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW)
            owner=psutil.Process(process.pid);created=owner.create_time()
            row={'command':command,'pid':owner.pid,'creation_time':created,'started_utc':now(),
                'maximum_observed_tree_rss':0,'minimum_available_bytes':None,'stopped_for_memory':False}
            while process.poll() is None:
                try:
                    current=psutil.Process(owner.pid);assert current.create_time()==created
                    # xmake may change its own cwd while loading/building a
                    # target. Identify our child by birth time, executable and
                    # exact launched arguments, not a mutable working directory.
                    assert Path(current.exe()).resolve()==Path(xmake).resolve()
                    assert current.cmdline()[1:]==command[1:]
                    observed_cwd=current.cwd()
                    observed=row.setdefault('observed_working_directories',[])
                    if observed_cwd not in observed:observed.append(observed_cwd)
                    children=current.children(recursive=True)
                    identities=[(child.pid,child.create_time()) for child in children]
                    available=psutil.virtual_memory().available
                    rss=current.memory_info().rss
                    for child in children:
                        try:rss+=child.memory_info().rss
                        except psutil.NoSuchProcess:pass
                    row['maximum_observed_tree_rss']=max(row['maximum_observed_tree_rss'],rss)
                    row['minimum_available_bytes']=available if row['minimum_available_bytes'] is None else min(row['minimum_available_bytes'],available)
                    if available<2.5*1024**3:
                        # Every target was enumerated from this exact build parent.
                        # Never use executable-name matching to stop another job.
                        for pid,stamp in reversed(identities):
                            try:
                                child=psutil.Process(pid)
                                if child.create_time()==stamp:child.terminate()
                            except psutil.NoSuchProcess:pass
                        if current.create_time()==created:current.terminate()
                        row['stopped_for_memory']=True
                except psutil.NoSuchProcess:pass
                time.sleep(.25)
            code=process.wait();row.update(exit_code=code,completed_utc=now());rows.append(row)
            if code or row['stopped_for_memory']:break
    unchanged=all(sha(repo/name)==h for name,h in before.items())
    report.update(exit_code=code,source_unchanged=unchanged,build_completed_utc=now(),processes=rows)
    if not code and unchanged and not any(r['stopped_for_memory'] for r in rows):
        report['executable_sha256']={}
        for target in targets:
            exe=out/(target+'.exe');shutil.copyfile(cache/'windows/x64/release'/(target+'.exe'),exe)
            report['executable_sha256'][exe.name]=sha(exe)
    manifest.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'}),flush=True)
    if code or not unchanged or any(r['stopped_for_memory'] for r in rows):raise SystemExit(code or 1)

if __name__=='__main__':main()
