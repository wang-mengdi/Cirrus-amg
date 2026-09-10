"""Prepare a new full-domain resolution while preserving the original flow settings.

This writes configuration inputs only. It never manufactures a flow run record.
"""
import argparse
import json
from pathlib import Path
import re
from run_twisted_solver import sha

def resized(source, ny):
    cfg=json.loads((source/'case_manifest.json').read_text())
    if sha(source/'a.conf')!=cfg['config_sha256']:raise ValueError('Source case changed')
    if ny<4 or cfg['shape']!=[2*cfg['ny'],cfg['ny'],cfg['ny']]:raise ValueError('Require the full twisted-pipe domain')
    text=(source/'a.conf').read_text()
    for axis,n in zip('xyz',(2*ny,ny,ny)):
        text,count=re.subn(r'^set int bs'+axis+r' \d+$',f'set int bs{axis} {n}',text,flags=re.M)
        if count!=1:raise ValueError('Unknown original mesh control')
    cfg['ny']=ny;cfg['shape']=[2*ny,ny,ny]
    return text,cfg

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--ny',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source=a.reference.resolve();out=a.output.resolve();text,cfg=resized(source,a.ny)
    out.mkdir(parents=True,exist_ok=False);(out/'a.conf').write_text(text)
    cfg['config_sha256']=sha(out/'a.conf')
    (out/'case_manifest.json').write_text(json.dumps(cfg,indent=2)+'\n')
    record={'scope':__doc__,'source_sha256':{str(f):sha(f) for f in (source/'a.conf',source/'case_manifest.json',Path(__file__).resolve())},
        'ny':a.ny,'shape':cfg['shape'],'config_sha256':cfg['config_sha256'],'no_flow_trajectory':True}
    (out/'resolution_preparation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))

if __name__=='__main__':main()
