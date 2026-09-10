"""Capture a completed physical prefix, optionally from an identified live native process."""
import argparse
import json
from pathlib import Path
from twisted_prefix_restart import create_checkpoint

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--step',type=int,required=True);p.add_argument('--pid',type=int);p.add_argument('--creation-time',type=float)
    a=p.parse_args();r=create_checkpoint(a.run,a.step,a.output,a.pid,a.creation_time)
    print(json.dumps({k:r[k] for k in ('format','physical_step','physical_time','cells','faces','state_sha256','parent_observation')}))
