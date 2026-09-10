"""Create an isolated ConvDiffScalExp translation unit; never change the baseline library.

The only insertion before the original RedistributeCutCells call is a diagnostic
hook with an opt-in single-block periodic residual halo repair. All operators
and the surrounding SIMPLE algorithm are compiled from the upstream sources.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aphros',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=args.aphros.resolve();out=args.output.resolve()
    if out.exists():raise ValueError('Use a fresh source directory')
    sources=[root/'src/solver/convdiffe.cpp',root/'src/solver/convdiffe.ipp',
             Path(__file__).with_name('twisted_explicit_residual.h').resolve()]
    body=sources[1].read_text(encoding='utf-8')
    anchor='    fclb = UEB::RedistributeCutCells(fclb, eb);'
    if body.count(anchor)!=1:raise ValueError('Upstream assembly anchor changed')
    body=body.replace('#include "convdiffe.h"','#include "convdiffe.h"\n#include "twisted_explicit_residual.h"')
    body=body.replace(anchor,'    static int twisted_call=0;\n    TwistedExplicitResidual(fclb,eb,m,twisted_call++);\n'+anchor)
    out.mkdir(parents=True)
    (out/'convdiffe.cpp').write_bytes(sources[0].read_bytes())
    (out/'convdiffe.ipp').write_text(body,encoding='utf-8')
    (out/'twisted_explicit_residual.h').write_bytes(sources[2].read_bytes())
    record={'scope':__doc__,'inputs':{str(p):sha(p) for p in sources},
            'generated':{str(p):sha(p) for p in out.iterdir()}}
    (out/'override_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(out/'convdiffe.cpp')


if __name__=='__main__':main()
