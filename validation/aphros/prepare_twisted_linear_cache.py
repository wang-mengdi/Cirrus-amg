"""Isolate the optional linear backend without rebuilding the Aphros library.

The original linear.cpp, linear.ipp and linear.h are copied unchanged. Only the
already optional TwistedSerialDirect backend header is replaced. The surrounding
Proj/SIMPLE discretization and updates remain in the original static library
unless --capture-pressure-faces is selected. That option copies the original
Embed Proj translation unit and adds only a read-only capture after its unchanged
face evaluation. It never rebuilds or edits the shared original library.
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
    parser.add_argument('--capture-pressure-faces',action='store_true',
                        help='Isolate original Embed Proj with one read-only face-expression hook')
    args=parser.parse_args()
    root=args.aphros.resolve();out=args.output.resolve()
    if out.exists():raise ValueError('Use a fresh source directory')
    folder=root/'src/linear'
    body=(folder/'linear.ipp').read_text(encoding='utf-8')
    if body.count('#include "twisted_direct.h"')!=1 or body.count('direct_.Solve(')!=1:
        raise ValueError('Expected the existing optional linear backend hook')
    sources=[folder/name for name in ('linear.cpp','linear.ipp','linear.h')]
    sources.extend(Path(__file__).with_name(name).resolve() for name in (
        'twisted_direct.h', 'twisted_pressure_snapshot.h', 'twisted_pressure_geometry.h',
        'twisted_pressure_face_snapshot.h'))
    if args.capture_pressure_faces:
        sources.extend(root/'src/solver'/name for name in ('proj_eb.cpp','proj.ipp','proj.h'))
    library=root/'src/libaphros_static.lib'
    out.mkdir(parents=True)
    for source in sources:(out/source.name).write_bytes(source.read_bytes())
    modifications={}
    if args.capture_pressure_faces:
        path=out/'proj.ipp';original=path.read_bytes()
        ending=b'\r\n' if b'\r\n' in original else b'\n'
        anchor=ending.join((b'      eb.LoopFaces([&](auto cf) { //',
                           b'        ffv[cf] = UEB::Eval(ctx->ffvc[cf], cf, fcp, eb);',
                           b'      });'))
        if original.count(anchor)!=1:raise ValueError('Original pressure face evaluation is not unique')
        include=b'#include "proj.h"'
        if original.count(include)!=1:raise ValueError('Expected unique proj.h include')
        modified=original.replace(include,include+ending+b'#include "twisted_pressure_face_snapshot.h"')
        modified=modified.replace(anchor,anchor+ending+
            b'      TwistedCapturePressureFaces(m, eb, ctx->ffvc, ffv, fcp, dt);')
        path.write_bytes(modified)
        modifications['proj.ipp']='Added header and read-only capture after unchanged original Eval loop'
    record={'scope':__doc__,'library':str(library),'library_sha256':sha(library),
            'inputs':{str(p):sha(p) for p in sources},'modifications':modifications,
            'generated':{str(p):sha(p) for p in out.iterdir()}}
    (out/'override_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(out/'linear.cpp')


if __name__=='__main__':main()
