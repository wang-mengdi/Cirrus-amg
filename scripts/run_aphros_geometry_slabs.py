"""Generate packed cut geometry with the original Aphros library on bounded z slabs; no flow is computed."""
import argparse
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from run_twisted_solver import sha, windows_memory
from twisted_geometry import HEADER


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--ny',type=int,required=True)
    parser.add_argument('--slab-depth',type=int,required=True)
    parser.add_argument('--geometry-template',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if os.name!='nt':raise ValueError('This runner measures Windows process memory counters')
    build=args.build.resolve(strict=True);out=args.output.resolve()
    manifest=json.loads((build/'build_manifest.json').read_text())
    assert manifest['exit_code']==0 and manifest['source_unchanged']
    exe=Path(manifest['executable']);assert sha(exe)==manifest['executable_sha256']
    template=args.geometry_template.resolve(strict=True)
    metadata=json.loads(template.read_text(encoding='utf-8-sig'))
    spec=metadata['geometry_spec']
    assert spec['extent']==[.25,.125,.125]
    for key,value in [('radius',.035),('amplitude',.015),('period',.25),('center_y',.0625),('center_z',.0625)]:
        assert spec[key]==value,(key,'Export driver uses the established fixed tube geometry')
    sources={str(p):sha(p) for p in [Path(__file__),build/'build_manifest.json',exe,template,Path(__file__).with_name('run_twisted_solver.py'),Path(__file__).with_name('twisted_geometry.py')]}
    for path,digest in manifest['source_sha256'].items():
        assert sha(Path(path))==digest,path
        sources[path]=digest
    out.mkdir(parents=True,exist_ok=False)
    geometry=out/'geometry'
    command=[str(exe),str(args.ny),str(args.slab_depth),str(geometry)]
    env=dict(os.environ,OMP_NUM_THREADS='2',OMP_WAIT_POLICY='PASSIVE')
    now=lambda:datetime.now(timezone.utc).isoformat()
    runtime={'scope':__doc__,'command':command,'started_utc':now(),'source_sha256':sources,'force_termination_enabled':False}
    (out/'run_manifest.json').write_text(json.dumps(runtime,indent=2)+'\n')
    memory={'samples':0}
    with (out/'run.log').open('w') as log, subprocess.Popen(command,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT) as process:
        runtime['pid']=process.pid
        import psutil
        runtime['creation_time']=psutil.Process(process.pid).create_time()
        (out/'run_manifest.json').write_text(json.dumps(runtime,indent=2)+'\n')
        while True:
            sample=windows_memory(process)
            if sample is not None:
                memory['samples']+=1
                for k,v in sample.items():memory[k]=max(memory.get(k,0),v)
            try:
                process.wait(timeout=.2)
                sample=windows_memory(process)
                if sample:
                    for k,v in sample.items():memory[k]=max(memory.get(k,0),v)
                break
            except subprocess.TimeoutExpired:pass
        code=process.returncode
    completion={'exit_code':code,'completed_utc':now(),'inputs_unchanged':all(sha(Path(p))==d for p,d in sources.items()),
                'resource_usage':memory,'packed_manifest_written':False,'flow_computed':False,'goal_complete':False}
    (out/'run_completion.json').write_text(json.dumps(completion,indent=2)+'\n')
    assert code==0 and completion['inputs_unchanged']
    result={k:metadata[k] for k in ['extent','geometry_spec','cells_columns','faces_columns','walls_columns']}
    result.update(format='aphros_cut_geometry_v2',finest_ny=args.ny,finest_h=.125/args.ny,
                  reference_translation_cells=[0,0,0],reference_translation=[0.,0.,0.],
                  wall_refinement_padding_fine_cells=2,source_sha256={str(out/'run_manifest.json'):sha(out/'run_manifest.json')},tables={})
    for name,width in [('cells',12),('faces',8),('walls',11),('polygons',8)]:
        path=geometry/f'geometry.{name}.bin'
        with path.open('rb') as stream:magic,rows,columns,endian=HEADER.unpack(stream.read(HEADER.size))
        assert (magic,columns,endian)==(b'CIRRCUT1',width,0x01020304)
        assert path.stat().st_size==HEADER.size+rows*columns*8
        result['tables'][name]={'file':path.name,'rows':rows,'columns':columns,'sha256':sha(path)}
    table=result['tables']['walls']
    wall=np.memmap(geometry/table['file'],dtype='<f8',mode='r',offset=HEADER.size,shape=(table['rows'],11))
    keys=np.asarray(wall[:,:3],dtype=np.int64);shape=np.array([2*args.ny,args.ny,args.ny])
    root_shape=(shape+15)//16;root_codes=set()
    for offset in itertools.product((-2,0,2),repeat=3):
        q=keys+offset;q[:,0]%=shape[0];q=q[np.all((q>=0)&(q<shape),axis=1)]//16
        codes=(q[:,0]*root_shape[1]+q[:,1])*root_shape[2]+q[:,2]
        root_codes.update(map(int,np.unique(codes)))
    result['refine_root_tiles']=[[int(c//(root_shape[1]*root_shape[2])),int(c//root_shape[2]%root_shape[1]),int(c%root_shape[2])] for c in sorted(root_codes)]
    result['baseline_polygons_binary']=str(geometry/'geometry.polygons.bin')
    result['polygons_columns']=['axis','i','j','k','vertex','x','y','z']
    (geometry/'geometry.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    completion.update(packed_manifest_written=True,geometry_manifest_sha256=sha(geometry/'geometry.json'),
                      packaging_completed_utc=now(),tables=result['tables'],refined_root_tiles=len(root_codes))
    (out/'run_completion.json').write_text(json.dumps(completion,indent=2)+'\n')
    print(json.dumps(completion))


if __name__=='__main__':main()
