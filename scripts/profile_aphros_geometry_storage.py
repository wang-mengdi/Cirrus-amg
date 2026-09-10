"""Inspect original serialized geometry without changing its grid or fluid domain.

Byte estimates describe MinGW long-double field storage, not measured process
peak memory. The physical-cell bounding box is an observation, not a cropped run.
"""
import argparse
import json
import mmap
from pathlib import Path
import struct
import numpy as np
from run_twisted_solver import sha

FIELDS=[('levelset','scalar'),('face_type','enum'),('face_polygon','polygon'),('face_area','scalar'),
    ('face_center','vector'),('cell_type','enum'),('wall_normal','vector'),('wall_plane','scalar'),
    ('wall_area','scalar'),('cell_volume','scalar'),('wall_center','vector'),('cell_center','vector'),
    ('wall_distance','scalar'),('wall_displacement','vector'),('stencil_volume','scalar')]

def inspect(path):
    with path.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as data:
        offset=0
        def unpack(fmt):
            nonlocal offset
            result=struct.unpack_from(fmt,data,offset);offset+=struct.calcsize(fmt);return result
        if unpack('<Q')[0]!=0x3154534f45475041:raise ValueError('Unknown geometry format')
        shape=[];spacing=[]
        for d in range(3):
            n,h=unpack('<qd');shape.append(n);spacing.append(h)
        raw_shape=tuple(n+5 for n in shape)
        raw_cells=int(np.prod(raw_shape));rows=[];bbox=None
        for expected,kind in FIELDS:
            length=unpack('<Q')[0]
            if not 0<length<=100:raise ValueError('Invalid field label size')
            label=data[offset:offset+length].decode();offset+=length
            if label!=expected:raise ValueError('Unexpected geometry field order')
            begin,end,halo=unpack('<qqi')
            expected_end=raw_cells*(3 if label.startswith('face_') else 1)
            if begin!=0 or end!=expected_end or not 0<=halo<=128:raise ValueError('Unsupported original single-block storage range')
            first=offset;vertices=0
            if kind=='polygon':
                for i in range(end):
                    n=unpack('<Q')[0]
                    if n>64:raise ValueError('Invalid polygon size')
                    vertices+=n;offset+=24*n
            else:
                width={'enum':4,'scalar':8,'vector':24}[kind]
                if label=='cell_type':
                    values=np.ndarray(raw_shape[::-1],dtype='<i4',buffer=data,offset=offset)
                    physical=values[2:2+shape[2],2:2+shape[1],2:2+shape[0]]
                    if not np.isin(physical,[0,1,2]).all():raise ValueError('Unknown physical cell type')
                    z,y,x=np.nonzero(physical!=2)
                    bbox={'fluid_cells':int(len(x)),'cut_cells':int(np.count_nonzero(physical==1)),
                        'minimum_index_xyz':[int(v.min()) for v in (x,y,z)],
                        'maximum_index_xyz':[int(v.max()) for v in (x,y,z)]}
                    del values,physical,x,y,z
                offset+=end*width
            if offset>len(data):raise ValueError('Truncated geometry data')
            resident_width={'enum':4,'scalar':16,'vector':48,'polygon':24}[kind]
            rows.append({'field':label,'kind':kind,'elements':end,'halo':halo,'serialized_payload_bytes':offset-first,
                'extended_storage_bytes':end*resident_width,'polygon_vertices':vertices,
                'extended_polygon_payload_bytes':vertices*48})
        if offset!=len(data):raise ValueError('Trailing geometry bytes')
    byname={r['field']:r for r in rows}
    cold=('levelset','face_polygon','cell_center','wall_distance','wall_displacement')
    released={name:byname[name]['extended_storage_bytes']+byname[name]['extended_polygon_payload_bytes'] for name in cold}
    released.update(driver_initial_levelset=byname['levelset']['extended_storage_bytes'],driver_initial_velocity=raw_cells*48)
    return {'shape_xyz':shape,'spacing_xyz':spacing,'raw_index_shape_xyz':raw_shape,'fields':rows,
        'physical_cell_bbox':bbox,'cold_storage_logical_bytes':released,'cold_storage_logical_bytes_total':sum(released.values()),
        'mesh_center_array_logical_bytes':raw_cells*4*48,
        'limits':'No storage or grid was changed. Logical storage estimates are not observed RSS savings or proof that a larger CFD run fits.'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--geometry',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();path=a.geometry.resolve();out=a.output.resolve()
    if out.exists():raise ValueError('Preserve previous inventory')
    complete=json.loads((path.parent/'run_completion.json').read_text());digest=sha(path)
    if complete['exit_code'] or digest!=complete['geometry_sha256']:raise ValueError('Require an unchanged original geometry capture')
    result=inspect(path);assert sha(path)==digest
    result.update(scope=__doc__,source_sha256={str(path):digest,str(path.parent/'run_completion.json'):sha(path.parent/'run_completion.json'),
        str(Path(__file__).resolve()):sha(Path(__file__))},goal_complete=False)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('fields','source_sha256')}))

if __name__=='__main__':main()
