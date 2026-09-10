"""Read legacy JSON or exact float64 packed cut geometry."""
import json
from pathlib import Path
import struct
import numpy as np
from run_twisted_solver import sha

HEADER = struct.Struct('<8sQII')


def load_geometry(path, verify=True, tables=('cells', 'faces', 'walls')):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if data['format'] == 'aphros_cut_geometry_v1':
        return data
    if data['format'] != 'aphros_cut_geometry_v2':
        raise ValueError('Unknown cut geometry format')
    for name, width in (('cells', 12), ('faces', 8), ('walls', 11)):
        if name not in tables:
            continue
        table = data['tables'][name]
        binary = path.parent/table['file']
        with binary.open('rb') as stream:
            magic, rows, columns, endian = HEADER.unpack(stream.read(HEADER.size))
        if (magic != b'CIRRCUT1' or endian != 0x01020304 or columns != width or
                rows != table['rows'] or columns != table['columns'] or
                binary.stat().st_size != HEADER.size+rows*columns*8):
            raise ValueError('Invalid packed geometry header or size')
        if verify and sha(binary) != table['sha256']:
            raise ValueError('Packed geometry content changed')
        data[name] = np.memmap(binary, dtype='<f8', mode='r', offset=HEADER.size, shape=(rows, columns))
    return data
