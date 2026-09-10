"""Read original Aphros wall polygons from CSV or streamed packed geometry.

The packed table contains every grid face as well as the embedded wall. Read it
in bounded chunks, verify the entire payload, and retain only wall vertices.
Coordinates and vertex order are preserved; no surface is reconstructed here.
"""
import hashlib
from pathlib import Path
import struct
import numpy as np

HEADER = struct.Struct('<8sQII')
COLUMNS = ['axis', 'i', 'j', 'k', 'vertex', 'x', 'y', 'z']


def _wall_rows(geometry, geometry_path, report, chunk_rows):
    table = geometry.get('tables', {}).get('polygons')
    if table is not None:
        source = (Path(geometry_path).resolve().parent/table['file']).resolve()
        declared = geometry.get('baseline_polygons_binary')
        if declared is not None and Path(declared).resolve() != source:
            raise ValueError('Packed polygon paths disagree')
        if geometry.get('polygons_columns') != COLUMNS:
            raise ValueError('Unknown packed polygon column layout')
        report.update(format='CIRRCUT1', source=str(source), chunk_rows=chunk_rows)
        digest = hashlib.sha256()
        with source.open('rb') as stream:
            header = stream.read(HEADER.size)
            if len(header) != HEADER.size:
                raise ValueError('Truncated polygon header')
            magic, count, width, endian = HEADER.unpack(header)
            if (magic != b'CIRRCUT1' or endian != 0x01020304 or width != 8 or
                    width != table['columns'] or count != table['rows'] or
                    source.stat().st_size != HEADER.size+count*width*8):
                raise ValueError('Invalid packed polygon header or size')
            digest.update(header)
            for offset in range(0, count, chunk_rows):
                expected = min(chunk_rows, count-offset)*width*8
                raw = stream.read(expected)
                if len(raw) != expected:
                    raise ValueError('Truncated polygon payload')
                digest.update(raw)
                rows = np.frombuffer(raw, dtype='<f8').reshape(-1, width)
                if (not np.isfinite(rows).all() or
                        not np.array_equal(rows[:, :5], np.rint(rows[:, :5])) or
                        np.any(np.abs(rows[:, :5]) > 2**53) or
                        np.any((rows[:, 0] < 0) | (rows[:, 0] > 3)) or
                        np.any(rows[:, 4] < 0)):
                    raise ValueError('Invalid packed polygon values')
                report['input_rows'] += len(rows)
                # Boolean selection copies only the wall rows of this chunk.
                for row in rows[rows[:, 0] == 3]:
                    yield row
            if stream.read(1):
                raise ValueError('Trailing polygon payload')
        report['source_sha256'] = digest.hexdigest()
        if report['source_sha256'] != table['sha256']:
            raise ValueError('Packed polygon content changed')
    else:
        if 'baseline_polygons' not in geometry:
            raise ValueError('No original wall polygon source')
        source = Path(geometry['baseline_polygons']).resolve()
        report.update(format='csv', source=str(source))
        digest = hashlib.sha256()
        with source.open('rb') as stream:
            header = stream.readline(); digest.update(header)
            if header.decode('utf-8-sig').strip().split(',') != COLUMNS:
                raise ValueError('Unknown CSV polygon column layout')
            for raw in stream:
                digest.update(raw)
                if not raw.strip():
                    continue
                report['input_rows'] += 1
                if raw.startswith(b'3,'):
                    row = np.array([float(value) for value in raw.split(b',')])
                    if (row.shape != (8,) or not np.isfinite(row).all() or
                            not np.array_equal(row[:5], np.rint(row[:5])) or
                            np.any(np.abs(row[:5]) > 2**53) or row[4] < 0):
                        raise ValueError('Invalid CSV wall polygon row')
                    yield row
        report['source_sha256'] = digest.hexdigest()
        expected = geometry.get('source_sha256', {}).get(str(source))
        if expected is not None and report['source_sha256'] != expected:
            raise ValueError('CSV polygon content changed')


def load_wall_polygons(geometry, geometry_path, chunk_rows=65536):
    """Return wall-cell keys and ordered float64 vertices, with payload evidence."""
    if not isinstance(chunk_rows, int) or chunk_rows < 1:
        raise ValueError('Require a positive integer chunk size')
    shift = np.asarray(geometry.get('reference_translation', [0, 0, 0]), dtype=float)
    shift_cells = np.asarray(geometry.get('reference_translation_cells', [0, 0, 0]))
    if (shift.shape != (3,) or shift_cells.shape != (3,) or
            not np.isfinite(shift).all() or not np.isfinite(shift_cells).all() or
            not np.array_equal(shift_cells, np.rint(shift_cells))):
        raise ValueError('Invalid polygon reference translation')
    shift_cells = shift_cells.astype(np.int64)
    extent = float(geometry['extent'][0]); h = float(geometry['finest_h'])
    if not np.isfinite([extent, h]).all() or extent <= 0 or h <= 0:
        raise ValueError('Invalid polygon periodic extent')
    nx = int(round(extent/h))
    if nx < 1 or abs(nx*h-extent) > extent*1e-12:
        raise ValueError('Polygon period is not an integer cell count')
    groups = {}; report = {'input_rows': 0, 'wall_vertices': 0}
    for row in _wall_rows(geometry, geometry_path, report, chunk_rows):
        key = row[1:4].astype(np.int64)+shift_cells
        wrap = key[0]//nx; key[0] %= nx
        point = row[5:8]+shift
        point[0] -= wrap*extent
        vertices = groups.setdefault(tuple(key), [])
        if int(row[4]) != len(vertices):
            raise ValueError('Wall polygon vertex order is missing, repeated or discontinuous')
        vertices.append(point)
        report['wall_vertices'] += 1
    for key, vertices in groups.items():
        if len(vertices) < 3:
            raise ValueError('Wall polygon has fewer than three vertices')
        groups[key] = np.asarray(vertices)
    expected = geometry.get('tables', {}).get('walls', {}).get('rows')
    if expected is None and 'walls' in geometry:
        expected = len(geometry['walls'])
    if expected is not None and len(groups) != expected:
        raise ValueError('Wall polygon count differs from cut geometry')
    report.update(wall_polygons=len(groups), all_payload_read=True,
                  retained_data='Only original wall vertices; no grid-face polygons retained')
    return groups, report
