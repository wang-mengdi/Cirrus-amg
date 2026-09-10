"""Test the CUDA AMG C ABI on independent 3D sparse systems and error recovery.

These known-solution checks validate the bridge, not Aphros flow alignment.
They use the existing 1e-12 explicit residual and 1e-8 solution-error gates.
"""
import argparse
import ctypes as ct
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import psutil
from scipy import sparse


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build, out = args.build.resolve(), args.output.resolve()
    if out.drive.lower() != 'd:':
        raise ValueError('Keep experiment output on D')
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((build/'build_manifest.json').read_text())
    assert manifest['passed'] and manifest['inputs_unchanged']
    library = build/'twisted_cuda_amg.dll'
    assert sha(library) == manifest['library_sha256']
    assert 'LNK4098' not in (build/'build.log').read_text()
    inputs = {str(p): sha(p) for p in [Path(__file__).resolve(), library,
              build/'build_manifest.json', build/'cuda_amg_bridge.cu']}
    cuda = Path(manifest['cuda'])
    for name in ['cudart64_12.dll', 'cusparse64_12.dll', 'nvJitLink_120_0.dll']:
        path = cuda/'bin'/name
        if path.exists():
            inputs[str(path)] = sha(path)
    (out/'launch.json').write_text(json.dumps({'scope': __doc__,
        'source_sha256': inputs, 'pid': os.getpid(),
        'creation_time': psutil.Process().create_time()}, indent=2)+'\n')
    deadline = time.monotonic()+1200
    while True:
        native = [p.pid for p in psutil.process_iter(['name'])
                  if p.info['name'] in ['simple_channel.exe', 'native_compact_gpu_audit.exe']]
        if not native and psutil.virtual_memory().available >= 4*2**30:
            break
        if time.monotonic() > deadline:
            raise RuntimeError('Bridge tests not launched: native experiment or low memory')
        time.sleep(2)
    directory = os.add_dll_directory(str(cuda/'bin'))
    dll = ct.CDLL(str(library))
    i32, f64, u64, ptr = ct.c_int32, ct.c_double, ct.c_uint64, ct.c_void_p
    dll.twisted_cuda_amg_abi_version.restype = ct.c_int
    assert dll.twisted_cuda_amg_abi_version() == 1
    create = dll.twisted_cuda_amg_create
    create.argtypes = [i32, i32, ptr, ptr, ptr, i32, ct.POINTER(ptr),
                       ct.POINTER(f64), ptr, u64]
    create.restype = ct.c_int
    solve = dll.twisted_cuda_amg_solve
    solve.argtypes = [ptr, i32, ptr, ptr, ct.POINTER(i32), ct.POINTER(f64),
                      ct.POINTER(f64), ptr, u64]
    solve.restype = ct.c_int
    destroy = dll.twisted_cuda_amg_destroy
    destroy.argtypes = [ptr]
    destroy.restype = None
    message = ct.create_string_buffer(2048)
    rows = []
    errors = []
    passed = False
    error = None
    try:
        side = 20
        eye = sparse.eye(side, format='csr')
        axis = sparse.diags([-np.ones(side-1), np.r_[1., np.full(side-2, 2.), 1.],
                             -np.ones(side-1)], [-1, 0, 1], format='csr')
        periodic = axis.tolil()
        periodic[0, 0] = periodic[-1, -1] = 2.
        periodic[0, -1] = periodic[-1, 0] = -1.
        laplacian = (sparse.kron(sparse.kron(eye, eye), periodic.tocsr())
                     + sparse.kron(sparse.kron(eye, axis), eye)
                     + sparse.kron(sparse.kron(axis, eye), eye)).tocsr()
        n = side**3
        index = np.arange(n)
        known = np.sin(.017*index)+.3*np.cos(.037*index)
        pressure = laplacian.tolil()
        pressure[0, :] = 0.
        pressure[:, 0] = 0.
        pressure[0, 0] = 1.
        pressure_known = known-known[0]
        diagonal = sparse.diags(.02+.01*np.sin(.013*index)**2)
        viscosity = .01*laplacian+diagonal
        shift = sparse.csr_matrix((np.ones(side),
            (np.arange(side), (np.arange(side)+1) % side)), shape=(side, side))
        advection = sparse.kron(sparse.kron(eye, eye), eye-shift).tocsr()
        general = viscosity+.003*advection
        for name, matrix, exact, symmetric in [
                ('pinned_pressure', pressure.tocsr(), pressure_known, True),
                ('implicit_viscosity', viscosity.tocsr(), known, True),
                ('nonsymmetric_transport', general.tocsr(), known, False)]:
            matrix.sum_duplicates()
            matrix.sort_indices()
            matrix.indices = np.asarray(matrix.indices, dtype=np.int32)
            matrix.indptr = np.asarray(matrix.indptr, dtype=np.int32)
            matrix.data = np.asarray(matrix.data, dtype=np.float64)
            path = out/(name+'.npz')
            sparse.save_npz(path, matrix)
            inputs[str(path)] = sha(path)
            handle, setup = ptr(), f64()
            code = create(n, matrix.nnz, matrix.indptr.ctypes.data, matrix.indices.ctypes.data,
                          matrix.data.ctypes.data, int(symmetric), ct.byref(handle),
                          ct.byref(setup), message, len(message))
            assert code == 0 and handle.value, message.value.decode()
            try:
                for variant in ['first', 'different_rhs', 'zero', 'after_rejected_rhs']:
                    target = exact if variant in ['first', 'after_rejected_rhs'] else (
                        -.37*exact if variant == 'different_rhs' else np.zeros(n))
                    rhs = np.asarray(matrix@target)
                    answer = np.full(n, np.nan)
                    iterations, residual, seconds = i32(), f64(), f64()
                    if variant == 'after_rejected_rhs':
                        invalid = rhs.copy()
                        invalid[3] = np.nan
                        rc = solve(handle, n, invalid.ctypes.data, answer.ctypes.data,
                            ct.byref(iterations), ct.byref(residual), ct.byref(seconds),
                            message, len(message))
                        assert rc != 0 and b'Nonfinite' in message.value
                        errors.append({'case': name, 'invalid_rhs_rejected': True,
                                       'message': message.value.decode()})
                    code = solve(handle, n, rhs.ctypes.data, answer.ctypes.data,
                        ct.byref(iterations), ct.byref(residual), ct.byref(seconds),
                        message, len(message))
                    assert code == 0, message.value.decode()
                    measured = float(np.linalg.norm(matrix@answer-rhs)/max(np.linalg.norm(rhs), 1e-300))
                    difference = float(np.linalg.norm(answer-target)/max(np.linalg.norm(target), 1e-300))
                    assert np.isfinite(answer).all() and measured < 1e-12 and difference < 1e-8
                    rows.append({'case': name, 'variant': variant, 'passed': True,
                        'rows': n, 'nonzeros': matrix.nnz, 'setup_seconds': setup.value,
                        'solve_seconds': seconds.value, 'iterations': iterations.value,
                        'reported_relative_residual': residual.value,
                        'independent_csr_relative_residual': measured,
                        'known_solution_relative_l2': difference})
            finally:
                destroy(handle)
        passed = True
    except Exception as exception:
        error = repr(exception)
        raise
    finally:
        unchanged = all(sha(p) == value for p, value in inputs.items())
        result = {'scope': __doc__, 'passed': passed and unchanged, 'tests': rows,
            'error_recovery': errors, 'error': error, 'source_sha256': inputs,
            'inputs_unchanged': unchanged, 'completed_utc': datetime.now(timezone.utc).isoformat(),
            'aphros_flow_alignment_checked': False, 'goal_complete': False}
        (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k: v for k, v in result.items() if k != 'source_sha256'}), flush=True)
        directory.close()


if __name__ == '__main__':
    main()
