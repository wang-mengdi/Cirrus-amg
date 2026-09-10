"""Archive pressure-roundoff evidence, including failed and interrupted runs."""
import argparse
import datetime
import gzip
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    runs = repo/'output/twisted'
    baseline = Path('D:/Dropbox/Agent-simulation/twisted-baseline')
    target = args.output.resolve()
    required_native = ('ours_proj16_pressure_trace_v1', 'ours_proj16_pressure_roundoff_v2',
                       'ours_proj64_pressure_roundoff_v2')
    for name in required_native:
        completion = json.loads((runs/name/'run_completion.json').read_text())
        if completion['exit_code'] != 0 or not all(completion[k] for k in (
                'executable_unchanged', 'config_unchanged', 'geometry_unchanged')):
            raise ValueError('Native completion/provenance failed: '+name)
    target.mkdir(parents=True, exist_ok=False)
    receipt = []

    def save(source, destination, scope):
        data = source.read_bytes()
        packed = len(data) > 2_000_000 and source.suffix in ('.csv', '.json', '.log')
        name = Path(str(destination)+('.gz' if packed else ''))
        output = gzip.compress(data, mtime=0) if packed else data
        path = target/name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise ValueError('Duplicate archive target: '+str(name))
        path.write_bytes(output)
        receipt.append({'path': name.as_posix(), 'source': str(source.resolve()),
                        'source_sha256': digest(data), 'sha256': digest(output),
                        'size': len(output), 'gzip': packed, 'scope': scope})

    def tree(root, destination, scope, suffixes=('.csv', '.json', '.log', '.txt', '.py', '.cmd', '.h', '.ipp', '.cpp', '.conf')):
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.suffix in suffixes:
                save(p, Path(destination)/p.relative_to(root), scope)

    reports = ('projection_ns32_steady_original_pair_v1.json',
               'aphros_proj32_steady_mass_diagnosis_v1.json',
               'pressure_trace_interrupted128_summary_v1.json',
               'pressure_trace_n16_regression_v1.json',
               'aphros_pressure_default_regression_v1.json',
               'pressure_roundoff_n16_cpu_pair_v2.json',
               'pressure_roundoff_n64_cpu_pair_v2.json',
               'pressure_roundoff_n64_aphros_pair_v2.json',
               'pressure_floor_n16_v1.json', 'pressure_floor_n128_trace_v1.json',
               'pressure_floor_n128_trace_final_v1.json',
               'pressure_floor_n16_guard_v2.json', 'pressure_floor_n64_guard_v2.json',
               'pressure_floor_n128_guard_initial_v2.json', 'pressure_floor_n128_counterfactual_v1.json',
               'pressure_roundoff_n128_iteration1_pair_v2.json',
               'paraview_roundoff64_premature_read_v2.json')
    for name in reports:
        save(runs/name, Path('reports')/name, 'completed comparison or captured diagnostic; inspect its own result')
        inputs = runs/(Path(name).stem+'_inputs')
        if inputs.exists():
            tree(inputs, Path('reports')/inputs.name, 'captured diagnostic bytes; not a completed flow')
    save(runs/'pressure_floor_checker_v1.py', 'checkers/pressure_floor_checker_v1.py', 'checker source at earlier report time')
    for name in ('check_twisted_projection_floor.py', 'check_twisted_mass.py',
                 'check_twisted_projection_iteration.py',
                 'compare_twisted.py', 'check_twisted_time_pair.py',
                 'check_twisted_projection_flux.py', 'run_twisted_solver.py',
                 'snapshot_twisted_reference.py',
                 'record_twisted_roundoff_checkpoint.py'):
        save(repo/'scripts'/name, Path('checkers')/name, 'current reproducibility source')
    for name in ('pressure_trace_build_v1', 'pressure_roundoff_build_v2'):
        tree(runs/name, Path('builds')/name, 'immutable native compiled source snapshot')
    tree(baseline/'projection_pressure_build_v1', 'builds/projection_pressure_build_v1',
         'immutable independent driver and linear backend; unchanged original solver library')
    for name in required_native:
        tree(runs/name, Path('native')/name, 'completed native time sequence')
    tree(runs/'aphros_proj32_pressure_tol2e15_step2_v1', 'aphros/pressure_tol2e15_step2',
         'verified completed step 2 of a longer original reference; actual mass check failed')
    for name in ('ours_proj128_material_gpu_v2', 'ours_proj128_pressure_trace_v1'):
        root = runs/name
        if json.loads((root/'run_completion.json').read_text())['exit_code'] == 0:
            raise ValueError('Expected an interrupted diagnostic run')
        for p in root.iterdir():
            if p.is_file() and p.suffix in ('.json', '.csv', '.log'):
                save(p, Path('interrupted')/name/p.name, 'interrupted; no completed flow claim')
    for name in ('navier_stokes_proj_n32_steady_cache3_v1',
                 'navier_stokes_proj_n16_pressure_default_v1',
                 'navier_stokes_proj_n32_steady_direct_tol15_v1',
                 'navier_stokes_proj_n32_steady_pressure_tol15_v1'):
        root = baseline/name
        completion = json.loads((root/'run_completion.json').read_text(encoding='utf-8-sig'))
        tree(root, Path('aphros')/name, 'terminal original reference; exit_code='+str(completion['exit_code']))
    for root in (runs/'ours_proj128_pressure_roundoff_v2',
                 baseline/'navier_stokes_proj_n128_implicit_amg_material_v1',
                 baseline/'navier_stokes_proj_n32_steady_pressure_tol2e15_v1'):
        # Only immutable launch information from jobs that may still be live.
        for name in ('case.json', 'a.conf', 'case_manifest.json', 'run_manifest.json'):
            if (root/name).exists():
                save(root/name, Path('launches')/root.name/name, 'launch information only; completion not certified')
    for name in ('proj.ipp', 'approx_eb.h', 'approx_eb.ipp'):
        save(baseline/'aphros/src/solver'/name, Path('original_solver')/(name+'.txt'), 'unchanged original Aphros solver source')
    # The two large raw 128 iteration dumps remain local. Their complete row
    # comparison, executable provenance, and exact hashes are retained above.
    external = []
    pair = json.loads((runs/'pressure_roundoff_n128_iteration1_pair_v2.json').read_text())
    for name, expected in pair['source_sha256'].items():
        path = Path(name)
        if path.parent.name == 'iter_1' and path.suffix == '.csv':
            if digest(path.read_bytes()) != expected:
                raise ValueError('Raw intermediate dump changed: '+name)
            external.append({'path': name, 'sha256': expected, 'size': path.stat().st_size,
                             'scope': 'Raw iteration 1 only; retained locally, not copied into this archive'})
    (target/'receipt.json').write_text(json.dumps({
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope': 'Pressure arithmetic diagnostics and completed regression runs. Physical grid/time convergence and full 128 comparison remain open.',
        'files': receipt, 'external_intermediate_files': external}, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'files': len(receipt), 'stored_bytes': sum(r['size'] for r in receipt)}))


if __name__ == '__main__':
    main()
