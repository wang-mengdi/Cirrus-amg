"""Preserve CGS2 linear and physical-step validation without claiming convergence."""
import argparse
import gzip
import json
from pathlib import Path
import shutil

from run_twisted_solver import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    runs = repo/'output/twisted'
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    files, retained = [], {}

    def retain(path, expected):
        path = Path(path).resolve()
        if sha(path) != expected:
            raise ValueError('Verified input changed: '+str(path))
        if str(path) in retained and retained[str(path)]['source_sha256'] != expected:
            raise ValueError('Conflicting retained provenance')
        retained[str(path)] = {'source': str(path), 'source_sha256': expected,
                               'size': path.stat().st_size, 'scope': 'completed scoped validation provenance'}

    def collect(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ('source_sha256', 'executed_input_sha256') and isinstance(value, dict):
                    for path, expected in value.items():
                        retain(path, expected)
                else:
                    collect(value)
        elif isinstance(node, list):
            for value in node:
                collect(value)

    def save(path, destination, scope):
        path = Path(path).resolve()
        value = sha(path)
        compressed = path.stat().st_size > 200000
        target = out/(destination+('.gz' if compressed else ''))
        target.parent.mkdir(parents=True, exist_ok=True)
        if compressed:
            with path.open('rb') as source, target.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, compresslevel=6) as sink:
                shutil.copyfileobj(source, sink)
        else:
            shutil.copyfile(path, target)
        if sha(path) != value:
            raise ValueError('Archive source changed')
        files.append({'path': target.relative_to(out).as_posix(), 'source': str(path),
                      'source_sha256': value, 'sha256': sha(target), 'size': target.stat().st_size,
                      'gzip': compressed, 'scope': scope})

    names = ('cgs2_benchmark16_v1/benchmark.json', 'cgs2_benchmark64_v1/benchmark.json',
             'cgs2_build_n16_default_cpu_pair_v1.json', 'cgs2_n16_mgs2_time_pair_v1.json',
             'cgs2_n16_cpu_time_pair_v1.json', 'cgs2_n16_aphros_cg_pair_v1.json',
             'aphros_n16_extended_cg_native_pair_v1.json',
             'cgs2_n64_native_checks_v1.json', 'cgs2_n64_aphros_extended_pair_v1.json')
    for name in names:
        path = runs/name
        report = json.loads(path.read_text())
        if not report['passed']:
            raise ValueError('Scoped validation failed: '+name)
        collect(report)
        save(path, 'checks/'+name.replace('/', '_'), 'completed linear or physical-step check; no spatial convergence claim')
    for n in (16, 64):
        root = runs/f'cgs2_benchmark{n}_v1'
        for path in sorted(root.rglob('*')):
            if path.is_file() and (path.suffix == '.json' or path.suffix == '.log') and path.name != 'benchmark.json':
                save(path, f'linear{n}/'+path.relative_to(root).as_posix(), 'alternating fresh audit configurations and actual results')
    for name in ('ours_proj16_mgs2_regression_v1', 'ours_proj16_cgs2_steps8_v1', 'ours_proj64_cgs2_dt005_v1'):
        root = runs/name
        completion = json.loads((root/'run_completion.json').read_text())
        if completion['exit_code'] or not all(completion[k] for k in ('executable_unchanged', 'config_unchanged', 'geometry_unchanged')):
            raise ValueError('Native trajectory did not complete unchanged')
        for filename in ('case.json', 'run_manifest.json', 'run_completion.json', 'transient_summary.json',
                         'projection_method.json', 'time_history.csv', 'gpu_linear.csv', 'run.log', 'mesh_cells.csv', 'mesh_faces.csv'):
            save(root/filename, name+'/'+filename, 'actual completed native trajectory')
        for step in sorted(root.glob('step_*')):
            for filename in ('metrics.json', 'solution.csv', 'walls.csv', 'flux.csv', 'sections.csv', 'history.csv', 'acceleration.csv'):
                save(step/filename, name+'/'+step.name+'/'+filename, 'actual completed physical step')
    reference = Path('D:/Dropbox/Agent-simulation/twisted-baseline/navier_stokes_proj_n16_extended_cg_v3')
    for name in ('a.conf', 'case_manifest.json', 'run_manifest.json', 'run_completion.json', 'tube_b0_time.csv',
                 'proj_final_b0_cells.csv', 'proj_final_b0_faces.csv', 'tube_final_b0_walls.csv'):
        save(reference/name, 'reference16_cg/'+name, 'completed original CG reference with original geometry and extended scalars')
    save(runs/'cgs2_build_v1/build_manifest.json', 'build/build_manifest.json', 'actual native build')
    launch = runs/'ours_proj128_cgs2_steps2_v1'
    config = json.loads((runs/'config_ours_proj128_cgs2_steps2_v1.json').read_text())
    original = json.loads((runs/'ours_proj128_anderson_failure_2steps_v1/case.json').read_text())
    if {k:v for k,v in config.items() if k not in ('output','gpu_orthogonalization')} != {k:v for k,v in original.items() if k!='output'}:
        raise ValueError('128 launch changed physical settings')
    for path in (runs/'config_ours_proj128_cgs2_steps2_v1.json', launch/'run_manifest.json'):
        save(path, 'native128_launch/'+path.name, 'launch only; no completed 128 validation claim')
    for name in ('benchmark_twisted_orthogonalization.py', 'check_twisted_time_pair.py',
                 'check_twisted_native_run.py', 'record_twisted_orthogonalization_checkpoint.py'):
        save(repo/'scripts'/name, 'sources/'+name, 'executed validation or preservation source')
    save(repo/'validation/twisted/GPU_BATCHED_ORTHOGONALIZATION.md', 'scope.md', 'findings and limits')
    receipt = {'scope': __doc__, 'goal_complete': False, 'files': files, 'retained_runtime_files': list(retained.values())}
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'archive_files': len(files), 'retained_runtime_files': len(retained), 'goal_complete': False}))


if __name__ == '__main__':
    main()
