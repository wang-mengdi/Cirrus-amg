"""Freeze storage equivalence and independent CFD checks, retaining failed refinement."""
import datetime
import json
from pathlib import Path
import shutil
import subprocess
from run_twisted_solver import sha


def main():
    repo = Path(__file__).resolve().parents[1]
    source = repo/'output/twisted'
    target = repo/'validation/twisted/results/storage_128_checkpoint'
    if target.exists():
        raise ValueError('Preserve the historical checkpoint')
    reports = {
        'independent_uniform128_stokes': ('compare_uniform128_aphros.json', True),
        'independent_adaptive128_stokes': ('compare_adaptive128_aphros.json', True),
        'independent_adaptive64_steady_ns': ('compare_ns64_large_dt2_steady_aphros.json', True),
        'minimal_driver_ns32': ('minimal_driver_ns32_steady.json', True),
        'compact_stokes16': ('compact_rows_stokes16_pair.json', True),
        'compact_ns32': ('compact_rows_ns32_pair.json', True),
        'compact_stokes64': ('compact_rows_stokes64_pair.json', True),
        'packed_stokes16': ('packed_geometry_stokes16_pair.json', True),
        'legacy_reader_stokes16': ('packed_reader_legacy_stokes16_pair.json', True),
        'packed_ns64': ('packed_geometry_ns64_pair.json', True),
        'packed_geometry64': ('packed_geometry64_identity.json', True),
        'packed_geometry128': ('packed_geometry128_identity.json', True),
        'truncated_input': ('packed_geometry_truncation_check.json', True),
        'packed_ns64_paraview': ('ours_ns64_large_dt2_98p03_packed_v1/paraview_time_readback.json', True),
        'uniform64_128_refinement': ('refinement_uniform64_128/refinement.json', False),
        'ns32_64_refinement': ('refinement_ns32_64_large_dt/refinement.json', False),
    }
    values = {name: json.loads((source/path).read_text(encoding='utf-8-sig'))
              for name, (path, _) in reports.items()}
    verified = {}

    def verify(path, expected):
        path = Path(path)
        if not path.is_absolute():
            path = repo/path
        key = str(path.resolve(strict=True))
        if key not in verified:
            verified[key] = sha(path)
        if verified[key] != expected:
            raise ValueError(f'Evidence source changed: {path}')

    for name, (_, expected) in reports.items():
        report = values[name]
        if report['passed'] != expected:
            raise ValueError(f'Unexpected validation outcome: {name}')
        for path, digest in report.get('source_sha256', {}).items():
            verify(path, digest)
        if 'sha256_reference_candidate' in report:
            roots = []
            for runtime in report['runtime']:
                manifest, completion = runtime['manifest'], runtime['completion']
                if completion['exit_code'] or not all(completion.get(k, True) for k in
                        ('executable_unchanged', 'config_unchanged', 'geometry_unchanged')):
                    raise ValueError('Storage comparison contains an invalid run')
                verify(manifest['executable'], manifest['executable_sha256'])
                verify(manifest['config'], manifest['config_sha256'])
                for path, digest in manifest.get('geometry_input_sha256', {}).items():
                    verify(path, digest)
                case = json.loads(Path(manifest['config']).read_text(encoding='utf-8-sig'))
                root = Path(case['output'])
                roots.append(root if root.is_absolute() else repo/root)
            for path, digests in report['sha256_reference_candidate'].items():
                for root, digest in zip(roots, digests):
                    verify(root/path, digest)
    for name in ('independent_uniform128_stokes', 'independent_adaptive128_stokes',
                 'independent_adaptive64_steady_ns'):
        if not values[name]['complete_reference_run_checked']:
            raise ValueError('Full reference comparison required')
    build = json.loads((source/'packed_geometry_v1_build.json').read_text())
    verify(build['executable'], build['executable_sha256'])
    for name, digest in build['source_sha256'].items():
        verify(repo/'simple'/name, digest)
    selected = [source/name for name in ('compact_rows_v1_build.json', 'packed_geometry_v1_build.json')]
    for run in ('ours_stokes64_memory_original_v2', 'ours_stokes64_memory_compact_v1',
                'ours_ns64_large_dt2_98p03_packed_v1', 'ours_ns64_large_dt2_98p03_legacy_reader_v1'):
        root = source/run
        completion = json.loads((root/'run_completion.json').read_text())
        if completion['exit_code']:
            raise ValueError('Cannot preserve an unfinished storage experiment as completed')
        selected += [root/name for name in ('case.json', 'run_manifest.json', 'run_completion.json',
                                           'embedded_operator_checks.json')]
        for name in ('transient_summary.json', 'time_history.csv'):
            if (root/name).exists():
                selected.append(root/name)
        selected += list(root.rglob('metrics.json'))
    selected.append(source/'ours_ns64_adaptive_large_dt2_aa5_v1/step_0002/cut_geometry.png')
    record = {
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'completed_goal': False,
        'scope': 'Unchanged numerical results with compact storage/packed geometry; completed independent Stokes128 and steady NS64 comparisons; failed physical refinement retained',
        'parent_commit_at_capture': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        'source_sha256': {str(p.relative_to(repo)): sha(p) for p in sorted(
            list((repo/'simple').glob('*'))+list((repo/'scripts').glob('*twisted*'))+
            [repo/'validation/check_twisted_packed_input.py', repo/'validation/twisted/IMPLEMENTATION.md',
             repo/'validation/twisted/ns64_refinement.json', repo/'validation/twisted/ns128_refinement.json']) if p.is_file()},
        'evidence': {name: {'source': str(source/path), 'sha256': sha(source/path), 'passed': expected}
                     for name, (path, expected) in reports.items()},
        'verified_existing_source_sha256': verified,
        'artifact_sha256': {str(p.relative_to(source)): sha(p) for p in selected},
        'provenance_limits': ['Older Stokes128 runs predate the runtime-manifest wrapper; original comparisons record final-field hashes and final iteration history, not a newly invented runtime manifest'],
        'remaining': ['Complete matched .98/.03 NS64 and NS128 independent references and native NS128 run',
                      'Evaluate the matched NS64 to NS128 physical probe refinement',
                      'Meet physical near-wall grid-convergence and sampling-sensitivity requirements'],
    }
    target.mkdir(parents=True)
    for name, value in values.items():
        (target/(name+'.json')).write_text(json.dumps(value, indent=2)+'\n')
    for path in selected:
        destination = target/path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    (target/'checkpoint.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'checkpoint': str(target), 'reports': len(reports), 'verified_source_files': len(verified), 'completed_goal': False}))


if __name__ == '__main__':
    main()
