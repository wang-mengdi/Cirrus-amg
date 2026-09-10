"""Plot manufactured cut-wall accuracy and area-weighted local error tails."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np

from compare_twisted import read
from run_twisted_solver import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous wall-accuracy plots')
    report_path = args.audit/'wall_accuracy.json'
    report = json.loads(report_path.read_text())
    sources = {str(report_path.resolve()): sha(report_path), str(Path(__file__).resolve()): sha(Path(__file__))}
    grids = report['grids']
    ny = np.array([g['ny'] for g in grids])
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.5))
    fig.subplots_adjust(left=.08, right=.98, top=.80, bottom=.23, wspace=.30)
    fig.suptitle('Twisted tube: the current wall closure converges at first order', fontsize=14, y=.97)
    fig.text(.5, .90, 'Analytic divergence-free, no-slip velocity field; operator audit, not the constant-force flow solution', ha='center', fontsize=10)
    ax = axes[0]
    errors = np.array([g['manufactured_wall_shear']['relative_l2'] for g in grids])*100
    q95 = np.array([g['pointwise_relative_error']['area_weighted_quantiles']['0.95'] for g in grids])*100
    ax.loglog(ny, errors, 'o-', color='#b3261e', label='Area-weighted L2 error')
    ax.loglog(ny, q95, 's-', color='#955c00', label='95th percentile by wall area')
    ax.loglog(ny, errors[-1]*ny[-1]/ny, '--', color='#777777', label='First-order slope')
    for x, y in zip(ny, errors):
        ax.annotate(f'{y:.2f}%', (x, y), xytext=(0, -17), textcoords='offset points', ha='center', fontsize=9)
    ax.set_xticks(ny, labels=[str(n) for n in ny])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(min(errors)*.74, max(q95)*1.2)
    ax.set(xlabel='Finest grid resolution ny', ylabel='Wall-shear error (%)', title='Grid refinement of the unchanged wall operator')
    ax.legend(fontsize=9)
    ax.grid(alpha=.25, which='both')
    ax = axes[1]
    threshold = np.geomspace(.001, 1, 250)
    for g, color in zip(grids, ['#955c00', '#007d80', '#006f9b']):
        path = args.audit/f'wall_accuracy_n{g["ny"]}.csv'
        sources[str(path.resolve())] = sha(path)
        values = read(path)
        area, err = values['area'], values['relative_point_error']
        tail = np.array([area[err > t].sum()/area.sum()*100 for t in threshold])
        ax.loglog(threshold*100, np.maximum(tail, 1e-6), color=color, label=f'ny={g["ny"]}')
    ax.set(xlabel='Local relative wall-shear error (%)', ylabel='Wall area above that error (%)', ylim=(1e-3, 105),
           title='Small wall patches retain large local errors')
    ax.legend(fontsize=9)
    ax.grid(alpha=.25, which='both')
    fine = grids[-1]
    fig.text(.08, .105, f'ny={fine["ny"]}: maximum local error {100*fine["pointwise_relative_error"]["maximum"]:.1f}%; '
             f'area above 25% error {100*fine["pointwise_relative_error"]["wall_area_fraction_above"]["0.25"]:.5f}%.', fontsize=10)
    fig.text(.08, .055, 'The replay matches actual C++ derivatives to about 1e-15. Geometry and wall-closure errors are both included.\n'
             'These diagnostics do not replace the independent steady Aphros comparison or the physical grid/time-step gates.', fontsize=9)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for suffix in ('png', 'svg'):
        fig.savefig(out/f'wall_accuracy.{suffix}', dpi=160)
    plt.close(fig)
    manifest = {'source_sha256': sources, 'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out/'plot_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'output': str(out)}))


if __name__ == '__main__':
    main()
