"""Plot completed pressure-sensitivity diagnostics without claiming convergence."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from run_twisted_solver import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve earlier plots')
    report = json.loads(args.report.read_text())
    pairs, runs = report['pairs'], report['runs']
    labels = [f'{p["dt"][0]*1e3:g} -> {p["dt"][1]*1e3:g}' for p in pairs]
    x = np.arange(len(pairs))
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6), layout='constrained')
    fig.suptitle('Twisted tube: pressure remains sensitive to time step on the 16 grid', fontsize=15)
    ax = axes[0, 0]
    for field, title, color in [('pressure', 'Pressure', '#b3261e'), ('cell_pressure_gradient', 'Cell pressure gradient', '#955c00'),
                                 ('velocity', 'Velocity', '#006f9b')]:
        ax.plot(x, [100*p[field]['relative_l2'] for p in pairs], 'o-', label=title, color=color)
    ax.axhline(.25, color='#777777', ls='--', lw=1, label='Pressure gate: 0.25%')
    ax.set(xticks=x, xticklabels=labels, xlabel='Time-step halving (ms)', ylabel='Volume-weighted relative L2 difference (%)',
           title='Completed steady fields at t = 0.64 s')
    ax.legend(fontsize=9)
    ax = axes[0, 1]
    ax.bar(x, [100*p['cut_pressure_difference_energy_fraction'] for p in pairs], color='#bb6633', label='Pressure-difference energy')
    ax.axhline(100*pairs[0]['cut_fluid_volume_fraction'], color='#444444', ls='--', label='Cut-cell share of fluid volume')
    ax.set(xticks=x, xticklabels=labels, xlabel='Time-step halving (ms)', ylabel='Cut-cell share (%)', ylim=(0, 100),
           title='Pressure differences are concentrated near the wall')
    ax.legend(fontsize=9)
    ax = axes[1, 0]
    dt = np.array([r['dt'] for r in runs])*1e3
    ax.loglog(dt, [r['interpolated_velocity_divergence_volume_rms'] for r in runs], 'o-', color='#955c00', label='Flux from interpolated cell velocity')
    ax.loglog(dt, [r['actual_flux_divergence_volume_rms'] for r in runs], 'o-', color='#006f9b', label='Stored conservative face flux')
    ax.set(xlabel='Time step (ms)', ylabel='Volume RMS divergence (1/s)', title='Use stored face flux for the mass check')
    ax.legend(fontsize=8.5)
    ax = axes[1, 1]
    ax.axis('off')
    max_replay = max(r['face_flux_identity_relative_l2'] for r in runs)
    text = ('Replayed from actual final u, p and face flux\n\n'
            r'$q_f=S_f(Iu)_f+\frac{\Delta t}{\rho}S_f[(IG_c p)_f-(G_f p)_f]$'+'\n\n'
            f'Maximum relative flux replay difference: {max_replay:.2e}\n'
            'All four stored face fluxes pass the unchanged mass gates.\n'
            'The pressure time-step gate remains FAILED.\n\n'
            'Scope: uniform cut grid, constant density and body force.\n'
            'Only the first pair uses the same executable; later binaries\n'
            'and Anderson settings differ and are recorded in the report.\n'
            'This identity is not an isolated causal test or a fine-grid\n'
            'accuracy certificate. BCG also depends on the time step.')
    ax.text(0, .98, text, va='top', fontsize=10, linespacing=1.5, transform=ax.transAxes)
    for ax in axes.flat[:3]:
        ax.grid(axis='y', alpha=.2)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for suffix in ('png', 'svg'):
        fig.savefig(out/f'pressure_timestep.{suffix}', dpi=160)
    plt.close(fig)
    manifest = {'source_sha256': {str(args.report.resolve()): sha(args.report), str(Path(__file__).resolve()): sha(Path(__file__))},
                'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out/'plot_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'output': str(out)}))


if __name__ == '__main__':
    main()
