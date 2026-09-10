"""Plot manufactured-field boundary consistency, separate from flow accuracy."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.analysis/'wall_consistency.json').read_text())
    grids = report['grids']
    h = np.array([g['h'] for g in grids])*1000
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for key, label, color in [('total_zero_wall_error', 'Current linear wall', '#374d70'),
                              ('quadratic_zero_boundary_error', 'Candidate 3D quadratic wall', '#087e8b'),
                              ('quadratic_exact_boundary_error', 'Quadratic with exact boundary value', '#b05627')]:
        values = np.array([g[key]['relative_l2'] for g in grids])*100
        axes[0].loglog(h, values, 'o-', label=label, color=color)
    axes[0].set_xlabel('Grid spacing (mm)'); axes[0].set_ylabel('Wall-gradient relative L2 error (%)')
    axes[0].grid(True, which='both', alpha=.2); axes[0].legend(fontsize=8)
    axes[0].set_title('Manufactured scalar; not a computed flow')
    fine = grids[-1]
    keys = ['wall_position_contribution', 'one_sided_difference_contribution', 'linear_fit_contribution']
    values = [fine[k]['relative_l2']*100 for k in keys]
    bars = axes[1].bar(['Wall position', 'One-sided difference', 'Linear fit'], values, color=['#aa6150', '#557695', '#669891'])
    axes[1].bar_label(bars, fmt='%.2f%%', padding=3)
    axes[1].set_ylabel('Contribution norm / exact gradient norm (%)')
    axes[1].set_title(f'Current linear closure at h={h[-1]:.4f} mm')
    axes[1].set_ylim(0, max(values)*1.3)
    axes[1].text(.02, .94, 'Norms do not add; signed terms can cancel.', transform=axes[1].transAxes, fontsize=8, va='top')
    fig.savefig(args.analysis/'wall_consistency.png', dpi=160)
    print(args.analysis/'wall_consistency.png')


if __name__ == '__main__':
    main()
