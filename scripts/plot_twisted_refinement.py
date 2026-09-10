"""Plot an existing physical-grid refinement report without changing its gates."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_twisted_solver import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    report=json.loads(args.report.read_text());root=args.report.parent
    wall=np.genfromtxt(root/'wall_probes.csv',delimiter=',',names=True)
    distance=report['near_wall_velocity_by_distance_m'];radii=np.array([float(d) for d in distance])*1000
    velocity=np.array([row['relative_l2'] for row in distance.values()])*100
    coarse=np.column_stack([wall['tau_'+d+'_coarse'] for d in 'xyz'])
    fine=np.column_stack([wall['tau_'+d+'_fine'] for d in 'xyz'])
    scale=np.sqrt(np.mean(np.sum(fine*fine,axis=1)))
    shear_error=np.linalg.norm(coarse-fine,axis=1)/scale*100
    if len(wall)!=128:raise ValueError('Expected the documented 8 by 16 analytic wall probes')
    levels=[round(.125/h) for h in report['finest_h']]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(11.8,8.5),layout='constrained')
    fig.suptitle(f'Steady twisted tube: finest grid {levels[0]} to {levels[1]}\nSpatial convergence remains unverified',fontsize=16)
    ax=axes[0,0];values=np.array(report['volume_flux'])*1e6
    bars=ax.bar([str(n) for n in levels],values,color=['#8295ad','#205f92'],width=.55)
    ax.set(ylabel='Volume flow (mL/s)',xlabel='Finest cells across box y',title='Section flow: no probe interpolation')
    ax.set_ylim(0,max(values)*1.18)
    ax.bar_label(bars,fmt='%.3f',padding=5)
    ax.text(.5,.1,f'Difference: {100*report["flow_relative_difference_to_fine"]:.2f}%\nGate: {100*report["limits"]["flow_relative"]:.2f}%',transform=ax.transAxes,ha='center')
    ax=axes[0,1];ax.plot(radii,velocity,'o-',color='#b64636',lw=2)
    ax.axhline(100*report['limits']['near_wall_velocity_relative_l2'],color='#606060',ls='--',label='1% gate')
    ax.set(xlabel='Inward normal distance (mm)',ylabel='Velocity difference, relative L2 (%)',title='Velocity at fixed physical probes',ylim=(0,max(velocity)*1.2))
    for x,y in zip(radii,velocity):ax.annotate(f'{y:.2f}%',(x,y),xytext=(3,7),textcoords='offset points')
    ax.legend(loc='upper right',frameon=False);ax.grid(axis='y',alpha=.2)
    ax=axes[1,0]
    heat=ax.imshow(shear_error.reshape(8,16).T,origin='lower',aspect='auto',cmap='magma',extent=(-.0625,.9375,-.0625,1.9375))
    ax.set(xlabel='Axial position x / L',ylabel='Angular position / pi',title='Wall shear vector difference / fine-grid RMS')
    fig.colorbar(heat,ax=ax,label='%')
    ax=axes[1,1];ax.axis('off')
    sensitivities=report['sampling_sensitivity']
    nearest=str(min(float(d) for d in distance))
    local=report.get('sampling_sensitivity_by_distance_m')
    sampling_text=('Nearest-probe velocity sampling sensitivity:\n'+
        ' / '.join(f'grid {n}: {100*s[nearest]["velocity"]["relative_l2"]:.2f}%' for n,s in zip(levels,local))+'\n\n') if local else ''
    text=(f'Wall shear relative L2 difference: {100*report["wall_shear"]["relative_l2"]:.2f}%\n'
          f'Wall shear convergence gate: {100*report["limits"]["wall_shear_relative_l2"]:.2f}%\n\n'
          'Wall interpolation sensitivity (32 vs 64 neighbors):\n'
          f'  grid {levels[0]}: {100*sensitivities[0]["wall_shear"]["relative_l2"]:.2f}%\n'
          f'  grid {levels[1]}: {100*sensitivities[1]["wall_shear"]["relative_l2"]:.2f}%\n'
          f'  gate: {100*report["limits"]["sampling_shear_relative_l2"]:.2f}%\n\n'
          +sampling_text+
          'Both fields satisfy the steady-state criteria.\n'
          'These are differences between grids, not known\n'
          'errors against an exact solution. Sampling\n'
          'uncertainty remains explicit and fails its gate.')
    ax.text(.02,.95,text,va='top',linespacing=1.65)
    for ext in ('png','svg'):fig.savefig(args.output/('refinement.'+ext),dpi=170)
    plt.close(fig)
    inputs=[args.report,root/'wall_probes.csv',Path(__file__)]
    (args.output/'plot_manifest.json').write_text(json.dumps({'scope':__doc__,
        'source_sha256':{str(p.resolve()):sha(p) for p in inputs},
        'output_sha256':{name:sha(args.output/name) for name in ('refinement.png','refinement.svg')}},indent=2)+'\n')
    print(args.output/'refinement.png')


if __name__=='__main__':main()
