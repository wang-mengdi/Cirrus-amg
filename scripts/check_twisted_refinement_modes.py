"""Check real-data refinement regression and explicit pseudo/physical input scope.

The same-grid 64 pair exercises probe analysis, not spatial convergence.
No new CFD trajectories or relaxed spatial/sampling tolerances are used.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from analyze_twisted_refinement import load, completed_steady_iteration
from run_twisted_solver import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('previous-refinement','physical-regression','same-grid-regression',
                 'physical32','physical64','steady16','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    out=a.output.resolve()
    if out.drive.lower()!='d:':raise ValueError('Keep validation outputs on D')
    out.mkdir(parents=True,exist_ok=False)
    reports=[json.loads((root/'refinement.json').read_text()) for root in
             (a.previous_refinement,a.physical_regression,a.same_grid_regression)]
    old,new,same=reports
    keys=('passed','checks','limits','finest_h','fluid_cells','coarse_fine_faces','volume_flux',
          'flow_relative_difference_to_fine','near_wall_velocity_by_distance_m','wall_shear',
          'sampling_sensitivity','sampling_sensitivity_by_distance_m','pressure_by_distance_m',
          'fit_condition_numbers','probe_definition','trajectory_provenance')
    for key in keys:assert old[key]==new[key], 'Historical physical result changed: '+key
    assert not new['passed'] and new['spatial_convergence_checked'] and not new['spatial_convergence_passed']
    for name in ('velocity_probes.csv','wall_probes.csv'):
        assert sha(a.previous_refinement/name)==sha(a.physical_regression/name),name
    assert same['limits']==old['limits']
    assert not same['spatial_convergence_checked'] and not same['spatial_convergence_passed']
    assert not same['independent_aphros_alignment_checked']
    assert same['input_iteration_modes']==['physical_trajectory','steady_pseudo_iteration']
    assert same['trajectory_provenance'][1] is None and same['steady_iteration_provenance'][0] is None
    proof=same['steady_iteration_provenance'][1]
    assert proof['passed'] and proof['validation_mode']=='state_only' and not proof['physical_trajectory_claimed']
    differences=[same['flow_relative_difference_to_fine'],same['wall_shear']['relative_l2']]
    differences.extend(row['relative_l2'] for key in ('near_wall_velocity_by_distance_m','pressure_by_distance_m')
                       for row in same[key].values())
    assert max(differences)<1e-6
    # A same-grid field match must not hide the existing wall-sampling failure.
    assert not same['passed'] and not same['checks']['sampling_sensitivity']
    cases=[]

    def rejects(name,call,expected):
        try:call()
        except ValueError as error:
            assert expected in str(error),(name,str(error))
            cases.append({'name':name,'passed':True,'diagnostic':str(error)})
        else:raise AssertionError('Accepted invalid input: '+name)

    rejects('pseudo_without_explicit_mode',lambda:load(a.steady16/'iterate_0017'), 'explicit steady-iteration')
    rejects('physical_with_pseudo_mode',lambda:load(a.physical64,True), 'explicit steady-iteration')
    rejects('nonfinal_steady_iteration',lambda:completed_steady_iteration(a.steady16/'iterate_0016'), 'actual final steady iteration')
    script=Path(__file__).with_name('analyze_twisted_refinement.py')
    for name,coarse,fine,options,expected in (
        ('same_grid_without_regression_scope',a.physical64,a.physical64,[], 'strictly finer grid'),
        ('different_mesh_as_same_grid',a.physical32,a.physical64,['--same-grid-regression'], 'identical native mesh')):
        result=subprocess.run([sys.executable,str(script),'--coarse',str(coarse),'--fine',str(fine),
            '--output',str(out/name),*options],capture_output=True,text=True)
        assert result.returncode and expected in result.stderr and not (out/name).exists()
        (out/(name+'.log')).write_text(result.stdout+result.stderr)
        cases.append({'name':name,'passed':True,'exit_code':result.returncode})
    inputs=[Path(__file__),script,Path(__file__).with_name('check_twisted_steady_iteration.py')]
    inputs.extend(root/name for root in (a.previous_refinement,a.physical_regression,a.same_grid_regression)
                  for name in ('refinement.json','velocity_probes.csv','wall_probes.csv'))
    report={'passed':True,'scope':__doc__,'physical_32_to_64_numbers_and_probes_unchanged':True,
            'physical_spatial_failure_preserved':True,'same_grid_maximum_probe_difference':max(differences),
            'same_grid_does_not_claim_spatial_convergence':True,'same_grid_sampling_failure_preserved':True,
            'cases':cases,'source_sha256':{str(path.resolve()):sha(path) for path in inputs}}
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key not in ('source_sha256','cases','scope')}))


if __name__=='__main__':main()
