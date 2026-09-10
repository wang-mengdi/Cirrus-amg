"""Preserve failed spatial-convergence checks, their sampling audit and verified ParaView outputs."""
import argparse,json,shutil
from pathlib import Path
from run_twisted_solver import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();repo=Path(__file__).resolve().parents[1];runs=repo/'output/twisted'
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);files=[];retained={}
    def keep(path,scope):
        path=Path(path).resolve();retained[str(path)]={'source':str(path),'source_sha256':sha(path),'size':path.stat().st_size,'scope':scope}
    def save(path,dest,scope):
        path=Path(path).resolve();target=out/dest;target.parent.mkdir(parents=True,exist_ok=True);value=sha(path)
        shutil.copyfile(path,target)
        if sha(path)!=value:raise ValueError('Archive input changed')
        files.append({'path':target.relative_to(out).as_posix(),'source':str(path),'source_sha256':value,
                      'sha256':sha(target),'size':target.stat().st_size,'gzip':False,'scope':scope})
    save(runs/'anderson_failure_build_v1/build_manifest.json','build/build_manifest.json','unchanged build used by native128 and the time-step-halving run')
    for version in (1,2,3):
        root=runs/f'projection_grid32_64_steady_v{version}';report=json.loads((root/'refinement.json').read_text())
        if report['passed'] or report['checks']['flow'] or report['checks']['near_wall_velocity'] or report['checks']['wall_shear']:
            raise ValueError('Expected explicitly failed spatial refinement checks')
        for path in root.iterdir():
            if path.is_file():save(path,f'refinement/v{version}/'+path.name,'FAILED spatial refinement; preserve actual computed probes and differences')
        for source,value in report['source_sha256'].items():
            if sha(Path(source))!=value:raise ValueError('Compared field or trajectory changed')
            keep(source,'actual steady field, completed trajectory provenance or immutable executed input')
    for name in ('analyze_twisted_refinement_before_provenance_v1.py.txt','analyze_twisted_refinement_before_distance_gate_v2.py.txt',
                 'check_refinement_distance_gate_v1.py','refinement_distance_gate_checks_v1.json'):
        save(runs/name,'sampling_audit/'+name,'actual earlier checker or completed real-data gate audit')
    if not json.loads((runs/'refinement_distance_gate_checks_v1.json').read_text())['passed']:raise ValueError('Gate audit failed')
    plot=runs/'projection_grid32_64_plot_v3';manifest=json.loads((plot/'plot_manifest.json').read_text())
    for source,value in manifest['source_sha256'].items():
        if sha(Path(source))!=value:raise ValueError('Plot input changed')
    for name,value in manifest['output_sha256'].items():
        if sha(plot/name)!=value:raise ValueError('Rendered plot changed')
    for path in plot.iterdir():
        if path.is_file():save(path,'plot/'+path.name,'rendered and visually inspected scientific plot of the failed spatial checks')
    step=runs/'ours_proj128_anderson_failure_2steps_v1/step_0002'
    pv=json.loads((step/'paraview_step_readback.json').read_text());geometry=json.loads((step/'paraview_export.json').read_text())
    if not pv['passed'] or not geometry['geometry_verified']:raise ValueError('ParaView readback or geometry verification failed')
    for source,value in pv['source_sha256'].items():
        if sha(Path(source))!=value:raise ValueError('ParaView field changed')
        keep(source,'actual ParaView artifact or its original solver field')
    for name in ('paraview_step_readback.json','paraview_export.json','metrics.json','case.json'):
        save(step/name,'paraview128/'+name,'completed transient step at t=0.01, not a steady state')
    for name in ('run_manifest.json','run_completion.json'):
        save(step.parent/name,'paraview128/'+name,'completed native two-step trajectory provenance')
    launch=runs/'ours_proj64_timehalf_anderson_v1'
    for name in ('case.json','run_manifest.json'):save(launch/name,'timehalf_launch/'+name,'launch only; no completed temporal sensitivity claim')
    save(runs/'config_ours_proj64_timehalf_anderson_v1.json','timehalf_launch/input.json','actual time-step-halving configuration')
    for name in ('analyze_twisted_refinement.py','plot_twisted_refinement.py','check_twisted_projection_timestep.py',
                 'export_twisted_paraview.py','check_twisted_paraview_step.py','record_twisted_spatial_checkpoint.py'):
        save(repo/'scripts'/name,'sources/'+name,'reproduction or verification source')
    save(repo/'validation/twisted/SPATIAL_CONVERGENCE_32_64.md','scope.md','failed spatial acceptance and remaining work')
    receipt={'scope':__doc__,'spatial_convergence_passed':False,'goal_complete':False,'files':files,'retained_runtime_files':list(retained.values())}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'archive_files':len(files),'retained_files':len(retained),'spatial_convergence_passed':False,'goal_complete':False}))


if __name__=='__main__':main()
