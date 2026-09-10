"""Check extended arithmetic feasibility on actual worst-cell expressions; no flow-field edits."""
import argparse,csv,json,subprocess
from pathlib import Path
from run_twisted_solver import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells-report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();repo=Path(__file__).resolve().parents[1]
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source=repo/'validation/aphros/pressure_precision_probe.cpp'
    compiled=out/source.name;compiled.write_bytes(source.read_bytes())
    compiler=Path('C:/ProgramData/mingw64/mingw64/bin/g++.exe')
    report=json.loads(args.cells_report.read_text())
    if not report['diagnostic_consistent']:raise ValueError('Need consistent pressure cells')
    compiler_version=subprocess.check_output([str(compiler),'--version'],text=True)
    flags=['-std=c++17','-O2','-fno-fast-math','-ffp-contract=off','-static']
    exe=out/'pressure_precision_probe.exe'
    build=subprocess.run([str(compiler),*flags,str(compiled),'-o',str(exe)],capture_output=True,text=True)
    (out/'build.log').write_text(build.stdout+build.stderr)
    if build.returncode:raise RuntimeError('Precision probe compilation failed')
    data=out/'input.csv'
    # Recover the normalization rate from the actual fsum balance, avoiding a
    # separately typed physical scale. All probe coordinates/coefficients remain original.
    with data.open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(('cell','sign','volume','rate','e0','e1','b','p0','p1'))
        for i,p in enumerate(report['worst_cells']):
            rate=abs(p['actual_flux_net_fsum'])/p['volume']/p['actual_relative_divergence']
            for f in p['faces']:
                writer.writerow((i,f['outward_sign'],p['volume'],rate,*[f[k] for k in ('e0','e1','b','p0','p1')]))
    run=subprocess.run([str(exe),str(data)],capture_output=True,text=True)
    (out/'output.csv').write_text(run.stdout);(out/'run.log').write_text(run.stderr)
    rows=list(csv.DictReader(run.stdout.splitlines()))
    passed=(run.returncode==0 and len(rows)==len(report['worst_cells']) and
            all(int(r['extended_digits'])>int(r['double_digits']) and float(r['local_extended_relative_divergence'])<1e-7 for r in rows))
    result={'passed':passed,'scope':__doc__,'coupled_pressure_system_solved':False,
            'original_flow_fields_modified':False,'compiler':str(compiler),'compiler_version':compiler_version,
            'compiler_sha256':sha(compiler),'flags':flags,'build_exit_code':build.returncode,'run_exit_code':run.returncode,
            'maximum_local_extended_relative_divergence':max((float(r['local_extended_relative_divergence']) for r in rows),default=None),
            'source_sha256':{str(p):sha(p) for p in (source,compiled,args.cells_report,data,exe,out/'output.csv')},
            'runner_sha256':sha(Path(__file__))}
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
