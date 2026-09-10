"""Build an isolated optional double-AMG candidate with unchanged extended Aphros equations and residual correction.

Only the approximate linear solve may use double. Original assembled long-double
matrices, RHS, compatibility handling, six refinement attempts and all original
row checks are retained. The existing extended AMG path remains the default.
This candidate is not an accepted independent reference until actual validation.
"""
import argparse
import datetime
import difflib
import json
from pathlib import Path
import shutil
import subprocess
from build_extended_reference import sha


def select_mixed_backend(body):
    def replace(old,new):
        nonlocal body
        if body.count(old)!=1:raise ValueError('Unknown mixed-AMG patch context: '+old[:80])
        body=body.replace(old,new)
    replace('  struct Factorization {', '''#ifdef APHROS_TWISTED_HAVE_AMGCL
  using DoubleBackend=amgcl::backend::builtin<double>;
  using DoubleAmg=amgcl::make_solver<amgcl::amg<DoubleBackend,
      amgcl::coarsening::smoothed_aggregation,amgcl::relaxation::spai0>,
      amgcl::solver::cg<DoubleBackend>>;
  using DoubleGeneralAmg=amgcl::make_solver<amgcl::amg<DoubleBackend,
      amgcl::coarsening::smoothed_aggregation,amgcl::relaxation::spai0>,
      amgcl::solver::bicgstab<DoubleBackend>>;
#endif
  struct Factorization {''')
    replace('    bool symmetric=true,use_amg=false;', '    bool symmetric=true,use_amg=false,double_amg=false;')
    replace('    std::unique_ptr<GeneralAmg> general_amg;', '''    std::unique_ptr<GeneralAmg> general_amg;
    std::unique_ptr<DoubleAmg> double_solver;
    std::unique_ptr<DoubleGeneralAmg> double_general_solver;''')
    replace('  size_t compatibility_calls_=0;', '''  size_t compatibility_calls_=0;
  std::ofstream mixed_log_;
  size_t mixed_calls_=0;''')
    selector='    const bool requested_amg=std::getenv("APHROS_TWISTED_AMG")!=nullptr;'
    replace(selector,selector+'''
    const bool requested_double_amg=requested_amg && std::getenv("APHROS_TWISTED_DOUBLE_AMG")!=nullptr;''')
    replace('      if(factor.use_amg!=requested_amg)return false;',
            '      if(factor.use_amg!=requested_amg || factor.double_amg!=requested_double_amg)return false;')
    replace('      use_amg_=requested_amg;trace("symmetry-temporaries-released");',
            '      use_amg_=requested_amg;factor.double_amg=requested_double_amg;trace("symmetry-temporaries-released");')
    begin='        Eigen::SparseMatrix<Scal,Eigen::RowMajor> rowmajor=cached_matrix;trace("amg-rowmajor-ready");'
    end='''          general_amg_.reset(new GeneralAmg(rowmajor,p));
        }'''
    replace(begin,'''        if(factor.double_amg) {
          Eigen::SparseMatrix<double,Eigen::RowMajor> rowmajor=cached_matrix.template cast<double>();
          // Conversion belongs only to the approximate solve. Keep the full
          // extended matrix for residual correction and original row checks.
          for(int outer=0;outer<cached_matrix.outerSize();++outer)
            for(typename Sparse::InnerIterator it(cached_matrix,outer);it;++it) {
              const double value=static_cast<double>(it.value());
              if(!std::isfinite(value)||(value==0 && it.value()!=Scal(0)))
                throw std::runtime_error("Double AMG coefficient conversion lost finite support");
            }
          trace("amg-rowmajor-ready");
          if(symmetric_) {
            typename DoubleAmg::params p;p.solver.tol=1e-13;p.solver.maxiter=500;
            factor.double_solver.reset(new DoubleAmg(rowmajor,p));
          } else {
            typename DoubleGeneralAmg::params p;p.solver.tol=1e-13;p.solver.maxiter=500;
            factor.double_general_solver.reset(new DoubleGeneralAmg(rowmajor,p));
          }
        } else {
'''+begin)
    replace(end,end+'\n        }')
    trace='<<" rows="<<n<<" nnz="<<matrix_nonzeros'
    replace(trace,trace+'<<" double_amg="<<factor.double_amg')
    begin='        std::vector<Scal> source(b.data(),b.data()+n),answer(n,0.);'
    end='        x=Eigen::Map<Dense>(answer.data(),n)*scale;'
    replace(begin,'''        if(factor.double_amg) {
          std::vector<double> source(n),answer(n,0.);
          for(int i=0;i<n;++i)source[i]=static_cast<double>(b[i]/scale);
          const auto info=symmetric_?(*factor.double_solver)(source,answer):(*factor.double_general_solver)(source,answer);
          linear_iterations+=int(std::get<0>(info));
          for(int i=0;i<n;++i)x[i]=Scal(answer[i])*scale;
        } else {
'''+begin)
    replace(end,end+'\n        }')
    trace='    trace("original-rows-checked");'
    replace(trace,trace+'''
    if(factor.double_amg) {
      if(!mixed_log_.is_open()) {
        mixed_log_.open("mixed_precision.csv");
        mixed_log_<<std::setprecision(21)<<"call,system,rows,refinements,linear_iterations,original_row_residual,tolerance\\n";
      }
      mixed_log_<<++mixed_calls_<<','<<system.GetName()<<','<<n<<','<<refinements<<','<<linear_iterations
                <<','<<residual<<','<<tolerance<<'\\n';
      mixed_log_.flush();if(!mixed_log_.good())throw std::runtime_error("Mixed AMG diagnostic write failed");
    }''')
    return body

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base, out = args.reference_build.resolve(), args.output.resolve()
    manifest = json.loads((base/'build_manifest.json').read_text())
    if not manifest['passed'] or not manifest['source_unchanged']:
        raise ValueError('Require a completed unchanged reference build')
    for path, value in manifest['source_sha256'].items():
        if sha(Path(path)) != value:
            raise ValueError('Reference build source changed: ' + path)
    if sha(base/'twisted_extended.exe') != manifest['executable_sha256']:
        raise ValueError('Reference executable changed')
    drivers = [r for r in manifest['results'] if Path(r['source']).name == 'driver.cpp']
    if len(drivers) != 1:
        raise ValueError('Require one driver translation unit')
    old = Path(drivers[0]['source']).parent
    compiler = Path(drivers[0]['command'][0])
    if sha(compiler) != manifest['compiler_sha256']:
        raise ValueError('Reference compiler changed')
    out.mkdir(parents=True, exist_ok=False)
    src = out/'sources'
    shutil.copytree(old, src)
    relative = Path('src/linear/twisted_direct.h')
    original = (old/relative).read_text()
    candidate = select_mixed_backend(original)
    (src/relative).write_text(candidate)
    changes = [p.relative_to(old).as_posix() for p in old.rglob('*')
               if p.is_file() and sha(p) != sha(src/p.relative_to(old))]
    if changes != [relative.as_posix()]:
        raise ValueError('Unexpected source change: ' + repr(changes))
    (out/'backend.patch').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), candidate.splitlines(True),
        fromfile='reference/'+relative.as_posix(), tofile='candidate/'+relative.as_posix())))
    prepared = json.loads((src/'prepare_manifest.json').read_text())
    prepared['source_sha256'][str(Path(__file__).resolve())] = sha(Path(__file__))
    keys = [k for k in prepared['prepared_source_sha256']
            if Path(k).as_posix() == relative.as_posix()]
    if len(keys) != 1:
        raise ValueError('Require one original prepared backend hash')
    prepared['prepared_source_sha256'][keys[0]] = sha(src/relative)
    prepared['mixed_amg_scope'] = __doc__
    (src/'prepare_manifest.json').write_text(json.dumps(prepared, indent=2)+'\n')
    before = {str(p): sha(p) for p in src.rglob('*') if p.is_file()}
    shutil.copyfile(__file__, out/'build_source.py.txt')
    rows, rebuilt, dependency_rows = [], [], []
    now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    for row_index, original_row in enumerate(manifest['results']):
        obj = Path(original_row['object'])
        if original_row['exit_code'] or sha(obj) != original_row['object_sha256']:
            raise ValueError('Reference object changed: ' + str(obj))
        source = src/Path(original_row['source']).relative_to(old)
        target = out/obj.name
        command = [a.replace(str(old), str(src)) for a in original_row['command']]
        command[command.index('-o')+1] = str(target)
        # Ask the same compiler, with the same include paths and defines, for
        # all user-header dependencies. Do not infer inclusion from filenames.
        dep_command = command[:]
        dep_command.remove('-c')
        oi = dep_command.index('-o')
        del dep_command[oi:oi+2]
        dep_command += ['-MM', '-MT', 'verified_object']
        dependency = out/f'dependency_{row_index:02d}.txt'
        with dependency.open('w') as log:
            code = subprocess.run(dep_command, stdout=log, stderr=subprocess.STDOUT).returncode
        if code:
            raise ValueError('Compiler dependency scan failed; retain output')
        dep_text = dependency.read_text().replace('\\\n', ' ').replace('\\', '/')
        affected = (src/relative).as_posix() in dep_text
        dependency_rows.append({'source': str(source), 'command': dep_command,
                                'output': str(dependency), 'sha256': sha(dependency),
                                'exit_code': code, 'affected': affected})
        row = dict(original_row)
        row.update(source=str(source), object=str(target))
        if affected:
            logpath = out/(obj.name+'.log')
            start = now()
            with logpath.open('w') as log:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
            row.update(command=command, started_utc=start, completed_utc=now(),
                       exit_code=code, object_sha256=sha(target) if target.exists() else None,
                       log_sha256=sha(logpath))
            if code:
                raise ValueError('Mixed AMG compilation failed; retain output')
            rebuilt.append(source.relative_to(src).as_posix())
        else:
            shutil.copyfile(obj, target)
            row.update(reused_from=str(obj), original_source=original_row['source'],
                       object_sha256=sha(target))
        rows.append(row)
        print(json.dumps({'source': source.relative_to(src).as_posix(),
                          'recompiled': affected, 'completed': row_index+1}), flush=True)
    if not rebuilt:
        raise ValueError('No translation unit consumed the changed backend')
    exe = out/'twisted_extended.exe'
    command = [str(compiler), '-fopenmp', '-static', '-Wl,--gc-sections',
               *[r['object'] for r in rows], '-lpsapi', '-o', str(exe)]
    with (out/'link.log').open('w') as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    unchanged = all(sha(Path(p)) == h for p, h in before.items())
    original_unchanged = all(sha(Path(p)) == h for p, h in manifest['source_sha256'].items())
    result = {'scope': __doc__, 'passed': code == 0 and unchanged and original_unchanged,
              'source_unchanged': unchanged, 'original_source_unchanged': original_unchanged,
              'source_sha256': before, 'compiler_sha256': sha(compiler), 'results': rows,
              'dependencies': dependency_rows, 'link_command': command, 'link_exit_code': code,
              'executable_sha256': sha(exe) if exe.exists() else None, 'completed_utc': now(),
              'reference_build': str(base), 'reference_build_manifest_sha256': sha(base/'build_manifest.json'),
              'recompiled_translation_units': rebuilt, 'reused_objects': len(rows)-len(rebuilt),
              'backend_patch_sha256': sha(out/'backend.patch')}
    (out/'build_manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ('source_sha256', 'results', 'dependencies', 'link_command')}), flush=True)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
