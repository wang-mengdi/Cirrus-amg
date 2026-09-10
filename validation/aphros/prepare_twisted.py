"""Configure an isolated Aphros worktree for the curved-pipe comparison.

Adds a prescribed level-set initialization hook and diagnostic dumps.
An explicit optional fix initializes the previously undefined cut-wall flux.
Run after applying the existing aphros-simple-dumps.patch to the worktree.
"""
import argparse
from pathlib import Path
import shutil


def replace_once(path, old, new):
    text = path.read_text(encoding='utf-8')
    if new in text:
        return
    if text.count(old) != 1:
        raise ValueError(f'Expected one patch location in {path}')
    path.write_text(text.replace(old, new), encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aphros',type=Path,required=True)
    parser.add_argument('--fix-wall-initialization',action='store_true',
                        help='Explicitly repair upstream SIMPLE uninitialized embedded-wall flux')
    parser.add_argument('--direct-backend',action='store_true',
                        help='Add an optional serial Eigen backend; requires Eigen include path at build time')
    parser.add_argument('--fix-flux-halo',action='store_true',
                        help='Add explicit runtime opt-in for missing SIMPLE face-flux halo exchange')
    args=parser.parse_args()
    root=args.aphros.resolve()
    if args.direct_backend:
        shutil.copyfile(Path(__file__).with_name('twisted_direct.h'),root/'src/linear/twisted_direct.h')
        replace_once(root/'src/linear/linear.ipp','#include "linear.h"',
                     '#include "linear.h"\n#include "twisted_direct.h"')
        replace_once(root/'src/linear/linear.ipp',
                     '  using Owner = SolverConjugate<M>;','  using Owner = SolverConjugate<M>;\n  TwistedSerialDirect<M> direct_;\n  Info direct_info_{};')
        replace_once(root/'src/linear/linear.ipp',
                     '    auto sem = m.GetSem(__func__);\n    struct {\n      FieldCell<Scal> fcu;\n      FieldCell<Scal> fcr;',
                     '''    if (std::getenv("APHROS_TWISTED_DIRECT")) {
      auto direct_sem = m.GetSem("twisted-direct");
      if (direct_sem("solve")) {
        direct_info_ = direct_.Solve(fc_system,fc_sol,m,conf.tol);
        m.Comm(&fc_sol,M::CommStencil::direct_one);
      }
      return direct_info_;
    }
    auto sem = m.GetSem(__func__);
    struct {
      FieldCell<Scal> fcu;
      FieldCell<Scal> fcr;''')
    shutil.copyfile(Path(__file__).with_name('twisted_diagnostics.h'),root/'src/solver/twisted_diagnostics.h')
    replace_once(root/'src/solver/simple.ipp','#include "simple.h"',
                 '#include "simple.h"\n#include "twisted_diagnostics.h"')
    replace_once(root/'src/solver/convdiffi.ipp','#include "convdiffi.h"',
                 '#include "convdiffi.h"\n#include "twisted_diagnostics.h"')
    replace_once(root/'src/solver/convdiffi.ipp',
                 '            ffq[cf] += ffu[cf] * ffv[cf];\n          });\n        }',
                 '            ffq[cf] += ffu[cf] * ffv[cf];\n          });\n        }\n        TwistedAdvectionDump(m, eb, fcu, ffv, ffq);')
    replace_once(root/'src/solver/simple.ipp','    UpdateDerivedConditions();\n\n    fcfcd_',
                 '    UpdateDerivedConditions();\n    TwistedGeometryDump(m, eb);\n\n    fcfcd_')
    replace_once(root/'src/solver/simple.ipp','    if (sem("debug-final")) {',
                 '    if (sem("debug-final")) {\n      TwistedWallDump(m, eb, cd_->GetVelocity(Step::time_curr), me_vel_, *owner_->fcd_);')
    replace_once(root/'src/solver/simple.ipp','    cd_ = GetConvDiff<EB>()(par.conv, m, eb, cdargs);',
                 '    cd_ = GetConvDiff<EB>()(par.conv, m, eb, cdargs);\n    TwistedTimeDump(m, eb, args.fcvel, owner_->GetTime(), true);')
    replace_once(root/'src/solver/simple.ipp',
                 '      TwistedWallDump(m, eb, cd_->GetVelocity(Step::time_curr), me_vel_, *owner_->fcd_);',
                 '      TwistedWallDump(m, eb, cd_->GetVelocity(Step::time_curr), me_vel_, *owner_->fcd_);\n      TwistedTimeDump(m, eb, cd_->GetVelocity(Step::time_curr), owner_->GetTime(), false);')
    if args.fix_wall_initialization:
        replace_once(root/'src/solver/simple.ipp',
                     '    FieldFaceb<Scal> fev(m);\n\n    const Scal rh',
                     '''    // Initialize ALL faces, including embedded walls. The upstream
    // allocation leaves scalar memory undefined and the loop below only writes
    // Cartesian faces; otherwise cut walls inject arbitrary continuity flux.
    FieldFaceb<Scal> fev(m, 0.);
    eb.LoopFaces([&](auto cf) { fev[cf] = fftv[cf].dot(eb.GetSurface(cf)); });

    const Scal rh''')
    if args.fix_flux_halo:
        replace_once(root/'src/solver/simple.ipp',
                     '      fev_.iter_prev = fev_.iter_curr;\n    }\n\n    UpdateBc(sem);',
                     '''      fev_.iter_prev = fev_.iter_curr;
    }

    // The embedded upwind interpolation reads supporting faces across a
    // periodic/block seam. LoopFaces updates only inner faces; communicate
    // their volume flux before using its sign on supporting faces.
    // Explicit opt-in preserves an executable path for the upstream audit.
    if (std::getenv("APHROS_TWISTED_FIX_FLUX_HALO")) {
      if (sem.Nested("volume-flux-halo")) {
        CommFieldFace(fev_.iter_prev, m);
      }
    }

    UpdateBc(sem);''')
    replace_once(root/'src/util/posthook_default.cpp',
                 'void InitEmbedHook(FieldNode<typename M::Scal>&, const Vars&, M&) {}',
                 '''void InitEmbedHook(FieldNode<typename M::Scal>& phi, const Vars& var, M& m) {
  // Benchmark geometry only: positive inside a periodically curved circular tube.
  auto enabled = var.Int.Find("twisted_pipe");
  if (!enabled || !*enabled) return;
  const double radius = var.Double["twisted_radius"];
  const double amplitude = var.Double["twisted_amplitude"];
  const double period = var.Double["twisted_period"];
  const double cy = var.Double["twisted_center_y"];
  const double cz = var.Double["twisted_center_z"];
  for (auto node : m.AllNodes()) {
    const auto x = m.GetNode(node);
    const double a = 2 * M_PI * x[0] / period;
    const double y = x[1] - cy - amplitude * std::sin(a);
    const double z = (M::dim > 2 ? x[2] : 0.) - cz - amplitude * std::cos(a);
    phi[node] = radius - std::sqrt(y*y + z*z);
  }
  phi.SetHalo(2);
}''')
    print(f'Prepared diagnostic/geometry hooks in {root}; wall initialization fix={args.fix_wall_initialization}')


if __name__=='__main__':
    main()
