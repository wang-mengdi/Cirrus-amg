"""Generate reproducible Aphros cases for one periodic unit of a twisted tube."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    repo=Path(__file__).resolve().parents[1]
    parser.add_argument('--root',type=Path,default=Path('D:/Dropbox/Agent-simulation/twisted-baseline'))
    parser.add_argument('--spec',type=Path,default=repo/'validation/twisted/case.json')
    parser.add_argument('--ny',type=int,nargs='+',default=[16,32])
    parser.add_argument('--max-iterations',type=int,default=5000)
    parser.add_argument('--velocity-relaxation',type=float,default=.7)
    parser.add_argument('--pressure-relaxation',type=float,default=.3)
    parser.add_argument('--pressure-linear-tolerance',type=float,default=1e-13)
    parser.add_argument('--iteration-tolerance',type=float,default=1e-11)
    parser.add_argument('--convection',action='store_true',help='Enable Navier-Stokes advection (SIMPLE: FOU; Proj: BCG)')
    parser.add_argument('--fluid-solver',choices=('simple','proj'),default='simple')
    parser.add_argument('--projection-diffusion-iterations',type=int,default=1)
    parser.add_argument('--time-step',type=float,default=1.)
    parser.add_argument('--time-steps',type=int,default=1)
    parser.add_argument('--momentum-mode',choices=('imp','exp'),default='imp',help='Original Aphros conv solver selection')
    parser.add_argument('--suffix',default='',help='Case directory suffix, preserving earlier runs')
    args=parser.parse_args()
    if not (0<args.velocity_relaxation<=1 and 0<args.pressure_relaxation<=1):
        parser.error('Relaxation factors must be in (0,1]')
    if args.max_iterations<1 or not 0<args.pressure_linear_tolerance<1:
        parser.error('Require positive iteration count and linear tolerance in (0,1)')
    if not 0<args.iteration_tolerance<1:parser.error('Iteration tolerance must be in (0,1)')
    if not 0<args.time_step<1e9 or args.time_steps<1:parser.error('Invalid physical time stepping')
    if args.momentum_mode=='exp' and not args.convection:parser.error('Explicit-mode validation currently requires Navier-Stokes with a physical time step')
    if args.projection_diffusion_iterations<1:parser.error('Projection diffusion iterations must be positive')
    if args.fluid_solver=='proj' and (not args.convection or args.momentum_mode!='imp'):
        parser.error('Projection driver currently requires Navier-Stokes with implicit diffusion')
    root=args.root.resolve(); spec=json.loads(args.spec.read_text(encoding='utf-8'))
    if spec['force'][1:]!=[0,0]:parser.error('This benchmark generator requires axial forcing')
    for ny in args.ny:
        if ny<8 or ny&(ny-1): parser.error('ny must be a power of two >= 8')
        nx=round(ny*spec['extent'][0]/spec['extent'][1])
        family='navier_stokes' if args.convection else 'stokes'
        if args.fluid_solver=='proj':family+='_proj'
        case=root/f'{family}_n{ny}{args.suffix}'
        if case.exists() and any(case.iterdir()):parser.error('Preserve existing cases; choose a fresh suffix')
        case.mkdir(parents=True,exist_ok=True)
        config=f'''include {(root/'aphros/deploy/scripts/sim_base.conf').as_posix()}
set string backend local
set int spacedim 3
set int dim 3
set int px 1
set int py 1
set int pz 1
set int bx 1
set int by 1
set int bz 1
set int bsx {nx}
set int bsy {ny}
set int bsz {ny}
set double extent {spec['extent'][0]}
set int loc_periodic_x 1
set int loc_periodic_y 1
set int loc_periodic_z 1
set int hypre_periodic_x 1
set int hypre_periodic_y 0
set int hypre_periodic_z 0
del bc_xm
del bc_xp
del bc_ym
del bc_yp
del bc_zm
del bc_zp
set int enable_bc 1
set string bc_path "inline
wall 0 0 0 {{ box 0 0 0 10 }}
"
set int enable_embed 1
set string eb_init none
set int twisted_pipe 1
set double twisted_radius {spec['radius']}
set double twisted_amplitude {spec['amplitude']}
set double twisted_period {spec['period']}
set double twisted_center_y {spec['center_y']}
set double twisted_center_z {spec['center_z']}
set int embed_smoothen_iters 0
set int enable_advection 0
set int CHECKNAN 1
set int enable_surftens 0
set string fluid_solver {args.fluid_solver}
set int stokes {0 if args.convection else 1}
set string convsc fou
set string conv {args.momentum_mode}
set int explviscous 0
set double rho1 {spec['rho']}
set double rho2 {spec['rho']}
set double mu1 {spec['rho']*spec['nu']}
set double mu2 {spec['rho']*spec['nu']}
set vect force {spec['force'][0]} 0 0
set vect gravity 0 0 0
set string vel_init zero
set double vrelax {args.velocity_relaxation}
set double prelax {args.pressure_relaxation}
set int second_order 0
set double dt0 {args.time_step}
set double dtmax {args.time_step}
set double tmax {args.time_step*args.time_steps}
del cfl
set int min_iter 1
set int max_iter {args.max_iterations}
set double tol {args.iteration_tolerance}
set string linsolver_gen conjugate
set string linsolver_symm conjugate
set double hypre_gen_tol 1e-13
set double hypre_symm_tol {args.pressure_linear_tolerance}
set int hypre_gen_maxiter 5000
set int hypre_symm_maxiter 5000
set int hypre_symm_miniter 0
set int linsolver_gen_maxnorm 1
set int linsolver_symm_maxnorm 1
set int verbose_stages 0
set int verbose_conf_unused 0
set int linreport 0
set string dumpformat plain
set string dumplist vx vy vz p
set double dump_field_t0 1e10
set int dumplast 1
set int dumpbc 1
'''
        if args.fluid_solver=='proj':
            config+=f'''set int proj_bcg 1
set int proj_redistr_adv 0
set int proj_diffusion_iters {args.projection_diffusion_iterations}
set int proj_diffusion_consistent_guess 1
'''
        (case/'a.conf').write_text(config,encoding='utf-8')
        (case/'case_manifest.json').write_text(json.dumps({'spec':spec,'ny':ny,
            'shape':[nx,ny,ny], 'velocity_relaxation':args.velocity_relaxation,
            'pressure_relaxation':args.pressure_relaxation,
            'pressure_linear_tolerance':args.pressure_linear_tolerance,
            'iteration_tolerance':args.iteration_tolerance,
            'fluid_solver':args.fluid_solver,
            'convection':args.convection,'convection_scheme':('bcg' if args.fluid_solver=='proj' else 'fou') if args.convection else 'none',
            'projection_parameters':{'bcg':1,'redistr_adv':0,'diffusion_iters':args.projection_diffusion_iterations,
                                     'diffusion_consistent_guess':1} if args.fluid_solver=='proj' else None,
            'momentum_mode':args.momentum_mode,
            'time_step':args.time_step,'time_steps':args.time_steps,
            'spec_sha256':hashlib.sha256(args.spec.read_bytes()).hexdigest(),
            'config_sha256':hashlib.sha256((case/'a.conf').read_bytes()).hexdigest()},indent=2)+'\n',encoding='utf-8')
        print(case)


if __name__=='__main__': main()
