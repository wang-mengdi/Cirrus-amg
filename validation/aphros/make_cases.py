from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'aphros/deploy/scripts/sim_base.conf'

for ny in (8, 16, 32):
    folder = ROOT / f'periodic_n{ny}'
    folder.mkdir(exist_ok=True)
    config = f'''include {BASE.as_posix()}
set string backend local
set int spacedim 2
set int dim 2
set int px 1
set int py 1
set int pz 1
set int bx 1
set int by 1
set int bz 1
set int bsx {8*ny}
set int bsy {ny}
set int bsz 1
set double extent 1
set int loc_periodic_x 1
set int loc_periodic_y 1
set int hypre_periodic_x 1
set int hypre_periodic_y 0
del bc_xm
del bc_xp
del bc_zm
del bc_zp
set int enable_bc 1
set string bc_path "inline
wall 0 0 0 {{ box 0 0 0 10 }}
"
set int enable_advection 0
set int CHECKNAN 1
set int enable_surftens 0
set string fluid_solver simple
set int stokes 1
set int explviscous 0
set double rho1 1
set double rho2 1
set double mu1 0.01
set double mu2 0.01
set vect force 1 0 0
set vect gravity 0 0 0
set string vel_init zero
set double vrelax 0.7
set double prelax 0.3
set int second_order 0
set double dt0 1
set double dtmax 1
set double tmax 1
del cfl
set int min_iter 1
set int max_iter 10000
set double tol 1e-12
set string linsolver_gen conjugate
set string linsolver_symm conjugate
set double hypre_gen_tol 1e-14
set double hypre_symm_tol 1e-14
set int hypre_gen_maxiter 5000
set int hypre_symm_maxiter 5000
set int hypre_symm_miniter 0
set int linsolver_gen_maxnorm 1
set int linsolver_symm_maxnorm 1
set int verbose_stages 0
set int verbose_conf_unused 0
set int linreport 0
set string dumpformat plain
set string dumplist vx vy p
set double dump_field_t0 1e10
set int dumplast 1
set int dumpbc 1
'''
    (folder / 'a.conf').write_text(config, encoding='utf-8')
    if ny in (8, 16):
        duct = ROOT/f'duct_3d_n{ny}'
        duct.mkdir(exist_ok=True)
        (duct/'a.conf').write_text(config + f'''
set int spacedim 3
set int dim 3
set int bsz {ny}
set int loc_periodic_z 1
set int hypre_periodic_z 0
''')
    if ny == 8:
        three = ROOT/'periodic_3d_n8'
        three.mkdir(exist_ok=True)
        three_config = config + '''
set int spacedim 3
set int dim 3
set int bsz 8
set int loc_periodic_z 1
set int hypre_periodic_z 1
'''
        (three/'a.conf').write_text(three_config)
        perturb = ROOT/'perturb_3d_n8'
        perturb.mkdir(exist_ok=True)
        (perturb/'a.conf').write_text(three_config + '''
set string vel_init simple-perturb
set double simple_perturb_amplitude 0.01
set double simple_perturb_height 0.125
''')
print('Generated periodic channel and four-wall square-duct cases')
