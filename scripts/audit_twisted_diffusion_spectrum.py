"""Audit scalar diffusion stability on the actual assembled curved cut-cell grid.

For du/dt = -L*u, a negative-real-part eigenvalue of L gives a growing mode.
This is a zero-wall scalar operator test, not a coupled Navier-Stokes result.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs
from scipy.linalg import eig
from run_twisted_solver import sha
from check_twisted_mass import read


def load_operators(root):
    meta=json.loads((root/'operators/manifest.json').read_text())
    matrices={}
    for name,shape in meta['shapes'].items():
        path=root/'operators'/(name+'.csv')
        if shape['nonzeros']:
            entries=read(path)
            if len(entries)!=shape['nonzeros']:raise ValueError('Unexpected sparse record count')
            matrix=sparse.coo_matrix((entries['value'],(entries['row'].astype(int),entries['column'].astype(int))),
                shape=(shape['rows'],shape['columns'])).tocsr()
            if matrix.nnz!=len(entries):raise ValueError('Duplicate sparse coordinates')
        else:
            if path.read_text().strip()!='row,column,value':raise ValueError('Nonempty zero matrix dump')
            matrix=sparse.csr_matrix((shape['rows'],shape['columns']))
        matrices[name]=matrix
    return matrices,read(root/'operators/mesh_cells.csv'),read(root/'operators/mesh_faces.csv')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dense',action='store_true',help='Complete spectrum for bounded small grids (at most 4000 cells)')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    config=json.loads((args.run/'case.json').read_text())
    matrices,cells,faces=load_operators(args.run)
    n=len(cells)
    if args.dense and n>4000:raise ValueError('Dense spectrum is limited to small diagnostic grids')
    if not np.array_equal(cells['id'],np.arange(n)):raise ValueError('Expected ordered cell IDs')
    R=matrices['redistribution']
    diffusion=config['nu']*sparse.diags(1/cells['volume'])*(matrices['compact_diffusion']+matrices['deferred_diffusion'])
    report={'scope':__doc__,'momentum_mode':config['momentum_mode'],'cells':n,'matrices':{},
            'eigensolver':'dense LAPACK complete spectrum' if args.dense else 'ARPACK smallest real part',
            'redistribution_column_sum_error':float(abs(np.asarray(R.sum(axis=0)).ravel()-1).max())}
    for label,matrix in [('redistribution',R),('diffusion',diffusion)]:
        if args.dense:
            values,vectors=eig(matrix.toarray(),check_finite=True)
            np.savetxt(args.output/(label+'_spectrum.csv'),np.column_stack((values.real,values.imag)),
                delimiter=',',header='real,imaginary',comments='',fmt='%.17g')
        else:
            values,vectors=eigs(matrix,k=6,which='SR',tol=1e-10,maxiter=10000,v0=np.sin(np.arange(n)*.713)+.5)
        order=np.argsort(values.real)[:6];values=values[order];vectors=vectors[:,order]
        errors=[]
        scale=float(np.max(np.asarray(abs(matrix).sum(axis=1))))
        for value,v in zip(values,vectors.T):
            error=float(np.linalg.norm(matrix@v-value*v)/(scale*np.linalg.norm(v)))
            if error>1e-9:raise ValueError('Unverified eigenpair residual')
            errors.append(error)
        report['matrices'][label]={'eigenvalues':[{'real':float(v.real),'imaginary':float(v.imag)} for v in values],
            'eigenpair_residuals_scaled_by_matrix_infinity_norm':errors,'matrix_infinity_norm':scale,
            'negative_real_mode_found':bool(values[0].real < -1e-8*scale)}
        v=vectors[:,0]/np.max(abs(vectors[:,0]))
        np.savetxt(args.output/(label+'_mode.csv'),np.column_stack([cells[k] for k in ('x','y','z','h','volume')]+[v.real,v.imag]),
            delimiter=',',header='x,y,z,h,volume,mode_real,mode_imaginary',comments='',fmt='%.17g')
        (args.output/'partial_spectrum.json').write_text(json.dumps(report,indent=2)+'\n')
    report['audit_completed']=True
    report['source_sha256']={str(p.resolve()):sha(p) for p in (args.run/'operators').glob('*') if p.is_file()}
    for name in ('case.json','run_manifest.json','run_completion.json'):report['source_sha256'][str((args.run/name).resolve())]=sha(args.run/name)
    (args.output/'spectrum.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['matrices'],indent=2))


if __name__=='__main__':main()
