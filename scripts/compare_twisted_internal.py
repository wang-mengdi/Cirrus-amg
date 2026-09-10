"""Auxiliary Cirrus adaptive/uniform comparison; NOT an independent CFD baseline."""
import argparse
import json
from pathlib import Path
import numpy as np
from compare_twisted import read,ordered,vector,error,transfer

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adaptive',type=Path,required=True)
    parser.add_argument('--uniform',type=Path,required=True)
    args=parser.parse_args()
    case=json.loads((args.adaptive/'case.json').read_text(encoding='utf-8-sig'))
    gp=Path(case['embedded_geometry']);mp=gp.with_suffix('.meta.json')
    geo=json.loads((mp if mp.exists() else gp).read_text())
    shift=np.array(geo.get('reference_translation',[0,0,0]))
    ours=ordered(read(args.adaptive/'solution.csv'));ref=read(args.uniform/'solution.csv')
    def translated(data):
        data=data.copy()
        for d,name in enumerate(('x','y','z')):data[name]+=shift[d]
        data['x']%=geo['extent'][0]
        return ordered(data)
    ref=translated(ref);h=geo['finest_h'];shape=[int(round(x/h)) for x in geo['extent']]
    sampled,trans=transfer(ours,ref,shape,h)
    volumes=ours['volume'];p=ours['p']-np.average(ours['p'],weights=volumes)
    pr=sampled[:,3]-np.average(sampled[:,3],weights=volumes)
    walls=ordered(read(args.adaptive/'walls.csv'));wr=translated(read(args.uniform/'walls.csv'))
    assert np.allclose(vector(walls,['x','y','z']),vector(wr,['x','y','z']),rtol=0,atol=1e-13)
    assert np.allclose(walls['area'],wr['area'],rtol=1e-12,atol=0)
    cut=np.isin(ours['id'],walls['owner'])
    metrics=[json.loads((p/'metrics.json').read_text()) for p in (args.adaptive,args.uniform)]
    report={'scope':'Internal Cirrus uniform/adaptive check; independent Aphros comparison is still required',
            'both_converged':all(m['converged'] for m in metrics),'transfer':trans,
            'velocity':error(vector(ours,['u','v','w']),sampled[:,:3],volumes),
            'pressure':error(p,pr,volumes),
            'cut_cell_velocity':error(vector(ours,['u','v','w'])[cut],sampled[cut,:3],volumes[cut]),
            'wall_shear':error(vector(walls,['tau_x','tau_y','tau_z']),vector(wr,['tau_x','tau_y','tau_z']),wr['area']),
            'volume_flux_relative_difference':abs(metrics[0]['volume_flux']/metrics[1]['volume_flux']-1)}
    (args.adaptive/'internal_uniform_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
