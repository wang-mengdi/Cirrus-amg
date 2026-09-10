"""Read all comparison fields through actual ParaView and reproduce the accepted weighted errors."""
import argparse
import json
from pathlib import Path

import numpy as np
from paraview import servermanager
from paraview.simple import XMLUnstructuredGridReader,XMLPolyDataReader,Delete
from vtkmodules.util.numpy_support import vtk_to_numpy
from run_twisted_solver import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--visualization',type=Path,required=True)
    a=p.parse_args();root=a.visualization.resolve();output=root/'paraview_difference_readback.json'
    if output.exists():raise ValueError('Preserve earlier ParaView readback evidence')
    manifest=json.loads((root/'difference_export.json').read_text());assert manifest['passed']
    hashes={str(root/'difference_export.json'):sha(root/'difference_export.json'),str(Path(__file__).resolve()):sha(Path(__file__))}
    for name,digest in manifest['output_sha256'].items():
        assert sha(root/name)==digest,name;hashes[str(root/name)]=digest
    expected=np.load(root/'expected_fields.npz');actual={};counts={}
    for prefix,name,reader_type in [('cells','solution.vtu',XMLUnstructuredGridReader),('walls','walls.vtp',XMLPolyDataReader)]:
        reader=reader_type(FileName=[str(root/name)]);reader.UpdatePipeline();data=servermanager.Fetch(reader)
        counts[prefix]=data.GetNumberOfCells()
        for key in expected.files:
            group,field=key.split('__')
            if group!=prefix or field in ('CutMask','Area'):continue
            values=vtk_to_numpy(data.GetCellData().GetArray(field)).copy()
            assert np.isfinite(values).all() and np.array_equal(values,expected[key]),key
            actual[key]=values
        Delete(reader)
    assert counts=={'cells':manifest['cells'],'walls':manifest['walls']}

    def relative(difference,reference,weights):
        if difference.ndim==1:difference=difference[:,None];reference=reference[:,None]
        return float(np.sqrt(np.sum(weights*np.sum(difference*difference,axis=1))/
                             np.sum(weights*np.sum(reference*reference,axis=1))))

    volume=actual['cells__FluidVolume'];cut=expected['cells__CutMask']
    differences={'velocity':relative(actual['cells__VelocityDifference'],actual['cells__AphrosVelocity'],volume),
                 'pressure':relative(actual['cells__PressureDifference'],actual['cells__AphrosPressureMeanZero'],volume),
                 'cut_cell_velocity':relative(actual['cells__VelocityDifference'][cut],actual['cells__AphrosVelocity'][cut],volume[cut]),
                 'wall_shear':relative(actual['walls__WallShearDifference'],actual['walls__AphrosWallShear'],expected['walls__Area'])}
    for key,value in differences.items():
        assert np.isclose(value,manifest['field_errors'][key]['relative_l2'],rtol=1e-11,atol=1e-16),key
    if any(sha(Path(path))!=digest for path,digest in hashes.items()):raise ValueError('Readback sources changed')
    report={'passed':True,'scope':__doc__,'cells':counts,'arrays_read_bitwise':len(actual),
            'reconstructed_relative_l2':differences,'source_sha256':hashes}
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'}))


if __name__=='__main__':main()
