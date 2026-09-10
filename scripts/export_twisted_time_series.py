"""Export completed physical steps and ParaView collections with actual times."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    summary = json.loads((run/'transient_summary.json').read_text())
    if not summary['converged']:
        raise ValueError('Physical time sequence did not finish')
    with (run/'time_history.csv').open() as f:
        history = list(csv.DictReader(f))
    if len(history) != summary['steps_completed'] or any(row['inner_converged'] != 'true' for row in history):
        raise ValueError('Time sequence contains an unconverged or missing step')
    repo = Path(__file__).resolve().parents[1]
    collections = {}
    exported = []
    for name in ('solution', 'walls'):
        root = ET.Element('VTKFile', type='Collection', version='0.1', byte_order='LittleEndian')
        collections[name] = (root, ET.SubElement(root, 'Collection'))
    for row in history:
        folder = f'step_{int(row["step"]):04d}'
        case = json.loads((run/folder/'case.json').read_text())
        if case['physical_time'] != float(row['time']):
            raise ValueError('Step metadata differs from the physical time history')
        metrics=json.loads((run/folder/'metrics.json').read_text())
        if not metrics['converged']:
            raise ValueError('Step metrics are not converged')
        step=int(row['step']);stride=summary.get('output_stride',1)
        expected=step==1 or step==summary['steps_completed'] or step%stride==0
        if metrics.get('field_output_written',True)!=expected:
            raise ValueError('Field output differs from its declared schedule')
        if not expected:
            continue
        subprocess.run([sys.executable, str(repo/'scripts/export_twisted_paraview.py'), '--run', str(run/folder)],
                       cwd=repo, check=True, stdout=subprocess.DEVNULL)
        check = json.loads((run/folder/'paraview_export.json').read_text())
        if not check['geometry_verified']:
            raise ValueError('Cut geometry export did not verify')
        exported.append(step)
        for name, extension in (('solution', 'vtu'), ('walls', 'vtp')):
            ET.SubElement(collections[name][1], 'DataSet', timestep=row['time'], group='', part='0', file=f'{folder}/{name}.{extension}')
    for name, (root, _) in collections.items():
        ET.indent(root)
        ET.ElementTree(root).write(run/f'{name}.pvd', encoding='utf-8', xml_declaration=True)
    if 'field_output_steps' in summary and exported!=summary['field_output_steps']:
        raise ValueError('Exported fields differ from summary')
    print(json.dumps({'physical_steps': len(history), 'exported_steps':exported, 'last_time': float(history[-1]['time']),
                      'velocity_collection': str(run/'solution.pvd'), 'wall_collection': str(run/'walls.pvd')}))


if __name__ == '__main__':
    main()
