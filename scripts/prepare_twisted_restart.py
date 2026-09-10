"""Create a lossless projection checkpoint from a completed native GPU run."""
import argparse
import json
from pathlib import Path
from twisted_restart import create_checkpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = create_checkpoint(args.run, args.step, args.output)
    print(json.dumps({k: result[k] for k in ('physical_step', 'physical_time', 'cells', 'faces', 'state_sha256')}))
