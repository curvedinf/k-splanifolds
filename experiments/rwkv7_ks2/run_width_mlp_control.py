"""One width-matched MLP seed-11 control, then four-offset validation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).parent
    out = args.out / 'mlp_11'
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'run.log').open('w') as log:
        subprocess.run([sys.executable, str(root / 'run_spread_gpu.py'), '--coding', 'learned',
                        '--data', str(root / 'gpu_runs/data'), '--out', str(out), '--variant', 'mlp',
                        '--width', str(args.width), '--layers', '4', '--ffn-charts', '32',
                        '--seed', '11', '--steps', '15000'], stdout=log, stderr=subprocess.STDOUT, check=True)
    with (out / 'tiled.log').open('w') as log:
        subprocess.run([sys.executable, str(root / 'evaluate_spread.py'), '--runs', str(args.out),
                        '--arm', 'mlp', '--seeds', '11'], stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads((out / 'results.json').read_text())
    result['tiled_evaluation'] = json.loads((args.out / 'tiled_mlp.json').read_text())[0]
    result['output_lr_multiplier'] = 1.
    (args.out / 'combined.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(width=args.width, parameters=result['parameters'],
                          evaluation=result['evaluation'], tiled_val=result['tiled_evaluation']['loss'],
                          seconds=result['training_seconds'])), flush=True)


if __name__ == '__main__':
    main()
