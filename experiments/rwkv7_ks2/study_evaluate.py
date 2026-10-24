"""Full-shard evaluation with the study model factory restored."""
import argparse
import json
from pathlib import Path
import sys
import evaluate_spread
from study_models import build_model

if __name__=='__main__':
 p=argparse.ArgumentParser(add_help=False);p.add_argument('--config',type=Path,required=True)
 opts,remaining=p.parse_known_args();cfg=json.loads(opts.config.read_text())
 evaluate_spread.run.Model=lambda seed,variant,**kw:build_model(seed,cfg)
 evaluate_spread.install=lambda model,coding:model
 sys.argv=[sys.argv[0],*remaining];evaluate_spread.main()
