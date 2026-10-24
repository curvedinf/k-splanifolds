"""Full-shard evaluation of one mux hypothesis with exact architecture restored."""
import argparse
import sys
import evaluate_spread
from mux_ablations import MODES, DEFAULT_CODE_SEED, build_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mode',choices=MODES,required=True)
    parser.add_argument('--code-seed',type=int,default=DEFAULT_CODE_SEED)
    opts,remaining = parser.parse_known_args()
    evaluate_spread.run.Model = lambda seed,variant,**kwargs:build_model(seed,kwargs['ffn_charts'],opts.mode,opts.code_seed)
    evaluate_spread.install = lambda model,coding:model
    sys.argv = [sys.argv[0],*remaining]
    evaluate_spread.main()
