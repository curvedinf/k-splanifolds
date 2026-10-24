"""Restore the selected RWKV/ROSA architecture for full-shard evaluation."""
import argparse
import sys
import evaluate_spread
from rwkv8_rosa import MODES,RWKV8Model

if __name__=='__main__':
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mode',choices=MODES,required=True)
    opts,remaining=parser.parse_known_args()
    evaluate_spread.run.Model=lambda seed,variant,**kwargs:RWKV8Model(seed,opts.mode,kwargs['ffn_charts'])
    evaluate_spread.install=lambda model,coding:model
    sys.argv=[sys.argv[0],*remaining]
    evaluate_spread.main()
