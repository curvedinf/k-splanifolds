"""Restore the exact selected RWKV mux arm for full-shard evaluation."""
import argparse
import sys
import evaluate_spread
from rwkv_mux import MODES,RWKVMuxModel

if __name__=='__main__':
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mode',choices=MODES,required=True)
    opts,remaining=parser.parse_known_args()
    evaluate_spread.run.Model=lambda seed,variant,**kwargs:RWKVMuxModel(seed,opts.mode,kwargs['ffn_charts'])
    evaluate_spread.install=lambda model,coding:model
    sys.argv=[sys.argv[0],*remaining]
    evaluate_spread.main()
