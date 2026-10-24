"""Full-shard evaluation with learned shared-mux gains restored."""
import argparse
import sys
import evaluate_spread
from single_mux_gains import MuxGainModel


if __name__=='__main__':
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--gain-placement',choices=['input','output','dynamic'],required=True)
    opts,remaining=parser.parse_known_args()
    evaluate_spread.run.Model = lambda seed,variant,**kwargs:MuxGainModel(seed,kwargs['ffn_charts'],opts.gain_placement)
    evaluate_spread.install = lambda model,coding:model
    sys.argv=[sys.argv[0],*remaining]
    evaluate_spread.main()
