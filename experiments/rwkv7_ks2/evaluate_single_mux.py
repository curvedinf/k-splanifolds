"""Full-shard evaluation with exactly one mux per layer."""
import evaluate_spread
from single_mux import SingleMuxModel


if __name__=='__main__':
    evaluate_spread.run.Model = lambda seed,variant,**kwargs:SingleMuxModel(seed,kwargs['ffn_charts'])
    evaluate_spread.install = lambda model,coding:model
    evaluate_spread.main()
