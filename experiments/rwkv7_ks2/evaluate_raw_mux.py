"""Restore raw-coordinate history before full-shard evaluation."""
import evaluate_spread
from raw_mux import RawMuxModel


if __name__ == '__main__':
    evaluate_spread.run.Model = lambda seed, variant, **kwargs: RawMuxModel(seed, kwargs['ffn_charts'])
    evaluate_spread.install = lambda model, coding: model
    evaluate_spread.main()
