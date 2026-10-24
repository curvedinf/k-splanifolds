"""Restore direct attention skips for full-shard evaluation."""
import evaluate_spread
from raw_mux_skip import RawSkipModel


if __name__ == '__main__':
    evaluate_spread.run.Model = lambda seed, variant, **kwargs: RawSkipModel(seed, kwargs['ffn_charts'])
    evaluate_spread.install = lambda model, coding: model
    evaluate_spread.main()
