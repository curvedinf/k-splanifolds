"""Full-shard evaluation with the side-stream architecture restored exactly."""
import argparse
import sys
import evaluate_spread
from side_stream import SideModel


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--side-width', type=int, required=True)
    options, remaining = parser.parse_known_args()
    evaluate_spread.run.Model = lambda seed, variant, **kwargs: SideModel(seed, options.side_width)
    evaluate_spread.install = lambda model, coding: model
    sys.argv = [sys.argv[0], *remaining]
    evaluate_spread.main()


if __name__ == '__main__':
    main()
