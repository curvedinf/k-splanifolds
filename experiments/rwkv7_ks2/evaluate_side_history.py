"""Restore history wiring before running the established full-shard evaluator."""
import evaluate_side
from side_history import HistoryModel


if __name__ == '__main__':
    evaluate_side.SideModel = HistoryModel
    evaluate_side.main()
