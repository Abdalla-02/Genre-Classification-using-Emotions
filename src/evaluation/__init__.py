"""Evaluation: multi-label metrics and cross-validated scoring."""

from src.evaluation.crossval import N_SPLITS, evaluate, evaluate_folds
from src.evaluation.metrics import METRICS, multilabel_metrics

__all__ = ["evaluate", "evaluate_folds", "N_SPLITS", "multilabel_metrics", "METRICS"]
