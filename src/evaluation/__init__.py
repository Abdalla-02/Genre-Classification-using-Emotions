"""Evaluation: multi-label metrics and cross-validated scoring."""

from src.evaluation.crossval import N_SPLITS, evaluate, evaluate_folds
from src.evaluation.metrics import (
    METRICS,
    macro_prf,
    multilabel_metrics,
    per_genre_f1,
)
from src.evaluation.repeated import (
    N_REPEATS,
    RepeatedGroupKFold,
    corrected_paired_t,
    diff_ci,
    nb_p_limit,
    repeat_ci,
    repeat_means,
    win_rate,
)

__all__ = [
    "evaluate", "evaluate_folds", "N_SPLITS", "multilabel_metrics", "METRICS", "macro_prf", "per_genre_f1",
    # repeated CV + significance (see repeated.py)
    "RepeatedGroupKFold", "N_REPEATS", "repeat_ci", "repeat_means", "diff_ci",
    "corrected_paired_t", "nb_p_limit", "win_rate",
]
