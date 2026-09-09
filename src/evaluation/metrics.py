"""Multi-label evaluation metrics.

Three complementary metrics (each covers a blind spot of the others):
  * exact_match  -- subset accuracy: the whole 8-genre vector must be correct (strict)
  * hamming_loss -- fraction of individually wrong labels (looks good under imbalance)
  * macro_f1     -- unweighted mean F1 across genres (fair to rare genres)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)

METRICS = ["exact_match", "hamming_loss", "macro_f1"]


def multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "exact_match": accuracy_score(y_true, y_pred),  # subset accuracy
        "hamming_loss": hamming_loss(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def macro_prf(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Macro-averaged precision / recall / F1 -- the reporting format of Ma et al. (2021).

    Their Table 4 reports precision, recall and F1 computed per genre and then
    macro-averaged. Reproducing that triple (rather than F1 alone) is what makes our
    cross-dataset numbers directly comparable with theirs; F1 alone hides whether a
    difference comes from over-prediction (recall up, precision down) or the reverse.
    """
    kw = dict(average="macro", zero_division=0)
    return {
        "macro_precision": precision_score(y_true, y_pred, **kw),
        "macro_recall": recall_score(y_true, y_pred, **kw),
        "macro_f1": f1_score(y_true, y_pred, **kw),
    }


def per_genre_f1(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-column F1 (no averaging), for the per-genre breakdown tables."""
    return f1_score(y_true, y_pred, average=None, zero_division=0)
