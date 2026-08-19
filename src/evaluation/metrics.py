"""Multi-label evaluation metrics.

Three complementary metrics (each covers a blind spot of the others):
  * exact_match  -- subset accuracy: the whole 8-genre vector must be correct (strict)
  * hamming_loss -- fraction of individually wrong labels (looks good under imbalance)
  * macro_f1     -- unweighted mean F1 across genres (fair to rare genres)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, hamming_loss

METRICS = ["exact_match", "hamming_loss", "macro_f1"]


def multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "exact_match": accuracy_score(y_true, y_pred),  # subset accuracy
        "hamming_loss": hamming_loss(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
