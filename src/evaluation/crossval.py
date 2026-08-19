"""Cross-validated evaluation of multi-label genre classifiers.

Two CV schemes:
  * "group"  -- GroupKFold by soundtrack (film): clips of the same film never split
                across train/test. This is the leakage-safe, REPORTED scheme.
  * "strat"  -- StratifiedKFold, stratified on a single-label proxy (``first_genre``,
                since sklearn's StratifiedKFold needs a 1-D target). Allows same-film
                clips in both folds -> the "leaky" comparison. The gap
                (strat - group) quantifies how much film-identity leakage inflates
                scores.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from src import config
from src.evaluation.metrics import METRICS, multilabel_metrics
from src.models import build_classifier

N_SPLITS = 5


def _splits(X, Y, groups, strat_labels, scheme: str, n_splits: int):
    if scheme == "group":
        # leakage-safe: same-film clips never split across train/test
        return GroupKFold(n_splits=n_splits).split(X, Y, groups)
    if scheme == "strat":
        # stratified on a single-label proxy; ignores film grouping (leaky)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.SEED)
        return skf.split(X, strat_labels)
    if scheme == "kfold":
        # plain shuffled split; isolates the grouping effect vs "group", and is the
        # right scheme when rows are independent (e.g. one row per film, Blockbuster)
        return KFold(n_splits=n_splits, shuffle=True, random_state=config.SEED).split(X)
    raise ValueError(scheme)


def evaluate_folds(X, Y, groups, strat_labels, clf_name: str, scheme: str,
                   n_splits: int = N_SPLITS) -> dict[str, np.ndarray]:
    """Run one classifier under one CV scheme; return {metric: per-fold score array}.

    Fold splits depend only on X/groups/strat_labels (not the feature values), so two
    feature sets evaluated with the same scheme get identical folds -> their per-fold
    scores can be paired for a significance test.
    """
    per_fold = {m: [] for m in METRICS}
    for train_idx, test_idx in _splits(X, Y, groups, strat_labels, scheme, n_splits):
        clf = build_classifier(clf_name)
        clf.fit(X[train_idx], Y[train_idx])
        pred = np.asarray(clf.predict(X[test_idx]))
        fold = multilabel_metrics(Y[test_idx], pred)
        for m in METRICS:
            per_fold[m].append(fold[m])
    return {m: np.asarray(v, dtype=float) for m, v in per_fold.items()}


def evaluate(X, Y, groups, strat_labels, clf_name: str, scheme: str,
             n_splits: int = N_SPLITS) -> dict[str, tuple[float, float]]:
    """Run one classifier under one CV scheme; return {metric: (mean, std)} over folds."""
    folds = evaluate_folds(X, Y, groups, strat_labels, clf_name, scheme, n_splits)
    return {m: (float(v.mean()), float(v.std())) for m, v in folds.items()}
