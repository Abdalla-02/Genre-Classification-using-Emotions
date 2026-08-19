"""Multi-label genre classifiers (Binary Relevance).

Each classifier predicts the 8-genre binary vector. Logistic Regression and Random
Forest follow the Binary Relevance strategy (one independent binary model per genre);
the MLP uses native multi-output sigmoid outputs. A ``most_frequent`` dummy is included
as a sanity floor (useful for reading Hamming Loss under class imbalance).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config

CLASSIFIERS = ["logreg", "rf", "mlp", "dummy"]


class _SingleClassSafe(BaseEstimator, ClassifierMixin):
    """Wrap a binary estimator so a single-class training fold degrades gracefully.

    With a rare genre (e.g. Documentary, base rate ~0.05), a CV training split can
    contain zero positives, and estimators like LogisticRegression raise on one-class
    input. This wrapper then predicts that constant class instead of erroring.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            self._constant = self.classes_[0]
            self._fitted = None
        else:
            self._constant = None
            self._fitted = clone(self.estimator).fit(X, y)
        return self

    def predict(self, X):
        if self._constant is not None:
            return np.full(X.shape[0], self._constant, dtype=int)
        return self._fitted.predict(X)


def build_classifier(name: str):
    """Return a fresh (unfitted) estimator that maps features -> (n, 8) label matrix."""
    if name == "logreg":
        # Binary Relevance: one L2 logistic regression per genre, class-balanced,
        # wrapped so single-class training folds (rare genres) don't crash.
        base = _SingleClassSafe(LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=config.SEED
        ))
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", MultiOutputClassifier(base)),
        ])

    if name == "rf":
        # Random Forest natively supports multi-label indicator targets.
        return RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=config.SEED,
            n_jobs=-1,
        )

    if name == "mlp":
        # One hidden layer; sklearn MLP handles a binary-indicator y as multi-label
        # (independent sigmoid outputs + cross-entropy). No class_weight in sklearn MLP.
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128,),
                alpha=1e-3,
                max_iter=1000,
                random_state=config.SEED,
            )),
        ])

    if name == "dummy":
        return DummyClassifier(strategy="most_frequent")

    raise ValueError(f"unknown classifier '{name}'")
