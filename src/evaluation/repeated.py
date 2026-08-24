"""Repeated group-wise cross-validation + the significance machinery it enables.

Motivation
----------
Every headline comparison in this thesis (emotion vs AST, predicted-emotion vs
PCA-8 control, ...) was previously estimated from a *single* 5-fold GroupKFold run.
Five folds give five numbers, so the standard error of a mean Macro-F1 is large and
the paired t-test over five folds is badly underpowered -- margins of +0.06 came out
"not significant" purely for lack of resamples, which is a statement about the
experiment, not about the models.

This module adds R x K repeated group k-fold: the film -> fold assignment is
re-randomised R times, giving R*K score samples instead of K while never splitting a
film across train/test. Two statistics are provided, both standard for repeated CV:

* :func:`repeat_ci` -- a t-interval built from the R *per-repeat* means. Folds inside
  one repeat share training data and are correlated, but the R repeat-means are much
  closer to independent, so this is the conservative, defensible interval to report.
* :func:`corrected_paired_t` -- the Nadeau & Bengio (2003) corrected resampled t-test.
  A naive paired t-test over all R*K fold differences treats overlapping training sets
  as independent and is anti-conservative (it will happily report p<0.001 for noise);
  the correction inflates the variance by the train/test overlap factor.

sklearn 1.5 has no shuffled ``GroupKFold``, hence the splitter here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from src import config

N_SPLITS = 5
N_REPEATS = 10


class RepeatedGroupKFold:
    """R x K group k-fold with a re-randomised group -> fold assignment per repeat.

    Within a repeat, groups are visited in random order and each is assigned to the
    currently smallest fold (GroupKFold's greedy balancing, with the deterministic
    size-descending order replaced by a random one). This keeps folds close to equal
    size while making the partitions genuinely different across repeats.
    """

    def __init__(self, n_splits: int = N_SPLITS, n_repeats: int = N_REPEATS,
                 random_state: int = config.SEED):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits * self.n_repeats

    def _fold_ids(self, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        uniq = np.unique(groups)
        sizes = {g: int((groups == g).sum()) for g in uniq}
        order = rng.permutation(uniq)
        load = np.zeros(self.n_splits, dtype=int)
        assign: dict = {}
        for g in order:
            f = int(np.argmin(load))
            assign[g] = f
            load[f] += sizes[g]
        return np.array([assign[g] for g in groups])

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("RepeatedGroupKFold requires 'groups'")
        groups = np.asarray(groups)
        idx = np.arange(len(groups))
        for r in range(self.n_repeats):
            rng = np.random.default_rng(self.random_state + 1000 * r)
            fold_of = self._fold_ids(groups, rng)
            for f in range(self.n_splits):
                test = idx[fold_of == f]
                train = idx[fold_of != f]
                yield train, test


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
@dataclass
class Interval:
    mean: float
    lo: float
    hi: float

    def __str__(self) -> str:
        return f"{self.mean:.3f} [{self.lo:.3f}, {self.hi:.3f}]"


def repeat_means(scores: np.ndarray, n_splits: int = N_SPLITS) -> np.ndarray:
    """Collapse a flat (n_repeats*n_splits,) score vector to one mean per repeat."""
    return np.asarray(scores, float).reshape(-1, n_splits).mean(axis=1)


def repeat_ci(scores: np.ndarray, n_splits: int = N_SPLITS,
              alpha: float = 0.05) -> Interval:
    """95% t-interval over the per-repeat means (the conservative interval we report)."""
    m = repeat_means(scores, n_splits)
    mean = float(m.mean())
    if len(m) < 2:
        return Interval(mean, float("nan"), float("nan"))
    se = float(m.std(ddof=1) / np.sqrt(len(m)))
    half = float(stats.t.ppf(1 - alpha / 2, len(m) - 1) * se)
    return Interval(mean, mean - half, mean + half)


def corrected_paired_t(a: np.ndarray, b: np.ndarray, n_train: int, n_test: int
                       ) -> tuple[float, float]:
    """Nadeau & Bengio (2003) corrected resampled paired t-test on fold differences.

    The naive paired t-test assumes the R*K differences are independent; they are not,
    because the training sets overlap heavily. The correction replaces the variance
    term ``var/n`` with ``var * (1/n + n_test/n_train)``. Returns ``(t, p)`` two-sided.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    var = float(d.var(ddof=1))
    if var == 0.0:
        return float("nan"), float("nan")
    denom = np.sqrt(var * (1.0 / n + n_test / n_train))
    t = float(d.mean() / denom)
    p = float(2 * stats.t.sf(abs(t), n - 1))
    return t, p


def nb_p_limit(a: np.ndarray, b: np.ndarray, n_train: int, n_test: int) -> float:
    """The p-value :func:`corrected_paired_t` would converge to with infinite repeats.

    In the Nadeau-Bengio denominator ``var * (1/n + n_test/n_train)`` only the ``1/n``
    term shrinks as repeats are added; the overlap term ``n_test/n_train`` does not.
    So the test has a floor: dropping ``1/n`` gives the best p-value that ANY number of
    repeats could reach on this corpus. If that floor is still above 0.05, no amount of
    extra resampling can make the comparison significant -- only more films can. This
    turns "not significant" from a dead end into a statement about what the data can
    support.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    var = float(d.var(ddof=1))
    if var == 0.0:
        return float("nan")
    t = float(d.mean() / np.sqrt(var * (n_test / n_train)))
    return float(2 * stats.norm.sf(abs(t)))


def diff_ci(a: np.ndarray, b: np.ndarray, n_splits: int = N_SPLITS,
            alpha: float = 0.05) -> Interval:
    """95% t-interval for the paired difference a-b, over per-repeat mean differences."""
    return repeat_ci(np.asarray(a, float) - np.asarray(b, float), n_splits, alpha)


def win_rate(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of folds where a strictly beats b (nonparametric, assumption-free)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float((a > b).mean())
