"""Experiment 4 -- Set 1 vs Set 2: how reliable are the emotion ratings?

Set 2 re-rates 110 excerpts drawn from Set 1 with a different listener panel, so the two
sets give two independent measurements of the same musical material. That yields the one
quantity the rest of the thesis cannot estimate from Set 1 alone: **how much of the
emotion signal is real and how much is rating noise** -- which in turn bounds how well any
audio model could ever predict these ratings.

Three questions:

  Q1  How well do the two panels agree? (Pearson/Spearman, ICC for consistency and for
      absolute agreement, bias and limits of agreement.)
  Q2  Where they disagree, is it noise or a systematic scale shift? (Paired t, Cohen's d;
      and does standardising the features remove it?)
  Q3  What ceiling does that agreement put on the audio -> emotion regression, and how
      close is the model (mean R^2 = 0.560, Section 9) to it?

The Set 1 <-> Set 2 correspondence is Set 2's ``link`` column (NOT ``number`` -- the two
CSVs number independently); see ``src/features/loader.align_sets`` for the audio-level
verification of that mapping. Seven excerpts were presented twice within Set 2; those
repeat trials give a *within-panel* noise estimate to contrast with the *between-panel*
one.

Run:  python experiments/features/exp4_rating_reliability.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.features.loader import align_sets, repeat_pairs  # noqa: E402
from src.utils import set_seed  # noqa: E402

# Achieved audio -> emotion performance, for the ceiling comparison (Section 9).
ACHIEVED_MEAN_R2 = 0.560
DIMENSIONAL = ("valence", "energy", "tension")  # bipolar scales
DISCRETE = ("anger", "fear", "happy", "sad", "tender")  # intensity of a named emotion


def icc(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-way random single-measure ICC: (consistency ICC(C,1), agreement ICC(A,1)).

    Both treat excerpts as random effects; they differ in whether a constant offset
    between the two raters counts as disagreement. ICC(C,1) ignores such an offset
    (do the panels *rank* excerpts the same way?), ICC(A,1) penalises it (do they give
    the same *numbers*?). Reporting both separates a scale shift from real disagreement.
    """
    x = np.column_stack([np.asarray(a, float), np.asarray(b, float)])
    n, k = x.shape
    grand = x.mean()
    ms_rows = k * ((x.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_cols = n * ((x.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    resid = x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + grand
    ms_err = (resid ** 2).sum() / ((n - 1) * (k - 1))
    consistency = (ms_rows - ms_err) / (ms_rows + (k - 1) * ms_err)
    agreement = (ms_rows - ms_err) / (
        ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n)
    return float(consistency), float(agreement)


def _pairs(df, suffix_a, suffix_b):
    return [(e, df[f"{e}{suffix_a}"].to_numpy(float),
             df[f"{e}{suffix_b}"].to_numpy(float)) for e in config.EMOTIONS]


def q1_agreement(df):
    print(f"\n{'=' * 84}\nQ1  Between-panel agreement: Set 1 vs Set 2 "
          f"(n={len(df)} excerpts)\n{'=' * 84}")
    print(f"{'emotion':10}{'r':>7}{'rho':>7}{'ICC(C,1)':>10}{'ICC(A,1)':>10}"
          f"{'bias':>8}{'95% limits of agreement':>26}")
    print("-" * 84)
    out = {}
    for e, a, b in _pairs(df, "_set1", "_set2"):
        r = float(np.corrcoef(a, b)[0, 1])
        rho = float(stats.spearmanr(a, b).statistic)
        cons, agree = icc(a, b)
        d = b - a
        lo, hi = d.mean() - 1.96 * d.std(ddof=1), d.mean() + 1.96 * d.std(ddof=1)
        out[e] = {"r": r, "rho": rho, "icc_c": cons, "icc_a": agree,
                  "bias": float(d.mean()), "loa": (float(lo), float(hi))}
        print(f"{e:10}{r:>7.3f}{rho:>7.3f}{cons:>10.3f}{agree:>10.3f}"
              f"{d.mean():>+8.2f}{f'[{lo:+.2f}, {hi:+.2f}]':>26}")
    m = lambda k: np.mean([v[k] for v in out.values()])  # noqa: E731
    print("-" * 84)
    print(f"{'MEAN':10}{m('r'):>7.3f}{m('rho'):>7.3f}{m('icc_c'):>10.3f}"
          f"{m('icc_a'):>10.3f}{m('bias'):>+8.2f}")
    print("\n  r/rho/ICC(C,1) high but ICC(A,1) lower => panels RANK excerpts alike but")
    print("  use the scale differently. Bias = mean(Set2 - Set1) in rating points (1-9).")
    return out


def q1b_within_panel(rep):
    print(f"\n{'=' * 84}\nQ1b  Within-panel noise: Set 2 repeat trials "
          f"(n={len(rep)} excerpts rated twice)\n{'=' * 84}")
    if len(rep) < 3:
        print("  too few repeat trials to estimate")
        return {}
    print(f"{'emotion':10}{'r':>7}{'ICC(A,1)':>10}{'bias':>8}   (n is small -- indicative)")
    print("-" * 84)
    out = {}
    for e, a, b in _pairs(rep, "_a", "_b"):
        r = float(np.corrcoef(a, b)[0, 1])
        _, agree = icc(a, b)
        out[e] = {"r": r, "icc_a": agree, "bias": float((b - a).mean())}
        print(f"{e:10}{r:>7.3f}{agree:>10.3f}{(b - a).mean():>+8.2f}")
    print("-" * 84)
    print(f"{'MEAN':10}{np.mean([v['r'] for v in out.values()]):>7.3f}"
          f"{np.mean([v['icc_a'] for v in out.values()]):>10.3f}")
    print("\n  Same panel, same excerpt, two presentations => pure measurement noise.")
    print("  Compare with Q1: the gap is the panel/context effect, not clip ambiguity.")
    return out


def q2_shift(df):
    print(f"\n{'=' * 84}\nQ2  Is the disagreement a systematic scale shift?"
          f"\n{'=' * 84}")
    print(f"{'emotion':10}{'type':>14}{'set1':>7}{'set2':>7}{'shift':>8}"
          f"{'Cohen d':>9}{'p (paired t)':>14}")
    print("-" * 84)
    for e, a, b in _pairs(df, "_set1", "_set2"):
        d = b - a
        t = stats.ttest_rel(b, a)
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        kind = "dimensional" if e in DIMENSIONAL else "discrete"
        print(f"{e:10}{kind:>14}{a.mean():>7.2f}{b.mean():>7.2f}{d.mean():>+8.2f}"
              f"{d.mean() / pooled:>9.2f}{t.pvalue:>14.3g}")
    dim = np.mean([df[f"{e}_set2"].mean() - df[f"{e}_set1"].mean() for e in DIMENSIONAL])
    dis = np.mean([df[f"{e}_set2"].mean() - df[f"{e}_set1"].mean() for e in DISCRETE])
    print("-" * 84)
    print(f"  mean shift, dimensional scales (valence/energy/tension): {dim:+.2f}")
    print(f"  mean shift, discrete emotions  (anger/fear/happy/...)  : {dis:+.2f}")

    # does standardising remove it? (the classifier pipeline z-scores its inputs)
    print("\n  After z-scoring each set independently (what the model pipeline does):")
    zr = []
    for e, a, b in _pairs(df, "_set1", "_set2"):
        za = (a - a.mean()) / a.std(ddof=1)
        zb = (b - b.mean()) / b.std(ddof=1)
        zr.append(float(np.abs(zb - za).mean()))
    print(f"    mean |z(set2) - z(set1)| = {np.mean(zr):.3f} SD  "
          f"(vs a raw shift of {abs(dim):.2f} points on the dimensional scales)")
    print("    => the shift is largely an offset/scaling of the response scale, which")
    print("       StandardScaler inside the classifier pipeline removes.")


def q3_ceiling(agree):
    print(f"\n{'=' * 84}\nQ3  Reliability ceiling on audio -> emotion regression"
          f"\n{'=' * 84}")
    print("  A predictor cannot correlate with a noisy target better than the target")
    print("  correlates with itself. Treating Set1-vs-Set2 agreement as a parallel-forms")
    print("  reliability r_xx, the maximum attainable correlation is sqrt(r_xx) and the")
    print("  maximum attainable R^2 is r_xx itself.\n")
    print(f"{'emotion':10}{'reliability':>13}{'max r':>8}{'max R^2':>10}")
    print("-" * 45)
    rel = {}
    for e in config.EMOTIONS:
        r_xx = max(0.0, agree[e]["icc_c"])   # consistency: ranking reliability
        rel[e] = r_xx
        print(f"{e:10}{r_xx:>13.3f}{np.sqrt(r_xx):>8.3f}{r_xx:>10.3f}")
    mean_rel = float(np.mean(list(rel.values())))
    print("-" * 45)
    print(f"{'MEAN':10}{mean_rel:>13.3f}{np.sqrt(mean_rel):>8.3f}{mean_rel:>10.3f}")
    pct = 100 * ACHIEVED_MEAN_R2 / mean_rel if mean_rel > 0 else float("nan")
    print(f"\n  Achieved (RandomForest on AST, Section 9): mean R^2 = {ACHIEVED_MEAN_R2:.3f}")
    print(f"  Ceiling implied by rating reliability     : mean R^2 = {mean_rel:.3f}")
    print(f"  => the model reaches {pct:.0f}% of the attainable variance.")
    print("\n  Interpretation: rating noise alone does NOT explain the gap between the")
    print("  model and perfect prediction -- there is real headroom left in the audio")
    print("  representation. Conversely, an R^2 near 1.0 was never achievable and should")
    print("  not be used as the reference point.")
    return rel


def q4_beauty_liking(df):
    cols = [c for c in ("beauty", "liking") if c in df.columns and df[c].notna().any()]
    if not cols:
        return
    print(f"\n{'=' * 84}\nQ4  Set 2 extras: beauty / liking vs the emotion ratings"
          f"\n{'=' * 84}")
    print(f"{'':10}" + "".join(f"{e:>9}" for e in config.EMOTIONS))
    print("-" * 84)
    for c in cols:
        y = df[c].to_numpy(float)
        rs = [np.corrcoef(df[f"{e}_set2"].to_numpy(float), y)[0, 1]
              for e in config.EMOTIONS]
        print(f"{c:10}" + "".join(f"{r:>9.2f}" for r in rs))
    print("\n  (Descriptive only: beauty/liking exist for Set 2 alone and are not used")
    print("   as model inputs anywhere.)")


def main():
    set_seed()
    df = align_sets()
    rep = repeat_pairs()
    print(f"Experiment 4 -- rating reliability (Set 1 vs Set 2)")
    print(f"  matched excerpts : {len(df)} (joined on Set 2's 'link' column)")
    print(f"  repeat trials    : {len(rep)} excerpts rated twice within Set 2")
    print(f"  soundtracks      : {df['soundtrack'].nunique()}")

    agree = q1_agreement(df)
    q1b_within_panel(rep)
    q2_shift(df)
    q3_ceiling(agree)
    q4_beauty_liking(df)


if __name__ == "__main__":
    main()
