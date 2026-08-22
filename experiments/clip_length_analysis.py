"""Open Question #2: does clip length correlate with genre?

AST truncates/pads every clip to ~10.24 s, so if duration differs systematically by
genre, that padding/truncation could introduce a spurious genre-length confound. This
script tests for such an association on the cleaned Set 1.

  * one-way ANOVA of duration across the descriptive primary genre;
  * per-genre (multi-label) Welch t-tests: clips WITH vs WITHOUT each genre,
    with a Bonferroni-corrected significance threshold;
  * a boxplot saved to results/clip_length_by_genre.png.

Run:  python experiments/clip_length_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import config  # noqa: E402
from src.features import durations_from_audio, load_set1  # noqa: E402


def main() -> None:
    df = load_set1()
    dur = durations_from_audio(df)
    df = df.merge(dur, on="number")

    d = df["duration_sec"]
    print("=== Clip duration (Set 1, n=%d) ===" % len(df))
    print(f"  overall: mean={d.mean():.2f}s  std={d.std():.2f}s  "
          f"min={d.min():.2f}s  max={d.max():.2f}s")
    print(f"  AST target window ~10.24s  ->  {(d > 10.24).mean() * 100:.0f}% of clips "
          f"are truncated")

    # 1) one-way ANOVA across the (alphabetically-)first genre. NB: IMDb order is
    # ~alphabetical, not relevance, so this partition is a coarse descriptive view;
    # the per-genre multi-label t-tests below are the order-independent, robust check.
    groups = [g["duration_sec"].values for _, g in df.groupby("first_genre")]
    F, p = stats.f_oneway(*groups)
    print("\n=== One-way ANOVA: duration ~ first_genre (alphabetical, descriptive) ===")
    print(f"  F={F:.3f}  p={p:.4f}  -> "
          f"{'SIGNIFICANT' if p < 0.05 else 'no significant'} genre-length association")
    print("  per first genre:")
    summ = (df.groupby("first_genre")["duration_sec"]
              .agg(["count", "mean", "std"]).round(2))
    print(summ.to_string())

    # 2) multi-label per-genre Welch t-tests (with vs without), Bonferroni-corrected
    print("\n=== Per-genre Welch t-test (with vs without), multi-label ===")
    alpha = 0.05 / len(config.PRIMARY_GENRES)
    print(f"  Bonferroni threshold alpha={alpha:.4f}")
    for g in config.PRIMARY_GENRES:
        with_g = df.loc[df[g] == 1, "duration_sec"]
        without_g = df.loc[df[g] == 0, "duration_sec"]
        t, pg = stats.ttest_ind(with_g, without_g, equal_var=False)
        flag = "  <-- significant" if pg < alpha else ""
        print(f"    {g:12} n={len(with_g):3d}  mean_with={with_g.mean():5.2f}  "
              f"mean_without={without_g.mean():5.2f}  p={pg:.4f}{flag}")

    # 3) boxplot
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    df.boxplot(column="duration_sec", by="first_genre", ax=ax, grid=False, rot=30)
    ax.axhline(10.24, color="red", ls="--", lw=1, label="AST window ~10.24s")
    ax.set_title("Clip duration by primary genre (Set 1)")
    ax.set_xlabel("primary genre")
    ax.set_ylabel("duration (s)")
    ax.legend()
    plt.suptitle("")
    out = config.RESULTS_DIR / "clip_length_by_genre.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nboxplot saved -> {out}")


if __name__ == "__main__":
    main()
