"""Clip-length handling: does using the FULL clip (windowed pooling) beat truncating
to the first 10.24 s? Extracts all 10.24 s windows (50% overlap) per clip once, then
compares pooling strategies on two downstream tasks.

Configs:
  first  -- window 0 only  == the current baseline (first 10.24 s, ~60% of mean clip)
  center -- middle window   (literature "middle of track")
  mean   -- average all windows == the full clip
  max    -- per-dim max over windows

Downstream: Experiment 5 (TARGET, 12-class, GroupKFold) and the direct genre baseline
(AST -> 8-genre multi-label, GroupKFold Macro-F1).

Run:  python notebooks/exp_clip_length.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.evaluation import evaluate  # noqa: E402
from src.features import (  # noqa: E402
    AstEmbedder,
    extract_window_embeddings,
    genre_matrix,
    load_set1,
    pool_windows,
)
from src.utils import set_seed  # noqa: E402

MODES = ["first", "center", "mean", "max"]


def exp5_score(X, y, groups):
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, random_state=config.SEED))
    pred = cross_val_predict(pipe, X, y, cv=GroupKFold(5).split(X, y, groups))
    return accuracy_score(y, pred), f1_score(y, pred, average="macro", zero_division=0)


def main():
    set_seed()
    df_full = load_set1(clean=False)   # 360 clips (TARGET)
    df_gen = load_set1()               # 346 cleaned clips (genre)

    print("Extracting AST window embeddings (resumable; first run is slow) ...")
    embedder = AstEmbedder()
    nwin = extract_window_embeddings(df_full, "set1", embedder=embedder)
    print(f"windows/clip: min={nwin.min()} max={nwin.max()} mean={nwin.mean():.2f} "
          f"(1 window = first 10.24s; 'mean' pools all -> uses the full clip)")

    y_target = df_full["TARGET"].to_numpy()
    g_full = df_full["soundtrack"].to_numpy()
    Yg = genre_matrix(df_gen)
    g_gen = df_gen["soundtrack"].to_numpy()
    strat = df_gen["first_genre"].to_numpy()

    print(f"\n{'mode':7} | {'Exp5 acc':>8} {'Exp5 F1':>8} | {'genre MacroF1 (group)':>22}")
    print("-" * 54)
    results = {}
    for mode in MODES:
        Xf = pool_windows(df_full, "set1", mode=mode)
        Xg = pool_windows(df_gen, "set1", mode=mode)
        acc, f1 = exp5_score(Xf, y_target, g_full)
        gmf1 = evaluate(Xg, Yg, g_gen, strat, "logreg", "group")["macro_f1"][0]
        results[mode] = (acc, f1, gmf1)
        tag = "  <- current baseline" if mode == "first" else ""
        print(f"{mode:7} | {acc:8.3f} {f1:8.3f} | {gmf1:22.3f}{tag}")

    best_t = max(results, key=lambda m: results[m][1])
    best_g = max(results, key=lambda m: results[m][2])
    print(f"\nbest on Exp5 (F1): {best_t}   |   best on genre (MacroF1): {best_g}")


if __name__ == "__main__":
    main()
