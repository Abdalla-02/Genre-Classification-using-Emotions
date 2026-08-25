"""Blockbuster (Ma et al. 2021): VGGish vs MFCC for multi-label genre classification.

Supervisor task: compare VGGish and MFCC features (from the Blockbuster dataset).
Same Binary-Relevance classifiers and metrics as the Eerola genre experiment; 5-fold
KFold (films are independent, so no grouping needed).

Run:  python experiments/cross_dataset/exp_blockbuster.py
"""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation import evaluate  # noqa: E402
from src.features.blockbuster import load_blockbuster  # noqa: E402
from src.models import CLASSIFIERS  # noqa: E402
from src.utils import set_seed  # noqa: E402

CLFS = [c for c in CLASSIFIERS if c != "mlp"]  # mlp can't be class-balanced -> skip


def _block(title, X, Y):
    print(f"\n{'=' * 66}\n{title}   (X={X.shape})\n{'=' * 66}")
    header = f"{'clf':7} | {'ExactMatch':>11} | {'Hamming':>8} | {'MacroF1':>8}"
    print(header + "\n" + "-" * len(header))
    scores = {}
    for clf in CLFS:
        r = evaluate(X, Y, None, None, clf, "kfold")
        scores[clf] = r
        em, ha, f1 = r["exact_match"], r["hamming_loss"], r["macro_f1"]
        print(f"{clf:7} | {em[0]:.3f}±{em[1]:.2f} | {ha[0]:.3f}±{ha[1]:.2f} | "
              f"{f1[0]:.3f}±{f1[1]:.2f}")
    return scores


def main():
    set_seed()
    d = load_blockbuster()
    Y = d["Y"]
    n, k = Y.shape
    print(f"films={n}  genres={k} {d['genres']}")
    print("genre base rates: " + "  ".join(
        f"{g}={r:.2f}" for g, r in zip(d["genres"], Y.mean(axis=0))))

    vgg = _block("VGGish (128-dim learned embedding)", d["X_vggish"], Y)
    mfc = _block("MFCC (78-dim hand-crafted)", d["X_mfcc"], Y)

    print(f"\n{'=' * 66}\nVGGish vs MFCC  (Macro-F1, best classifier per feature set)\n{'=' * 66}")
    for name, sc in (("VGGish", vgg), ("MFCC", mfc)):
        best = max(sc, key=lambda c: sc[c]["macro_f1"][0])
        print(f"  {name:7}: best={best:7} MacroF1={sc[best]['macro_f1'][0]:.3f}")


if __name__ == "__main__":
    main()
