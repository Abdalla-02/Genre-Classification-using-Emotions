"""Waveform-domain vs spectrogram-domain audio representations (supervisor request #3).

Every representation used so far starts from a time-frequency image: AST and CLAP from a
log-mel spectrogram, VGGish from log-mel patches, the MIR baseline from STFT descriptors.
That leaves a confound in the thesis's central claim. "Emotion features are at least as
good as learned audio embeddings" could always have been answered with "you only tried
*spectrogram* embeddings" -- perhaps the mel front-end, not the representation, is what
limits the direct-audio baseline.

wav2vec 2.0 removes that confound: its feature encoder is a stack of 1-D convolutions
over the raw 16 kHz sample sequence, with no spectral transform anywhere. This script
scores it against the four spectrogram-based representations on both stages of the
pipeline:

  Stage 1  audio -> 8 emotions            RandomForest, GroupKFold, mean R^2
           (this is the "feature-extraction accuracy" metric -- see
            docs/latex/evaluation_protocol.tex)
  Stage 2  audio -> genre                 5-genre subset, repeated GroupKFold with
           and                            nested-CV-tuned C, Macro-F1 + corrected
           audio -> emotion -> genre      paired tests against the emotion features

Honest caveat carried into the write-up: wav2vec 2.0 was pretrained on read speech
(LibriSpeech), while AST and VGGish were pretrained on AudioSet, which contains music.
The comparison therefore varies the input domain AND the pretraining corpus. A wav2vec 2.0
result that is merely competitive is consequently more interesting than a losing one -- a
loss cannot be attributed to the waveform front-end alone.

Run:  python experiments/features/exp_waveform_vs_spectrogram.py
      python experiments/features/exp_waveform_vs_spectrogram.py --stage1-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.evaluation.repeated import (  # noqa: E402
    RepeatedGroupKFold,
    corrected_paired_t,
    diff_ci,
    nb_p_limit,
    repeat_ci,
    win_rate,
)
from src.features import (  # noqa: E402
    add_derived_features,
    assemble_from_cache,
    build_emotion_features,
    genre_matrix,
    load_set1,
)
from src.models import build_classifier, select_logreg_C  # noqa: E402
from src.utils import set_seed  # noqa: E402

RESULTS = config.RESULTS_DIR / "waveform_vs_spectrogram.json"

# name -> (cache dir, input domain, pretraining corpus)
REPRESENTATIONS = {
    "wav2vec2-768":  (config.W2V_EMBEDDINGS_DIR, "RAW WAVEFORM", "speech (LibriSpeech)"),
    "AST-768":       (config.EMBEDDINGS_DIR, "log-mel spectrogram", "AudioSet"),
    "VGGish-128":    (config.VGGISH_EMBEDDINGS_DIR, "log-mel spectrogram", "AudioSet"),
    "CLAP-512":      (config.CLAP_EMBEDDINGS_DIR, "log-mel spectrogram", "audio-text pairs"),
    "MIR-103":       (config.MIR_EMBEDDINGS_DIR, "STFT descriptors", "none (hand-crafted)"),
}

# MusiCNN needs a separate TensorFlow environment (see extract_musicnn.py), so it is
# included only when its cache is present. Everything else runs without TensorFlow.
if (config.MUSICNN_EMBEDDINGS_DIR / "set1").is_dir():
    REPRESENTATIONS["MusiCNN-200"] = (
        config.MUSICNN_EMBEDDINGS_DIR, "log-mel spectrogram", "music tagging (MSD)")
EMO = "emotion(11) [ground truth]"


def stage1(df, groups) -> dict:
    """Emotion regression: the metric that ranks feature extractors."""
    Y = df[config.EMOTIONS].to_numpy(float)
    cv = GroupKFold(5)
    print("=" * 84)
    print("STAGE 1 -- audio -> 8 emotions   (RandomForest, GroupKFold by film, n=360)")
    print("  mean R^2 = the 'feature-extraction accuracy': the share of human rating")
    print("  variance each representation recovers. Reliability ceiling 0.897 (Exp 4).")
    print("=" * 84)
    print(f"{'representation':16}{'input domain':22}{'dim':>5}{'meanR2':>8}{'RMSE':>7}"
          f"{'% of ceiling':>13}")
    print("-" * 84)
    out = {}
    for name, (cache, domain, pretrain) in REPRESENTATIONS.items():
        X = assemble_from_cache(df, "set1", cache_dir=cache)
        model = RandomForestRegressor(n_estimators=300, random_state=config.SEED, n_jobs=-1)
        pred = cross_val_predict(model, X, Y, cv=cv.split(X, Y, groups))
        r2 = [r2_score(Y[:, j], pred[:, j]) for j in range(len(config.EMOTIONS))]
        rmse = [mean_squared_error(Y[:, j], pred[:, j]) ** 0.5
                for j in range(len(config.EMOTIONS))]
        out[name] = {"dim": int(X.shape[1]), "input_domain": domain,
                     "pretraining": pretrain, "mean_r2": float(np.mean(r2)),
                     "mean_rmse": float(np.mean(rmse)),
                     "per_emotion_r2": dict(zip(config.EMOTIONS, map(float, r2)))}
        print(f"{name:16}{domain:22}{X.shape[1]:>5}{np.mean(r2):>8.3f}"
              f"{np.mean(rmse):>7.2f}{np.mean(r2) / 0.897:>12.0%}")

    print(f"\nper-emotion R^2:")
    print(f"{'representation':16}" + "".join(f"{e[:7]:>8}" for e in config.EMOTIONS))
    for name, r in out.items():
        print(f"{name:16}" + "".join(f"{r['per_emotion_r2'][e]:>8.3f}"
                                     for e in config.EMOTIONS))
    return out


def stage2(df, groups, repeats: int, splits: int) -> dict:
    """Genre classification on the 5-genre subset, nested-CV-tuned, repeated GroupKFold."""
    Y8 = genre_matrix(df)
    idx = [config.PRIMARY_GENRES.index(x) for x in config.GENRE_SUBSET]
    keep = Y8[:, idx].sum(1) >= 1
    dfk = df[keep].reset_index(drop=True)
    Y = Y8[keep][:, idx]
    g = dfk["soundtrack"].to_numpy()
    Emo = dfk[config.EMOTIONS].to_numpy(float)

    feats = {n: assemble_from_cache(dfk, "set1", cache_dir=c)
             for n, (c, _, _) in REPRESENTATIONS.items()}
    feats[EMO] = dfk[config.FEATURE_COLS].to_numpy(float)

    folds = list(RepeatedGroupKFold(splits, repeats).split(np.zeros(len(dfk)), None, g))
    n_test = float(np.mean([len(te) for _, te in folds]))
    n_train = len(dfk) - n_test

    print("\n" + "=" * 84)
    print(f"STAGE 2 -- audio -> genre   ({repeats}x{splits} repeated GroupKFold, "
          f"nested-CV-tuned C)")
    print(f"  5-genre subset {config.GENRE_SUBSET}, n={len(dfk)} clips / "
          f"{dfk['soundtrack'].nunique()} films")
    print("=" * 84)

    scores = {n: np.zeros(len(folds)) for n in feats}
    scores["wav2vec2 -> predicted emotion(11)"] = np.zeros(len(folds))
    Cs: dict[str, list] = {n: [] for n in scores}

    for i, (tr, te) in enumerate(folds):
        for name, X in feats.items():
            C = select_logreg_C(X[tr], Y[tr], g[tr])
            Cs[name].append(C)
            pred = build_classifier("logreg", C=C).fit(X[tr], Y[tr]).predict(X[te])
            scores[name][i] = f1_score(Y[te], pred, average="macro", zero_division=0)
        # the emotion bottleneck driven by the WAVEFORM front-end
        Xw = feats["wav2vec2-768"]
        reg = RandomForestRegressor(n_estimators=300, random_state=config.SEED,
                                    n_jobs=-1).fit(Xw[tr], Emo[tr])
        Ztr = build_emotion_features(reg.predict(Xw[tr]))
        Zte = build_emotion_features(reg.predict(Xw[te]))
        C = select_logreg_C(Ztr, Y[tr], g[tr])
        Cs["wav2vec2 -> predicted emotion(11)"].append(C)
        pred = build_classifier("logreg", C=C).fit(Ztr, Y[tr]).predict(Zte)
        scores["wav2vec2 -> predicted emotion(11)"][i] = f1_score(
            Y[te], pred, average="macro", zero_division=0)
        print(f"  fold {i + 1}/{len(folds)}", end="\r", flush=True)
    print(" " * 30, end="\r")

    order = sorted(scores, key=lambda k: -scores[k].mean())
    print(f"{'arm':36}{'MacroF1':>9}  {'95% CI':>20}{'C':>7}")
    print("-" * 74)
    rows = {}
    for name in order:
        ci = repeat_ci(scores[name], splits)
        c = max(set(Cs[name]), key=Cs[name].count)
        rows[name] = {"mean": ci.mean, "ci_lo": ci.lo, "ci_hi": ci.hi, "C_mode": c}
        print(f"{name:36}{ci.mean:>9.3f}  [{ci.lo:>7.3f}, {ci.hi:>7.3f}]{c:>7g}")

    pairs = [(EMO, "wav2vec2-768"), ("AST-768", "wav2vec2-768"),
             ("VGGish-128", "wav2vec2-768"), ("CLAP-512", "wav2vec2-768"),
             ("wav2vec2 -> predicted emotion(11)", "wav2vec2-768"),
             (EMO, "wav2vec2 -> predicted emotion(11)")]
    print(f"\n{'comparison':60}{'diff':>7}{'p':>8}{'p_lim':>8}{'win':>6}")
    print("-" * 89)
    cmp = []
    for a, b in pairs:
        d = diff_ci(scores[a], scores[b], splits)
        _, p = corrected_paired_t(scores[a], scores[b], n_train, n_test)
        plim = nb_p_limit(scores[a], scores[b], n_train, n_test)
        w = win_rate(scores[a], scores[b])
        print(f"{a + ' vs ' + b:60}{d.mean:>+7.3f}{p:>8.3f}"
              f"{'*' if p < 0.05 else ' '}{plim:>7.3f}{w:>6.0%}")
        cmp.append({"a": a, "b": b, "diff": d.mean, "ci_lo": d.lo, "ci_hi": d.hi,
                    "p_corrected": p, "p_limit": plim, "win_rate": w})
    print("\n  p = Nadeau-Bengio corrected resampled paired t-test; * = p<0.05.")
    return {"arms": rows, "comparisons": cmp,
            "per_fold": {k: v.tolist() for k, v in scores.items()},
            "n_clips": int(len(dfk)), "n_films": int(dfk["soundtrack"].nunique())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--stage1-only", action="store_true")
    args = ap.parse_args()
    set_seed()

    df_all = load_set1(clean=False)          # all 360 clips carry emotion ratings
    out = {"stage1": stage1(df_all, df_all["soundtrack"].to_numpy())}

    if not args.stage1_only:
        df = add_derived_features(load_set1())
        out["stage2"] = stage2(df, df["soundtrack"].to_numpy(), args.repeats, args.splits)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
