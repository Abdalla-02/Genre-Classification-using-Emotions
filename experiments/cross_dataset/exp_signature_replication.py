"""Do the Eerola emotion-genre signatures replicate on Blockbuster? (RQ3, external validity)

Section 7b established a per-genre emotional signature on Eerola -- horror is marked by
fear (Cohen's d +0.75), comedy by its absence, and so on -- using HUMAN ratings on 346
clips. That is the thesis's interpretability claim, and so far it rests entirely on the
corpus it was derived from. A signature that only exists in its own training data is a
description of that data, not a finding about film music.

This script tests it on an independent corpus with no emotion annotations at all. The
Eerola-trained VGGish->emotion regressor predicts emotions for every Blockbuster cue; the
same Cohen's d is then computed there, and the two sets of effect sizes are compared. A
replication requires the signs to agree and the pattern to correlate -- nothing about the
target corpus enters the emotion model, so any agreement is genuine transfer.

Two granularities are reported because they answer different questions:
  * per FILM (n=110)  -- the inferentially honest unit; films are independent.
  * per CUE  (n=4664) -- 42x more observations, so a far more stable estimate of the
    effect, but cues within a film are not independent and the n must not be read as
    statistical power.

All six shared genres can be compared. Note that sci-fi and romance are absent from the
eight modelled Eerola genres but ARE present in its raw IMDb annotations (19 and 51 clips)
-- which is exactly why the shared label space of section 15.1 exists -- so the Eerola
reference signatures are computed over all six. The two smallest, sci-fi (n=19) and
Blockbuster horror (n=11 films), carry correspondingly noisy estimates.

Run:  python experiments/cross_dataset/exp_signature_replication.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.features import assemble_from_cache, load_set1, load_set1_shared  # noqa: E402
from src.features.blockbuster import aggregate_cues, load_blockbuster_cues  # noqa: E402
from src.utils import set_seed  # noqa: E402

RESULTS = config.RESULTS_DIR / "signature_replication.json"

# All six genres of the shared label space are comparable: the Eerola side is computed
# from load_set1_shared(), which carries sci-fi and romance from the raw IMDb annotations.
COMPARABLE = config.SHARED_GENRES


def cohens_d(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Effect size per column between the rows with and without the label."""
    a, b = values[mask], values[~mask]
    if len(a) < 2 or len(b) < 2:
        return np.full(values.shape[1], np.nan)
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(0, ddof=1) + (nb - 1) * b.var(0, ddof=1))
                     / (na + nb - 2))
    pooled = np.where(pooled == 0, np.nan, pooled)
    return (a.mean(0) - b.mean(0)) / pooled


def table(title, d_by_genre, genres):
    print(f"\n{title}")
    print(f"{'genre':10}" + "".join(f"{e[:7]:>8}" for e in config.EMOTIONS))
    print("-" * (10 + 8 * len(config.EMOTIONS)))
    for g in genres:
        print(f"{g:10}" + "".join(f"{v:>+8.2f}" for v in d_by_genre[g]))


def main() -> None:
    set_seed()

    # ---- Eerola reference signatures, from HUMAN ratings ------------------ #
    df = load_set1_shared()
    E_eerola = df[config.EMOTIONS].to_numpy(float)
    d_eerola = {g: cohens_d(E_eerola, df[g].to_numpy(bool)) for g in config.SHARED_GENRES}

    # ---- the emotion model: trained on Eerola only ------------------------ #
    full = load_set1(clean=False)
    Xe = assemble_from_cache(full, "set1", cache_dir=config.VGGISH_EMBEDDINGS_DIR)
    reg = RandomForestRegressor(n_estimators=300, random_state=config.SEED,
                                n_jobs=-1).fit(Xe, full[config.EMOTIONS].to_numpy(float))

    # ---- Blockbuster, predicted emotions at both granularities ------------ #
    cue = load_blockbuster_cues()
    Y, films = cue["Y"], cue["films"]
    idx = cue["cue_film_vggish"]
    E_cue = reg.predict(cue["X_vggish"])
    E_film = aggregate_cues(E_cue, idx, len(films))

    d_cue = {g: cohens_d(E_cue, Y[idx][:, j].astype(bool))
             for j, g in enumerate(cue["genres"])}
    d_film = {g: cohens_d(E_film, Y[:, j].astype(bool))
              for j, g in enumerate(cue["genres"])}

    print("=" * 78)
    print("EMOTION-GENRE SIGNATURES -- does the Eerola pattern reappear on Blockbuster?")
    print("=" * 78)
    print(f"Eerola     : {len(df)} clips, HUMAN ratings")
    print(f"Blockbuster: {len(films)} films / {len(E_cue)} cues, "
          f"emotions PREDICTED by an Eerola-trained model (no target annotation exists)")

    table("Cohen's d -- EEROLA (human ratings, per clip)", d_eerola, config.SHARED_GENRES)
    table("Cohen's d -- BLOCKBUSTER (predicted, per film, n=110)", d_film, cue["genres"])
    table("Cohen's d -- BLOCKBUSTER (predicted, per cue, n=4664)", d_cue, cue["genres"])

    # ---- agreement on the four comparable genres -------------------------- #
    print("\n" + "=" * 78)
    print("REPLICATION on the genres present in both label spaces")
    print("=" * 78)
    print(f"{'genre':10}{'r(d)':>8}{'sign agree':>12}{'Eerola top':>14}"
          f"{'Blockbuster top':>18}")
    print("-" * 64)
    out = {}
    for g in COMPARABLE:
        a, b = d_eerola[g], d_film[g]
        ok = np.isfinite(a) & np.isfinite(b)
        r = float(np.corrcoef(a[ok], b[ok])[0, 1])
        agree = float((np.sign(a[ok]) == np.sign(b[ok])).mean())
        ta = config.EMOTIONS[int(np.nanargmax(np.abs(a)))]
        tb = config.EMOTIONS[int(np.nanargmax(np.abs(b)))]
        sa = "+" if a[config.EMOTIONS.index(ta)] > 0 else "-"
        sb = "+" if b[config.EMOTIONS.index(tb)] > 0 else "-"
        out[g] = {"r": r, "sign_agreement": agree,
                  "eerola_top": f"{sa}{ta}", "blockbuster_top": f"{sb}{tb}",
                  "d_eerola": a.tolist(), "d_blockbuster_film": b.tolist(),
                  "d_blockbuster_cue": d_cue[g].tolist()}
        print(f"{g:10}{r:>+8.2f}{agree:>11.0%}{sa + ta:>14}{sb + tb:>18}")

    n_cells = len(COMPARABLE) * len(config.EMOTIONS)
    allr = float(np.corrcoef(
        np.concatenate([d_eerola[g] for g in COMPARABLE]),
        np.concatenate([d_film[g] for g in COMPARABLE]))[0, 1])
    allagree = float(np.mean([np.sign(d_eerola[g]) == np.sign(d_film[g])
                              for g in COMPARABLE]))
    print("-" * 64)
    print(f"{'POOLED':10}{allr:>+8.2f}{allagree:>11.0%}"
          f"   ({n_cells} genre x emotion cells)")

    print("\nNote: Blockbuster's film-level d values are systematically LARGER than "
          "Eerola's,\nbecause averaging ~39 cues shrinks within-group variance and "
          "inflates the effect size.\nThe cue-level table is the one to compare against "
          "Eerola's clip-level d -- same unit,\nsame order of magnitude. What replicates "
          "is the PATTERN, not the absolute size.")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({
        "emotions": config.EMOTIONS,
        "eerola_d": {g: d_eerola[g].tolist() for g in config.SHARED_GENRES},
        "blockbuster_d_film": {g: d_film[g].tolist() for g in cue["genres"]},
        "blockbuster_d_cue": {g: d_cue[g].tolist() for g in cue["genres"]},
        "replication": out,
        "pooled_r": allr, "pooled_sign_agreement": allagree,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
