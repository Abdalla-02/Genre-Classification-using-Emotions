"""Box office vs. what the soundtrack says (supervisor request, exploratory).

"Check the correlation between box office revenue and genre classification from the
soundtrack." That can mean three different things, so all three are tested:

  A. Is a film's soundtrack EASIER TO CLASSIFY when the film grossed more?
     Hypothesis: big-budget films score more conventionally, so their music is more
     stereotypically genre-coded. Measured as the per-film mean of the out-of-fold
     per-clip F1 from the audio -> emotion -> genre pipeline, against log10 gross.
  B. Do the EMOTIONS of the soundtrack track gross? Per-film mean of the 8 human
     ratings (and of the model's predictions) against log10 gross.
  C. Does GENRE track gross? Descriptive: gross of films carrying each genre vs. not.

Everything is Spearman (gross is heavy-tailed; log10 is used for plots but Spearman is
rank-based anyway) with a permutation p-value, and the year of release is reported as a
partial-correlation control because gross is not inflation-adjusted and older films are
systematically lower.

This is EXPLORATORY and the write-up must say so: n is about 37 films, the figures mix
worldwide and domestic-only grosses (the ``scope`` column in box_office.csv says which),
nothing is inflation-adjusted, and 37 films across four decades is not a sample from
which to claim an effect. It can rule an effect in as "worth a proper study" or out as
"nothing visible at this size"; it cannot establish one.

Two films are excluded because their IMDb id in the enriched CSV points at the wrong film
(see fetch_box_office.py): ``Blanc`` and ``Pride and Prejudice``.

Run:  python experiments/diagnostics/exp_box_office.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.features import (  # noqa: E402
    add_derived_features,
    assemble_from_cache,
    build_emotion_features,
    genre_matrix,
    load_set1,
)
from src.models import build_classifier, select_logreg_C  # noqa: E402
from src.utils import set_seed  # noqa: E402

RESULTS = config.RESULTS_DIR / "box_office.json"
BOX = config.PROCESSED_DIR / "box_office.csv"
MISATTRIBUTED = {"tt1179025", "tt0032943"}   # Blanc -> Adele Blanc-Sec; P&P -> 1940 film


def spearman_perm(x, y, n_perm: int = 5000, seed: int = config.SEED):
    """Spearman rho with a permutation p-value (exact enough at this n)."""
    rho = stats.spearmanr(x, y).correlation
    rng = np.random.default_rng(seed)
    null = np.array([stats.spearmanr(x, rng.permutation(y)).correlation
                     for _ in range(n_perm)])
    p = float((np.abs(null) >= abs(rho)).mean())
    return float(rho), p


def partial_spearman(x, y, z):
    """Spearman correlation of x and y after removing the rank-linear effect of z."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    def resid(a, b):
        A = np.column_stack([np.ones_like(b), b])
        return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    return float(stats.pearsonr(resid(rx, rz), resid(ry, rz))[0])


def per_film_pipeline_f1(df, Y, groups) -> pd.Series:
    """Out-of-fold per-clip F1 from audio -> RF emotion -> LogReg genre, averaged per film."""
    X = assemble_from_cache(df, "set1", cache_dir=config.VGGISH_EMBEDDINGS_DIR)
    E = df[config.EMOTIONS].to_numpy(float)
    clip_f1 = np.zeros(len(df))
    for tr, te in GroupKFold(5).split(X, Y, groups):
        reg = RandomForestRegressor(n_estimators=300, random_state=config.SEED,
                                    n_jobs=-1).fit(X[tr], E[tr])
        Ztr, Zte = build_emotion_features(reg.predict(X[tr])), build_emotion_features(reg.predict(X[te]))
        C = select_logreg_C(Ztr, Y[tr], groups[tr])
        pred = np.asarray(build_classifier("logreg", C=C).fit(Ztr, Y[tr]).predict(Zte))
        for k, i in enumerate(te):   # per-clip F1 over the label vector
            clip_f1[i] = f1_score(Y[i], pred[k], zero_division=0)
    return pd.Series(clip_f1, index=df.index).groupby(df["soundtrack"]).mean()


def main() -> None:
    set_seed()
    df = add_derived_features(load_set1())
    Y = genre_matrix(df)
    groups = df["soundtrack"].to_numpy()

    box = pd.read_csv(BOX)
    box = box[(box["currency"] == "USD") & box["gross"].notna()
              & ~box["imdb_id"].isin(MISATTRIBUTED)]
    box = box.set_index("soundtrack")
    box["log_gross"] = np.log10(box["gross"])

    print("=" * 76)
    print("BOX OFFICE vs THE SOUNDTRACK  (exploratory)")
    print("=" * 76)
    print(f"films with usable gross: {len(box)}  "
          f"({(box['scope'].str.startswith('worldwide')).sum()} worldwide, "
          f"{(box['scope'] == 'domestic only').sum()} domestic-only); "
          f"years {int(box['year'].min())}-{int(box['year'].max())}")

    # ---- per-film quantities ------------------------------------------------
    film = pd.DataFrame({"pipeline_f1": per_film_pipeline_f1(df, Y, groups)})
    for e in config.EMOTIONS:
        film[f"gt_{e}"] = df.groupby("soundtrack")[e].mean()
    for j, g in enumerate(config.PRIMARY_GENRES):
        film[f"genre_{g}"] = pd.Series(Y[:, j], index=df.index).groupby(df["soundtrack"]).max()
    film["n_clips"] = df.groupby("soundtrack").size()
    film = film.join(box[["log_gross", "gross", "year", "scope"]], how="inner")
    n = len(film)
    print(f"films in the analysis (have both clips and a gross): {n}\n")

    out = {"n_films": int(n), "years": [int(film['year'].min()), int(film['year'].max())]}

    # ---- A. classification quality vs gross ---------------------------------
    rho, p = spearman_perm(film["log_gross"], film["pipeline_f1"])
    prho = partial_spearman(film["log_gross"], film["pipeline_f1"], film["year"])
    print("A. Is a higher-grossing film's soundtrack easier to classify?")
    print(f"   Spearman rho(log gross, per-film pipeline F1) = {rho:+.3f}  p = {p:.3f}"
          f"   | controlling for year: {prho:+.3f}")
    out["A_f1_vs_gross"] = {"rho": rho, "p_perm": p, "rho_partial_year": prho}

    # ---- B. emotions vs gross ------------------------------------------------
    print("\nB. Do the soundtrack's emotions track gross?  (human ratings, per-film mean)")
    print(f"   {'emotion':10}{'rho':>8}{'p':>8}{'| year-ctrl':>12}")
    out["B_emotion_vs_gross"] = {}
    for e in config.EMOTIONS:
        rho, p = spearman_perm(film["log_gross"], film[f"gt_{e}"])
        prho = partial_spearman(film["log_gross"], film[f"gt_{e}"], film["year"])
        star = " *" if p < 0.05 else ""
        print(f"   {e:10}{rho:>+8.3f}{p:>8.3f}{prho:>+12.3f}{star}")
        out["B_emotion_vs_gross"][e] = {"rho": rho, "p_perm": p, "rho_partial_year": prho}
    print("   (8 tests: a single p<0.05 is what chance produces about a third of the time)")

    # ---- C. genre vs gross ---------------------------------------------------
    print("\nC. Gross by genre (films carrying the genre vs not; Mann-Whitney)")
    print(f"   {'genre':12}{'n_with':>7}{'median$ with':>14}{'median$ without':>17}{'p':>8}")
    out["C_genre_vs_gross"] = {}
    for g in config.PRIMARY_GENRES:
        m = film[f"genre_{g}"] == 1
        if m.sum() < 3 or (~m).sum() < 3:
            print(f"   {g:12}{int(m.sum()):>7}   (too few films to test)")
            continue
        a, b = film.loc[m, "gross"], film.loc[~m, "gross"]
        p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
        print(f"   {g:12}{int(m.sum()):>7}{a.median()/1e6:>12.0f}M{b.median()/1e6:>15.0f}M"
              f"{p:>8.3f}{' *' if p < 0.05 else ''}")
        out["C_genre_vs_gross"][g] = {"n_with": int(m.sum()), "median_with": float(a.median()),
                                      "median_without": float(b.median()), "p_mw": float(p)}

    # ---- D. is the anger-gross link just Action? ------------------------------
    # Action films are both angrier and higher-grossing. If the anger correlation is
    # mediated by genre it should vanish once Action membership is controlled for.
    print("\nD. Mediation check: does the anger-gross link survive controlling for Action?")
    out["D_mediation"] = {}
    for e in ("anger", "tension"):
        raw = stats.spearmanr(film["log_gross"], film[f"gt_{e}"]).correlation
        ctrl = partial_spearman(film["log_gross"], film[f"gt_{e}"], film["genre_Action"])
        non_action = film[film["genre_Action"] == 0]
        rho_na, p_na = spearman_perm(non_action["log_gross"], non_action[f"gt_{e}"])
        print(f"   {e:8} raw {raw:+.3f} | controlling for Action {ctrl:+.3f} | "
              f"within non-Action films only (n={len(non_action)}) {rho_na:+.3f}, p={p_na:.3f}")
        out["D_mediation"][e] = {"rho_raw": float(raw), "rho_partial_action": ctrl,
                                 "rho_non_action_only": rho_na, "p_non_action_only": p_na,
                                 "n_non_action": int(len(non_action))}

    # ---- the confound everyone should see ------------------------------------
    rho, p = spearman_perm(film["log_gross"], film["year"])
    print(f"\nconfound: rho(log gross, year) = {rho:+.3f} (p={p:.3f}) -- gross is not "
          f"inflation-adjusted and\n          older films are systematically lower, "
          f"which is why a year-controlled column is shown.")
    out["confound_year"] = {"rho": rho, "p_perm": p}

    film.reset_index().to_csv(config.PROCESSED_DIR / "box_office_per_film.csv", index=False)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {RESULTS} and box_office_per_film.csv")


if __name__ == "__main__":
    main()
