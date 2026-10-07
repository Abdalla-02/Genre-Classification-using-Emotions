"""Draw every results figure for the thesis from the saved results files.

Supervisor, fourth meeting: "show the results as a graph or diagram, not just a table".
Every figure here is drawn from ``results/*.json`` (and, for the box-office scatter, the
per-film CSV an experiment writes next to its JSON), never from a number typed in by
hand, so a figure can never disagree with the table beside it. Re-run this script after
re-running an experiment.

Output: ``figures/<name>.pdf`` (vector, for LaTeX) and ``figures/<name>.png`` (preview).
The LaTeX that places them is ``docs/latex/results_figures.tex``.

Run:  python experiments/make_figures.py            # all figures whose inputs exist
      python experiments/make_figures.py zero_shot   # just one
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
OUT = REPO / "figures"

# One colour per route, used in every figure (Okabe-Ito, colour-blind safe).
COL = {"emotion": "#D55E00", "direct": "#0072B2", "pca": "#7F7F7F",
       "ceiling": "#009E73", "other": "#56B4E9", "floor": "#000000"}
WIDTH = 5.9          # inches, the text width of the thesis template

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def load(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  figures/{name}.pdf")


def floor_line(ax, x, label, style="--"):
    """A chance floor as a vertical line; call ``floor_legend`` once afterwards."""
    ax.axvline(x, color=COL["floor"], lw=0.8, ls=style, zorder=0, label=f"{label} ({x:.3f})")


def floor_legend(ax, loc="lower right"):
    ax.legend(loc=loc, frameon=True, framealpha=0.9, edgecolor="none", fontsize=7)


def route_colour(name: str) -> str:
    n = name.lower()
    if "ground-truth" in n or "ratings" in n or "ceiling" in n:
        return COL["ceiling"]
    if "emotion" in n:
        return COL["emotion"]
    if "pca" in n:
        return COL["pca"]
    return COL["direct"]


# --------------------------------------------------------------------------- #
def fig_indomain():
    """Eerola 5 genres: every representation, film-level macro-F1 with film-bootstrap CI.

    The clip-level score of the same predictions is drawn as a hollow marker beside it."""
    d = load("film_level")
    if not d:
        return
    arms = d["subset5"]["arms"]
    label = {
        "ground-truth emotion(11) [ceiling]": "emotion from human ratings (ceiling)",
        "VGGish -> predicted emotion(11), OOF-trained": "emotion route (VGGish)",
        "wav2vec2 -> predicted emotion(11), OOF-trained": "emotion route (wav2vec 2.0)",
        "VGGish-128 (direct)": "direct: VGGish",
        "AST-768": "direct: AST",
        "CLAP-512": "direct: CLAP",
        "MusiCNN-200": "direct: MusiCNN",
        "MIR-103": "direct: hand-crafted MIR",
        "wav2vec2-768": "direct: wav2vec 2.0",
        "PCA-8(VGGish) [control]": "PCA-8 control (VGGish)",
    }
    rows = sorted(((label[k], arms[k]) for k in label if k in arms),
                  key=lambda kv: kv[1]["mean"])
    fig, ax = plt.subplots(figsize=(WIDTH, 3.0))
    for i, (name, v) in enumerate(rows):
        c = route_colour(name)
        ax.errorbar(v["mean"], i, xerr=[[v["mean"] - v["ci_lo"]], [v["ci_hi"] - v["mean"]]],
                    fmt="o", color=c, ms=5, capsize=2, lw=1.2)
        ax.scatter(v["clip_level_mean"], i, s=18, facecolors="none", edgecolors=c, lw=0.9,
                   zorder=3, label="clip level, same predictions" if i == 0 else None)
        ax.text(v["ci_hi"] + 0.004, i, f"{v['mean']:.3f}", va="center", fontsize=7)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    floor_line(ax, arms["dummy"]["mean"], "most-frequent baseline")
    floor_line(ax, d["subset5"]["random_guess_film"], "base-rate random guess", ":")
    floor_legend(ax, "lower right")
    ax.set_xlabel("film-level macro-F1, 41 films (bars: 95% film bootstrap)")
    hi = max(v["ci_hi"] for _, v in rows)
    ax.set_xlim(0.14, hi + 0.05)
    save(fig, "indomain_eerola5")


def fig_repeat_spread():
    """Stability: the 10 film-level repeat scores per route, and the means under 5 seeds."""
    d = load("film_level")
    if not d:
        return
    pr = d["subset5"]["per_repeat_film"]
    seeds = d["stability"]["per_seed"]
    keys = [("ground-truth emotion(11) [ceiling]", "ratings\n(ceiling)"),
            ("VGGish -> predicted emotion(11), OOF-trained", "emotion\nroute"),
            ("VGGish-128 (direct)", "direct\nVGGish"),
            ("AST-768", "direct\nAST"),
            ("PCA-8(VGGish) [control]", "PCA-8\ncontrol")]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    rng = np.random.default_rng(0)
    for i, (k, lab) in enumerate(keys):
        v = np.array(pr[k])
        c = route_colour(lab)
        ax.boxplot(v, positions=[i], widths=0.45, showfliers=False,
                   medianprops={"color": c}, boxprops={"color": c},
                   whiskerprops={"color": c}, capprops={"color": c})
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=9, color=c, alpha=0.7,
                   zorder=3, label="one repeat (seed 42)" if i == 0 else None)
        if seeds:
            sm = [seeds[s][k]["film"] for s in seeds]
            ax.scatter([i + 0.33] * len(sm), sm, marker="_", s=90, color="k", lw=1.2,
                       label="mean of 10 repeats, per seed" if i == 0 else None)
    ax.set_xticks(range(len(keys)), [k[1] for k in keys])
    ax.set_ylabel("film-level macro-F1")
    ax.legend(loc="lower left", frameon=False)
    save(fig, "stability_repeats")


def fig_metric_family():
    """The ranking of the routes under every metric (metrics_stability.json)."""
    m = load("metrics_stability")
    if not m:
        return
    show = [("film_level_macro_f1", "macro-F1\nper film\n(headline)"),
            ("macro_f1", "macro-F1\nper clip"), ("macro_f1_tuned_threshold",
            "macro-F1,\ntuned cut"), ("macro_average_precision", "macro AP\n(no cut)"),
            ("macro_roc_auc", "macro\nROC-AUC"), ("micro_f1", "micro-F1"),
            ("samples_f1", "samples-F1")]
    arms = [("VGGish -> predicted emotion(11), OOF-trained", "emotion route"),
            ("VGGish-128 (direct)", "direct audio (VGGish)"),
            ("AST-768", "direct audio (AST)"),
            ("PCA-8(VGGish) [control]", "PCA-8 control")]
    colours = [COL["emotion"], COL["direct"], COL["other"], COL["pca"]]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.8))
    w = 0.19
    for j, ((k, lab), c) in enumerate(zip(arms, colours)):
        # distance above the base-rate random guess, so metrics with different
        # chance levels (0.5 for ROC-AUC) share one axis
        y = [m["metrics"][mt]["arms"][k]["mean"] - m["metrics"][mt]["base_rate_random_guess"]
             for mt, _ in show]
        e = [m["metrics"][mt]["arms"][k]["sd"] for mt, _ in show]
        ax.bar(np.arange(len(show)) + (j - 1.5) * w, y, w, yerr=e, color=c, label=lab,
               error_kw={"lw": 0.7, "capsize": 1.5})
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(show)), [s[1] for s in show])
    ax.set_ylabel("score − random guess")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=False)
    save(fig, "metric_family")


def fig_zero_shot():
    """Zero-shot Eerola -> Blockbuster, strict regime, with Ma et al.'s floors."""
    d = load("zero_shot")
    if not d:
        return
    z = d["zero_shot"]["strict (source scaler)"]
    rows = [("VGGish -> predicted emotion(11), per-cue", "emotion route"),
            ("PCA-8(VGGish) [control]", "PCA-8 control"),
            ("VGGish-128 direct", "direct audio (VGGish)")]
    fig, ax = plt.subplots(figsize=(WIDTH, 1.9))
    for i, (k, lab) in enumerate(rows[::-1]):
        v = z[k]
        ax.barh(i, v["macro_f1"], color=route_colour(lab), height=0.6)
        ax.text(v["macro_f1"] + 0.005, i, f"{v['macro_f1']:.3f}", va="center", fontsize=8)
    ax.set_yticks(range(len(rows)), [r[1] for r in rows[::-1]])
    floor_line(ax, d["blockbuster_baselines"]["plurality"]["macro_f1"], "most-frequent baseline")
    floor_line(ax, d["blockbuster_baselines"]["random_guess"]["macro_f1"],
               "base-rate random guess", ":")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, frameon=False,
              fontsize=7)
    b = d["bootstrap"]["strict (source scaler)"]
    p1 = b["VGGish -> predicted emotion(11), per-cue vs VGGish direct"]["p_two_sided"]
    p2 = b["predicted-emotion vs PCA-8 control"]["p_two_sided"]
    ax.text(0.99, 1.02, f"film bootstrap: emotion vs direct p = {p1:.3f}, "
            f"vs PCA-8 p = {p2:.3f}", transform=ax.transAxes, ha="right", fontsize=7)
    ax.set_xlabel("film-level macro-F1 on the 110 Blockbuster films (6 shared genres), "
                  "trained on Eerola only")
    ax.set_xlim(0, 0.6)
    save(fig, "zero_shot")


def fig_cross_dataset():
    """Both transfer directions and the pooled design (cross_dataset_cv.json)."""
    d = load("cross_dataset_cv")
    if not d:
        return
    keys = [("VGGish-128 direct", "direct audio"), ("PCA-8(VGGish) [control]", "PCA-8 control"),
            ("VGGish -> emotion (per cue) -> genre", "emotion route")]
    # every design scored per film: Blockbuster rows are films; Eerola clips are averaged
    # per film (the film_level blocks of designs 2 and 3)
    blocks = [(d["eerola_to_blockbuster"], "Eerola → Blockbuster", "macro_f1"),
              (d["blockbuster_to_eerola"]["film_level"], "Blockbuster → Eerola", "macro_f1"),
              (d["pooled"]["film_level"], "both in training (pooled CV)", "mean")]
    # error bars: bootstrap SD over test films for the two single splits, SD of the
    # repeat means for the pooled cross-validation
    sd_field = ["boot_sd", "boot_sd", "repeat_sd"]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.4))
    w = 0.26
    for j, (k, lab) in enumerate(keys):
        y = [b["arms"][k][f] for b, _, f in blocks]
        e = [b["arms"][k].get(s, 0.0) for (b, _, _), s in zip(blocks, sd_field)]
        ax.bar(np.arange(3) + (j - 1) * w, y, w, yerr=e, color=route_colour(lab), label=lab,
               error_kw={"lw": 0.8, "capsize": 2})
        for x, v, s in zip(np.arange(3) + (j - 1) * w, y, e):
            ax.text(x, v + s + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(range(3), [b[1] for b in blocks])
    ax.set_ylabel("film-level macro-F1 (6 genres)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15), frameon=False)
    ax.set_ylim(0, 0.7)
    save(fig, "cross_dataset")


def fig_blockbuster_cv():
    """Blockbuster in-domain, 10x5 cross-validation over films: mean +- SD over the repeats."""
    d = load("blockbuster_deep")
    if not d:
        return
    pf = d["logreg"]["per_fold"]
    n_rep = d["design"]["n_repeats"]
    rows = [("emotion(11), per-cue -> pooled", "emotion route (per cue)"),
            ("VGGish, instance majority voting", "VGGish, majority vote over cues"),
            ("VGGish-128", "VGGish, film average"),
            ("PCA-8(VGGish) [control]", "PCA-8 control"),
            ("MIR-140 (full)", "hand-crafted MIR (140)"),
            ("MFCC-78", "MFCC subset (78)"),
            ("random-8(VGGish) [control]", "random 8-d projection")]
    m = [np.mean(pf[k]) for k, _ in rows]
    sd = [np.asarray(pf[k]).reshape(n_rep, -1).mean(1).std(ddof=1) for k, _ in rows]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    x = np.arange(len(rows))
    cols = [route_colour(lab) if "random" not in lab and "MIR" not in lab and "MFCC" not in lab
            else COL["other"] for _, lab in rows]
    ax.bar(x, m, 0.6, yerr=sd, color=cols, error_kw={"lw": 0.8, "capsize": 2})
    for xi, v, s in zip(x, m, sd):
        ax.text(xi, v + s + 0.01, f"{v:.3f}", ha="center", fontsize=7)
    ax.axhline(d["random_guess_floor"], color=COL["floor"], lw=0.8, ls=":",
               label=f"base-rate random guess ({d['random_guess_floor']:.3f})")
    ax.set_xticks(x, [lab for _, lab in rows], rotation=25, ha="right")
    ax.set_ylabel("macro-F1 over the 110 films")
    ax.set_ylim(0, 0.72)
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    save(fig, "blockbuster_cv")


def fig_per_genre():
    """Film-level F1 per genre and route (five genres); the best route per genre is boxed."""
    d = load("film_level")
    if not d or "per_genre_precision" not in next(iter(d["subset5"]["arms"].values())):
        return
    b = d["subset5"]
    rows = [("ground-truth emotion(11) [ceiling]", "ratings (ceiling)"),
            ("VGGish -> predicted emotion(11), OOF-trained", "emotion route"),
            ("PCA-8(VGGish) [control]", "PCA-8 control"),
            ("MusiCNN-200", "MusiCNN"), ("VGGish-128 (direct)", "VGGish"),
            ("CLAP-512", "CLAP"), ("MIR-103", "MIR"), ("AST-768", "AST"),
            ("wav2vec2-768", "wav2vec 2.0"), ("dummy", "most frequent")]
    genres = b["genres"]
    films = b["films_per_genre"]
    M = np.array([b["arms"][k]["per_genre_f1"] for k, _ in rows])
    fig, ax = plt.subplots(figsize=(WIDTH, 3.0))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    real = [i for i, (k, _) in enumerate(rows) if k not in ("ground-truth emotion(11) [ceiling]", "dummy")]
    for j in range(M.shape[1]):
        best = max(real, key=lambda i: M[i, j])
        ax.add_patch(plt.Rectangle((j - 0.5, best - 0.5), 1, 1, fill=False, ec=COL["emotion"], lw=1.8))
        for i in range(M.shape[0]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(genres)), [f"{g}\n({films[g]} films)" for g in genres])
    ax.set_yticks(range(len(rows)), [lab for _, lab in rows])
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="film-level F1")
    save(fig, "per_genre")


def fig_stage1():
    """Stage 1: R^2 per emotion for every audio representation (heatmap)."""
    d = load("waveform_vs_spectrogram")
    if not d:
        return
    s1 = d["stage1"]
    order = sorted(s1, key=lambda k: -s1[k]["mean_r2"])
    emos = list(next(iter(s1.values()))["per_emotion_r2"])
    M = np.array([[s1[k]["per_emotion_r2"][e] for e in emos] + [s1[k]["mean_r2"]]
                  for k in order])
    fig, ax = plt.subplots(figsize=(WIDTH, 2.3))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=0.8, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] > 0.5 else "black",
                    fontweight="bold" if j == M.shape[1] - 1 else "normal")
    ax.set_xticks(range(len(emos) + 1), emos + ["mean"])
    ax.set_yticks(range(len(order)), [k.replace("-", " (") + ")" for k in order])
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="$R^2$")
    save(fig, "stage1_r2")


def fig_ablation():
    """Leave-one-emotion-out and theory-driven subsets: difference to all eight (film level)."""
    d = load("film_level")
    if not d:
        return
    comps = [c for c in d["ablation"]["comparisons"]]
    comps.sort(key=lambda c: c["diff"])
    fig, ax = plt.subplots(figsize=(WIDTH, 3.0))
    for i, c in enumerate(comps):
        name = c["a"]
        col = COL["emotion"] if name.startswith("without") else COL["other"]
        ax.errorbar(c["diff"], i, xerr=[[c["diff"] - c["ci_lo"]], [c["ci_hi"] - c["diff"]]],
                    fmt="o", color=col, ms=4, capsize=2, lw=1)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(comps)), [c["a"] for c in comps])
    ax.set_xlabel("difference to all eight emotions (film-level macro-F1; bars: 95% bootstrap)")
    save(fig, "ablation")


def fig_signatures():
    """Emotion signature of each genre (Cohen's d), Eerola vs Blockbuster side by side."""
    d = load("signature_replication")
    if not d:
        return
    emos, genres = d["emotions"], list(d["eerola_d"])
    A = np.array([d["eerola_d"][g] for g in genres])
    B = np.array([d["blockbuster_d_cue"][g] for g in genres])
    lim = float(np.ceil(max(abs(A).max(), abs(B).max()) * 10) / 10)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.6), sharey=True)
    for ax, M, t in ((axes[0], A, "Eerola (human ratings)"),
                     (axes[1], B, "Blockbuster (predicted, per cue)")):
        im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=6.5)
        ax.set_xticks(range(len(emos)), emos, rotation=45, ha="right")
        ax.set_title(t)
        ax.spines[:].set_visible(False)
    axes[0].set_yticks(range(len(genres)), genres)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="Cohen's d (genre vs rest)")
    save(fig, "signatures")


def fig_box_office():
    """Blockbuster: gross against predicted emotion, and what budget does to it."""
    d = load("box_office_blockbuster")
    csv = REPO / "data" / "processed" / "Blockbuster" / "box_office_per_film.csv"
    if not d or not csv.exists():
        return
    f = pd.read_csv(csv).dropna(subset=["gross"])
    B = d["B_emotion_vs_gross"]
    emos = ["valence", "energy", "tension", "anger", "fear", "happy", "sad", "tender"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 2.6),
                                 gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.35})
    act = f["genre_action"] == 1
    a1.scatter(f.loc[~act, "emo_anger"], f.loc[~act, "gross"] / 1e6, s=10,
               color=COL["direct"], alpha=0.7, label="other films")
    a1.scatter(f.loc[act, "emo_anger"], f.loc[act, "gross"] / 1e6, s=10,
               color=COL["emotion"], alpha=0.7, label="Action films")
    a1.set_yscale("log")
    a1.set_xlabel("mean predicted anger per film")
    a1.set_ylabel("worldwide gross (million USD)")
    r = B["mean anger"]
    a1.set_title(f"ρ = {r['spearman']:+.2f}", loc="left")
    a1.legend(frameon=False, loc="lower right", fontsize=7)
    x = np.arange(len(emos))
    raw = [B[f"mean {e}"]["spearman"] for e in emos]
    bud = [B[f"mean {e}"]["spearman_partial_budget"] for e in emos]
    a2.bar(x - 0.2, raw, 0.4, color=COL["emotion"], label="Spearman ρ with gross")
    a2.bar(x + 0.2, bud, 0.4, color=COL["pca"], label=f"controlling for budget (n={d['n_with_budget']})")
    a2.axhline(0, color="k", lw=0.8)
    a2.set_xticks(x, emos, rotation=45, ha="right")
    a2.set_ylim(-0.5, 0.5)
    a2.legend(frameon=False, loc="upper left", fontsize=7)
    save(fig, "box_office_blockbuster")


FIGURES = {"indomain": fig_indomain, "stability": fig_repeat_spread,
           "metrics": fig_metric_family, "zero_shot": fig_zero_shot,
           "cross_dataset": fig_cross_dataset, "stage1": fig_stage1,
           "ablation": fig_ablation, "signatures": fig_signatures,
           "box_office": fig_box_office, "blockbuster_cv": fig_blockbuster_cv,
           "per_genre": fig_per_genre}


def main() -> None:
    want = sys.argv[1:] or list(FIGURES)
    for k in want:
        FIGURES[k]()


if __name__ == "__main__":
    main()
