"""Render any results/*.json as Markdown tables, ready to paste into a report.

The experiments print their tables to the terminal AND save every per-fold score to
``results/<name>.json``. The JSON is the archival record: the printed output scrolls away,
but the JSON lets any number be re-derived, re-checked or re-formatted months later
without re-running a single model. This script turns it back into readable tables.

    python experiments/show_results.py                      # list what is available
    python experiments/show_results.py statistical_power    # print its tables
    python experiments/show_results.py --all                # everything
    python experiments/show_results.py blockbuster_deep --out report.md

The renderer is shape-driven rather than file-specific: it recognises the block types the
experiments emit and renders whichever it finds, so a new experiment that follows the same
conventions needs no change here. The shapes are

  arms          {name: {mean | macro_f1, ...}}      scores, with CI / p / C if present
  comparisons   [{a, b, diff, p, ...}] or {label: {diff | diff_mean, p, ...}}
  stage1        {name: {mean_r2, mean_rmse, ...}}   emotion regression
  grid          {row: {field: scalar}}              anything tabular that is not a score
  list grid     {row: [v1 ... vn]}                  columns named by a matching list
  scalars       {field: scalar}                     a plain key/value block
  per_fold      summarised, never dumped

``audit_consistency.py`` checks that every results file still renders a non-trivial set of
blocks, so a future experiment that invents a new shape is caught rather than silently
dropped -- which is exactly what happened to the zero-shot and cross-dataset tables before
the grid and dict-comparison shapes were added.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402

RESULTS_DIR = config.RESULTS_DIR

# One-line description per results file, shown by the listing and as a table caption.
DESCRIPTIONS = {
    "statistical_power": "Original Eerola run, per-fold metric: 10x5 repeated GroupKFold, "
                         "default vs tuned C. Superseded for levels by cv_corrected (section 11)",
    "cv_corrected": "Eerola re-run with the two CV corrections: pooled per-repeat "
                    "macro-F1 and out-of-fold emotion training; old vs new side by side "
                    "(section 21)",
    "blockbuster_deep": "Blockbuster under the primary protocol: repeated CV, cue-level "
                        "arms, full 140-feature MIR (section 17)",
    "zero_shot": "Train on Eerola, test on Blockbuster without training on it; plus the "
                 "in-domain reproduction of Ma et al. (section 15)",
    "cross_dataset_cv": "Cross-validation in every direction: Eerola->Blockbuster, the "
                        "reverse, and both corpora pooled (section 19)",
    "waveform_vs_spectrogram": "wav2vec 2.0 (raw waveform) against the five other "
                               "representations; its genre part is superseded by cv_corrected (section 16)",
    "w2v_layer_sweep": "Which wav2vec 2.0 layer to pool -- the control behind section 16",
    "signature_replication": "Do the Eerola emotion-genre signatures reappear on "
                             "Blockbuster? (section 17.5)",
    "emotion_ablation": "Which emotions carry the genre signal? Leave-one-out and "
                        "theory-motivated subsets (section 18)",
    "box_office": "Does box-office gross relate to the soundtrack or to its genre? "
                  "Exploratory, n=37 (section 20)",
    "model_search": "Does any other model or feature combination beat the baseline? "
                    "(section 7d)",
}

# Column axes for list-valued blocks whose meaning cannot be recovered from the file.
# Everything else takes its headers from a same-length list of strings in the same file
# (`shared_genres`, `emotions`, ...), which is why this table has a single entry.
LIST_AXES = {"ma2021": ["precision", "recall", "Macro-F1"]}

SCORE_KEYS = ("mean", "macro_f1")
DIFF_KEYS = ("diff", "diff_mean", "diff_vs_baseline")
P_KEYS = ("p_corrected", "p_two_sided", "p_perm", "p_mw", "p")


def md_table(header: list[str], rows: list[list[str]]) -> str:
    align = ["---"] + ["---:"] * (len(header) - 1)
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def fmt(v, nd=3) -> str:
    if isinstance(v, bool) or v is None:
        return str(v)
    if isinstance(v, float):
        return f"{v:.{nd}f}" if abs(v) < 1e5 else f"{v:,.0f}"
    return str(v)


def first_key(d: dict, keys) -> str | None:
    return next((k for k in keys if k in d), None)


def is_scalar(v) -> bool:
    return v is None or isinstance(v, (int, float, str, bool))


# --------------------------------------------------------------------------- #
# Block recognisers
# --------------------------------------------------------------------------- #
def is_arms(v) -> bool:
    return (isinstance(v, dict) and bool(v)
            and all(isinstance(x, dict) and first_key(x, SCORE_KEYS) for x in v.values()))


def is_stage1(v) -> bool:
    return (isinstance(v, dict) and bool(v)
            and all(isinstance(x, dict) and "mean_r2" in x for x in v.values()))


def is_comparisons(v) -> bool:
    """A list of comparison records, or a dict of them keyed by label."""
    def one(x):
        return isinstance(x, dict) and first_key(x, DIFF_KEYS) and first_key(x, P_KEYS)
    if isinstance(v, list):
        return bool(v) and one(v[0])
    return isinstance(v, dict) and bool(v) and all(one(x) for x in v.values())


def is_grid(v) -> bool:
    """A dict of dicts of scalars -- one row per key, one column per field."""
    return (isinstance(v, dict) and bool(v)
            and all(isinstance(x, dict) and any(is_scalar(y) for y in x.values())
                    for x in v.values()))


def is_list_grid(v) -> bool:
    """A dict of equal-length numeric lists -- one row per key, one column per position."""
    return (isinstance(v, dict) and bool(v)
            and all(isinstance(x, list) and x and all(is_scalar(y) for y in x)
                    for x in v.values())
            and len({len(x) for x in v.values()}) == 1)


# --------------------------------------------------------------------------- #
# Block renderers
# --------------------------------------------------------------------------- #
def render_arms(name: str, arms: dict) -> str:
    key = first_key(next(iter(arms.values())), SCORE_KEYS)
    has = {k: any(k in d for d in arms.values())
           for k in ("macro_precision", "ci_lo", "diff_vs_ref", "p_corrected",
                     "C_mode", "C", "C_selected")}
    header = ["arm", "Macro-F1"]
    if has["macro_precision"]:
        header += ["precision", "recall"]
    if has["ci_lo"]:
        header += ["95% CI"]
    if has["diff_vs_ref"]:
        header += ["vs ref"]
    if has["p_corrected"]:
        header += ["p"]
    c_field = next((k for k in ("C_mode", "C", "C_selected") if has[k]), None)
    if c_field:
        header += ["C"]

    rows = []
    for arm, d in sorted(arms.items(), key=lambda kv: -kv[1][key]):
        row = [arm, fmt(d[key])]
        if has["macro_precision"]:
            row += [fmt(d.get("macro_precision")), fmt(d.get("macro_recall"))]
        if has["ci_lo"]:
            lo = d.get("ci_lo")
            row.append(f"[{lo:.3f}, {d['ci_hi']:.3f}]"
                       if isinstance(lo, float) and lo == lo else "-")
        if has["diff_vs_ref"]:
            v = d.get("diff_vs_ref")
            row.append(f"{v:+.3f}" if isinstance(v, (int, float)) else "reference")
        if has["p_corrected"]:
            p = d.get("p_corrected")
            row.append(f"{p:.3f}{' *' if p < 0.05 else ''}"
                       if isinstance(p, (int, float)) else "-")
        if c_field:
            row.append(fmt_C(d.get(c_field)))
        rows.append(row)
    return f"**{name}**\n\n" + md_table(header, rows)


def fmt_C(v) -> str:
    """A tuned C is either one value or a [[C, folds], ...] histogram; show the mode."""
    if isinstance(v, list) and v:
        best = max(v, key=lambda p: p[1])
        return f"{best[0]:g} ({best[1]}/{sum(c for _, c in v)} folds)"
    return fmt(v, 4) if v is not None else "-"


def render_comparisons(name: str, comps) -> str:
    items = list(comps) if isinstance(comps, list) else \
        [{**v, "_label": k} for k, v in comps.items()]
    rows = []
    for c in items:
        p = c.get(first_key(c, P_KEYS))
        star = " *" if isinstance(p, (int, float)) and p < 0.05 else ""
        label = c.get("_label") or (f"{c['a']} vs {c['b']}" if "a" in c else c.get("arm", "?"))
        diff = c.get(first_key(c, DIFF_KEYS))
        ci = (f"[{c['ci_lo']:.3f}, {c['ci_hi']:.3f}]"
              if isinstance(c.get("ci_lo"), float) else "")
        rows.append([label, f"{diff:+.3f}", ci, f"{p:.3f}{star}",
                     fmt(c.get("p_limit", ""), 3),
                     f"{c['win_rate']:.0%}" if "win_rate" in c else ""])
    keep = [i for i in range(6) if i < 2 or any(r[i] for r in rows)]
    header = [h for i, h in enumerate(
        ["comparison", "diff", "95% CI", "p", "p_limit", "win rate"]) if i in keep]
    rows = [[r[i] for i in keep] for r in rows]
    return (f"**{name}**\n\n" + md_table(header, rows)
            + "\n\n*`*` marks p < 0.05. `p_corrected` is the Nadeau-Bengio corrected "
              "paired t-test; `p_two_sided` a bootstrap over films; `p_perm` a "
              "permutation test.*")


def render_stage1(name: str, d: dict) -> str:
    rows = [[k, v["input_domain"], str(v["dim"]), fmt(v["mean_r2"]),
             fmt(v["mean_rmse"], 2), f"{v['mean_r2'] / 0.897:.0%}"]
            for k, v in sorted(d.items(), key=lambda kv: -kv[1]["mean_r2"])]
    return (f"**{name}**\n\n"
            + md_table(["representation", "input domain", "dim", "mean R2", "RMSE",
                        "% of 0.897 ceiling"], rows))


def render_grid(name: str, d: dict) -> str:
    cols: list[str] = []
    for row in d.values():
        cols += [k for k, v in row.items() if is_scalar(v) and k not in cols]
    if not cols:
        return ""
    rows = [[k] + [fmt(row.get(c, "")) for c in cols] for k, row in d.items()]
    return f"**{name}**\n\n" + md_table([""] + cols, rows)


def render_list_grid(name: str, d: dict, axes: dict[int, list[str]]) -> str:
    n = len(next(iter(d.values())))
    head = LIST_AXES.get(name.split("/")[-1].strip()) or axes.get(n) or \
        [str(i + 1) for i in range(n)]
    rows = [[k] + [fmt(v) for v in vals] for k, vals in d.items()]
    return f"**{name}**\n\n" + md_table([""] + list(head), rows)


def render_scalars(name: str, d: dict) -> str:
    rows = [[k, fmt(v)] for k, v in d.items() if is_scalar(v)]
    if not rows:
        return ""
    return f"**{name or 'summary'}**\n\n" + md_table(["key", "value"], rows)


def walk(node, path: str, out: list, axes: dict[int, list[str]]) -> None:
    """Recurse, emitting a table wherever a recognised block shape appears."""
    for recogniser, renderer in ((is_arms, render_arms), (is_stage1, render_stage1),
                                 (is_comparisons, render_comparisons)):
        if recogniser(node):
            out.append(renderer(path, node))
            return
    if is_list_grid(node):
        out.append(render_list_grid(path, node, axes))
        return
    if is_grid(node) and not any(isinstance(v, dict) and any(
            isinstance(w, dict) for w in v.values()) for v in node.values()):
        out.append(render_grid(path, node))
        return
    if isinstance(node, dict):
        scal = render_scalars(path, node)
        if scal:
            out.append(scal)
        for k, v in node.items():
            if k == "per_fold":
                n = len(next(iter(v.values()))) if v else 0
                out.append(f"*(`{path} / per_fold` holds the raw {n} per-fold scores for "
                           f"{len(v)} arms -- the material for any re-analysis.)*")
                continue
            if isinstance(v, (dict, list)) and not is_scalar(v):
                walk(v, f"{path} / {k}" if path else k, out, axes)


def render_file(stem: str) -> str:
    path = RESULTS_DIR / f"{stem}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    # headers for list-valued blocks: any top-level list of strings, by length
    axes = {len(v): v for v in data.values()
            if isinstance(v, list) and v and all(isinstance(x, str) for x in v)}
    parts = [f"## `{stem}.json`"]
    if stem in DESCRIPTIONS:
        parts.append(DESCRIPTIONS[stem])
    blocks: list[str] = []
    walk(data, "", blocks, axes)
    parts += [b for b in blocks if b]
    return "\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", help="results file stem, e.g. statistical_power")
    ap.add_argument("--all", action="store_true", help="render every results file")
    ap.add_argument("--out", help="write Markdown to this file instead of stdout")
    args = ap.parse_args()

    stems = sorted(p.stem for p in RESULTS_DIR.glob("*.json"))
    if not args.name and not args.all:
        print(f"results in {RESULTS_DIR}:\n")
        for s in stems:
            print(f"  {s:26} {DESCRIPTIONS.get(s, '')}")
        print("\nrun:  python experiments/show_results.py <name> [--out report.md]")
        return

    chosen = stems if args.all else [args.name]
    missing = [c for c in chosen if c not in stems]
    if missing:
        sys.exit(f"no such results file: {missing[0]} (available: {', '.join(stems)})")

    text = "\n\n---\n\n".join(render_file(c) for c in chosen)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
