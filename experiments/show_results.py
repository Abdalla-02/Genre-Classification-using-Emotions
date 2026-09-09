"""Render any results/*.json as Markdown tables, ready to paste into a report.

The experiments print their tables to the terminal AND save every per-fold score to
``results/<name>.json``. The JSON is the archival record: the printed output scrolls away,
but the JSON lets any number be re-derived, re-checked or re-formatted months later
without re-running a single model. This script turns it back into readable tables.

    python experiments/show_results.py                      # list what is available
    python experiments/show_results.py statistical_power    # print its tables
    python experiments/show_results.py --all                # everything
    python experiments/show_results.py blockbuster_deep --out report.md

The renderer is shape-driven rather than file-specific: it recognises the three block
types the experiments emit -- an "arms" table (scores with confidence intervals), a
"comparisons" table (paired significance tests) and a "per_fold" block (the raw scores) --
so a new experiment that follows the same conventions is rendered without changing this
file.
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
    "statistical_power": "Eerola headline results: 10x5 repeated GroupKFold, default vs "
                         "nested-CV-tuned C (progress log section 11)",
    "blockbuster_deep": "Blockbuster under the primary protocol: repeated CV, cue-level "
                        "arms, full 140-feature MIR (section 17)",
    "zero_shot": "Train on Eerola, test on Blockbuster without training on it; plus the "
                 "in-domain reproduction of Ma et al. (section 15)",
    "waveform_vs_spectrogram": "wav2vec 2.0 (raw waveform) against the four "
                               "spectrogram-based representations (section 16)",
    "w2v_layer_sweep": "Which wav2vec 2.0 layer to pool -- the control behind section 16",
    "signature_replication": "Do the Eerola emotion-genre signatures reappear on "
                             "Blockbuster? (section 17.5)",
    "model_search": "Does any other model or feature combination beat the baseline? "
                    "(section 7d)",
}


def md_table(header: list[str], rows: list[list[str]]) -> str:
    align = ["---"] + ["---:"] * (len(header) - 1)
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def fmt(v, nd=3) -> str:
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


# --------------------------------------------------------------------------- #
# Block renderers, selected by shape
# --------------------------------------------------------------------------- #
def is_arms(v) -> bool:
    return (isinstance(v, dict) and v
            and all(isinstance(x, dict) and "mean" in x for x in v.values()))


def is_comparisons(v) -> bool:
    return (isinstance(v, list) and v and isinstance(v[0], dict)
            and "diff" in v[0] and ("p_corrected" in v[0] or "p" in v[0]))


def is_stage1(v) -> bool:
    return (isinstance(v, dict) and v
            and all(isinstance(x, dict) and "mean_r2" in x for x in v.values()))


def render_arms(name: str, arms: dict) -> str:
    rows = []
    for arm, d in sorted(arms.items(), key=lambda kv: -kv[1]["mean"]):
        ci = (f"[{d['ci_lo']:.3f}, {d['ci_hi']:.3f}]"
              if d.get("ci_lo") is not None and d["ci_lo"] == d["ci_lo"] else "-")
        extra = fmt(d["C_mode"], 4) if "C_mode" in d else ""
        rows.append([arm, fmt(d["mean"]), ci] + ([extra] if extra else []))
    header = ["arm", "Macro-F1", "95% CI"] + (["C"] if len(rows[0]) == 4 else [])
    return f"**{name}**\n\n" + md_table(header, rows)


def render_comparisons(name: str, comps: list) -> str:
    rows = []
    for c in comps:
        p = c.get("p_corrected", c.get("p"))
        star = " *" if p is not None and p < 0.05 else ""
        label = f"{c['a']} vs {c['b']}" if "a" in c else c.get("arm", "?")
        diff = c.get("diff", c.get("diff_vs_baseline"))
        rows.append([label, f"{diff:+.3f}", f"{p:.3f}{star}",
                     fmt(c.get("p_limit", ""), 3), f"{c.get('win_rate', 0):.0%}"])
    return (f"**{name}**\n\n"
            + md_table(["comparison", "diff", "p", "p_limit", "win rate"], rows)
            + "\n\n*p = Nadeau-Bengio corrected paired t-test; `*` marks p < 0.05.*")


def render_stage1(name: str, d: dict) -> str:
    rows = [[k, v["input_domain"], str(v["dim"]), fmt(v["mean_r2"]),
             fmt(v["mean_rmse"], 2), f"{v['mean_r2'] / 0.897:.0%}"]
            for k, v in sorted(d.items(), key=lambda kv: -kv[1]["mean_r2"])]
    return (f"**{name}**\n\n"
            + md_table(["representation", "input domain", "dim", "mean R2", "RMSE",
                        "% of 0.897 ceiling"], rows))


def render_scalars(name: str, d: dict) -> str:
    rows = [[k, fmt(v, 3)] for k, v in d.items()
            if isinstance(v, (int, float, str, bool))]
    if not rows:
        return ""
    return f"**{name or 'summary'}**\n\n" + md_table(["key", "value"], rows)


def walk(node, path: str, out: list, depth: int = 0) -> None:
    """Recurse, emitting a table wherever a recognised block shape appears."""
    if is_arms(node):
        out.append(render_arms(path, node))
        return
    if is_stage1(node):
        out.append(render_stage1(path, node))
        return
    if is_comparisons(node):
        out.append(render_comparisons(path, node))
        return
    if isinstance(node, dict):
        scal = render_scalars(path, node)
        if scal and depth <= 1:
            out.append(scal)
        for k, v in node.items():
            if k == "per_fold":
                n = len(next(iter(v.values()))) if v else 0
                out.append(f"*(`{path} / per_fold` holds the raw {n} per-fold scores for "
                           f"{len(v)} arms -- the material for any re-analysis.)*")
                continue
            if isinstance(v, (dict, list)):
                walk(v, f"{path} / {k}" if path else k, out, depth + 1)


def render_file(stem: str) -> str:
    path = RESULTS_DIR / f"{stem}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    parts = [f"## `{stem}.json`"]
    if stem in DESCRIPTIONS:
        parts.append(DESCRIPTIONS[stem])
    blocks: list[str] = []
    walk(data, "", blocks)
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
