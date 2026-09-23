# The `results/` files — what they are and how to use them

Every experiment prints its tables to the terminal. That output scrolls away, and a number
quoted in a document six weeks later cannot be checked against it. So the important
experiments **also write every per-fold score to `results/<name>.json`**.

This is not a convenience — it is the mechanism that catches errors. Section 7d of the
progress log records a results table that sat in the documentation for weeks with **no
source script and no saved output**; it could not be re-derived, and when it was finally
rewritten as a runnable experiment its central claim turned out to be wrong. Anything
worth quoting is worth saving in machine-readable form.

---

## Quick use

```bash
python experiments/show_results.py                          # list all result files
python experiments/show_results.py statistical_power        # print its tables as Markdown
python experiments/show_results.py --all --out report.md    # everything into one file
```

`show_results.py` turns a JSON back into Markdown tables that paste straight into a
report, an e-mail to the supervisor, or the thesis. It is shape-driven, so a new
experiment that follows the same conventions renders without touching it.

---

## The files

| file | what it answers | progress log |
|---|---|---|
| `cv_corrected.json` | **The Eerola results to quote.** The same folds as `statistical_power.json`, re-scored with the two CV corrections (macro-F1 on pooled out-of-fold predictions; out-of-fold emotion training). All six representations and the ablation on one protocol, old and new metric side by side, and a check that the old numbers reproduce exactly. | §21 |
| `statistical_power.json` | The **original headline Eerola results** (per-fold metric), kept as the record §21 corrects. Does emotion beat direct audio, and is it significant? Default vs nested-CV-tuned `C`. | §11, §11b |
| `blockbuster_deep.json` | Blockbuster under the **same protocol** as Eerola: repeated CV, tuned `C`, cue-level arms, full 140-feature MIR. | §17 |
| `zero_shot.json` | **Train on Eerola, test on Blockbuster without training on it.** Also the in-domain reproduction of Ma et al. (2021). | §15 |
| `waveform_vs_spectrogram.json` | Does a **raw-waveform** model (wav2vec 2.0) match spectrogram front-ends? Both pipeline stages. | §16 |
| `w2v_layer_sweep.json` | Which wav2vec 2.0 layer to pool — the control that makes §16's negative result defensible. | §16.2 |
| `signature_replication.json` | Do the **emotion→genre signatures** found on Eerola reappear on an independent corpus? | §17.5 |
| `emotion_ablation.json` | **Which emotions carry the genre signal?** Leave-one-out over the eight, plus theory-motivated subsets. | §18 |
| `cross_dataset_cv.json` | Cross-validation in **every direction**: Eerola→Blockbuster, the reverse, and both corpora pooled. | §19 |
| `box_office.json` | Does box-office gross relate to the soundtrack, or only to the genre? Exploratory, n = 37. | §20 |
| `model_search.json` | Does any other model or feature combination beat the baseline? (No.) | §7d |

`clip_length_by_genre.png` is the one non-JSON file here — a figure from
`experiments/features/clip_length_analysis.py`, kept because §5–§6 argue from it.

---

## The block types

Every file is built from a handful of recurring shapes, which is why one renderer handles
them all. The three below carry the headline results; four smaller ones
(`grid`, list grids, plain scalar blocks and the emotion-regression `stage1` block) are
described at the end.

### `arms` — one score per approach

```json
"arms": {
  "VGGish-128": { "mean": 0.593, "ci_lo": 0.573, "ci_hi": 0.613 }
}
```

`mean` is the average Macro-F1 over all folds. `ci_lo`/`ci_hi` are the **95 % confidence
interval computed over the per-repeat means**, not over the individual folds — folds inside
one repeat share training data and are not independent observations, so using all 50 would
understate the interval.

**Single-split experiments key the score `macro_f1` instead of `mean`** and carry
`macro_precision` / `macro_recall` beside it, because there are no folds to average: the
zero-shot arms in `zero_shot.json` and the two transfer directions in
`cross_dataset_cv.json` are each one train-once-test-once evaluation, and their uncertainty
comes from the bootstrap over films rather than from a fold spread. An arm may also carry
`diff_vs_ref` and `p_corrected` (`emotion_ablation.json`) when every arm is compared
against one reference rather than pairwise.

### `comparisons` — one paired significance test per pair

```json
{ "a": "emotion(11), per-cue -> pooled", "b": "emotion(11), pooled -> per-film",
  "diff": 0.059, "ci_lo": 0.045, "ci_hi": 0.073,
  "p_corrected": 0.020, "p_limit": 0.013, "win_rate": 0.92 }
```

| field | meaning |
|---|---|
| `diff` | mean Macro-F1 of `a` minus that of `b`, over the same folds |
| `ci_lo`, `ci_hi` | 95 % interval of that difference |
| `p_corrected` | Nadeau–Bengio corrected paired *t*-test. A plain *t*-test over 50 overlapping folds is anti-conservative; the correction inflates the variance by the train/test overlap factor |
| `p_limit` | the *p* this test converges to with **infinite** repeats. If `p_limit > 0.05`, more computation can never make the comparison significant — only more films can |
| `win_rate` | fraction of folds on which `a` beats `b`; parameter-free and often more intuitive than *p* |

A comparison block is a **list** when the pairs are arbitrary (`a` vs `b`), and a **dict
keyed by the comparison's name** when there is a fixed set of them — `zero_shot.json →
bootstrap`, where the difference is `diff_mean` and the *p* is `p_two_sided` because it
comes from a 2000-sample bootstrap over the 110 films rather than from folds.

### `per_fold` — the raw scores

```json
"per_fold": { "VGGish-128": [0.58, 0.61, 0.55, ...] }
```

One entry per fold per arm, on the **same folds for every arm**, which is what makes the
comparisons properly paired. This is the material for any re-analysis: a different
significance test, a different aggregation, a box plot, or simply checking that a reported
mean is what the folds actually say.

---

## Recipes

**Re-derive a number quoted in the thesis.**

```python
import json
d = json.load(open("results/blockbuster_deep.json"))
print(d["logreg"]["arms"]["MIR-140 (full)"]["mean"])          # 0.585
```

**Check that a mean matches its folds.**

```python
import json, numpy as np
d = json.load(open("results/statistical_power.json"))["subset5"]["tuned"]
for arm, folds in d["per_fold"].items():
    print(f"{arm:40} stored {d['arms'][arm]['mean']:.3f}  recomputed {np.mean(folds):.3f}")
```

**Plot the fold distribution** (a box plot shows overlap that a mean ± CI hides):

```python
import json, numpy as np, matplotlib.pyplot as plt
pf = json.load(open("results/blockbuster_deep.json"))["logreg"]["per_fold"]
plt.boxplot(list(pf.values()), labels=list(pf), vert=False); plt.xlabel("Macro-F1")
plt.tight_layout(); plt.savefig("results/blockbuster_folds.png", dpi=150)
```

**The smaller shapes**, all rendered by the same script:

- **`stage1`** — emotion regression rather than classification, so it holds `mean_r2`,
  `mean_rmse` and `per_emotion_r2` rather than Macro-F1
  (`waveform_vs_spectrogram.json → stage1`).
- **grid** — `{row: {field: scalar}}`, for anything tabular that is not a score:
  `box_office.json → C_genre_vs_gross` (one row per genre), `→ B_emotion_vs_gross` (one
  per emotion), `signature_replication.json → replication` (one per genre).
- **list grid** — `{row: [v₁ … vₙ]}`. The columns are named by a same-length list of
  strings elsewhere in the same file, which is why `shared_genres` and `emotions` are
  stored at the top level: `signature_replication.json` holds Cohen's *d* vectors
  (`eerola_d`, `blockbuster_d_film`, `blockbuster_d_cue`, each `{genre: [8 values]}` in
  `config.EMOTIONS` order) and `zero_shot.json → ma2021` holds the numbers **published by
  Ma et al.**, as `[precision, recall, Macro-F1]`, so our reproduction can be read next to
  the paper's own figures.
- **scalars** — a plain `{field: value}` block: `design`, `summary`, and the single
  bootstrap results in `cross_dataset_cv.json` (`emotion_vs_direct`, `emotion_vs_pca8`).

If a new experiment invents a shape none of these covers, `show_results.py` would silently
drop it — so `audit_consistency.py` checks that every results file still renders a
non-trivial number of blocks. That check exists because the zero-shot and cross-dataset
tables *were* silently dropped for several weeks.

---

## Regenerating

Every file is reproducible from a clone — seed 42, cached embeddings committed:

```bash
python experiments/evaluation/exp_statistical_power.py       # statistical_power.json
python experiments/evaluation/exp_cv_corrected.py            # cv_corrected.json (~25 min, 12 cores)
python experiments/cross_dataset/exp_blockbuster_deep.py     # blockbuster_deep.json
python experiments/cross_dataset/exp_zero_shot.py            # zero_shot.json
python experiments/cross_dataset/exp_signature_replication.py
python experiments/cross_dataset/exp_cross_dataset_cv.py      # cross_dataset_cv.json
python experiments/genre/exp_emotion_ablation.py              # emotion_ablation.json
python experiments/features/exp_waveform_vs_spectrogram.py
python experiments/features/exp_w2v_layer_sweep.py
python experiments/diagnostics/exp_model_search.py
python experiments/diagnostics/exp_box_office.py              # box_office.json
```

`box_office.json` additionally needs `data/processed/Eerola_DB/box_office.csv`, which is
committed; regenerate it with `python experiments/features/fetch_box_office.py` only if the
figures need refreshing, since that script goes out to Wikidata and Box Office Mojo.

> **Caution (known issue, §12.6).** A short debug run overwrites the authoritative file:
> `exp_statistical_power.py --repeats 2` silently replaces `results/statistical_power.json`
> with a 10-fold version. Check `design.n_repeats` in the JSON before quoting from it — the
> reported figures use `n_repeats: 10`.
