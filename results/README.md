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
python experiments/show_results.py film_level               # print its tables as Markdown
python experiments/show_results.py --all --out report.md    # everything into one file
```

`show_results.py` turns a JSON back into Markdown tables that paste straight into a
report, an e-mail to the supervisor, or the thesis. It is shape-driven, so a new
experiment that follows the same conventions renders without touching it.

---

## The files

"In the thesis" says whether the thesis quotes the file: **yes**, **superseded** (kept as
the record of a corrected result; quote its successor instead) or **no** (analysis kept in
the repository but left out of the thesis).

| file | what it answers | in the thesis | progress log |
|---|---|---|---|
| `film_level.json` | **The Eerola headline (since 7 Oct 2026): every genre evaluation scored per film.** Clip probabilities averaged per film, genre predicted at ≥ 0.5, macro-F1 over the 41 (43) films. Same arms, folds and seeds as `cv_corrected.json`; every arm and comparison also carries its clip-level score, the clip-level Nadeau–Bengio p and `p_limit`; 5 and 8 genres, ablation, stability over five seeds, per-genre F1, film counts per genre. Its reproduction check (107 numbers of `cv_corrected` and `metrics_stability`) must hold. | yes (all in-domain genre results, Tables and Figures of Ch. 5) | §23 |
| `cv_corrected.json` | The **clip-level** Eerola results: the same folds as `statistical_power.json`, re-scored with the two CV corrections (macro-F1 on pooled out-of-fold predictions; out-of-fold emotion training). All six representations and the ablation on one protocol, old and new metric side by side. Until 7 Oct the headline; now the clip-level check beside `film_level.json`. | yes, as the clip-level values | §21 |
| `metrics_stability.json` | **Is macro-F1 the right metric, and how stable are the numbers?** Every route under 13 multi-label metrics (incl. tuned thresholds, macro AP, ROC-AUC, film-level F1), paired film-bootstrap tests per metric, and the spread over repeats, folds, films and five master seeds. Seed 42 reproduces `cv_corrected.json` exactly. | yes (Section 5.4, metric table, stability figure) | §22.1 |
| `statistical_power.json` | The **original headline Eerola results** (per-fold metric), kept as the record §21 corrects. Default vs nested-CV-tuned `C`; Nadeau–Bengio `p_limit`. | superseded: `cv_corrected.json` repeats its Nadeau–Bengio p and `p_limit` (`old_metric_*` fields), so quote from there | §11, §11b |
| `blockbuster_deep.json` | Blockbuster under the **same protocol** as Eerola: repeated CV, tuned `C`, cue-level arms, full 140-feature MIR. | yes (Blockbuster in-domain) | §17 |
| `zero_shot.json` | **Train on Eerola, test on Blockbuster without training on it** (quote the strict, source-scaler regime). Also the in-domain reproduction of Ma et al. (2021). | yes (the main transfer result, 0.511 vs 0.407) | §15 |
| `cross_dataset_cv.json` | Transfer in **every direction**: Eerola→Blockbuster, the reverse, and both datasets pooled. The pooled design was re-run at its default 5×5 on 7 Oct 2026 (the earlier file held a 3×5 run). The `film_level` blocks of the reverse and pooled designs (Eerola clips averaged per film) are what the thesis quotes; the clip-level blocks beside them are unchanged. Note: the clip-level reverse-direction bootstrap resamples clips, not films (§23.3). | yes (cross-dataset figure, Discussion table) | §19, §22.4, §23.3 |
| `signature_replication.json` | Do the **emotion→genre signatures** found on Eerola reappear on Blockbuster (r = 0.836)? | yes (signatures figure) | §17.5 |
| `emotion_ablation.json` | **Which emotions carry the genre signal?** Leave-one-out over the eight, plus theory-motivated subsets. | superseded by the `ablation` block of `cv_corrected.json` | §18 |
| `waveform_vs_spectrogram.json` | Does a **raw-waveform** model (wav2vec 2.0) match spectrogram front-ends? Both pipeline stages. | yes (Stage-1 table) | §16 |
| `w2v_layer_sweep.json` | Which wav2vec 2.0 layer to pool, the control that makes §16's negative result defensible. | yes (Methods, one sentence) | §16.2 |
| `model_search.json` | Does any other model or feature combination beat the baseline? (No.) | yes (Methods: other classifiers tried, none better) | §7d |
| `box_office.json` | Does box-office gross relate to the soundtrack, or only to the genre? Eerola, exploratory, n = 37. | **no** (author's decision, 7 Oct 2026) | §20, §22.2 |
| `box_office_blockbuster.json` | The same question on the **110 Blockbuster films**: gross against per-film classification quality under six per-film metrics (in-domain and zero-shot), predicted emotions, genre, cross-validated prediction of gross, and all 140 MIR descriptors, with Benjamini–Hochberg correction. | **no** (draft text kept in `docs/latex/held_back/box_office.tex`) | §22.3 |

`clip_length_by_genre.png` is the one non-JSON file here — a figure from
`experiments/features/clip_length_analysis.py`, kept because §5–§6 argue from it. The
thesis figures are in `figures/` (see "Running the experiments" below).

Some numbers in the thesis come from experiments that only print their output; they are
listed, with their scripts, in `docs/verification_checklist.md` (section D).

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

What `mean` averages depends on the file's scoring protocol (log §21, §23):

- **Pooled per repeat, per film** (`film_level.json`, the thesis headline): the out-of-fold
  clip probabilities of each repeat are averaged per film, a genre is predicted at ≥ 0.5,
  and macro-F1 is computed over the films; `mean` is the average of the 10 repeat scores
  (listed in `per_repeat_film`). `ci_lo`/`ci_hi` are the film bootstrap, `repeat_sd`,
  `repeat_min`, `repeat_max` describe the repeats, `per_genre_f1` follows the block's
  `genres`, and `clip_level_mean` (with its interval) is the clip-level score of the same
  predictions. Its comparisons carry `p_two_sided` (film bootstrap at film level) plus
  `clip_level_diff`, `clip_level_p_two_sided`, `clip_level_nb_p` and
  `clip_level_nb_p_limit`.
- **Pooled per repeat, per clip** (`cv_corrected.json`, `metrics_stability.json`): macro-F1 is
  computed once per repeat on all clips' out-of-fold predictions, and `mean` is the average
  of those 10 repeat scores. The thesis quotes it as the clip-level check. `ci_lo`/`ci_hi` are the
  95 % interval from the film bootstrap; `pooled_repeat_min`/`_max` (or `sd`, `min`, `max`
  in `metrics_stability.json`) describe the 10 repeats. `cv_corrected.json` also keeps the
  old per-fold score of the same arm as `old_metric_mean`, and `C_selected` counts how
  often each regularisation strength was chosen.
- **Per fold** (`statistical_power.json`, `blockbuster_deep.json`, `emotion_ablation.json`):
  `mean` is the average over all 50 folds, and `ci_lo`/`ci_hi` are the 95 % interval over
  the per-repeat means, not over the individual folds. Folds inside one repeat share
  training data and are not independent, so using all 50 would understate the interval.
  On Eerola this per-fold score is biased low by 0.02–0.03, because a fold that lacks a
  rare genre scores it 0; on Blockbuster (110 films) the bias is negligible.

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

The fields above are the **Nadeau–Bengio** form (`statistical_power.json`,
`blockbuster_deep.json`, `emotion_ablation.json`). The pooled-protocol files test with the
**paired film bootstrap** instead:

```json
{ "a": "VGGish -> predicted emotion(11), OOF-trained", "b": "VGGish-128 (direct)",
  "diff": 0.040, "ci_lo": 0.001, "ci_hi": 0.084, "p_two_sided": 0.042,
  "repeat_win_rate": 1.0, "old_metric_p_corrected": 0.151, "old_metric_p_limit": 0.130 }
```

`p_two_sided` comes from resampling films with replacement (2000 draws in
`cv_corrected.json`, 1000 per metric in `metrics_stability.json`, which adds a `metric`
field), and `repeat_win_rate` is the share of the 10 repeats that `a` wins. In
`cv_corrected.json` the `old_metric_*` fields carry the Nadeau–Bengio test of the same pair
on the per-fold scores. **The thesis reports both tests** (bootstrap p = 0.042,
Nadeau–Bengio p = 0.151 for the comparison above).

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

Pooled-protocol files add **`per_repeat_pooled`**: `{arm: [10 values]}`, the score of each
repeat on its pooled out-of-fold predictions. These are the numbers whose mean is quoted.

---

## Recipes

**Re-derive a number quoted in the thesis.**

```python
import json
d = json.load(open("results/film_level.json"))["subset5"]
print(d["arms"]["VGGish -> predicted emotion(11), OOF-trained"]["mean"])   # 0.469, the emotion route (film level)
d = json.load(open("results/blockbuster_deep.json"))
print(d["logreg"]["arms"]["MIR-140 (full)"]["mean"])          # 0.585
```

**Check that a mean matches its repeats.**

```python
import json, numpy as np
d = json.load(open("results/cv_corrected.json"))["subset5"]
for arm, reps in d["per_repeat_pooled"].items():
    print(f"{arm:48} stored {d['arms'][arm]['mean']:.3f}  recomputed {np.mean(reps):.3f}")
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
- **metric family** — `metrics_stability.json → metrics → <metric>` holds one `arms` block
  per metric (each arm `mean`, `sd`, `min`, `max` over the 10 repeats) plus that metric's
  `definition`, `lower_is_better` and simulated `base_rate_random_guess`.
  `route_ranking_by_metric` lists the four audio routes best first under each metric.
- **stability** — `metrics_stability.json → stability`: under `seed_42`, per arm, the SD
  over the 10 repeats (`repeat_sd`), over the 50 single folds (`fold_sd`) and the film
  bootstrap SE (`film_bootstrap_se_approx`, the 95 % interval width / 3.92); under
  `seeds`, the 10-repeat mean for each of five master seeds and their SD (`sd_of_means`).
  `docs/current_state.md` (section 3) explains what each of these measures.

If a new experiment invents a shape none of these covers, `show_results.py` would silently
drop it — so `audit_consistency.py` checks that every results file still renders a
non-trivial number of blocks. That check exists because the zero-shot and cross-dataset
tables *were* silently dropped for several weeks.

---

## Running the experiments

Run every command from the **repository root** with the project environment
(`.venv/Scripts/python.exe` on Windows, shown as `python` below). Everything is seeded
(42) and reads the committed embedding caches, so a re-run reproduces the file it writes
exactly. Reduced debug runs (`--repeats 2`, `--quick`) write a separate
`*.partial.json` and never replace the file the documents quote.

### Experiments that write a results file

```bash
# --- Eerola, in-domain ------------------------------------------------------------------
python experiments/evaluation/exp_film_level.py              # film_level.json  (~38 min, 12 cores) -- THE numbers to quote (film level)
python experiments/evaluation/exp_cv_corrected.py            # cv_corrected.json  (~27 min, 12 cores) -- the clip-level values
python experiments/evaluation/exp_metrics_stability.py       # metrics_stability.json  (~19 min, 12 cores) -- metric family + stability
python experiments/evaluation/exp_statistical_power.py       # statistical_power.json  (the original per-fold run)
python experiments/genre/exp_emotion_ablation.py             # emotion_ablation.json
python experiments/diagnostics/exp_model_search.py           # model_search.json
python experiments/features/exp_waveform_vs_spectrogram.py   # waveform_vs_spectrogram.json
python experiments/features/exp_w2v_layer_sweep.py           # w2v_layer_sweep.json  (see requirements below)
python experiments/features/clip_length_analysis.py          # clip_length_by_genre.png  (needs raw audio)

# --- Blockbuster, and transfer between the two corpora ----------------------------------
python experiments/cross_dataset/exp_blockbuster_deep.py     # blockbuster_deep.json
python experiments/cross_dataset/exp_zero_shot.py            # zero_shot.json
python experiments/cross_dataset/exp_cross_dataset_cv.py     # cross_dataset_cv.json
python experiments/cross_dataset/exp_signature_replication.py  # signature_replication.json

# --- Box office (exploratory; not in the thesis) ----------------------------------------
python experiments/features/fetch_box_office.py              # OPTIONAL, needs internet: refreshes data/processed/Eerola_DB/box_office.csv
python experiments/diagnostics/exp_box_office.py             # box_office.json  (+ box_office_per_film.csv)
python experiments/features/fetch_box_office_blockbuster.py  # OPTIONAL, needs internet: refreshes data/processed/Blockbuster/box_office.csv
python experiments/diagnostics/exp_box_office_blockbuster.py # box_office_blockbuster.json  (+ Blockbuster/box_office_per_film.csv)

# --- Figures for the thesis, drawn from the files above ------------------------------------
python experiments/make_figures.py                           # figures/*.pdf and *.png
```

Both box-office analyses work offline: they read the committed
`data/processed/Eerola_DB/box_office.csv` and `data/processed/Blockbuster/box_office.csv`.
Run the fetch scripts first only if you want fresh figures. They query Wikidata and Box
Office Mojo, so the grosses (and with them the results) can change if either source has
been updated since. The Blockbuster fetch reuses the Wikidata matches of an earlier run
unless given `--fresh`, because resolving 110 titles from scratch takes up to half an
hour under Wikimedia's rate limit.

`make_figures.py` draws every figure from the results files, never from typed-in
numbers, so re-run it after re-running an experiment. The eight figures in the thesis are
`indomain_eerola5`, `stability_repeats`, `metric_family`, `stage1_r2`, `ablation`,
`signatures`, `zero_shot` and `cross_dataset`. It also draws `box_office_blockbuster`,
which the thesis no longer uses.

### Experiments that only print their tables

These write no results file. Their output goes to the terminal; the progress log
(`docs/README.md`) records what each one found.

```bash
python experiments/features/verify_data.py              # data sanity check: clip and film counts (seconds)
python experiments/features/exp4_rating_reliability.py  # two listener panels agree: the R^2 ceiling of 0.897
python experiments/emotion/exp_emotion_regression.py    # stage 1: audio -> emotion, per representation
python experiments/emotion/exp_emotion_improve.py       # attempts to improve stage 1
python experiments/genre/exp5_target.py                 # which genre target to use
python experiments/genre/exp_genre.py                   # stage 2: genre from emotion (8 genres)
python experiments/genre/exp_genre_subset.py            # the 5-genre subset
python experiments/genre/exp_emotion_genre.py           # Cohen's d signatures per genre (quoted in Fundamentals)
python experiments/diagnostics/exp_error_analysis.py    # per-genre errors, over-prediction
python experiments/diagnostics/exp_threshold_fix.py     # one global threshold, older protocol (per-genre tuning: metrics_stability.json)
python experiments/diagnostics/exp_learning_curve.py    # would more data help?
python experiments/diagnostics/exp_clip_length.py       # full-clip vs 10.24 s window (needs raw audio)
python experiments/cross_dataset/exp_blockbuster.py         # first Blockbuster baseline (superseded by blockbuster_deep)
python experiments/cross_dataset/exp_blockbuster_emotion.py # first emotion bridge (superseded by blockbuster_deep)
```

To keep a copy of a print-only experiment's output, redirect it:
`python experiments/genre/exp_emotion_genre.py > emotion_genre.txt`.

### Requirements that a fresh clone does not meet

Most experiments need nothing beyond the repository. Three need the **raw Eerola audio**,
which is not in the repository for copyright reasons (`data/raw/README.md` explains
where to get it):

| experiment | why |
|---|---|
| `clip_length_analysis.py` | measures clip durations from the audio files |
| `exp_clip_length.py` | embeds every 10.24 s window of each clip (cached locally, git-ignored) |
| `exp_w2v_layer_sweep.py` | needs all 13 wav2vec 2.0 layers per clip; they are cached locally but git-ignored (~13 MB per clip), so on a fresh clone it re-extracts them from the audio |

### Rebuilding the embedding caches (rarely needed)

Every experiment reads the per-clip embeddings in `data/processed/Eerola_DB/embeddings/`,
which are committed. Re-extract them only after changing an extractor; this needs the
raw audio and downloads the pretrained models:

```bash
python experiments/features/extract_features.py --model ast   # also: vggish, clap, mir, wav2vec2
.venv-musicnn/Scripts/python.exe experiments/features/extract_musicnn.py   # separate TensorFlow environment
```

### Check the documents afterwards

```bash
python experiments/audit_consistency.py   # do the documents still quote what results/*.json says?
python experiments/show_results.py --all --out report.md   # every results file as Markdown tables
python experiments/export_current_state.py   # docs/current_state.md -> .docx and .pdf (PDF needs LibreOffice)
```
