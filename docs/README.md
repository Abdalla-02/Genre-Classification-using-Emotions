# Implementation Progress Log

Supervisor-facing summary of what has been implemented, the key results, the decisions
taken (with justification), and what is planned next. Kept up to date as work proceeds.

---

## 1. Environment & data pipeline (done)

- **Data:** Eerola & Vuoskoski (2011) film-soundtrack set. Set 1 cleaned to **346 clips**
  (any-primary-present rule: keep a clip if any IMDb genre is one of the 8 modelled
  genres; drop no-genre + Fantasy/Mystery-only). **43 films** (`soundtrack`), used as the
  cross-validation grouping key.
- **Genre labels are MULTI-LABEL** over 8 genres (Action, Crime, Drama, Adventure, Comedy,
  Biography, Documentary, Horror). Positive counts: Drama 239, Action 103, Crime 98,
  Adventure 97, Comedy 40, Biography 24, Horror 28, Documentary 17.
- **Data-integrity checks** assert these counts on load, so any drift fails loudly
  (this caught a stray blank row Excel appended to the CSV).
- **Reproducibility:** fixed seed (42) throughout; embeddings cached to disk so nothing
  is re-extracted per run.

## 2. Feature extraction (done)

- **Primary:** AST (Audio Spectrogram Transformer, `MIT/ast-finetuned-audioset`), 128-dim
  log-mel input, mean-pooled to one **768-dim** embedding per clip. 10.24 s window.
- **Baselines:** CLAP (HTS-AT audio encoder, 512-dim; extracted, not yet used in a genre
  comparison), **VGGish** (128-dim, postprocessed — the shared feature with Blockbuster,
  see §10), and **librosa-MIR** (103-dim hand-crafted MFCC/chroma/spectral, a classic-MER
  baseline). All four extractors share one interface and cache to `data/processed/`.
- **Note:** for *emotion* prediction (§9), with a non-linear regressor AST and VGGish tie
  (~0.56 R^2); VGGish is used for the cross-dataset bridge only because it is reproducible on
  Blockbuster (§10), not because AST is weaker.
- **Clip-length investigation (see §6).**

## 3. Experiments implemented & results

All genre results use **GroupKFold by film** (leakage-safe) as the reported metric, with
StratifiedKFold / plain KFold as leaky comparators to quantify data leakage. Metrics:
Exact Match, Hamming Loss, **Macro-F1** (the fair metric under class imbalance).

| Exp | Question | Result (headline) |
|-----|----------|-------------------|
| **5 — TARGET sanity check** | embedding → 12 balanced emotion categories | ~0.22 GroupKFold acc vs 0.083 chance → pipeline works end-to-end |
| **Genre: WITH emotion** | 11 ground-truth emotion features → genre (RQ1) | Macro-F1 **0.272** (LogReg) vs 0.102 dummy → emotions do predict genre |
| **Genre: WITHOUT emotion** | 768-dim AST embedding → genre (baseline, RQ2b) | Macro-F1 **0.258** (LogReg) |
| **Blockbuster: VGGish vs MFCC** | Ma et al. 2021 features → genre (supervisor #1) | VGGish **0.582** vs MFCC 0.455 Macro-F1 |

### Key finding — data leakage (methodological contribution)
The AST baseline scores **0.44 Macro-F1 under leaky KFold but only 0.26 under GroupKFold
(a +0.18 gap)** — i.e. ~40% of its apparent performance is the model memorising *which
film* a clip came from. The 11 emotion features barely leak (+0.05 gap). So emotion
features **generalise across films; raw AST embeddings partly memorise film identity.**
This is a strong argument for the interpretable emotion intermediate.

### Honest caveat on WITH vs WITHOUT
Under leakage-safe evaluation the emotion (0.272) vs AST (0.258) difference is **within
fold-noise** (paired t-test p=0.62 for LogReg; classifier-dependent). We therefore claim
*"emotion features are competitive with 768-dim AST embeddings while being interpretable
and far more leakage-robust,"* NOT that emotion strictly beats AST. Note the WITH-emotion
number uses **ground-truth** ratings → it is the ceiling of the emotion→genre path.

## 4. Models & methodology (done)

- **Binary Relevance** multi-label strategy: Logistic Regression, Random Forest (both via
  `MultiOutputClassifier`), MLP (native multi-output), + a most-frequent dummy floor.
  Class imbalance handled with `class_weight="balanced"`. Rare-genre CV folds made robust
  with a single-class-safe wrapper.
- **Evaluation** reports mean±std over folds across three CV schemes (group / kfold /
  strat) plus a paired significance test — added after a code review to avoid overclaiming.

## 5. Literature: clip length (supervisor #3, done)

Full comparison saved in `literature/clip_length_literature.md`. Summary: prior work uses **5–60 s**
segments with **30 s the de-facto standard** (Kim 2010; Kang & Herremans 2025; Bhattacharjee
2024 shows 30 s empirically optimal). Our clips (10–31 s, mean 17 s) fall within range;
the AST 10.24 s window sits at the lower-but-established end.

## 6. Decision: clip-length handling (done — keep 10.24 s window)

Because truncating to 10.24 s discards ~40% of the average clip, we empirically tested
**full-clip windowed pooling** (tile each clip into 10.24 s windows, ~2.8 windows/clip on
average, and pool: first / center / mean / max). Result on Exp 5 and the genre baseline:

| Pooling | Exp5 F1 | Genre Macro-F1 |
|---------|---------|----------------|
| first (10.24 s, current) | 0.215 | 0.258 |
| mean (full clip) | 0.220 | 0.260 |
| center | 0.208 | 0.264 |
| max | 0.208 | 0.253 |

**All differences ≤0.006 — within noise.** Using the full clip does **not** improve results,
because these are short, homogeneous music cues (the first 10.24 s already characterises
the clip; unlike multi-scene trailers where more audio helps). **Decision: keep the single
10.24 s window** — simpler, and empirically justified by this test. The comparison itself
is a reportable Methods result.

## 7. Supervisor to-do status

| # | Item | Status |
|---|------|--------|
| 1 | VGGish vs MFCC (Blockbuster) | ✅ done |
| 2 | Genre classification (with vs without emotion) | ✅ done (first pass) |
| 3 | Clip length in the literature | ✅ done |
| 4 | Emotion–genre relationship | ✅ done — empirical RQ3 (§7b) + literature (`latex/emotion_genre_relationship.tex`) |
| 5 | Content chapters | ⬜ writing (student) |
| 6 | Emotion regression | ✅ done (§9) — AST vs VGGish vs MIR, + cross-dataset bridge (§10) |

## 7b. RQ3 — which emotions predict which genres (done)

Per-genre LogReg coefficients + RF importances + Cohen's d agree on an emotional
signature per genre (`experiments/exp_emotion_genre.py`):

| Genre | Signature | Cohen's d |
|-------|-----------|-----------|
| Horror | high fear | +0.75 |
| Biography | high happy | +0.67 (n=24) |
| Comedy | low fear / high valence | −0.61 |
| Drama | high valence | +0.57 |
| Documentary | low anger/arousal | −0.51 (n=17) |
| Action | low happy / high tension | −0.50 |
| Crime | high fear / low sad | +0.40 |
| Adventure | none distinctive | ±0.25 |

Narrative: **fear is the master discriminator** (threat genres vs light genres), valence
the second axis. Each predictable genre has an interpretable affective signature — the
interpretability payoff of the emotion intermediate. **Adventure has no distinctive
emotion, which explains its near-chance predictability.** Caveat: tiny-n genres
(Biography, Documentary) have wobbly LR coefficients from feature collinearity.

## 7c. Error analysis & why the score is what it is (done)

`experiments/exp_error_analysis.py` — per-genre FP/FN breakdown (WITH emotion, GroupKFold).
Diagnosis: the balanced LogReg predicts **3.49 labels/clip vs 1.87 true** (heavy
over-prediction) → low precision. Rare genres are essentially never caught (Biography
100% FN, Documentary 94% FN); Adventure fails both ways (no emotional signature).
Contrast: the AST baseline predicts a realistic 1.85 labels/clip but *misses* the
distinctive rare genres (Horror recall 0.68→0.21, Comedy 0.62→0.12).

`experiments/exp_threshold_fix.py` — tested whether curbing the over-prediction helps:

| Config | pred/clip | Macro-F1 |
|--------|-----------|----------|
| balanced thr 0.5 (current) | 3.49 | **0.281** (best) |
| balanced thr 0.6 | 1.85 | 0.203 |
| unbalanced thr 0.5 | 1.04 | 0.118 |
| per-genre tuned (optimistic ceiling) | — | 0.350 |

**Result: the fix makes it worse.** Reducing over-prediction cuts recall on the rare
distinctive genres (Horror/Comedy F1 → 0) and collapses toward predicting only Drama.
The balanced model is therefore **near the honest ceiling**; the modest Macro-F1 reflects
genuine data/signal limits (rare genres n=17–24, no signature for Adventure, genre
overlap), **not** a calibration mistake. Only per-genre threshold tuning shows headroom
(~0.35), and only as an optimistic upper bound.

## 7d. Accuracy-improvement search (done)

Tested standard levers for multi-label + imbalance + small n; Macro-F1 (GroupKFold,
WITH-emotion unless noted). **None beats the baseline (0.281).**

| Approach | Macro-F1 |
|----------|----------|
| LogReg + emotion (baseline) | **0.281** |
| LogReg + AST | 0.277 |
| Emotion + AST fusion (768) | 0.261 |
| Emotion + PCA-AST fusion (best) | 0.279 |
| HistGradientBoosting | 0.244 |
| Classifier Chains | 0.225 |

Findings: fusion hurts (AST adds noise once film-leakage removed); classifier chains hurt
(empirically confirms the thesis's rejection of chains for small n); gradient boosting
overfits. The 0.28 is a **robust ceiling** confirmed from four angles (audit, error
analysis, threshold test, model/feature search). The only honest ways to a higher number
are reframings, not model fixes: **genre-subset (5 learnable genres) -> 0.394**, or
film-level aggregation -> 0.34.

## 7e. Would more data help? Learning curve (done)

`experiments/exp_learning_curve.py` — train on increasing fractions of the films
(subsampled by soundtrack), Macro-F1 on the GroupKFold held-out folds.

| Train data | ~clips | Emotion Macro-F1 | AST Macro-F1 |
|-----------|--------|------------------|--------------|
| 40% | 111 | 0.241 | 0.204 |
| 55% | 152 | 0.253 | 0.226 |
| 70% | 194 | 0.261 | 0.239 |
| 85% | 235 | 0.263 | 0.248 |
| 100% | 277 | **0.272** | **0.258** |

**Both curves are still rising at 100% — neither has plateaued.** So performance is
**dataset-size-limited (not only signal-limited) in this range**: a larger corpus would
raise both models. Nuances: (1) **AST is more data-hungry** (steeper slope, +0.054 vs
emotion's +0.031) — at a much larger scale it might catch up; (2) **emotion is more
sample-efficient** and wins at every data size; (3) the marginal rate is modest (~+0.01
Macro-F1 per ~40 extra clips), so a *substantially* larger dataset (2-3x+) would be needed
for a meaningful jump. Caveat: the curve locates us on the rising part but not where it
flattens — "more data helps" is supported, "how much" is an extrapolation. Ties to the
rare-genre point: a larger dataset would rescue Biography/Documentary (n<25), but not
Adventure (no emotional signature).

## 7f. Genre-subset reframing (done)

Literature (6 papers, `literature/genre_subset_literature.md`): genre count scales with dataset
size (~100 films -> 4-6 genres: Austin, Ma; 10k films -> 18: Mangolin). Our corpus is
smaller than all of them, so 5-6 genres is scale-appropriate; the closest analogue
(Ma 2021, 110 film soundtracks) reduced IMDb's 24 genres to 6.

Experiment (`experiments/exp_genre_subset.py`): 5-genre subset {Action, Crime, Drama,
Comedy, Horror} (329 clips with >=1 of the 5), GroupKFold, LogReg:

| | Macro-F1 (8) | Macro-F1 (5-subset) |
|---|---|---|
| WITH emotion | 0.272 | **0.388** |
| WITHOUT (AST) | 0.258 | 0.323 |
| dummy | 0.102 | 0.164 |

On the subset emotion (0.388) exceeds AST (0.323) by +0.064 -- larger and consistent, but
NOT significant at 5 folds (paired-t p=0.16, underpowered). **Metric caveat:** Exact Match
and Hamming are uninformative here (the dummy scores best on both, due to balanced
over-prediction under imbalance) -- report Macro-F1 as the headline. LaTeX justification in
`latex/genre_subset.tex`. Report both full-8 and subset-5 transparently.

## 9. Experiment 1 — emotion regression (done)

`experiments/exp_emotion_regression.py` (feature comparison) and `exp_emotion_improve.py`
(regressor comparison). **The regressor matters greatly:** on VGGish, Ridge gives mean
R^2 = 0.37, SVR-RBF 0.54, **RandomForest 0.56** — so RF is used throughout. Feature
comparison (RF, GroupKFold, mean R^2 over the 8 emotions, 360 clips):

| Feature (RandomForest) | mean R^2 |
|------------------------|----------|
| AST-768 | 0.560 |
| VGGish-128 | 0.558 |
| MIR-librosa (103) | 0.490 |

Findings: with a non-linear regressor **AST and VGGish tie (~0.56)** and MIR is slightly
behind. (The earlier "VGGish >> AST for emotion" was an artifact of the linear Ridge
baseline overfitting AST's 768 dims.) Emotion is moderately predictable — 0.56 is within the
MER range (cf. Kang & Herremans 2025); arousal-type emotions are easiest, happy/sad hardest
(the MER "valence problem"). VGGish remains the feature for the cross-dataset bridge (§10)
because it is the only one reproducible on Blockbuster, **not** because AST is weak.

## 10. Cross-dataset emotion bridge + "is emotion special?" control (done)

`experiments/exp_blockbuster_emotion.py`, `exp_emotion_improve.py`. VGGish is the one
feature both datasets produce identically. A VGGish->8-emotion **RandomForest** regressor is
trained on all Eerola and applied to predict emotions on the target set; genre is then
classified from the 11 emotion features. The key control is **PCA-8(VGGish)** — a generic
8-d compression — to test whether the *emotion* bottleneck beats any low-dimensional one.

**Eerola, 5-genre subset (GroupKFold):**

| Approach | Macro-F1 |
|----------|----------|
| **VGGish -> PREDICTED emotion(11) -> genre (RF)** | **0.406** |
| ground-truth emotion(11) -> genre (ceiling) | 0.388 |
| PCA-8(VGGish) [control] | 0.350 |
| VGGish-128 direct | 0.328 |
| emotion + VGGish fusion | 0.313 |
| dummy | 0.164 |

**Blockbuster, 6 genres (KFold):** direct VGGish 0.582; predicted-emotion 0.565; PCA-8
control 0.559; random-8 0.482; dummy 0.058.

Conclusions (corrected from the earlier Ridge-based, over-tempered version):
- **On the primary Eerola set the emotion bottleneck is genuinely best:** predicted-emotion
  (0.406) beats the PCA-8 control (0.350) and the direct embedding (0.328), and matches the
  ground-truth ceiling (0.388) — so it is *not* merely an interpretable relabelling of a
  generic compression. On the easier Blockbuster it essentially ties PCA-8 (0.565 vs 0.559).
- **Predicted ~ ground-truth** (0.406 vs 0.388): RF-predicted emotions are as
  genre-informative as the human ratings (likely denoising the per-clip ratings).
- **Fusion (emotion + VGGish) hurts** (0.313) — the two are not complementary.
- **Face validity (Blockbuster, cross-dataset):** horror->fear (+0.99), romance->tender
  (+1.05), comedy->happy, drama->tender, action->anger — textbook-correct; the transferred
  emotions are meaningful, not noise.
- **Caveats:** 5-fold margins (~±0.05) make "beats control" suggestive, not significant;
  Blockbuster has no ground-truth emotions to validate the predictions directly.

This answers RQ2b with the *predicted* pipeline on both datasets and rules out the "any 8-d
bottleneck" alternative on the primary set.

## 8. Planned next

- **Content chapters** (#5) — student.
- **Complete RQ2b on Eerola:** run the predicted-emotion pipeline on Eerola itself
  (VGGish -> predicted emotion -> genre vs the AST baseline), to mirror the Blockbuster
  bridge on the primary dataset.
- **CLAP** is extracted but still unused in any comparison — wire it into the genre
  experiment or drop it.
- Optional: Blockbuster extensions (full 140-MIR, mean+std pooling); Set 1 vs Set 2 diff.

## Repository map

- `src/features/` — data loading/cleaning; AST, CLAP, VGGish, MIR extractors; Blockbuster loader
- `src/models/` — multi-label classifiers
- `src/evaluation/` — metrics + cross-validated scoring
- `experiments/` — runnable experiments: `verify_data`, `extract_features`, `exp5_target`,
  `exp_genre`, `exp_genre_subset`, `exp_emotion_genre`, `exp_error_analysis`,
  `exp_threshold_fix`, `exp_learning_curve`, `exp_clip_length`, `clip_length_analysis`,
  `exp_blockbuster`, `exp_emotion_regression`, `exp_blockbuster_emotion`
- `docs/` — this log; `docs/literature/` (literature reviews); `docs/latex/` (LaTeX snippets)
- `data/processed/Eerola_DB/` — cached embeddings (AST, CLAP, VGGish, MIR, AST-windows)
