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
- **Baselines:** CLAP (HTS-AT audio encoder, 512-dim; now a full genre baseline and the
  weakest of the three embeddings, see §11b), **VGGish** (128-dim, postprocessed — the shared feature with Blockbuster,
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

> **Protocol note.** This table is the ORIGINAL single 5-fold GroupKFold run at
> sklearn's default `C=1.0`. It is kept because early write-ups quote it, but
> **§11/§11b supersede it** — every arm has since been re-scored over 50 leakage-safe
> folds with nested-CV-tuned regularisation. Quote §11b in the thesis.

| Exp | Question | Result (headline, single 5-fold, C=1.0) |
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

### WITH vs WITHOUT — SUPERSEDED, see §11b
This section used to conclude that the emotion (0.272) vs AST (0.258) difference was
**within fold-noise** (paired t-test p=0.62), and that we could therefore claim only that
*"emotion features are competitive with AST embeddings"* — not that emotion beats AST.

**That conclusion no longer holds.** The p=0.62 came from a single 5-fold run at the
default `C=1.0`: five samples cannot resolve a difference of this size, and the default
regularisation handicapped the high-dimensional AST arm more than the 11-feature emotion
arm. Under 10x5 repeated GroupKFold with nested-CV-tuned `C` (§11b) the same comparison
gives **0.300 vs 0.257, diff +0.043, p=0.031, winning 92% of the 50 folds** — a
significant advantage for the emotion features. The progression
**p=0.62 -> 0.092 (repeated) -> 0.031 (repeated + tuned)** is itself a useful
methodological illustration; it is written up in `latex/statistical_power.tex`.

Still true: the WITH-emotion number uses **ground-truth** ratings, so it is the ceiling of
the emotion->genre path. The *predicted*-emotion pipeline (§10, §11b) is the honest
end-to-end figure, and it matches that ceiling.

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

### Research-question status
- **RQ1 — can emotion predict genre?** Yes. Ground-truth emotion → genre reaches Macro-F1
  **0.397** on the 5-genre subset and **0.300** on all 8 (§11b, the authoritative protocol),
  against dummy floors of 0.166 and 0.101, with interpretable predictors (§7b).
- **RQ2a — emotion↔genre relationships?** Done — per-genre Cohen's d signatures (§7b).
- **RQ2b — does the emotion intermediate help vs direct audio?** Yes on the full task,
  borderline on the subset. All figures below are §11b (10x5 repeated GroupKFold,
  nested-CV-tuned C — quote these, not the older single-run numbers in §3/§10):
  - **8 genres: significant.** Emotion 0.300 vs AST 0.257, +0.043, **p=0.031**, winning
    92% of 50 folds.
  - **5-genre subset: consistent but borderline.** The predicted-emotion pipeline (VGGish
    → RF-emotion → genre, 0.394) leads the PCA-8 control (0.343, +0.052, p=0.060, 88% of
    folds), the AST baseline (0.345, +0.050, p=0.078) and direct VGGish (0.350, +0.044,
    p=0.104). Direction and magnitude are stable; 5%-level significance is not reached.
    Note `p_limit`=0.045 for the control comparison, i.e. *below* 0.05 — unlike the
    untuned protocol, more repeats could eventually cross the threshold, which is why it
    is reported as borderline rather than as either result.
  - **The bottleneck costs nothing.** Predicted vs ground-truth emotion is
    −0.003 (p=0.855), so the pipeline does not depend on human ratings.
  - It also transfers cross-dataset to Blockbuster with face-valid emotions (§10).
- **RQ3 — strongest emotions per genre?** Done — LogReg coefficients + RF importance + Cohen's d (§7b).

## 7b. RQ3 — which emotions predict which genres (done)

Per-genre LogReg coefficients + RF importances + Cohen's d agree on an emotional
signature per genre (`experiments/genre/exp_emotion_genre.py`):

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

`experiments/diagnostics/exp_error_analysis.py` — per-genre FP/FN breakdown (WITH emotion, GroupKFold).
Diagnosis: the balanced LogReg predicts **3.49 labels/clip vs 1.87 true** (heavy
over-prediction) → low precision. Rare genres are essentially never caught (Biography
100% FN, Documentary 94% FN); Adventure fails both ways (no emotional signature).
Contrast: the AST baseline predicts a realistic 1.85 labels/clip but *misses* the
distinctive rare genres (Horror recall 0.68→0.21, Comedy 0.62→0.12).

`experiments/diagnostics/exp_threshold_fix.py` — tested whether curbing the over-prediction helps:

| Config | pred/clip | Macro-F1 (C=1.0) | Macro-F1 (tuned C=0.003) |
|--------|-----------|------------------|--------------------------|
| balanced thr 0.5 (current) | 3.49 / 3.68 | **0.281** (best) | **0.295** (best) |
| balanced thr 0.6 | 1.85 / 0.40 | 0.203 | 0.053 |
| balanced thr 0.7 | 0.67 / 0.00 | 0.091 | 0.000 |
| unbalanced thr 0.5 | 1.04 / 1.00 | 0.118 | 0.102 |
| per-genre tuned (optimistic ceiling) | — | 0.350 | 0.356 |

Re-run at the tuned operating point (§11b) the conclusion is **unchanged and stronger**:
raising the threshold collapses Macro-F1 to 0.053 (vs 0.203 at C=1.0), because the
regularised model's probabilities are compressed toward 0.5. Curbing over-prediction is
even less viable than first reported.

**Result: the fix makes it worse.** Reducing over-prediction cuts recall on the rare
distinctive genres (Horror/Comedy F1 → 0) and collapses toward predicting only Drama.
The balanced model is therefore **near the honest ceiling**; the modest Macro-F1 reflects
genuine data/signal limits (rare genres n=17–24, no signature for Adventure, genre
overlap), **not** a calibration mistake. Only per-genre threshold tuning shows headroom
(~0.35), and only as an optimistic upper bound.

## 7d. Accuracy-improvement search (done — REVISED, see the correction below)

`experiments/diagnostics/exp_model_search.py`, results in `results/model_search.json`.
8 genres, 5x5 repeated GroupKFold by film, Macro-F1.

> **Correction (2026-08-24).** The table originally recorded here had **no source script
> in the repository** — it could not be re-derived or re-checked, and its AST figure
> (0.277) contradicted §3's figure (0.258) for what should have been the same
> measurement. It was also measured entirely at sklearn's default `C=1.0`, which §11b
> later showed to be the *worst* setting for every feature set here. The claim that
> **"0.281 is a robust ceiling" was therefore wrong**: proper regularisation alone
> reaches 0.299. The comparison has been rewritten as a runnable experiment and is
> reported below at both operating points. The *qualitative* conclusion survives — nothing
> beats plain LogReg on the 11 emotion features — but the numbers and the strength of the
> claims change.

| Approach | A: default C=1.0 | B: nested-CV tuned | Δ |
|---|---|---|---|
| **LogReg + emotion(11)** (baseline) | 0.275 | **0.299** [0.284, 0.313] | +0.024 |
| emotion + PCA-8(AST) fusion (19) | 0.254 | 0.277 [0.257, 0.298] | +0.023 |
| LogReg + AST(768) | 0.245 | 0.260 [0.251, 0.269] | +0.015 |
| emotion + AST fusion (779) | 0.244 | 0.260 [0.249, 0.272] | +0.017 |
| ClassifierChain(LogReg) + emotion | 0.214 | 0.252 [0.240, 0.263] | +0.038 |
| HistGradientBoosting + emotion | 0.198 | 0.198 [0.191, 0.206] | ±0.000 |

Against the **tuned** baseline (Nadeau–Bengio corrected, 25 folds):

| Alternative | diff vs baseline | p | wins |
|---|---|---|---|
| emotion + PCA-8(AST) fusion | −0.021 | 0.213 | 16% |
| emotion + AST fusion | −0.038 | 0.072 | 16% |
| LogReg + AST | −0.039 | 0.076 | 12% |
| ClassifierChain | −0.047 | **0.024** | 12% |
| HistGradientBoosting | −0.100 | **<0.001** | 0% |

### What survives, and what changes

- **Survives: nothing beats the baseline.** Every alternative scores below plain LogReg on
  the 11 emotion features, and each wins at most 16% of the 25 folds.
- **Survives strongly: gradient boosting overfits** (−0.100, p<0.001) — the clearest
  negative result in the table, and unaffected by regularisation since it has no `C`.
- **Changed: the ceiling is 0.299, not 0.281.** Tuning `C` — a lever the original search
  never tested — beats the number that was called a robust ceiling. "Robust ceiling
  confirmed from four angles" was an overstatement; the honest claim is that *no
  alternative model or feature combination improves on a properly regularised baseline.*
- **Weakened: "fusion hurts".** At the tuned operating point fusion and the AST baseline
  are **not significantly worse** than the emotion baseline (p=0.072 and p=0.076); PCA
  fusion is nowhere near significance (p=0.213). Say "no combination improves on emotion
  features alone", not "adding AST actively hurts".
- **Weakened: "chains hurt".** Classifier chains gain the most from tuning (+0.038); the
  original margin was inflated by under-regularisation. They are still significantly
  worse (p=0.024), so the thesis's rejection of chains at small *n* stands, but on a
  smaller margin than first reported.

The routes to a materially higher number remain reframings rather than model fixes:
the 5-genre subset (§7f, §11b) or film-level aggregation.

## 7e. Would more data help? Learning curve (done)

`experiments/diagnostics/exp_learning_curve.py` — train on increasing fractions of the films
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

Experiment (`experiments/genre/exp_genre_subset.py`): 5-genre subset {Action, Crime, Drama,
Comedy, Horror} (329 clips with >=1 of the 5), GroupKFold, LogReg:

| | Macro-F1 (8) | Macro-F1 (5-subset) |
|---|---|---|
| WITH emotion | 0.272 | **0.388** |
| WITHOUT (AST) | 0.258 | 0.323 |
| dummy | 0.102 | 0.164 |

On the subset emotion (0.388) exceeds AST (0.323) by +0.064 -- larger and consistent, but
NOT significant at 5 folds (paired-t p=0.16, underpowered) -- see §11, which re-runs this
under 10x5 repeated GroupKFold and supersedes these single-run numbers. **Metric caveat:** Exact Match
and Hamming are uninformative here (the dummy scores best on both, due to balanced
over-prediction under imbalance) -- report Macro-F1 as the headline. LaTeX justification in
`latex/genre_subset.tex`. Report both full-8 and subset-5 transparently.

> *(There is no section 8. It was originally "Planned next"; that now sits
> un-numbered at the end as **Next steps**. Section numbers are stable labels —
> they are referenced from `docs/latex/` and from the thesis — so the gap is kept
> rather than renumbering 9-13.)*

## 9. Experiment 1 — emotion regression (done)

`experiments/emotion/exp_emotion_regression.py` (feature comparison) and `exp_emotion_improve.py`
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

`experiments/cross_dataset/exp_blockbuster_emotion.py`, `exp_emotion_improve.py`. VGGish is the one
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
- **Caveats:** the Eerola numbers here come from a single 5-fold run; §11 re-runs every arm
  under 10x5 repeated GroupKFold with fold-internal PCA and **supersedes them** (predicted
  0.406 -> 0.384, control 0.350 -> 0.346; ordering unchanged, "beats control" still not
  significant at 5%). Blockbuster has no ground-truth emotions to validate the predictions
  directly.

This answers RQ2b with the *predicted* pipeline on both datasets and rules out the "any 8-d
bottleneck" alternative on the primary set.

## 11. Statistical power — repeated CV + confidence intervals (done)

`experiments/evaluation/exp_statistical_power.py`, `src/evaluation/repeated.py`, results in
`results/statistical_power.json`, LaTeX in `latex/statistical_power.tex`.

**The problem this fixes.** Every comparison above rested on ONE 5-fold GroupKFold run.
Five folds cannot separate a +0.05 Macro-F1 margin from partition noise, so real,
consistent effects were being reported as "within noise" — a statement about the
*resolution of the experiment*, not about the models. Fix: **10 x 5 repeated GroupKFold**
(film→fold assignment re-randomised each repeat → 50 leakage-safe folds; sklearn 1.5 has
no shuffled GroupKFold, so `RepeatedGroupKFold` is implemented in `src/evaluation/`). All
arms score on the *same* folds → every comparison is properly paired. PCA and the emotion
regressor are now fit **inside** the training fold (stricter than the earlier full-data
PCA).

Two statistics, both standard for repeated CV:
- **95% CI over the 10 per-repeat means** (not the 50 folds — folds inside a repeat share
  training data and are not independent).
- **Nadeau & Bengio (2003) corrected resampled t-test.** A naive paired t-test over 50
  correlated folds is anti-conservative; the correction inflates the variance by the
  train/test overlap factor. Plus a nonparametric **win rate**.

### 5-genre subset (n=329, 41 films), repeated GroupKFold Macro-F1

*(All tables in §11 use sklearn's default `C=1.0`. §11b re-runs the same arms with
nested-CV-tuned `C` and is the protocol to quote; §11 is kept so the effect of tuning
is visible.)*

| Approach | Macro-F1 | 95% CI |
|---|---|---|
| **VGGish → PREDICTED emotion(11) → genre** | **0.384** | [0.375, 0.393] |
| ground-truth emotion(11) → genre (ceiling) | 0.379 | [0.370, 0.389] |
| PCA-8(VGGish) [control] | 0.346 | [0.333, 0.360] |
| VGGish-128 direct | 0.337 | [0.320, 0.353] |
| AST-768 | 0.320 | [0.305, 0.334] |
| CLAP-512 | 0.303 | [0.285, 0.320] |
| dummy | 0.166 | [0.165, 0.167] |

| Comparison | diff | 95% CI of diff | p (corrected) | p_limit | win |
|---|---|---|---|---|---|
| predicted-emotion vs PCA-8 control | +0.038 | [+0.024, +0.051] | 0.158 | 0.137 | 76% |
| predicted-emotion vs VGGish direct | +0.048 | [+0.033, +0.062] | 0.104 | 0.085 | 82% |
| predicted-emotion vs AST-768 | +0.065 | [+0.055, +0.075] | **0.021** | 0.013 | 92% |
| predicted-emotion vs ground truth | +0.005 | [−0.002, +0.012] | 0.791 | 0.782 | 50% |
| ground-truth emotion vs AST-768 | +0.060 | [+0.045, +0.075] | 0.074 | 0.058 | 82% |
| ground-truth emotion vs VGGish direct | +0.043 | [+0.028, +0.057] | 0.106 | 0.087 | 78% |

**Full 8 genres (n=346):** emotion 0.276 [0.270, 0.281] vs AST 0.240 [0.233, 0.247];
diff +0.036 [+0.028, +0.043], p=0.092, p_limit=0.074, wins 82% of folds. (The old
single-5-fold run reported p=0.62 for this same comparison.)

### What this establishes

1. **The ordering is stable, not a lucky split.** In every emotion-vs-embedding
   comparison the difference CI excludes zero and the emotion pipeline wins 76–92% of
   the 50 folds. (The predicted-vs-ground-truth CI straddles zero, which is the
   intended result there — see point 3.) This is the claim the single 5-fold run
   could not support, and it is now firm.
2. **Only emotion vs AST is significant** under the corrected test (+0.065, p=0.021).
   Predicted-emotion vs the PCA-8 bottleneck control (p=0.158) and vs direct VGGish
   (p=0.104) are **not** — direction and magnitude are supported, 5%-level significance
   is not. Report them as such.
3. **Predicted ≈ ground-truth is now a positive result, not a hedge** (+0.005, p=0.791,
   exactly 50% of folds). Audio-estimated emotions are as genre-informative as the human
   ratings — the precondition for the pipeline to work without emotion annotations.
4. **At this operating point, more repeats cannot fix the rest — only more films can.**
   `p_limit` is the p-value the corrected test converges to with *infinite* repeats (only
   the 1/n term shrinks; the train/test overlap term does not). Here p_limit is
   0.137 / 0.085 for the two non-significant comparisons, i.e. still above 0.05, and 10
   repeats is already within 0.02 of that floor, so the binding constraint is the 41 films
   — consistent with the learning curve (§7e, still rising at 100%).
   **Caveat: this conclusion is specific to the default `C`.** Once `C` is tuned (§11b)
   the predicted-emotion vs PCA-8 comparison has p_limit=0.045, *below* 0.05 — so under
   that protocol additional repeats could in principle cross the threshold. §11b reports
   it as borderline rather than resolving it by choosing a repeat count.

**Numbers that changed vs the single-run values.** predicted-emotion 0.406 → 0.384,
ground-truth 0.388 → 0.379, PCA-8 0.350 → 0.346, VGGish 0.328 → 0.337, AST 0.323 → 0.320,
8-genre emotion 0.272 → 0.276 / AST 0.258 → 0.240. The single-run 0.406 was an optimistic
partition (and used full-data PCA); **the repeated-CV numbers supersede it everywhere.**

### 11b. Hyperparameter tuning (nested CV) + CLAP baseline

Section A above uses sklearn's default `C=1.0`. That default turned out to be the **worst**
setting for every feature set (all of them prefer `C<=0.1`) — with n≈330 clips against
128–768 embedding dimensions the model was badly under-regularised. Tuning `C` on the test
folds would be cheating, so Section B chooses it by an **inner GroupKFold inside each
training fold** (nested CV, `select_logreg_C` in `src/models/classifiers.py`). The inner loop is grouped by
film too — otherwise C gets tuned against film-identity leakage and comes out too weak.

**CLAP** (extracted long ago, never used) is now a third embedding baseline. It is the
weakest of the three, but that is exactly its value: the direct-audio baseline is no
longer a single model a reader can dismiss as a poor choice.

| Arm | A: default C=1.0 | B: nested-CV tuned | Δ | C picked |
|---|---|---|---|---|
| VGGish → PREDICTED emotion(11) | 0.384 | **0.394** [0.389, 0.400] | +0.010 | 0.003 |
| ground-truth emotion(11) [ceiling] | 0.379 | **0.397** [0.388, 0.407] | +0.018 | 0.003 |
| VGGish-128 direct | 0.337 | 0.350 [0.335, 0.365] | +0.013 | 0.003 |
| AST-768 | 0.320 | 0.345 [0.328, 0.361] | +0.025 | 0.003 |
| PCA-8(VGGish) [control] | 0.346 | 0.343 [0.330, 0.355] | −0.004 | 0.1 |
| CLAP-512 | 0.303 | 0.335 [0.323, 0.347] | +0.033 | 0.003 |
| dummy | 0.166 | 0.166 | ±0.000 | — |

Tuned paired comparisons (5-genre subset):

| Comparison | diff | 95% CI | p | p_limit | win |
|---|---|---|---|---|---|
| predicted-emotion vs PCA-8 control | +0.052 | [+0.037, +0.066] | 0.060 | 0.045 | 88% |
| predicted-emotion vs AST-768 | +0.050 | [+0.034, +0.065] | 0.078 | 0.061 | 82% |
| predicted-emotion vs VGGish direct | +0.044 | [+0.030, +0.059] | 0.104 | 0.085 | 80% |
| predicted-emotion vs ground truth | −0.003 | [−0.012, +0.007] | 0.855 | 0.848 | 54% |
| ground-truth emotion vs CLAP-512 | +0.062 | [+0.046, +0.077] | **0.036** | 0.025 | 86% |
| ground-truth emotion vs AST-768 | +0.052 | [+0.036, +0.069] | 0.072 | 0.056 | 84% |
| AST-768 vs CLAP-512 | +0.010 | [−0.005, +0.024] | 0.752 | 0.741 | 62% |

**Full 8 genres, tuned:** emotion **0.300** [0.293, 0.306] vs AST **0.257** [0.251, 0.263];
diff +0.043, **p=0.031 (significant)**, p_limit=0.021, wins 92% of folds.

### What tuning changes

1. **The 8-genre claim becomes significant.** emotion vs AST goes p=0.62 (old single
   5-fold) → 0.092 (repeated) → **0.031 (repeated + tuned)**. This is now a defensible
   headline result rather than a hedge.
2. **Tuning helps the baselines more than emotion** (CLAP +0.033, AST +0.025 vs emotion
   +0.018) — as expected, high-dimensional embeddings suffer most from under-regularisation.
   The 5-genre emotion-vs-AST gap *narrows* slightly (0.060 → 0.052). Conclusions are
   unchanged, which is the point of checking: **the result is not an artefact of untuned
   baselines.**
3. **The three modern embeddings are statistically indistinguishable from each other**
   (AST 0.345, VGGish 0.350, CLAP 0.335; AST vs CLAP p=0.752). Emotion features beat all
   three. This is a much more robust framing than "emotion beats AST".
4. **Predicted ≈ ground-truth still holds** (−0.003, p=0.855) — the tuned pipeline does not
   depend on human emotion ratings.
5. **Borderline, do not overclaim:** predicted-emotion vs the PCA-8 control is p=0.060 with
   p_limit=0.045. Because p_limit sits just below 0.05, more repeats *would* eventually push
   it under the threshold — but that is arbitrating a conclusion by choosing the repeat
   count, so it is reported as **borderline**, not significant. The honest statement is that
   the emotion bottleneck beats a generic 8-d bottleneck consistently (88% of folds) at
   roughly the 5–6% level.

Note: `C=0.003` is the grid's lower edge for most arms, so the optimum may lie below it;
widening the grid is a loose end, though the flatness between 0.003 and 0.01 in the manual
sweep suggests little is left on the table.

## 12. Defects found during review (status noted per item)

Discovered while verifying paths after the `experiments/` reorganisation. **Neither affects
any result reported in this log** — both concern artifacts that nothing reads — but both are
traps for anyone reading the repository later.

1. **[FIXED] `set1_ast.npy` disagreed with the per-clip cache.** `features/extract_features.py`
   used to write `<set>_<model>.npy`, `<set>_numbers.npy` and `<set>_durations.csv`, but
   **no code read any of them**: every experiment calls `assemble_from_cache()`, which reads
   the per-clip embedding files. The committed `set1_ast.npy` differed from the
   stack of per-clip caches (max abs diff 3.35), i.e. it is stale — written from an earlier
   extraction run and never refreshed when the per-clip caches were regenerated. Re-running
   the script rewrites it and produces a large git diff on a file no experiment uses.
   *Fixed 2026-08-24:* all ten write-only artifacts deleted and no longer written;
   `data/processed/` reorganised to `embeddings/<model>/<set>/` so the per-clip caches are
   unambiguously the single source of truth.

2. **[FIXED] Clip durations depended on whether the cache was warm.** In `extract_embeddings`, a cache
   *miss* records `len(waveform)/sampling_rate` (decoded length) while a cache *hit* records
   `librosa.get_duration(path=...)` (mp3 header). These differ systematically by ~0.089 s per
   clip (encoder delay/padding). The committed CSV holds decoded values; any re-run now
   produces header values. The clip-length numbers quoted in §5 come from
   `clip_length_analysis.py` (`durations_from_audio`, computed fresh) and are unaffected.
   *Fixed 2026-08-24:* the durations CSV is no longer written at all, so the two code
   paths can no longer disagree in a persisted file. Clip-length figures come from
   `clip_length_analysis.py`, which computes durations fresh from the audio.

3. **[FIXED] VGGish and MIR caches were unreproducible.** `VggishEmbedder` and
   `MirEmbedder` existed and were exported but were instantiated by no script:
   `extract_features.py` registered only `ast` and `clap`. Four experiments *consume*
   those caches — including the §10 cross-dataset bridge and the §11 predicted-emotion
   arm, i.e. the RQ2b answer — so the central result could not have been rebuilt from the
   repository if the cache were lost. Both are now registered
   (`--model vggish`, `--model mir`) and verified to reproduce the cached matrices
   (360x128 and 360x103).
4. **[FIXED] The logistic regression had four separate definitions.**
   `exp_threshold_fix` and `exp_emotion_genre` constructed their own `LogisticRegression`,
   so changes to `build_classifier` (such as the `C` parameter added in §11b) never
   reached them. Both now call the single shared factory
   `src.models.build_binary_logreg`, and `_SingleClassSafe` gained `predict_proba` so the
   threshold analysis can use the shared wrapper. (`exp5_target` and the Exp-5 half of
   `exp_clip_length` keep their own estimator deliberately: those solve a *single-label
   12-class balanced* task, where the multi-label binary-relevance model does not apply.)
5. **[OPEN] Most experiments still print results instead of writing them.** Only
   `exp_statistical_power`, `exp_model_search` write JSON to `results/`. This is the
   mechanism that let §7d's unreproducible table sit in this log undetected.
6. **[OPEN] Short debug runs overwrite the authoritative results files.**
   `exp_statistical_power.py --repeats 2` silently replaces `results/statistical_power.json`.

## 13. Experiment 4 — rating reliability, Set 1 vs Set 2 (done)

`experiments/features/exp4_rating_reliability.py`; alignment helpers in
`src/features/loader.py` (`align_sets`, `repeat_pairs`).

Set 2 re-rates 110 excerpts drawn from Set 1 with a **different listener panel**, giving
two independent measurements of the same music. That yields the one quantity Set 1 alone
cannot provide: **how much of the emotion signal is real and how much is rating noise** —
which bounds how well any audio model could predict these ratings.

### The clip correspondence (a trap worth documenting)

The two CSVs number clips **independently**. The correspondence is Set 2's **`link`**
column, which holds the Set 1 `number`. Joining on `number` instead produces a table that
looks perfectly valid and is meaningless: mean rating correlation 0.157, soundtrack labels
agreeing on 2.7% of rows.

The `link` mapping was verified **against the audio**, not just the metadata: full-lag
normalised cross-correlation between each Set 2 clip and its linked Set 1 clip has median
**0.964** (10th pct 0.792) against median **0.027** for randomly paired clips; 109 of 110
rows sit far above the null's 99th percentile. The single failure (Set 2 #17) is dropped.
Seven Set 1 clips are linked by two Set 2 rows each — **repeat trials**, the same excerpt
presented twice to the same panel — which give a separate within-panel estimate. Averaging
those leaves **102 matched excerpts from 38 soundtracks**.

> **Do not confuse this 102 with the other one.** §1 reports Set 2 cleaning to *also*
> 102 clips (110 minus 6 with no IMDb genre minus 2 with no primary genre, spanning 36
> soundtracks). That is a pure numerical coincidence: this 102 comes from a different
> filter (110 minus the 1 audio mismatch, then averaging 7 repeat pairs) and is a
> *different set of clips* — the two overlap in 101 rows, with 1 unique to the cleaned
> set and 8 unique to the aligned set. The soundtrack counts differ (36 vs 38) because
> the cleaned set uses Set 2's own labels while the aligned set inherits Set 1's.

### Q1 — between-panel agreement (n=102)

| Emotion | r | ICC(C,1) consistency | ICC(A,1) agreement | bias (Set2−Set1) |
|---|---|---|---|---|
| valence | 0.876 | 0.871 | 0.707 | +1.13 |
| energy | 0.905 | 0.859 | 0.618 | +1.40 |
| tension | 0.945 | 0.913 | 0.736 | +1.23 |
| anger | 0.922 | 0.919 | 0.918 | +0.10 |
| fear | 0.933 | 0.931 | 0.929 | +0.17 |
| happy | 0.930 | 0.926 | 0.926 | +0.06 |
| sad | 0.877 | 0.869 | 0.822 | +0.60 |
| tender | 0.908 | 0.891 | 0.886 | +0.21 |
| **MEAN** | **0.912** | **0.897** | **0.818** | +0.61 |

High r and ICC(C,1) with a clearly lower ICC(A,1) means the panels **rank excerpts the
same way but use the scale differently** — and the gap is confined to the three
dimensional scales.

### Q1b — within-panel noise (7 repeat trials)

Mean r = **0.988**, ICC(A,1) = **0.987**, bias ≈ 0 on every emotion. The excerpts are
therefore *not* intrinsically ambiguous: presented twice to the same panel they get
essentially the same ratings. **The entire between-panel gap is a panel/context effect,
not clip ambiguity.** Caveat: n=7, and a repeat within one session is not fully
independent (listeners may recall the excerpt), so this is an upper bound on
within-panel reliability — treat as indicative.

### Q2 — the disagreement is a scale shift, not disagreement about content

| | mean shift | Cohen's d | p (paired t) |
|---|---|---|---|
| **dimensional** (valence, energy, tension) | **+1.25** | 0.68–0.89 | 1e-24 … 1e-31 |
| **discrete** (anger, fear, happy, tender) | **+0.23** | 0.03–0.12 | mostly n.s. |
| sad (discrete, exception) | +0.60 | 0.34 | 1e-09 |

Set 2's panel used the *bipolar* scales about 1.25 points higher while agreeing almost
exactly on *how much anger/fear/happiness* they heard. After independently z-scoring each
set — which is what `StandardScaler` inside the classifier pipeline does — the mean
absolute difference falls to **0.305 SD**. So the shift is an offset/scaling of the
response scale that the modelling pipeline already removes; it is a reporting caveat, not
a threat to the results.

### Q3 — the reliability ceiling (the headline)

A predictor cannot correlate with a noisy target better than the target correlates with
itself. Treating the between-panel consistency as a parallel-forms reliability `r_xx`, the
maximum attainable R² is `r_xx`. (Consistency rather than absolute agreement is the right
reliability here: the model is scored by R² against *one* set's ratings, and a constant
offset between panels does not reduce the variance explainable within a set.)

| | mean R² |
|---|---|
| Ceiling implied by rating reliability | **0.897** |
| Achieved (RandomForest on AST, §9) | 0.560 |
| **Fraction of attainable variance reached** | **62%** |

Two conclusions, and the second is the uncomfortable one:

1. **R² ≈ 1.0 was never achievable** — the honest reference point for the emotion
   regression is ~0.90, not 1.0. Reporting 0.56 against a ceiling of 0.90 is a materially
   different claim from reporting it against 1.0.
2. **Rating noise does NOT explain the model's shortfall.** At 62% of attainable variance
   there is real headroom left in the audio representation. This *closes off* the
   convenient excuse that the ratings are too noisy to predict — the limitation is the
   features/model, not the labels. It is consistent with §7e (learning curves still
   rising) and argues the same way: more/better data and representation, not a different
   classifier.

### Q4 — beauty / liking (Set 2 only, descriptive)

`beauty` correlates −0.81 with tension, −0.74 with fear, +0.78 with tender, +0.63 with
valence; `liking` follows the same pattern more weakly. Perceived beauty in film music
tracks calm positive affect. Not used as a model input anywhere.

## Next steps

- **Content chapters** (#5) — student; the LaTeX snippets in `docs/latex/` are ready to fold in.
- **More films** — §11 shows the remaining non-significance is corpus-bound, not
  resampling-bound (p_limit > 0.05 with infinite repeats), and §7e shows both learning
  curves still rising. Extending the corpus is the only lever left on the headline claims.
- **Remaining defects (§12, items 1-2 and 5-6 still open):** `extract_features.py` writes
  three artifacts nothing reads, one of which (`set1_ast.npy`) disagrees with the per-clip
  cache the experiments actually use; clip durations differ by ~0.09 s depending on whether
  the cache was warm; most experiments still only print their results; and a short debug
  run of `exp_statistical_power` overwrites the authoritative results file. None affects a
  reported result. Items 3-4 (VGGish/MIR reproducibility, the duplicated logistic-regression
  definitions) are fixed.
- **Tests** — there are none beyond the loader's count assertions. Three cheap ones would
  pay for themselves: loader counts, `align_sets` invariants (102 rows, link integrity),
  and `RepeatedGroupKFold` never splitting a film across folds.
- **Optional:** Blockbuster full-140-MIR / mean+std pooling; emotion-regressor tuning
  (only the classifier's C is tuned so far); widen the C grid below 0.003.
  ~~Set 1 vs Set 2 diff (Exp 4)~~ — done, see §13.

## Repository map

- `src/features/` — data loading/cleaning; AST, CLAP, VGGish, MIR extractors; Blockbuster loader
- `src/models/` — multi-label classifiers; `build_binary_logreg` (the single shared
  definition of the thesis's logistic regression); `select_logreg_C` (inner-CV
  hyperparameter selection for nested CV)
- `src/evaluation/` — metrics + cross-validated scoring; `repeated.py` (RepeatedGroupKFold,
  Nadeau-Bengio corrected t-test, per-repeat CIs)
- `experiments/` — runnable experiments, grouped by pipeline stage (index +
  per-script purpose in `experiments/README.md`). Run from the repository root.
  - `features/` — `verify_data`, `extract_features`, `clip_length_analysis`,
    `exp4_rating_reliability`
  - `emotion/` — `exp_emotion_regression`, `exp_emotion_improve`
  - `genre/` — `exp5_target`, `exp_genre`, `exp_genre_subset`, `exp_emotion_genre`
  - `diagnostics/` — `exp_error_analysis`, `exp_threshold_fix`, `exp_model_search`,
    `exp_learning_curve`, `exp_clip_length`
  - `cross_dataset/` — `exp_blockbuster`, `exp_blockbuster_emotion`
  - `evaluation/` — `exp_statistical_power` (supersedes the single-run numbers from
    `genre` / `cross_dataset`)
- `docs/` — this log; `docs/literature/` (literature reviews); `docs/latex/` (LaTeX snippets)
- `data/processed/Eerola_DB/embeddings/<model>/<set>/` — per-clip embedding caches, one
  folder per model (`ast`, `clap`, `vggish`, `mir`, `ast_windows`); the single source of
  truth, read via `assemble_from_cache` (layout documented in `src/config.py`).
  Download instructions for the raw data are in `data/raw/README.md`.
- `results/` — machine-readable experiment output (`statistical_power.json`,
  `model_search.json`: every per-fold score)
