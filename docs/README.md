# Implementation Progress Log

The **full technical log**: every experiment, every result, every decision with its
justification. Kept up to date as work proceeds.

> **Looking for the short version?** `current_state.md` is the supervisor briefing — the
> supervisor's notes and their status, the definitions and metrics explained, and the
> headline results, in about a tenth of the length. `results/README.md` explains what the
> machine-readable results in `results/` contain and how to render them.

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
  weakest of the three spectrogram embeddings, see §11b), **VGGish** (128-dim, postprocessed — the shared feature with Blockbuster,
  see §10 and §15), **librosa-MIR** (103-dim hand-crafted MFCC/chroma/spectral, a classic-MER
  baseline), and **wav2vec 2.0** (768-dim, the only representation that reads the **raw
  waveform** rather than a spectrogram — see §16). All five extractors share one interface
  and cache to `data/processed/`.
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

> **Do NOT read the last row against the two above it.** They are different tasks and the
> numbers are not comparable: 8 genres vs 6, one ~10 s clip per row vs one whole film per
> row, and random-guess floors of 0.233 vs 0.292. Putting 0.272 next to 0.582 invites
> exactly the conclusion the supervisor warned about ("the classifier must have the same
> genre classes in both datasets to be fair"). The comparable pairing is in **§15.2**,
> where both corpora are scored in the *same* 6-genre space: Eerola 0.315 vs Blockbuster
> 0.620, each against its own floor. Blockbuster is the easier corpus — film-level bags
> average out per-cue noise, and its genres have far higher base rates.
>
> The MFCC figure is also superseded: it uses only the 78 MFCC-family columns of Ma et
> al.'s 140 MIR features, because the supervisor's question was specifically about MFCC.
> Against the **full** MIR set the gap largely disappears — see §17.

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
  - **Strongest evidence: cross-corpus transfer (§15).** Trained on Eerola and applied
    zero-shot to Blockbuster, the predicted-emotion pipeline beats the direct embedding
    by **+0.086 macro-F1, 95% CI [+0.017, +0.153], p=0.012** (paired bootstrap over the
    110 target films). The in-domain margin roughly doubles under domain shift and
    becomes unambiguous — which is where an interpretable intermediate should help most,
    since an emotion value means the same thing in both corpora and an embedding
    dimension does not. The PCA-8 bottleneck control gains only +0.019 (p=0.423) in the
    same setting, so the advantage is not dimensionality reduction.
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

*(§16 extends this table with CLAP-512 at 0.561 and the raw-waveform wav2vec2-768 at
0.323, and adds the % -of-reliability-ceiling column.)*

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

> **SUPERSEDED by §17.** These are a single 5-fold run at default `C`, with the emotion
> bridge applied at film level. Under 10x5 repeated KFold with nested-CV-tuned `C` and
> per-cue bridging the ordering changes: emotion 0.616 > VGGish 0.593 > PCA-8 0.587.
> Quote §17.

Conclusions (corrected from the earlier Ridge-based, over-tempered version):
- **On the primary Eerola set the emotion bottleneck is genuinely best:** predicted-emotion
  (0.406) beats the PCA-8 control (0.350) and the direct embedding (0.328), and matches the
  ground-truth ceiling (0.388) — so it is *not* merely an interpretable relabelling of a
  generic compression. On the easier Blockbuster it essentially ties PCA-8 (0.565 vs 0.559)
  — but see §17.3: at the correct (per-cue) granularity and under the proper protocol the
  Blockbuster ordering becomes emotion 0.616 > VGGish 0.593 > PCA-8 0.587, still a tie
  statistically but no longer behind.
- **Predicted ~ ground-truth** (0.406 vs 0.388): RF-predicted emotions are as
  genre-informative as the human ratings (likely denoising the per-clip ratings).
- **Fusion (emotion + VGGish) hurts** (0.313) — the two are not complementary.
- **Face validity (Blockbuster, cross-dataset):** horror->fear (+0.99), romance->tender
  (+1.05), comedy->happy, drama->tender, action->anger — textbook-correct; the transferred
  emotions are meaningful, not noise. **§17.5 turns this from an eyeball check into a
  measured replication:** the full Eerola Cohen's-d signature matrix correlates r=0.84 with
  the Blockbuster one across 48 genre x emotion cells, 92% sign agreement.
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

## 14. Second round of supervisor comments (status)

| # | Item (as given) | Status |
|---|-----------------|--------|
| A | Test on Blockbuster **without training on it** -- train on Eerola, compare against the paper's own results | done -- section 15; extended in section 17 (proper protocol, cue level, full MIR, signature replication) |
| B | Explain the evaluation process and which metrics produced the feature-extraction accuracy | done -- `latex/evaluation_protocol.tex` |
| C | Try a **waveform** extractor, not only spectrogram | done -- section 16 |
| D | The classifier must have the **same genre classes in both datasets** to be fair | done -- section 15.1 (prerequisite for A) |
| E | Emotion-genre relationship in the literature, **including in music** | done -- `latex/emotion_genre_relationship.tex`, new "Evidence from music research" block |

## 15. Cross-dataset transfer: one shared genre space, zero-shot (items D and A)

`experiments/cross_dataset/exp_zero_shot.py`, results in `results/zero_shot.json`.

### 15.1 The shared label space (item D)

Comparing the two corpora previously meant comparing an 8-genre task (Eerola) with a
6-genre task (Blockbuster) -- different classes, so different difficulty, so no fair
comparison and no possibility of moving a trained model between them. Both are now
relabelled into **one identical 6-genre space**, `config.SHARED_GENRES` =
{action, drama, comedy, sci-fi, romance, horror}.

That list is not our choice: it is Ma et al.'s six reduced genres, and **their reduction
rule was recovered from their published `film_genre_master_list.csv` rather than guessed.**
It is the identity rule -- a film carries reduced genre *g* iff *g* appears in its raw IMDb
genre list -- verified on **110/110 films with zero mismatches**. That is exactly the
"any-present" rule already used for the 8-genre Eerola target, so the two corpora enter the
same space without any dataset-specific heuristic.

| genre | Eerola clips | Blockbuster films |
|---|---|---|
| action | 103 | 55 |
| drama | 239 | 44 |
| comedy | 40 | 37 |
| sci-fi | 19 | 36 |
| romance | 51 | 13 |
| horror | 28 | 11 |
| **total** | **319 clips / 41 films** | **110 films** |

Note this is *not* the 346-clip 8-genre set: Romance and Sci-Fi are in the shared space but
not in `PRIMARY_GENRES`, while Crime, Adventure, Biography and Documentary exist in Eerola
but not in Blockbuster. VGGish is the one representation both corpora produce identically,
and the two spaces are numerically compatible (both standard 0-255 post-processed VGGish;
per-dimension means correlate r=0.97), so a model fitted on one can be applied to the other.

### 15.2 Reference points -- do we reproduce the paper? (item A, part 1)

A transfer score is meaningless without knowing what in-domain performance looks like.
Ma et al.'s protocol (leave-one-film-out over the 110 films, macro-averaged P/R/F1) was
therefore re-run with our own simple pipeline (mean-pooled VGGish -> Binary-Relevance
logistic regression):

| | precision | recall | macro-F1 |
|---|---|---|---|
| Ma et al., VGGish + single-attention NN (their best) | 0.60 | 0.73 | **0.65** |
| Ma et al., VGGish + avg-pooling NN | 0.55 | 0.78 | 0.62 |
| Ma et al., VGGish + kNN Simple-MI | 0.64 | 0.59 | 0.61 |
| Ma et al., MIR + SVM Simple-MI | 0.60 | 0.52 | 0.56 |
| **ours: VGGish -> LogReg, LOO (reimplementation)** | 0.569 | 0.705 | **0.620** |
| Ma et al., random-guess baseline | 0.32 | 0.32 | 0.32 |
| ours: random-guess baseline (recomputed) | 0.294 | 0.293 | 0.292 |
| Ma et al., plurality-label baseline | 0.14 | 0.34 | 0.19 |
| ours: plurality-label baseline (recomputed) | 0.138 | 0.333 | 0.193 |

**Our reimplementation lands inside their published VGGish range (0.620 vs 0.61-0.65) and
both baselines reproduce to within 0.03.** The pipeline is therefore validated against an
external result, which is what makes the transfer number below interpretable: a low
transfer score cannot be blamed on a weak classifier.

> **Baseline caveat worth carrying into the thesis.** Ma et al.'s "random guess" draws each
> genre independently at its base rate and scores macro-F1 0.32; our usual `most_frequent`
> dummy predicts one fixed label set and scores 0.19. These are different floors. Quoting
> our dummy against their random-guess would overstate the margin by >0.1 F1. Both are now
> computed on both corpora.

For orientation, Eerola in-domain on the same six classes (GroupKFold by film) scores
macro-F1 **0.315** against its own random-guess floor of 0.250. Blockbuster is simply an
easier corpus (film-level bags, higher base rates), so scores must be read against each
corpus's own floor, never against each other.

### 15.3 Zero-shot: trained on Eerola, tested on Blockbuster (item A, part 2)

Every arm is fitted on Eerola alone and predicts all 110 Blockbuster films. **No
Blockbuster label is seen during training.** Where the genre stage consumes predicted
emotions, those are produced *out-of-fold* on Eerola, so it is trained on the same noisy
quantity it meets at test time rather than on near-perfect in-sample predictions.

| approach (trained on Eerola only) | precision | recall | macro-F1 |
|---|---|---|---|
| **VGGish -> predicted emotion(11), per-cue -> genre** | 0.488 | 0.651 | **0.511** |
| VGGish -> predicted emotion(11), film-level -> genre | 0.450 | 0.637 | 0.482 |
| emotion(11) -> genre, genre stage fitted on ratings | 0.445 | 0.474 | 0.434 |
| PCA-8(VGGish) [bottleneck control] | 0.534 | 0.467 | 0.421 |
| VGGish-128 direct | 0.576 | 0.442 | 0.407 |
| random guess (class frequencies) | 0.294 | 0.293 | 0.292 |
| *(in-domain reference from 15.2)* | *0.569* | *0.705* | *0.620* |

**Granularity of the bridge matters** (added after §17). A Blockbuster *cue* is a single
piece of film music of median 19 s — the same object as an Eerola clip — whereas a
film-level vector averages ~39 of them. Predicting emotion **per cue and pooling the
emotions afterwards** keeps the regressor on its training unit and is worth +0.029 here
(and +0.059 in-domain, p=0.020, §17.2). The per-cue row is the one to quote.

Because the split is single, the margins are quantified by a **paired bootstrap over the
110 target films** (2000 resamples, both arms re-scored on each resample) rather than
asserted:

| comparison | diff | 95% CI | p |
|---|---|---|---|
| **predicted-emotion (per-cue) vs VGGish direct** | **+0.106** | [+0.015, +0.186] | **0.018** |
| **predicted-emotion (per-cue) vs PCA-8 control** | **+0.092** | [+0.009, +0.171] | **0.032** |
| predicted-emotion (film-level) vs VGGish direct | +0.077 | [-0.007, +0.152] | 0.066 |
| PCA-8 control vs VGGish direct | +0.014 | [-0.032, +0.062] | 0.572 |

> **Which scaling regime these come from, and why it matters.** Only the *embedding* arms
> move between the two regimes; the emotion arms are in rating units and are identical in
> both. An earlier version of this table paired the strict-regime scores above with a
> bootstrap computed in the target-standardised regime, where the direct embedding is
> weaker (0.394 vs 0.407) — which flattered the margin by ~0.013. Everything here is now
> **strict (source scaler)**, the regime in which the baseline does *better*, i.e. the
> conservative choice. Under target-standardisation the same comparisons give +0.116
> (p<0.001) and +0.097 (p=0.006); both regimes are in `results/zero_shot.json`.

**The control comparison is significant either way.** With film-level bridging, emotion vs
the generic 8-d bottleneck was +0.068 at p=0.075 — borderline, and reported as such. At the
correct granularity it is **+0.092 at p=0.032** under the conservative regime. This is the
cleanest evidence in the project that the *emotion* bottleneck, not merely *a* bottleneck,
is what helps.

### 15.4 What this establishes

1. **Genre knowledge transfers across corpora at all.** 0.482 zero-shot against a 0.292
   random floor, and 78% of the 0.620 reached with in-domain training -- despite the
   training rows being ~10 s clips and the test rows whole films' cues. This is the first
   evidence in the project that the models learn something about genre rather than
   something about this corpus.
2. **The emotion bottleneck transfers *better* than the raw embedding** (+0.106,
   p=0.018), and it is the only arm that clears the bootstrap CI. In-domain, emotion vs
   embedding was a +0.04-0.05 effect hovering at p~0.03-0.10 (sections 11, 11b); under
   domain shift the margin roughly doubles and becomes significant. This is the strongest form of the RQ2b
   claim obtained so far, and it arrives in the setting where an interpretable intermediate
   should help most: "fear = 6.2" means the same thing in both corpora, whereas VGGish
   dimension 87 does not.
3. **The mechanism is visible in the scaling ablation.** Re-standardising each corpus with
   its own statistics moves the direct embedding (0.407 -> 0.394) and the PCA control
   (0.421 -> 0.413) but leaves the emotion arm **exactly unchanged at 0.482** -- it is
   expressed in rating units, so the domain shift is absorbed by construction rather than
   by a scaling trick.
4. **The bottleneck is not merely dimensionality reduction.** PCA-8 of the same embedding
   gains only +0.014 over the raw 128-d features (p=0.572), i.e. compressing to 8 dimensions
   is worth almost nothing; compressing to 8 *emotions* is worth +0.106. The gap between
   them is **+0.092, p=0.032 — significant**, unlike every in-domain version of this
   comparison.
5. **Training the genre stage on *predicted* emotions beats training it on ratings**
   (0.511 vs 0.434) when both are tested on predicted emotions. In-domain the two were
   equivalent (§11b); under transfer, matching the training distribution to the test
   distribution matters. This validates the out-of-fold design rather than merely
   permitting it — and it still means no emotion annotation is needed at deployment.
6. **Reverse direction works too but weakly:** Blockbuster -> Eerola gives 0.323 vs a 0.250
   floor. Expected -- 110 film-level rows is a much smaller and coarser training set.

Per-genre F1 for the per-cue emotion arm: action 0.822, sci-fi 0.607, drama 0.602,
romance 0.438, comedy 0.364, horror 0.237. The failure on horror (n=11 films) and comedy is notable
because those are the genres with the *strongest* emotional signatures in-domain (section
7b); the likely cause is that the emotion arm over-predicts (recall 0.637 vs precision
0.450), which costs most on the rarest classes.

**Literature check.** This degradation pattern is exactly what Eerola (2011, *Are the
emotions expressed in music genre-specific?*) predicts: models of musical emotion
generalise substantially worse across genres/repertoires than within them (valence
43% -> 16% of variance; arousal 62% -> 43%). Our 0.620 -> 0.482 is the same phenomenon at
the genre-prediction level, and it is now cited in
`latex/emotion_genre_relationship.tex`.

## 16. Waveform vs spectrogram representations (item C)

`experiments/features/exp_waveform_vs_spectrogram.py` and
`experiments/features/exp_w2v_layer_sweep.py`; results in
`results/waveform_vs_spectrogram.json` and `results/w2v_layer_sweep.json`.

### 16.1 Why this was a real gap

Every representation used up to this point begins by turning the audio into a
time-frequency image: AST and CLAP from a log-mel spectrogram, VGGish from log-mel
patches, the MIR baseline from STFT descriptors. So the thesis's central comparison had
only ever been run against *spectrogram* features, and a reader could reasonably answer
"emotion features match or beat learned embeddings" with "you only tried mel front-ends".

**wav2vec 2.0** (`facebook/wav2vec2-base`, self-supervised, not an ASR fine-tune) removes
that confound: its feature encoder is a stack of 1-D convolutions over the raw 16 kHz
sample sequence, with no spectral transform anywhere. Same 10.24 s excerpt, same
mean-pooling, same 768 dimensions as AST -- only the front-end differs.

**Confound that remains, and must be stated in the thesis:** wav2vec 2.0 was pretrained on
read speech (LibriSpeech), whereas AST and VGGish were pretrained on AudioSet, which
contains music. The comparison therefore varies input domain *and* pretraining corpus. A
wav2vec 2.0 loss cannot be attributed to the waveform front-end alone.

### 16.2 Required control: which layer to pool

Pooling the final layer -- the obvious default -- gives a mean R^2 of **0.161**, and
reporting that would have been a mistake. A self-supervised model's last layers specialise
toward its own pretraining objective and transfer poorly to tasks it was never trained
for. The sweep over all 13 layers (same protocol: RandomForest, GroupKFold, mean R^2):

| layer | 0 (CNN out) | 1 | **2** | 3 | 4 | 5 | 6 | 8 | 10 | 12 (final) |
|---|---|---|---|---|---|---|---|---|---|---|
| mean R^2 | 0.292 | 0.322 | **0.324** | 0.306 | 0.301 | 0.278 | 0.235 | 0.164 | 0.187 | 0.161 |

Monotone decline after layer 2, and **the layer choice alone is worth a factor of two** --
larger than the effect the thesis is trying to measure. Layer 2 is therefore used
throughout. Note it is chosen on the Stage-1 emotion task and then held fixed; it is not
tuned against any genre result.

### 16.3 Stage 1 -- emotion regression (the feature-extraction accuracy)

RandomForest, GroupKFold by film, 360 clips, mean R^2 over the 8 emotions:

| representation | input domain | dim | mean R^2 | RMSE | % of the 0.897 reliability ceiling |
|---|---|---|---|---|---|
| CLAP-512 | log-mel spectrogram | 512 | **0.561** | 1.22 | 62% |
| AST-768 | log-mel spectrogram | 768 | 0.560 | 1.22 | 62% |
| VGGish-128 | log-mel spectrogram | 128 | 0.558 | 1.23 | 62% |
| **MusiCNN-200** | log-mel spectrogram | 200 | **0.541** | 1.25 | 60% |
| MIR-103 | STFT descriptors | 103 | 0.490 | 1.31 | 55% |
| **wav2vec2-768** | **RAW WAVEFORM** | 768 | **0.323** | 1.52 | 36% |

**The waveform representation is clearly worse at recovering emotion** -- not only behind
all three spectrogram embeddings but behind the 103-dimensional hand-crafted MIR baseline.
The gap is uniform across emotions (e.g. fear 0.375 vs 0.601 for AST) rather than confined
to any one.

### 16.4 Stage 2 -- genre classification, and the interesting part

5-genre subset, 5x5 repeated GroupKFold, nested-CV-tuned C, Macro-F1:

| arm | Macro-F1 | 95% CI |
|---|---|---|
| emotion(11), ground truth | **0.391** | [0.376, 0.407] |
| **wav2vec2 -> predicted emotion(11) -> genre** | **0.376** | [0.367, 0.384] |
| VGGish-128 direct | 0.342 | [0.322, 0.363] |
| MusiCNN-200 direct | 0.340 | [0.323, 0.357] |
| MIR-103 direct | 0.339 | [0.322, 0.355] |
| AST-768 direct | 0.338 | [0.306, 0.371] |
| CLAP-512 direct | 0.335 | [0.314, 0.355] |
| **wav2vec2-768 direct** | **0.300** | [0.286, 0.313] |

| comparison | diff | p (Nadeau-Bengio) | win |
|---|---|---|---|
| ground-truth emotion vs wav2vec2 direct | +0.092 | **<0.001** | 100% |
| **wav2vec2 -> predicted emotion vs wav2vec2 direct** | **+0.076** | **0.003** | **100%** |
| ground-truth emotion vs wav2vec2 -> predicted emotion | +0.016 | 0.304 | 68% |
| VGGish direct vs wav2vec2 direct | +0.043 | 0.118 | 80% |
| AST direct vs wav2vec2 direct | +0.039 | 0.134 | 72% |
| CLAP direct vs wav2vec2 direct | +0.035 | 0.190 | 76% |

### 16.4b MusiCNN — the music-pretrained representation

Added after the supervisor asked whether a music-specific model would do better
(`experiments/features/extract_musicnn.py`, run in a separate TensorFlow environment;
see `current_state.md` section 7b for the dependency archaeology). MusiCNN is pretrained on
Million Song Dataset tagging, so its domain match is the best of any model here.

It scores **R^2 = 0.541** on emotion regression and **0.340** on genre: above MIR and far
above wav2vec 2.0, but below AST/CLAP/VGGish on Stage 1 and statistically inseparable from
all of them on Stage 2. Music-specific pretraining does not buy an advantage on this task.

This strengthens rather than weakens the thesis. The direct-audio baseline is now five
models spanning four pretraining regimes -- general audio events (AST, VGGish),
audio-text (CLAP), music tagging (MusiCNN) and none at all (hand-crafted MIR) -- and they
all land between 0.335 and 0.342, while the emotion features reach 0.391. "Emotion beats
the audio baseline" can no longer be dismissed as a poorly chosen baseline.

### 16.5 What this establishes

1. **The answer to the supervisor's question is: yes, tried, and the waveform front-end is
   worse.** wav2vec 2.0 is the weakest representation on both stages -- with the honest
   caveat of 16.1 that its speech pretraining is confounded with its input domain. The
   thesis's use of a spectrogram front-end is now an empirically supported choice rather
   than an unexamined default.
2. **The emotion bottleneck rescues it -- and this is the strongest in-domain evidence for
   RQ2b so far.** Routed through predicted emotions, the *worst* representation
   (0.300, last place) jumps to **0.376, second place**, ahead of every direct
   spectrogram embedding and statistically indistinguishable from the ground-truth
   emotion ceiling (+0.016, p=0.304). The gain over its own direct use is +0.076 at
   p=0.003, winning **100% of 25 folds**. The emotion intermediate is therefore not a
   property of one particular embedding: it helps most exactly where the raw
   representation is weakest.
3. **Emotion regression discriminates representations far more sharply than genre
   classification does.** The Stage-1 spread is huge (0.32 vs 0.56, a factor of 1.7) while
   the same representations' direct genre scores differ by only 0.035-0.043 and none of
   those differences reaches significance (p=0.12-0.19). Two consequences: (a) mean R^2 is
   the right metric for ranking feature extractors, which is what
   `latex/evaluation_protocol.tex` argues; (b) the genre task is close to its noise
   ceiling, consistent with sections 7c-7e -- a much better representation buys very little
   genre accuracy, but the emotion bottleneck still does.
4. **The layer-choice control matters methodologically** (16.2). Had the final layer been
   reported by default, wav2vec 2.0 would have scored 0.161 instead of 0.324 and the
   waveform front-end would have been dismissed on an artefact.

## 17. Blockbuster, tested properly (protocol fix, cue level, full MIR)

`experiments/cross_dataset/exp_blockbuster_deep.py` and
`exp_signature_replication.py`; results in `results/blockbuster_deep.json` and
`results/signature_replication.json`.

### 17.1 Why this was needed

Every Blockbuster number reported before this section came from **one 5-fold split at
sklearn's default `C`** — exactly the flaw §11 diagnosed for Eerola and then never applied
to the second corpus. Two further shortcuts were in place: only 78 of Ma et al.'s 140 MIR
features were used (the MFCC family, because the supervisor's question named MFCC), and
each film was collapsed to the mean of its cues, discarding the bag structure entirely.
This section re-runs the corpus under the primary protocol — **10x5 repeated KFold over
films, nested-CV-tuned `C`, per-repeat CIs, Nadeau-Bengio corrected tests** — and uses
both discarded resources.

Films are independent rows here, so plain KFold is correct; there is no film-grouping
problem as on Eerola. Cue-level arms still split **by film**, so a bag label cannot leak.

### 17.2 Results

| arm | Macro-F1 | 95% CI |
|---|---|---|
| VGGish, instance majority voting (Ma's IMV) | **0.621** | [0.607, 0.635] |
| **emotion(11), per-cue -> pooled** | **0.616** | [0.600, 0.632] |
| VGGish-128 (film-level) | 0.593 | [0.573, 0.613] |
| PCA-8(VGGish) [control] | 0.587 | [0.571, 0.604] |
| **MIR-140 (full hand-crafted set)** | **0.585** | [0.568, 0.601] |
| emotion(11), pooled -> per-film | 0.557 | [0.547, 0.568] |
| MFCC-78 (the subset used in §3) | 0.516 | [0.497, 0.535] |
| random-8(VGGish) [control] | 0.487 | [0.470, 0.505] |
| random-guess floor (Ma's definition) | 0.295 | — |
| dummy (most-frequent) | 0.035 | — |

| comparison | diff | p | win |
|---|---|---|---|
| **emotion per-cue vs emotion film-level** | **+0.059** | **0.020** | 92% |
| **PCA-8 vs random-8** | **+0.100** | **0.001** | 100% |
| VGGish vs MFCC-78 | +0.077 | 0.062 | 82% |
| MIR-140 vs MFCC-78 | +0.068 | 0.068 | 84% |
| **VGGish vs MIR-140** | **+0.009** | **0.787** | 50% |
| emotion per-cue vs VGGish | +0.023 | 0.536 | 60% |
| emotion per-cue vs PCA-8 | +0.029 | 0.369 | 64% |
| IMV vs VGGish film-level | +0.028 | 0.388 | 64% |

Our best arms (0.616-0.621) now sit squarely inside Ma et al.'s published VGGish range
(0.61-0.65), and the IMV arm — their own architecture — reproduces their average-pooling
result (0.62) exactly.

### 17.3 Two corrections to previously reported claims

1. **"VGGish beats hand-crafted MIR" does not survive.** §3 reported VGGish 0.582 vs MFCC
   0.455, a gap of +0.127 that looked decisive. Under the proper protocol it is 0.593 vs
   0.516 (+0.077, **p=0.062 — not significant**), and against the **full 140-feature MIR
   set** the gap essentially vanishes: **+0.009, p=0.787, winning 50% of 50 folds**. The
   original margin was an artefact of comparing a learned embedding against a deliberately
   restricted subset of the hand-crafted features. Ma et al. were more careful than our
   summary of them: they claim VGGish models "generally outperform" MIR ones, and their
   own best MIR model (0.61) is close to their best VGGish model (0.65). **The thesis must
   say that the learned embedding and the full hand-crafted set are statistically
   indistinguishable on this corpus.**
2. **The emotion bottleneck's Blockbuster standing improves, but stays a tie.** §10
   reported predicted-emotion 0.565 vs PCA-8 0.559 vs VGGish 0.582 — i.e. emotion slightly
   behind the embedding. At the correct granularity it is 0.616 vs 0.587 vs 0.593, i.e.
   nominally *ahead* of both, though neither difference is significant (p=0.369, p=0.536).
   The honest statement for the in-domain Blockbuster task remains **"competitive with,
   not better than"** — the corpus is too easy and too small to separate them. The
   *transfer* setting (§15.3) is where the separation appears.

### 17.4 The cue level is the right unit

The corpus contains **4664 individually scored music cues** (median 39 per film, median
duration 19 s). An Eerola clip is 10-31 s, mean 17 s — so a cue and a clip are the same
kind of object, while a film-level mean over 39 cues is not. Predicting emotion per cue
and pooling the *emotions* afterwards is therefore the granularity that matches the
regressor's training distribution, and it is worth **+0.059 in-domain (p=0.020, 92% of
folds)** and **+0.029 in zero-shot transfer** over pooling the embedding first. This is a
concrete, measurable instance of the domain-shift argument that was previously only
asserted from the per-dimension SD gap (50.4 vs 59.8).

The MIL structure itself adds little: instance majority voting (0.621) is not significantly
better than simply averaging the cues (0.593, p=0.388). So the gain comes from *where the
emotion model is applied*, not from modelling the bag.

### 17.5 Do the emotion-genre signatures replicate? (RQ3 external validity)

`exp_signature_replication.py`. The §7b signatures were derived on Eerola from **human
ratings**. Whether they are a property of film music or of that corpus is testable: the
Eerola-trained regressor predicts emotions for every Blockbuster cue, and the same Cohen's
d is recomputed there. Nothing about Blockbuster enters the emotion model, so any agreement
is genuine transfer.

| genre | r(d) | sign agreement | Eerola top emotion | Blockbuster top emotion |
|---|---|---|---|---|
| romance | +0.99 | 100% | +valence | -anger |
| action | +0.97 | 88% | -happy | +anger |
| sci-fi | +0.97 | 100% | +anger | +anger |
| horror | +0.95 | 88% | **+fear** | **+fear** |
| drama | +0.80 | 100% | +valence | -energy |
| comedy | +0.59 | 75% | -fear | +happy |
| **POOLED (48 genre x emotion cells)** | **+0.84** | **92%** | | |

**The signatures replicate.** Across 48 genre-by-emotion cells the two corpora's effect
sizes correlate at r=0.84 with 92% sign agreement, and horror's defining emotion is fear in
both. Where the "top" emotion differs it is a near-tie within a signature that is otherwise
the same shape (action is negative-valence/high-anger/low-happy in both; the two corpora
just rank anger and happy differently).

Caveat on magnitude: Blockbuster's **film-level** d values are roughly twice Eerola's
because averaging ~39 cues shrinks within-group variance and inflates the effect size. The
**cue-level** table is the one comparable to Eerola's clip-level d, and there the sizes
match closely (horror/fear +0.55 vs +0.78). What replicates is the pattern, not the
absolute magnitude.

This is the strongest available evidence that the interpretability claim generalises:
the emotional signatures are not an artefact of the Eerola panel or corpus.

## Next steps

- **Content chapters** — student; the LaTeX snippets in `docs/latex/` are ready to fold
  in. Five are now available: `evaluation_protocol`, `cross_dataset_transfer`,
  `emotion_genre_relationship`, `genre_subset`, `emotion_regression_bridge`,
  `statistical_power`, `rating_reliability`.
- **Bib entries to add on Overleaf** (cited by the new snippets, not yet in
  `bib/library.bib`): `hu2007exploring`, `laurier2009mood`, `eerola2011genrespecific`,
  `saari2016genre`, `nadeau2003inference`, `sokolova2009systematic`. Each snippet carries
  its BibTeX entry in a header comment. Note `eerola2011genrespecific` is a *different*
  paper from `eerola2011comparison` (same author, same year).
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
- `experiments/show_results.py` — render any `results/*.json` as Markdown tables;
  `experiments/audit_consistency.py` — verify the docs still agree with the saved results
- `experiments/` — runnable experiments, grouped by pipeline stage (index +
  per-script purpose in `experiments/README.md`). Run from the repository root.
  - `features/` — `verify_data`, `extract_features`, `clip_length_analysis`,
    `exp4_rating_reliability`, `exp_waveform_vs_spectrogram`, `exp_w2v_layer_sweep`
  - `emotion/` — `exp_emotion_regression`, `exp_emotion_improve`
  - `genre/` — `exp5_target`, `exp_genre`, `exp_genre_subset`, `exp_emotion_genre`
  - `diagnostics/` — `exp_error_analysis`, `exp_threshold_fix`, `exp_model_search`,
    `exp_learning_curve`, `exp_clip_length`
  - `cross_dataset/` — `exp_blockbuster`, `exp_blockbuster_emotion`, `exp_zero_shot`,
    `exp_blockbuster_deep`, `exp_signature_replication`
  - `evaluation/` — `exp_statistical_power` (supersedes the single-run numbers from
    `genre` / `cross_dataset`)
- `docs/` — this log; `current_state.md` (the supervisor briefing, Markdown — supersedes
  the older `current_state.docx`/`.pdf`); `results/README.md` (what each results JSON holds
  and how to render it); `docs/literature/` (literature reviews); `docs/latex/` (LaTeX
  snippets: `evaluation_protocol`, `cross_dataset_transfer`, `waveform_vs_spectrogram`,
  `emotion_genre_relationship`, `genre_subset`, `emotion_regression_bridge`,
  `statistical_power`, `rating_reliability` — each carries the BibTeX entries it needs in
  a header comment)
- `data/processed/Eerola_DB/embeddings/<model>/<set>/` — per-clip embedding caches, one
  folder per model (`ast`, `clap`, `vggish`, `mir`, `wav2vec2`, `ast_windows`); the single source of
  truth, read via `assemble_from_cache` (layout documented in `src/config.py`).
  Download instructions for the raw data are in `data/raw/README.md`.
- `results/` — machine-readable experiment output, every per-fold score:
  `statistical_power.json`, `model_search.json`, `zero_shot.json`,
  `waveform_vs_spectrogram.json`, `w2v_layer_sweep.json`, `blockbuster_deep.json`,
  `signature_replication.json`. Render any of them as Markdown tables with
  `python experiments/show_results.py <name>`; see `results/README.md`.
