# Supervisor briefing — current state

**Can a soundtrack's emotion predict a film's genre?**
Eerola & Vuoskoski (2011): 346 clips / 43 films · Ma et al. (2021) "Blockbuster": 110 films / 4664 cues
Implementation complete. Open item: content chapters.

> This document supersedes `current_state.docx` / `.pdf`. It is the Markdown version, kept
> up to date; the LaTeX in `docs/latex/` is for pasting into the thesis, and the full
> technical log with every experiment is `docs/README.md`.

---

## 1. Your notes — what was asked and what came of it

### Round 2 (most recent)

| # | Your note | Status | Where |
|---|---|---|---|
| A | Test on Blockbuster **without training on it**, train on Eerola, compare with the paper's own results | **Done.** Zero-shot reaches Macro-F1 **0.511** against a 0.292 chance floor — 82 % of what in-domain training achieves. Our in-domain reproduction of Ma et al. scores **0.620**, inside their published 0.61–0.65. | §4.3, §4.4 |
| B | Explain the evaluation process and which metrics produced the feature-extraction accuracy | **Done.** It is **mean R²** on emotion regression. Full explanation in §3. | §3 |
| C | Try a **waveform** extractor, not just spectrogram | **Done.** wav2vec 2.0 added. It is clearly worse (R² 0.323 vs 0.56) — but the emotion bottleneck lifts it from last place to second. | §4.5 |
| D | The classifier must have the **same genre classes in both datasets** to be fair | **Done, and it found a real error in our own reporting.** Both corpora now share one 6-genre space. | §4.2 |
| E | Emotion–genre relationship in the literature, **including in music** | **Done.** New literature section covering Hu & Downie (2007), Laurier et al. (2009), Eerola (2011), Saari et al. (2016). | §6 |

### Round 1 (earlier)

| # | Note | Status |
|---|---|---|
| 1 | VGGish vs MFCC on Blockbuster | Done — **but the conclusion changed**, see §5 |
| 2 | Genre classification with vs without emotion | Done |
| 3 | Clip length in the literature | Done — `docs/literature/clip_length_literature.md` |
| 4 | Emotion–genre relationship | Done — empirically and in the literature |
| 5 | Content chapters | **Open — this is the writing task** |
| 6 | Emotion regression, if time permits | Done |

### On note D — what it caught

You were right that this matters, and it was not only a design question. Our own progress
log had a results table putting **Eerola's 0.272 next to Blockbuster's 0.582** with nothing
marking them apart. Those are different label spaces (8 vs 6 genres), different units (a
10 s clip vs a whole film) and different chance levels (0.233 vs 0.292). A reader would
reasonably conclude Blockbuster works twice as well. Both places where this happened now
carry an explicit warning and point to the like-for-like comparison:

| same 6 genres, each against its own floor | Macro-F1 | chance |
|---|---|---|
| Eerola (clip level) | 0.315 | 0.250 |
| Blockbuster (film level) | 0.620 | 0.292 |

Blockbuster is genuinely the easier corpus — a film-level vector averages out per-cue
noise, and its genres have far higher base rates.

---

## 2. Definitions

Terms used throughout, in the order they become relevant.

### Data

**Clip** — one Eerola excerpt, 10–31 s (mean 17 s) of film music, rated by a listener
panel on 8 emotions. 360 in total; 346 after genre cleaning.

**Cue** — one continuous piece of music as it appears in a film, identified by Ma et al.
in the Blockbuster corpus. There are **4664 of them, median 39 per film, median duration
19 s**. A cue is the same *kind* of object as an Eerola clip; a film is a *bag* of cues.
This distinction turned out to matter — see §4.6.

**Bag / Multiple-Instance Learning** — a film carries a genre label, but the label applies
to the whole film, not to each individual cue. Ma et al. treat this as an MIL problem. Our
simplest handling ("Simple-MI") averages a film's cues into one vector.

### Modelling

**Multi-label classification** — a film belongs to several genres at once, so the target
is a binary vector rather than one class. A prediction can be partly right.

**Binary Relevance** — the strategy used: train one independent binary classifier per
genre, then stack their outputs. Simple, and empirically better here than classifier
chains (which are significantly worse, p = 0.024).

**`C` — the regularisation strength of the logistic regression.** This is the single most
consequential hyperparameter in the project, so it is worth being precise.

A logistic regression fits weights to features. With 128–768 embedding dimensions and only
~330 clips it can fit the training data almost perfectly by giving large weights to
dimensions that happen to correlate with genre in *this* sample — memorisation rather than
learning. Regularisation counteracts this by adding a penalty on large weights.

In scikit-learn the parameter is `C`, the **inverse** penalty strength:

- **large `C`** (e.g. 1.0) → weak penalty → complex model → overfits
- **small `C`** (e.g. 0.003) → strong penalty → simpler, smoother model → generalises better

sklearn's default is `C = 1.0`, and it turned out to be **the worst setting for every
feature set here**; all of them prefer `C ≤ 0.1`. Crucially, the default penalised the
high-dimensional audio baselines *more* than the 11-dimensional emotion features — i.e. in
the direction that would have flattered this thesis's hypothesis. Fixing it was necessary
to be fair to the baselines, and it narrowed our own margin slightly (§5).

`C` is never chosen by looking at test performance. It is selected by **nested
cross-validation** (below).

**Emotion bottleneck** — the name for the middle of the two-stage pipeline. "Bottleneck"
is the standard term for a point where information is forced through a deliberately narrow
channel:

```
direct:      audio ──► 128-768 numbers ─────────────────────────────► genre
bottleneck:  audio ──► 128-768 numbers ──► 8 EMOTIONS ──► 11 features ──► genre
                                          ^^^^^^^^^^
                                          everything must pass through here
```

The classifier in the second stage never sees the audio embedding at all. It sees only
eight numbers — valence, energy, tension, anger, fear, happy, sad, tender — so whatever
genre information survives has to be expressible *as emotion*. That is a large loss of
capacity: 768 dimensions down to 8, a 96-fold compression.

The point is that the compression is not arbitrary. Each surviving dimension is a quantity
a human rated on a 1–9 scale, which buys two things a generic compression cannot:

- **Interpretability** — a prediction can be explained ("this scores as Horror because
  fear is high"), and the per-genre signatures can be checked against film-scoring
  practice.
- **Portability** — "fear = 6.2" denotes the same perceptual quantity in any corpus,
  whereas "VGGish dimension 87 = 0.4" is meaningful only relative to the corpus the
  classifier was fitted on. This is why the advantage is largest under transfer.

**The control that makes this claim testable.** An obvious objection is that *any* 8-value
compression would help — by regularising a small dataset, not by being emotional. So a
**PCA-8 control** is run everywhere: the same embedding compressed to its 8 principal
components, same classifier, same folds. A **random-8** projection sits below that as a
floor. The result: compressing to 8 principal components buys +0.014 under transfer
(p = 0.572); compressing to 8 *emotions* buys +0.106. It is the emotions, not the
narrowness.

### Evaluation

**GroupKFold by film** — cross-validation where all clips of one film land on the same side
of the split. Necessary because clips from one film share instrumentation and share their
genre label by construction; a random split lets a model score well by recognising the
*film*. Measured effect: the AST baseline scores **0.44 ungrouped vs 0.26 grouped** — about
40 % of its apparent performance was film memorisation.

**Repeated cross-validation (10×5)** — the film→fold assignment is re-randomised 10 times,
giving 50 folds. One 5-fold run cannot separate a 0.05 difference from partition noise; an
early version of this project consequently reported real effects as "within noise".

**Nested cross-validation** — `C` is chosen by an *inner* GroupKFold **inside each training
fold**, so the test fold never participates in the choice. Without this, tuning `C` on test
performance would inflate every score.

**Out-of-fold prediction** — a prediction for a clip made by a model that never saw that
clip in training. Used both for scoring and for feeding the second stage realistic (noisy)
emotion inputs rather than near-perfect in-sample ones.

**In-domain vs zero-shot** — the two evaluation settings, and the difference is what a
result is allowed to claim.

|  | in-domain | zero-shot |
|---|---|---|
| train and test data | **the same corpus**, split into folds | **different corpora** |
| what is held out | some films of that corpus | *everything* — the whole target corpus |
| target labels used in training | yes (the training folds) | **none, ever** |
| repeated? | yes, 10×5 folds | no — one pass; uncertainty from a bootstrap over test films |
| what it shows | the signal exists **in this data** | the signal is a property of **film music** |
| our numbers | Eerola 0.315, Blockbuster 0.620 | Eerola → Blockbuster 0.511 |

In-domain answers "can a model learn this?" — but a model can score well in-domain by
picking up quirks of one corpus (which is exactly what film-identity leakage is, one level
down). Zero-shot answers "did it learn something general?" Nothing about Blockbuster
enters training: not its labels, not its feature statistics under the strict regime, not a
single fold. The model is fitted on Eerola, frozen, and pointed at 110 unseen films.

That is also why zero-shot is the more demanding test and why the emotion result there
carries the most weight: transfer is precisely where a corpus-specific representation
should fail and a semantically grounded one should survive.

**Zero-shot** — the model is trained on one corpus and applied to another with **no
further fitting and no target labels**. This is what your note A asked for.

**Nadeau–Bengio corrected *t*-test** — a plain paired *t*-test over 50 folds is
anti-conservative because the folds' training sets overlap; the correction inflates the
variance estimate accordingly. `p_limit` is the *p*-value the test converges to with
infinite repeats: **if `p_limit` > 0.05, no amount of extra computation can make the result
significant — only more films can.** `win rate` is the fraction of folds one arm beats the
other, which makes no distributional assumption.

**Reliability ceiling (0.897)** — Set 2 re-rates 110 Set 1 excerpts with a *different*
listener panel. The two panels agree at r ≈ 0.91, so a model cannot explain more than
~0.90 of the rating variance. **The honest reference point for R² is 0.90, not 1.0.**

---

## 3. How the evaluation actually works

This is note B in full.

### The two stages are scored with two different metrics

The pipeline is **audio → emotion → genre**, and the two stages are different problems.

| stage | problem | metric | why |
|---|---|---|---|
| 1. audio → 8 emotions | regression (values 1–9) | **mean R²** (+ RMSE) | continuous target |
| 2. emotion → genre | multi-label classification | **Macro-F1** | binary vectors, severe imbalance |

### Stage 1 — this is the "feature-extraction accuracy"

**Mean R² on emotion regression is the number that ranks the feature extractors.** It
measures exactly the property the pipeline needs from an audio representation: how much
affective information it carries. Genre Macro-F1 cannot serve that purpose, because it
confounds the representation with the second-stage classifier.

The procedure, concretely (`experiments/features/exp_waveform_vs_spectrogram.py`):

1. **Extract.** Each clip is decoded to mono at the model's sample rate, passed through a
   **frozen** pretrained model — no fine-tuning anywhere — and the frame-wise output is
   **mean-pooled over time** into one vector per clip. Vectors are cached to disk, so the
   numbers are reproducible without re-running any neural network.
2. **Regress.** One `RandomForestRegressor` (300 trees) predicts **all 8 emotions jointly**
   from that vector. All 360 clips are used here, not the 346-clip genre subset, because
   every clip has ratings.
3. **Cross-validate.** `cross_val_predict` with `GroupKFold(5)` grouped by film. Each clip
   therefore receives exactly one prediction, made by a model that never saw that clip —
   or any other clip from the same film.
4. **Score.** R² is computed **per emotion over all 360 pooled out-of-fold predictions**,
   then averaged over the 8 emotions.

Two details worth defending if asked:

- **R² is computed on the pooled predictions, not averaged per fold.** Averaging per-fold
  R² is unstable when a fold happens to contain little target variance (the denominator of
  R² is the variance of the fold's targets). Pooling uses the full-sample variance and is
  the standard choice with `cross_val_predict`.
- **The regressor mattered more than the features.** On VGGish: Ridge 0.37, SVR-RBF 0.54,
  RandomForest 0.56. An earlier version of this project used Ridge and concluded
  "VGGish ≫ AST for emotion" — a linear-model artefact. With a non-linear regressor the
  three pretrained embeddings tie at ≈ 0.56.

Result (mean R² over 8 emotions, GroupKFold by film, n = 360):

| representation | input domain | dim | mean R² | RMSE | % of the 0.897 ceiling |
|---|---|---:|---:|---:|---:|
| CLAP | log-mel spectrogram | 512 | **0.561** | 1.22 | 62 % |
| AST | log-mel spectrogram | 768 | 0.560 | 1.22 | 62 % |
| VGGish | log-mel spectrogram | 128 | 0.558 | 1.23 | 62 % |
| **MusiCNN** | log-mel spectrogram | 200 | **0.541** | 1.25 | 60 % |
| hand-crafted MIR | STFT descriptors | 103 | 0.490 | 1.31 | 55 % |
| **wav2vec 2.0** | **raw waveform** | 768 | **0.323** | 1.52 | 36 % |

RMSE is reported alongside because R² alone does not convey practical size: 1.22 means the
model is typically 1.2 points off on a 1–9 scale.

**What 0.56 means.** Not "56 % correct" — it is 62 % of the *attainable* variance, since
two human panels only agree at 0.90. And the remaining shortfall is **not** rating noise:
at 62 % of the ceiling there is real headroom in the audio representation, which closes off
the convenient excuse that the labels are too noisy.

### Stage 2 — genre classification

**Macro-F1** is the headline: F1 computed per genre, then averaged with **equal weight per
genre**. Under this imbalance (Drama 239 clips, Documentary 17) the alternatives are
actively misleading:

| model (8 genres, GroupKFold) | Exact Match | Hamming ↓ | Macro-F1 |
|---|---:|---:|---:|
| emotion(11) → LogReg | 0.006 | 0.431 | **0.272** |
| AST-768 → LogReg | 0.107 | 0.247 | 0.258 |
| dummy (always "Drama") | **0.116** | **0.185** | 0.102 |

The dummy wins both other metrics while being useless. Exact Match demands the *entire*
genre vector be right; Hamming rewards predicting almost nothing. Both are reported for
transparency, never quoted as a result.

**Two different floors, and they are not interchangeable.** Our `dummy` predicts the single
most frequent label set (always "Drama") → Macro-F1 0.102. Ma et al.'s *random guess*
predicts each genre independently at its base rate → 0.233 on our corpus, 0.32 on theirs.
Comparing our dummy against their random guess would overstate a margin by more than 0.1
F1. Both are now computed on both corpora.

---

## 4. Results

### 4.1 Eerola — the primary result

10×5 repeated GroupKFold, nested-CV-tuned `C`, 5-genre subset (329 clips / 41 films):

| approach | Macro-F1 | 95 % CI |
|---|---:|---|
| ground-truth emotion(11) — ceiling of the emotion path | **0.397** | [0.388, 0.407] |
| VGGish → predicted emotion(11) — the real pipeline | 0.394 | [0.389, 0.400] |
| VGGish-128, direct | 0.350 | [0.335, 0.365] |
| AST-768, direct | 0.345 | [0.328, 0.361] |
| PCA-8 of VGGish — "any 8-d bottleneck" control | 0.343 | [0.330, 0.355] |
| CLAP-512, direct | 0.335 | [0.323, 0.347] |
| dummy | 0.166 | [0.165, 0.167] |

On all 8 genres: **emotion 0.300 vs AST 0.257, p = 0.031** (dummy 0.101).

| comparison | Δ | p | folds won | verdict |
|---|---:|---:|---:|---|
| emotion vs AST, 8 genres | +0.043 | **0.031** | 92 % | significant |
| emotion vs CLAP, 5 genres | +0.062 | **0.036** | 86 % | significant |
| predicted emotion vs PCA-8 control | +0.052 | 0.060 | 88 % | borderline |
| predicted vs ground-truth emotion | −0.003 | 0.855 | 54 % | **the same** |
| AST vs CLAP | +0.010 | 0.752 | 62 % | the same |

Two things to emphasise: the three pretrained embeddings are **indistinguishable from each
other**, so the baseline is not a badly chosen model; and **predicted emotions work as well
as human ratings**, which is what makes the pipeline usable without annotations.

### 4.2 One shared genre space (note D)

Ma et al.'s six reduced genres — action, drama, comedy, sci-fi, romance, horror. Their
reduction rule was **recovered from their published master list rather than assumed**: a
film gets genre *g* iff *g* appears in its raw IMDb list, verified on **110/110 films with
zero mismatches**. That is the same "any-present" rule already used on Eerola, so both
corpora enter one space without any dataset-specific heuristic.

| genre | Eerola clips | Blockbuster films |
|---|---:|---:|
| action | 103 | 55 |
| drama | 239 | 44 |
| comedy | 40 | 37 |
| sci-fi | 19 | 36 |
| romance | 51 | 13 |
| horror | 28 | 11 |
| **total** | **319 clips / 41 films** | **110 films** |

VGGish is the one representation both corpora provide identically, and the two spaces are
numerically compatible (per-dimension means correlate r = 0.97), so a model fitted on one
can be applied to the other.

### 4.3 Reproducing Ma et al. before transferring (note A, part 1)

A transfer score is uninterpretable without knowing what in-domain performance looks like.
Their protocol — leave-one-film-out, macro-averaged P/R/F1 — re-run with our own simple
pipeline:

| | precision | recall | Macro-F1 |
|---|---:|---:|---:|
| Ma et al., VGGish + single-attention (their best) | 0.60 | 0.73 | **0.65** |
| Ma et al., VGGish + average pooling | 0.55 | 0.78 | 0.62 |
| Ma et al., VGGish + kNN Simple-MI | 0.64 | 0.59 | 0.61 |
| **ours: VGGish → LogReg, LOO** | 0.569 | 0.705 | **0.620** |
| random-guess floor (published / ours) | .32 / .294 | .32 / .293 | .32 / .292 |
| plurality floor (published / ours) | .14 / .138 | .34 / .333 | .19 / .193 |

We land inside their range and both floors reproduce to within 0.03. **The pipeline is
validated against an external published result**, so a low transfer score cannot be blamed
on a weak classifier.

### 4.4 Zero-shot transfer (note A, part 2)

Trained on Eerola alone, applied to all 110 Blockbuster films. No Blockbuster label is seen
in training.

| approach | precision | recall | Macro-F1 |
|---|---:|---:|---:|
| **VGGish → predicted emotion, per cue → genre** | 0.488 | 0.651 | **0.511** |
| the same, applied at film level | 0.450 | 0.637 | 0.482 |
| genre stage fitted on ratings instead | 0.445 | 0.474 | 0.434 |
| PCA-8 control | 0.534 | 0.467 | 0.421 |
| VGGish-128 direct | 0.576 | 0.442 | 0.407 |
| random-guess floor | 0.294 | 0.293 | 0.292 |
| *(in-domain reference, §4.3)* | *0.569* | *0.705* | *0.620* |

Because this is a single split with no folds, margins come from a **paired bootstrap over
the 110 target films** (2000 resamples):

| comparison | Δ | 95 % CI | p |
|---|---:|---|---:|
| **emotion vs VGGish direct** | **+0.106** | [+0.015, +0.186] | **0.018** |
| **emotion vs PCA-8 control** | **+0.092** | [+0.009, +0.171] | **0.032** |
| emotion (film-level) vs VGGish direct | +0.077 | [−0.007, +0.152] | 0.066 |
| PCA-8 control vs VGGish direct | +0.014 | [−0.032, +0.062] | 0.572 |

**This is the strongest result in the project.** In-domain, emotion vs the PCA-8 control was
borderline at best (p = 0.060). Under domain shift it is significant. And the reason is
interpretable: *"fear = 6.2" means the same thing in both corpora, whereas VGGish dimension
87 does not.* Compressing to 8 dimensions is worth +0.014; compressing to 8 **emotions** is
worth +0.106.

**A note on fairness.** The features can be scaled two ways: *strict* (one scaler fitted on
the training corpus) or *target-standardised* (each corpus z-scored with its own statistics
— unsupervised, so still zero-shot). Only the embedding arms move; the emotion arms are in
rating units and are identical in both. All figures above use **strict**, the regime in
which the baseline scores *better* (0.407 vs 0.394) — the conservative choice. Under
target-standardisation the same comparisons give +0.116 (p < 0.001) and +0.097 (p = 0.006).

That the emotion arm is unchanged across regimes is itself supporting evidence: it absorbs
the domain shift by construction rather than by a scaling trick.

### 4.5 Waveform vs spectrogram (note C)

Every representation used so far starts from a time–frequency image, so "emotion matches
learned embeddings" had only ever been tested against *spectrogram* front-ends. wav2vec 2.0
convolves the raw 16 kHz samples directly — no spectral transform anywhere.

**A control was needed first.** Pooling wav2vec 2.0's final layer gives R² = 0.161; pooling
its second transformer block gives **0.324**. The layer choice alone is worth a factor of
two — larger than the effect this thesis measures — so reporting the default would have
dismissed the waveform front-end on an artefact.

Genre results (5-genre subset, 5×5 repeated GroupKFold, tuned `C`):

| approach | Macro-F1 |
|---|---:|
| emotion(11), ground truth | 0.391 |
| **wav2vec 2.0 → predicted emotion → genre** | **0.376** |
| VGGish / MIR / AST / CLAP, direct | 0.335–0.342 |
| **wav2vec 2.0, direct** | **0.300** (last) |

So: the waveform model is the weakest representation — **and the emotion bottleneck lifts it
from last place to second**, +0.076 at p = 0.003, winning **100 % of 25 folds**, ending
statistically tied with the ground-truth ceiling. The benefit of the emotion intermediate is
not a property of one embedding; it is largest exactly where the representation is weakest.

*Honest caveat:* wav2vec 2.0 was pretrained on speech while AST and VGGish saw AudioSet
(which contains music), so input domain and pretraining corpus vary together. Its loss
cannot be attributed to the waveform front-end alone.

### 4.6 What a cue is, and why it improved the results

The Blockbuster corpus stores each film as a **bag of individually scored music cues** —
4664 of them, median 39 per film, **median duration 19 s**. We had been collapsing each
film's entire bag into a single averaged vector, discarding that structure.

The problem this created is specific. Our emotion regressor was trained on Eerola clips of
10–31 s (mean 17 s). A **cue is the same kind of object**; a mean over 39 cues is not — it
is much smoother (per-dimension SD 50.4 vs 59.9). So we were training on one kind of input
and predicting on another.

The fix is to change *where* the emotion model is applied:

- **before:** average the film's 128-d embeddings → predict emotions once
- **after:** predict emotions for **each cue** → average the resulting emotion vectors

Same data, same model, different order of operations:

| | in-domain Blockbuster | zero-shot |
|---|---:|---:|
| pool embeddings first, then predict | 0.557 | 0.482 |
| **predict per cue, then pool emotions** | **0.616** | **0.511** |
| gain | **+0.059** (p = 0.020, 92 % of folds) | **+0.029** |

This converts the domain-shift argument from an assertion into a measurement. Notably, the
MIL structure *itself* adds little — instance majority voting (0.621) is not significantly
better than plain averaging (0.593, p = 0.388). **The gain comes from applying the emotion
model at the granularity it was trained on, not from modelling the bag.**

### 4.7 Do the emotion signatures replicate? (RQ3 external validity)

The per-genre signatures were derived on Eerola from **human ratings**. Whether they
describe film music or merely that corpus is testable: predict emotions for all 4664
Blockbuster cues with the Eerola-trained model and recompute the same Cohen's *d*. No
Blockbuster annotation exists or is used, so any agreement is genuine transfer.

| genre | r(d) | sign agreement | Eerola top | Blockbuster top |
|---|---:|---:|---|---|
| romance | +0.99 | 100 % | +valence | −anger |
| action | +0.97 | 88 % | −happy | +anger |
| sci-fi | +0.97 | 100 % | +anger | +anger |
| horror | +0.95 | 88 % | **+fear** | **+fear** |
| drama | +0.80 | 100 % | +valence | −energy |
| comedy | +0.59 | 75 % | −fear | +happy |
| **pooled (48 cells)** | **+0.84** | **92 %** | | |

**The signatures replicate.** Horror is defined by fear in both corpora. This turns the
earlier eyeball "face validity" check into a measured replication, and it is the strongest
evidence that the interpretability claim generalises beyond one listener panel.

*Caveat:* Blockbuster's film-level *d* values are about twice Eerola's, because averaging
39 cues shrinks within-group variance and inflates the effect size. The **cue-level** table
is the comparable one (horror–fear +0.55 vs +0.78). What replicates is the pattern, not the
absolute magnitude.

---

## 5. Numbers that changed — and why

Several previously reported figures moved. **In every case the evaluation got stricter, not
the models better.** This is worth stating plainly rather than quietly updating.

| previously reported | now | why |
|---|---|---|
| emotion vs AST "within noise", p = 0.62 | **significant, p = 0.031** | 5 folds cannot resolve that gap; 50 folds + tuned `C` can |
| "0.281 is a robust ceiling" | ceiling is 0.299 | regularisation had never been tested; tuning alone beats it |
| predicted-emotion pipeline 0.406 | 0.394 | one optimistic split; 50 folds regress it |
| 338 clips | 346 clips | corrected inclusion rule |
| **VGGish 0.582 ≫ MFCC 0.455 on Blockbuster** | **VGGish 0.593 vs full MIR 0.585, p = 0.787** | see below |
| Blockbuster emotion 0.565, behind VGGish 0.582 | emotion 0.616, VGGish 0.593 | proper protocol + per-cue bridging |

### The VGGish-vs-MFCC correction

This one changes a claim, so it needs care. The original comparison used only the **78
MFCC-family columns** of Ma et al.'s 140 MIR features — reasonable, since your note asked
specifically about MFCC. But the resulting +0.127 gap was then read as "the learned
embedding beats hand-crafted features".

Against the **full 140-feature MIR set**, under repeated CV with tuned `C`:

| | Macro-F1 |
|---|---:|
| VGGish-128 | 0.593 |
| PCA-8(VGGish), bottleneck control | 0.587 |
| **MIR-140 (full hand-crafted set)** | **0.585** |
| MFCC-78 (the earlier subset) | 0.516 |

**VGGish vs MIR-140: +0.009, p = 0.787, winning 50 % of 50 folds — statistically
indistinguishable.** The 62 excluded features (roughness, key strength, pulse clarity,
spectral shape) are precisely the descriptors most associated with musical affect.

This is also closer to what Ma et al. actually report: their best MIR model reaches 0.61
against their best VGGish model's 0.65. The thesis should state that **the learned embedding
and the full hand-crafted set are indistinguishable on this corpus** — and it costs nothing,
because the result that *does* separate under transfer is the emotion bottleneck (§4.4).

---

## 6. Literature: emotion and genre, including in music (note E)

Ready to paste: `docs/latex/emotion_genre_relationship.tex`, section "Evidence from music
research". Four sources, four distinct points:

- **Hu & Downie (2007, ISMIR)** — association test over 3903 albums and 22 genres (7134
  genre–mood pairs) finds systematic consistencies in the genre–mood relationship. This
  motivated the standardised mood clusters that became the MIREX Audio Mood
  Classification ground truth.
- **Laurier, Sordo, Serrà & Herrera (2009, ISMIR)** — a semantic mood space derived from
  Last.fm tags recovers Russell's valence–arousal structure, with mood tags clustering
  along musical style.
- **Eerola (2011, JNMR)** — *by the same author as our corpus.* Across nine datasets
  spanning classical, film, popular and mixed genres, emotion models generalise far worse
  **across** genres than **within** them: valence 43 % → 16 % of variance explained,
  arousal 62 % → 43 %.
- **Saari et al. (2016, IEEE TAC)** — exploits the dependency constructively:
  *genre-adaptive* mood models beat genre-agnostic ones on valence, arousal and tension.

**Why this matters for our results:** Eerola (2011) predicts exactly the degradation we
measured. Our in-domain 0.620 → zero-shot 0.482/0.511 is the same phenomenon one level up,
at the genre-prediction level. The literature and the new experiment support each other.

New BibTeX entries needed on Overleaf (each snippet carries its entry in a header comment):
`hu2007exploring`, `laurier2009mood`, `eerola2011genrespecific`, `saari2016genre`,
`baevski2020wav2vec`, `nadeau2003inference`, `sokolova2009systematic`.

> `eerola2011genrespecific` is a **different paper** from `eerola2011comparison` (the
> dataset paper) — same author, same year. Keep both keys.

---

## 7. Why the method holds up

| decision | why it is defensible |
|---|---|
| Cross-validation grouped by film | AST scores 0.44 ungrouped vs 0.26 grouped — 40 % of its apparent performance was recognising the film. Emotion features leak only +0.05. |
| Macro-F1 as the headline | Under this imbalance the dummy wins Exact Match and Hamming. Macro-F1 weights every genre equally. |
| 50 folds, not 5 | Five folds cannot separate a 0.05 difference from partition noise. Significance uses Nadeau–Bengio, since overlapping training sets break the ordinary *t*-test. |
| `C` tuned by nested CV | The default was the worst setting for every feature set. Tuning helped the *baselines* more than emotion (CLAP +0.033 vs emotion +0.018), so it does not flatter us. |
| Both 5 and 8 genres reported | Genre count scales with corpus size (Austin: 4 over 98 films; Ma: 6 over 110). Biography (24) and Documentary (17) are too rare; Adventure has no emotional signature at all. |
| Binary relevance, not chains | Chains were tested and are significantly worse (p = 0.024) — rejected empirically. |
| Single 10.24 s window | Full-clip pooling (first/centre/mean/max) was tested; all differences ≤ 0.006. |
| Frozen extractors, no fine-tuning | Pure feature extraction, cached per clip. Every result reproduces from a clone without running a neural network. |
| Emotion model applied per cue | Worth +0.059 in-domain (p = 0.020) — matching the training unit is measurable, not cosmetic. |

### One technical detail worth pre-empting

The genre classifier is trained on the forest's **in-sample** emotion predictions
(R² ≈ 0.94 on training clips) but evaluated on **out-of-sample** ones (R² ≈ 0.60). That is
a train/test *distribution* mismatch, not leakage — no test clip or label influences either
fit, so the score remains an honest generalisation estimate. The direction matters: the
classifier is calibrated on clean features and then handed noisy ones, so 0.394 is if
anything a slight under-estimate.

The zero-shot experiment **fixes this properly**, training the genre stage on out-of-fold
predicted emotions. And it turns out to matter under transfer: training on predictions
scores 0.511 against 0.434 for training on ratings. Better to state this than to have it
drawn out.

---

## 7b. Extractors considered and not used

For completeness, since "why not model X?" is a fair question:

**MusiCNN — tested, and it works.** A good suggestion on the merits: it is the only
representation here pretrained on *music tagging* (Million Song Dataset), rather than
general audio events (AST, VGGish: AudioSet) or speech (wav2vec 2.0), and several of its
output tags are mood words. It is now part of the comparison.

Getting it running required an isolated environment, because MusiCNN is TensorFlow-only
while the rest of the project is deliberately PyTorch-only:

| obstacle | resolution |
|---|---|
| official `musicnn` pins `tensorflow>=1.14` **and `numpy<1.17`** (2019) — impossible on Python 3.12 | used `musicnn-keras`, a TF-2 Keras port of the same weights |
| `musicnn-keras` copied the same stale `numpy<1.17` pin, contradicting TF 2's own `numpy>=1.22` | installed with `--no-deps`; the code itself works fine with numpy 1.26 |
| its pip wheel ships no model weights, and loads them from a path relative to the *working directory* | weights fetched directly from the source repository and loaded explicitly |
| `essentia-tensorflow` (the other route) has no Windows wheel | not needed |

The result is clean and reproducible: Python 3.11 + TensorFlow 2.15.1 + numpy 1.26.4 in
`.venv-musicnn`, `experiments/features/extract_musicnn.py` writing 200-dimensional
penultimate-layer embeddings into the normal cache. **Nothing in `src/` imports
TensorFlow**; the two environments meet only at the cache directory, so the main project
stays PyTorch-only.

**And it does not win.** Emotion regression R² = 0.541 — above hand-crafted MIR (0.490)
and far above wav2vec 2.0 (0.323), but below AST, CLAP and VGGish (0.558–0.561). On genre
it scores 0.340, statistically inseparable from VGGish (0.342), MIR (0.339), AST (0.338)
and CLAP (0.335).

That negative result is worth more than a win would have been. The thesis's claim is no
longer "emotion beats AST" but **"emotion (0.391) beats every audio representation we
could find"** — general-purpose, audio-text, music-specific and hand-crafted alike, all
of which cluster at 0.335–0.342. A supervisor asking "did you just pick a weak baseline?"
now has a five-model answer. MusiCNN is also the best of all representations on *energy*
(R² 0.687), which is a sensible thing for a music-tagging model to be good at.

**YAMNet / OpenL3** — excluded for the same reason: TensorFlow, against the deliberate
PyTorch-only design that keeps the environment reproducible.

## 8. Open items

1. **Content chapters** — the writing task. Eight ready-to-paste LaTeX sections in
   `docs/latex/`: `evaluation_protocol`, `cross_dataset_transfer`,
   `waveform_vs_spectrogram`, `emotion_genre_relationship`, `genre_subset`,
   `emotion_regression_bridge`, `statistical_power`, `rating_reliability`.
2. **More films is the only remaining lever on the headline claims.** `p_limit` > 0.05 for
   the still-borderline comparisons means additional computation cannot help, and the
   learning curves are still rising at 100 % of the data.
3. **Minor known issues** (§12 of the technical log): a short debug run of
   `exp_statistical_power.py` overwrites the authoritative results file; most diagnostic
   experiments still only print rather than save.

---

## 9. Where everything is

| what | where |
|---|---|
| This briefing | `docs/current_state.md` |
| Full technical log, every experiment and result | `docs/README.md` |
| Thesis-ready LaTeX sections | `docs/latex/` |
| Literature reviews | `docs/literature/` |
| What the results JSONs mean, and how to render them | `results/README.md` |
| Runnable experiments, grouped by pipeline stage | `experiments/` (see its `README.md`) |

```bash
python experiments/show_results.py            # list every saved result
python experiments/show_results.py --all --out report.md   # all tables as Markdown
```

Seed 42 throughout; embeddings cached and committed, so every number reproduces from a
clone without re-running a model over the audio.
