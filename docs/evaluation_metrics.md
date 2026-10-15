# The evaluation explained: every metric, why it is there, and what it showed

Written 15 October 2026, for the author. It answers three questions: what Hamming loss
means, why the emotion route scores worst on it, and why each metric of the evaluation was
chosen. Section 8 holds a LaTeX paragraph that can go into Appendix A as it is.

Every number is from the five-genre Eerola subset (329 clips, 40 films), seed 42, mean of
the ten repeats, unless stated otherwise. Sources: `results/metrics_stability.json`
(13 metrics), `results/film_level.json` (headline, per genre) and
`results/hamming_breakdown.json` (the Hamming analysis of section 1, written by
`experiments/diagnostics/exp_hamming_breakdown.py`).

---

## 1. Hamming loss, and why the emotion route is worst on it

### 1.1 What it measures

Each clip gets five yes/no decisions, one per genre (Action, Crime, Drama, Comedy,
Horror). Hamming loss is **the share of these decisions that are wrong**:

    Hamming loss = (false positives + false negatives) / (clips × genres)

Lower is better. A clip that is truly {Drama, Crime} and predicted as {Drama, Horror}
has two wrong decisions out of five (Crime missed, Horror wrongly added), a Hamming loss
of 0.4.

Two properties matter here:

- **Every decision counts the same.** Wrongly adding Horror costs exactly as much as
  missing Drama. There is no notion of a genre being rare or important.
- **For a rare genre, saying "no" is almost always right.** Only 28 of the 329 clips are
  Horror (8.5 %). A model that *never* predicts Horror is wrong on just 8.5 % of the
  Horror decisions. Any attempt to find Horror films produces false alarms, and each false
  alarm raises the loss.

### 1.2 The numbers

| route | Hamming loss ↓ | genres predicted per clip | wrong "yes" per clip | wrong "no" per clip | macro-F1 (clips) |
|---|---:|---:|---:|---:|---:|
| most frequent (always "Drama") | **0.218** | 1.00 | 0.27 | 0.82 | 0.168 |
| direct AST | 0.311 | 1.68 | 0.84 | 0.71 | 0.368 |
| random guess (base rates) | 0.323 | 1.54 | — | — | 0.307 |
| direct VGGish | 0.358 | 1.85 | 1.05 | 0.74 | 0.366 |
| PCA-8 control | 0.396 | 2.04 | 1.24 | 0.75 | 0.364 |
| **emotion route** | 0.406 | **2.35** | **1.42** | 0.62 | **0.405** |
| ratings (ceiling) | 0.427 | 2.47 | 1.54 | 0.61 | 0.416 |

A clip truly has **1.54 genres** on average.

**The two rankings are almost exactly reversed.** The model that has learned nothing
(always "Drama") has the best Hamming loss and the worst macro-F1. The human ratings,
which are the best input there is, have the best macro-F1 and the worst Hamming loss.
That alone shows that Hamming loss and macro-F1 measure different things, not that one
route is "worse" in general.

### 1.3 Why the emotion route is worse on Hamming loss

**1. It says "yes" more often.** It predicts 2.35 genres per clip, against 1.54 true
genres and 1.85 for VGGish. More "yes" means more true genres found (wrong "no" 0.62
per clip, the fewest of all routes). It also means more false alarms (wrong "yes" 1.42
per clip, the most of the audio routes). Hamming loss counts both kinds of error the same,
so the extra false alarms outweigh the extra hits.

**2. All of the difference comes from the two rare genres.** Split by genre, the share of
wrong decisions is:

| genre (clips) | emotion route | direct VGGish | difference |
|---|---:|---:|---|
| Action (103) | 0.371 | 0.411 | emotion route **better** |
| Crime (98) | 0.473 | 0.468 | equal |
| Drama (239) | 0.356 | 0.396 | emotion route **better** |
| Comedy (40) | 0.438 | 0.286 | emotion route worse |
| Horror (28) | 0.403 | 0.235 | emotion route worse |

On the three common genres the emotion route makes *fewer* or equally many errors. The
gap comes only from Comedy and Horror. The emotion route predicts Comedy for 148 clips
(40 are Comedy; it finds 22 of them) and Horror for 135 clips (28 are Horror; it finds
15). VGGish predicts them for 78 and 67 clips and finds 12 and 9.

**3. Hamming loss rewards ignoring rare genres; macro-F1 punishes it.** A genre that is
never predicted has F1 = 0, and in macro-F1 that zero weighs as much as Drama's score.
In Hamming loss the same genre costs only its base rate (0.085 for Horror). So the
behaviour that wins macro-F1 (trying to find the rare genres) loses Hamming loss.

### 1.4 Why the emotion route says "yes" so often

- **Class-balanced weights.** The genre classifier is a logistic regression per genre
  with `class_weight="balanced"` (`src/models/classifiers.py`). The few positive clips of
  a rare genre are weighted up (Horror positives about ×5.9, negatives about ×0.55).
  This works like lowering that genre's decision threshold. It is applied to
  every route equally and was chosen so that rare genres are not ignored.
- **The penalty `C` is chosen for macro-F1.** An inner cross-validation on the training
  folds picks `C` by macro-F1, so every route is tuned towards finding genres, not
  towards making few errors.
- **In emotion space the genres overlap more.** With only 11 emotion features, a tense,
  fearful crime or thriller clip looks much like a horror clip, and a light, happy drama
  clip like a comedy one. The classifier cannot separate them sharply, so many clips end
  up with moderate probabilities. Under balanced weights, many of these moderate
  probabilities cross 0.5. In the 128 VGGish dimensions other acoustic details separate
  them a little better.
- **It is not caused by errors of the emotion regression.** The human ratings predict
  even more genres per clip (2.47) and have the highest Hamming loss (0.427). The effect
  belongs to the emotion space itself.

### 1.5 Is this a problem for the thesis? What to say

No, but it should be said openly, and the thesis already points the same way (Appendix
A.2: "macro recall 0.566 against 0.423 for VGGish").

> "The emotion route has the worst Hamming loss because it predicts more genres per
> clip: 2.35 against 1.54 true ones. That is the price of finding the rare genres. On
> Action and Drama it actually makes fewer errors than VGGish, and the whole Hamming gap
> comes from Comedy and Horror. Hamming loss is a poor headline here: a model that always
> says 'Drama' wins it, and the human ratings lose it. Under the metrics that weight the
> genres equally, the emotion route is first."

The honest counterpart, which the thesis also states: with **a decision threshold tuned
per genre**, the over-prediction is corrected for every route and the routes converge
(0.420 to 0.433 over clips; the emotion route lowest). Part of the emotion route's lead
at the default threshold is therefore an operating-point effect (section 5).

---

## 2. The design of the evaluation in one page

| decision | what was chosen | why |
|---|---|---|
| Unit that is scored | **the film** (clip probabilities averaged per film, then scored once); clip level beside it | genre is a property of the film; a film with 18 clips should not count nine times as much as one with 2 |
| Split | 10×5 cross-validation **grouped by film**, ten different shuffles | a film's clips sound alike; ungrouped splits leak (AST 0.44 ungrouped vs 0.23 grouped) |
| Hyper-parameter `C` | chosen inside each training fold (nested) | the test fold never influences a choice |
| Emotion features for training the genre model | out-of-fold predicted emotions | the classifier sees the same noisy emotions in training as at test time |
| Score per repeat | pooled over the five test folds, then averaged over repeats | per-fold macro-F1 is biased low (Comedy is missing from 17 of 50 folds) |
| Three label spaces (8 genres, 5 genres, 6 shared) | never compared with each other | different genres have different chance levels |
| Chance floor | base-rate random guess *and* most-frequent baseline | they differ by over 0.1; each metric has its own fair floor (section 4) |

---

## 3. Stage 1: emotion regression

| metric | what it means | why it is used | result (best) |
|---|---|---|---|
| **R²**, mean over the 8 emotions | share of the rating variance the model explains; 0 = always predicting the mean rating, 1 = perfect | standard in music-emotion recognition, comparable across emotions with different scales | CLAP 0.558, VGGish 0.553 |
| RMSE | typical error in rating points | gives the error in the units of the rating scale | 1.22 (CLAP) |
| % of ceiling | R² divided by 0.897, the agreement between two independent listener panels | a model cannot agree with the ratings better than the ratings agree with themselves | 62 % |

Why R² and not correlation: correlation ignores a constant offset or wrong scale; R²
punishes both. The "random" reference of Stage 1 is R² = 0 (predicting the mean rating),
since a random guess of a continuous rating has no natural definition.

---

## 4. Stage 2: genre classification, metric by metric

The thirteen metrics were all computed on **the same predictions**, so any difference
between them comes from the metric alone. "Random" is the base-rate random guess of that
metric; "dummy" is the most-frequent baseline (always Drama).

### 4.1 The headline

| metric | what it measures | why | emotion | VGGish | ratings | random | dummy |
|---|---|---|---:|---:|---:|---:|---:|
| **Macro-F1, film level** | F1 per genre over the 40 films, then the plain mean of the five | every genre counts the same, so a model must find Comedy and Horror, not only Drama; the film is the unit of genre | **0.460** | 0.367 | 0.491 | 0.309 | 0.168 |
| Macro-F1, clip level | the same over the 329 clips | the check beside the headline; film level was chosen after the clip results were known, and the thesis says so | 0.405 | 0.366 | 0.416 | 0.307 | 0.168 |

**Why F1 and not accuracy:** with 73 % Drama, accuracy rewards saying "Drama". F1
combines precision and recall, so it is high only if a model both finds a genre and is
right when it names it.

**Why the base-rate random guess is the floor for macro-F1:** it guesses each genre with
its true frequency, so it sometimes says "Horror" and gets a non-zero F1 on every genre.
The dummy never says "Horror", gets F1 = 0 on four genres and is an unrealistically low
floor for macro-F1 (but a hard one for Hamming loss and exact match).

### 4.2 How the F1 is reached: precision and recall

| metric | emotion | VGGish | reading |
|---|---:|---:|---|
| Macro precision | 0.363 | 0.341 | when it names a genre it is right about as often as VGGish |
| Macro recall | **0.566** | 0.423 | it finds many more of the true genres |

This is *why* the emotion route wins macro-F1: recall, mostly on the rare genres. It is
also why it loses Hamming loss (section 1). Reported because Ma et al. (Blockbuster)
report them, and because F1 alone does not show whether a model over- or under-predicts.

### 4.3 The threshold check

| metric | what changes | emotion | VGGish | PCA-8 | AST |
|---|---|---:|---:|---:|---:|
| Macro-F1, tuned threshold | each genre gets its own cut (0.05–0.95), chosen on the training folds only | 0.420 | 0.423 | 0.433 | 0.430 |

**Why:** 0.5 is a convention. Rare genres rarely reach a probability of 0.5, so the
default cut under-predicts them for some routes and not for others. Tuning the cut per
genre gives every route its best operating point. **Result:** the routes converge and the
emotion route is no longer ahead (−0.004, p = 0.813). So part of its lead is where it
puts its cut, not what it knows. This is the most important caveat of the in-domain
result, and it is stated in Chapter 5 and Appendix A.2.

### 4.4 Threshold-free: does the model rank correctly?

| metric | what it measures | emotion | VGGish | random |
|---|---|---:|---:|---:|
| Macro average precision | area under each genre's precision–recall curve, averaged | 0.370 | 0.357 | 0.320 |
| Macro ROC-AUC | chance that a true clip of a genre gets a higher score than a false one; 0.5 = chance | 0.592 | 0.542 | 0.500 |

**Why:** these ignore the cut entirely and ask only whether the probabilities are in the
right order. **Result:** the emotion route ranks first on both, but not significantly
(p = 0.352 and p = 0.063). Its ranking is at least as good as direct audio's, so its
default-threshold lead is not only a threshold artefact; but it is not clearly better
either.

### 4.5 Metrics dominated by the frequent genres (reported for transparency)

| metric | what it measures | emotion | VGGish | AST | random | dummy |
|---|---|---:|---:|---:|---:|---:|
| Weighted-F1 | F1 per genre weighted by genre frequency | **0.546** | 0.505 | 0.534 | 0.476 | 0.396 |
| Micro-F1 | one F1 over all 1645 clip–genre decisions pooled | 0.476 | 0.473 | 0.517 | 0.477 | **0.571** |
| Samples-F1 | F1 per clip over its genre list, averaged over clips | 0.479 | 0.453 | 0.497 | 0.437 | **0.574** |
| Jaccard | per clip: genres both predicted and true / genres predicted or true | 0.360 | 0.356 | 0.408 | 0.354 | **0.500** |
| Exact match | share of clips whose whole genre list is exactly right | 0.072 | 0.102 | 0.168 | 0.130 | **0.283** |
| Hamming loss ↓ | share of wrong clip–genre decisions | 0.406 | 0.358 | 0.311 | 0.323 | **0.218** |

**Why they are not the headline:** in five of these six, the dummy that always says
"Drama" beats every real model. A metric under which a model that has learned nothing
comes first cannot answer a question about Horror and Comedy. Among the real models they
prefer AST, the route that predicts the fewest genres (1.68 per clip).

**Why exact match is so low for everyone:** a clip is only counted as right if all five
decisions are right. The emotion route predicts 2.35 genres for clips that have 1.54, so
it almost never hits the exact list (7 % of clips). The dummy gets every clip that is
"Drama only" right (28 %).

**Why they are still reported:** leaving them out would look like choosing the metric
that suits the method. Showing them, together with the reason they favour the dummy, is
more convincing. Weighted-F1 is the one exception in this block, which the emotion route
wins (+0.042, p = 0.008), because it still scores F1 per genre.

### 4.6 Per genre

Film-level precision, recall and F1 for each genre (Chapter 5, Section 5.4). **Why:**
macro-F1 is an average of five very different genres; the per-genre view shows *where*
a route is strong. **Result:** the emotion route is best on Action (0.61), Crime (0.35)
and Comedy (0.34); MIR on Drama (0.82); CLAP on Horror (0.43). With 4 Comedy and 5 Horror
films, one film moves their F1 by about 0.2, so this part has no significance test.

---

## 5. Significance and spread

| tool | what it answers | why this one | where |
|---|---|---|---|
| **Paired film bootstrap** (2000 draws) | would the difference hold for another sample of films? Resamples whole films, keeps both routes' predictions paired | the only test that works at film level (a test fold holds about 8 films) | every film-level Δ and p |
| **Nadeau–Bengio corrected t-test** | would the difference hold for another training set? Uses the per-fold scores, corrected for overlapping training sets | the more conservative test; quoted beside the bootstrap at clip level so the friendlier test is never the only one | clip-level p_NB |
| p_lim | the smallest p-value Nadeau–Bengio can reach on this dataset, however many repeats | shows that more repeats cannot help, only more films (0.069–0.200 for the borderline comparisons) | Appendix A.1 |
| Repeats won | in how many of the ten repeats one route beats the other | easy to read, independent of any test | "won" column |
| SD over repeats | how much the score moves when films are shuffled into other folds (film level 0.034–0.045) | stability of the protocol | Appendix A.2 |
| Bootstrap SE | how much it depends on which films the dataset contains (0.027–0.038) | the main source of uncertainty | Appendix A.2 |
| Five master seeds | does anything the code controls change the result? (moves the mean by at most 0.019) | reproducibility | Appendix A.2 |

The main result in this frame: the emotion route is ahead of VGGish by +0.093 at film
level, in 9 of 10 repeats and under every seed, but p = 0.052. It is consistent, but on
40 films not proven.

---

## 6. Transfer between datasets

| design | metric | why |
|---|---|---|
| Zero-shot Eerola → Blockbuster | macro-F1 over the 110 films, P and R beside it; paired bootstrap over the 110 films | one fixed train–test split has no folds, so the bootstrap over test films is the test; P and R as in Ma et al. |
| Blockbuster → Eerola | macro-F1 over the 40 Eerola films; film bootstrap | same reason |
| Pooled (both in training) | repeated grouped CV over both datasets (5 repeats); Nadeau–Bengio | there are folds again |

Thresholds stay at 0.5 in every transfer: the target dataset has no labels that could
tune them, which is the realistic setting. Whether a threshold tuned on the *source*
dataset keeps the zero-shot lead (0.508 vs 0.403, p = 0.013) has not been tested; it is
open item 3 in `current_state.md`.

---

## 7. Summary: why macro-F1 over films is the headline

1. Genre is a property of the film, so the film is scored once.
2. The labels are imbalanced (Drama 73 %, Horror 8.5 %). Macro averaging gives every
   genre the same weight; a model that ignores rare genres is punished.
3. Under the alternatives that reward frequent genres (micro, samples, Jaccard, exact
   match, Hamming), a model that always says "Drama" wins. They do not measure the
   question.
4. It is the metric of the closest prior work (Ma et al.), so results are comparable.
5. Its weak spot, the fixed 0.5 threshold, is checked openly with tuned thresholds and two
   threshold-free metrics.

---

## 8. For Overleaf: a paragraph for Appendix A.2

Suggested place: `appendix/supplementary.tex`, Section "Does the Choice of Metric
Matter?", directly after the paragraph that ends "…which is why macro-$F_1$ is used.
Among the metrics that weight genres equally, …both threshold-free metrics." It uses
only numbers from `metrics_stability.json` and `hamming_breakdown.json`.

```latex
Hamming loss deserves a closer look, because under it the emotion route is the weakest
of the four audio routes (0.406) and even falls behind the random guess (0.323). The
reason is the number of genres it predicts: 2.35 per clip on average, against 1.54 true
genres, 1.85 for VGGish and 1.68 for AST. Hamming loss counts each of the five decisions
per clip equally, so an unnecessary positive costs as much as a missed genre, and for a
rare genre the lowest loss is reached by never predicting it, which costs only the
genre's base rate (0.085 for Horror). Split by genre, the emotion route makes fewer
errors than VGGish on Action (0.371 against 0.411) and Drama (0.356 against 0.396) and
about as many on Crime; the whole difference comes from Comedy (0.438 against 0.286) and
Horror (0.403 against 0.235). It predicts these two genres for 148 and 135 of the 329
clips, although only 40 and 28 carry them. This is the recall-driven operating point
described below, seen from the other side: macro-$F_1$ rewards finding a rare genre,
Hamming loss penalises every attempt. The human ratings predict even more genres per clip
(2.47) and have the highest Hamming loss of all (0.427), so the effect belongs to the
emotion features themselves, in which a tense crime clip resembles a horror clip, and not
to errors of the emotion regression.
```

If the paragraph goes in, copy the changed file to `docs/latex/chapters/supplementary.tex`
and check the compile on Overleaf.
