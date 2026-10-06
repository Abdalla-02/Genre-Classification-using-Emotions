# Supervisor briefing — update since the last meeting

**Can a soundtrack's emotion predict a film's genre?** This page covers only what happened
since the fourth meeting (24 September 2026): the six notes from that meeting, what came
of them, and the work on the thesis text that followed (state: 7 October 2026). Earlier rounds, the three routes (direct audio, emotion route, PCA-8
control), the definitions, every result table and the previous Q&A are in
`briefing_full.md`; every experiment in detail is in the technical log, `docs/README.md`
(section 22 for this round).

---

## 1. Your notes from the last meeting

| # | Your note | What was done | Result |
|---|---|---|---|
| 1 | Explain the F1 metric more; is there a better approach? | Every route scored under 13 metrics on the same predictions (section 2) | **Macro-F1 stays the right headline**, but the in-domain lead **depends on the decision threshold** |
| 2 | Standard deviation / variance of the cross-validation: stable and reproducible? | Spread over repeats, folds, films and five master seeds (section 3) | **Stable and reproducible**; the emotion lead is positive under every seed |
| 3 | Box office vs the Blockbuster dataset, under all metrics | Worldwide gross for all 110 films; 6 per-film metrics, emotions, genre, prediction, all 140 MIR features (section 4) | Classification quality: **no relation**. Emotions: **only through the budget**. Then **left out of the thesis** (7 Oct) |
| 4 | Write the rest of the chapters | Full drafts of every remaining chapter and the abstract (section 5) | **On Overleaf, compiles with 0 errors**; reviewed, style-revised, references checked. Abstract, conclusions and contributions held back for the author |
| 5 | Results as graphs, not only tables | Figures drawn directly from the results files (section 6) | Eight figures in the Evaluation chapter, with short captions |
| 6 | Figures from other papers are allowed, with the source | Noted (section 6) | The circumplex figure could now be the original |

---

## 2. Note 1 — the F1 metric, and whether there is a better one

### What F1 measures

For **one genre**, say Horror:

- **Precision** = of the clips the model *called* Horror, the share that really are.
- **Recall** = of the clips that really *are* Horror, the share the model found.
- **F1** = the harmonic mean of the two, `2·P·R / (P + R)`. It is high only when both are
  high: a model that calls everything Horror has recall 1 but poor precision, and one that
  calls almost nothing Horror has the opposite problem.

A clip can have several genres, so there is one F1 per genre, and they have to be
combined into one number. That is where the variants differ:

| averaging | how | what it rewards |
|---|---|---|
| **macro-F1** (ours) | F1 per genre, then the plain mean | every genre counts the same: Horror (28 clips) as much as Drama (239) |
| micro-F1 | pool all clip–genre decisions, one F1 | dominated by the frequent genres |
| weighted-F1 | F1 per genre, mean weighted by genre size | in between |
| samples-F1 | F1 per clip over its genre list, then the mean | dominated by the frequent genres |

### What the alternatives show

Same predictions, 5 genres, mean of 10 repeats (`results/metrics_stability.json`).
"Random" is a base-rate random guess, "most freq." always predicts the most common genres
(in practice: Drama).

| metric | emotion route | direct VGGish | direct AST | PCA-8 control | most freq. | random |
|---|---:|---:|---:|---:|---:|---:|
| **macro-F1 (headline)** | **0.417** | 0.377 | 0.371 | 0.372 | 0.168 | 0.308 |
| macro-F1, threshold tuned per genre | 0.440 | 0.429 | 0.427 | **0.441** | 0.168 | 0.308 |
| macro average precision (no threshold) | **0.372** | 0.361 | 0.359 | 0.356 | 0.309 | 0.321 |
| macro ROC-AUC (no threshold) | **0.608** | 0.567 | 0.571 | 0.567 | 0.500 | 0.501 |
| macro-F1 per film (clips averaged) | **0.469** | 0.375 | 0.307 | 0.418 | 0.169 | 0.301 |
| micro-F1 | 0.482 | 0.471 | 0.515 | 0.446 | **0.571** | 0.477 |
| exact match | 0.082 | 0.105 | 0.170 | 0.095 | **0.283** | 0.133 |
| Hamming loss (lower is better) | 0.404 | 0.359 | 0.314 | 0.393 | **0.218** | 0.323 |

**Three findings.**

1. **Macro-F1 is the right headline.** Micro-F1, samples-F1, exact match, Hamming loss and
   Jaccard are all *won by the most-frequent baseline*, a model that has learned nothing
   but "say Drama". A metric that ranks it first cannot answer a question about Horror or
   Comedy.
2. **Among the metrics that treat genres equally, the emotion route is first**: macro-F1,
   weighted-F1, both threshold-free metrics and the film-level score (0.469 vs 0.375,
   p = 0.041).
3. **But the in-domain lead depends on where the probability is cut.** Our classifiers
   say "yes, this genre" when their probability exceeds 0.5. If each genre's cut is
   instead tuned on the training data, every route improves and they converge
   (0.427–0.441). The emotion lead is then no longer significant (+0.011 vs VGGish,
   p = 0.44; level with PCA-8). The threshold-free metrics, which measure only how well
   clips are *ranked*, still put the emotion route first, but not significantly. At the
   default cut the emotion route wins mainly through **recall** (0.596 vs 0.439).

**The honest statement for the thesis:** in-domain, the emotion route loses nothing
against direct audio and is ahead at the standard operating point; it is not better under
every way of scoring. The **zero-shot** result (0.511 vs 0.407) is a different situation:
there are no labels on the target dataset to tune a threshold with. Whether that
advantage survives a threshold tuned on the *source* dataset has not been tested yet; it
is the one open check this raises (section 7).

---

## 3. Note 2 — standard deviation, stability, reproducibility

Macro-F1, 5 genres. Three sources of variation, measured separately:

| route | SD over the 10 repeats (fold assignment) | SD over the 50 single folds | film bootstrap SE (which films) | SD of the mean over 5 seeds |
|---|---:|---:|---:|---:|
| human ratings (ceiling) | 0.009 | 0.057 | 0.027 | 0.002 |
| **emotion route** | **0.011** | 0.054 | 0.024 | 0.003 |
| direct VGGish | 0.020 | 0.057 | 0.023 | 0.008 |
| direct AST | 0.023 | 0.050 | 0.018 | 0.001 |
| PCA-8 control | 0.015 | 0.061 | 0.026 | 0.004 |

- **Repeats** vary little (≤ 0.023); the emotion route is among the steadiest.
- **Single folds** vary a lot (≈ 0.055), because a fold holds only about 8 films. That is
  why one 5-fold run cannot resolve differences of 0.04, and why we repeat 10 times.
- **Which films are in the dataset** is the largest source (0.018–0.027). Only more
  films can reduce it.
- **Seeds.** The whole 10×5 protocol was re-run under five master seeds (42, 1, 2, 3, 4),
  which changes both the fold assignment and the random forest. The emotion route scores
  0.410–0.417 under every seed. Its lead over direct VGGish is positive under **every**
  seed (+0.037 to +0.059), as are its leads over AST and PCA-8. The PCA-8-vs-VGGish
  difference flips sign between seeds (−0.005 to +0.012), which is what "no effect" looks
  like.
- **Reproducibility.** Under seed 42 the new script reproduces the per-repeat scores of
  the main analysis (`cv_corrected.json`) exactly, for every route.

Figure: `figures/stability_repeats.pdf`.

---

## 4. Note 3 — box office on Blockbuster (done, then left out of the thesis)

**Data.** Blockbuster ships only title slugs, so each film was matched to Wikidata by
title and release year, and its **worldwide** gross taken from Box Office Mojo: all 110
films, one source, USD 9 million to 2.8 billion. The budget was found for 84 films.
Blockbuster has no ratings, so the emotions are those the Eerola-trained model predicts
per cue. Every correlation is Spearman with a permutation p-value and a
Benjamini–Hochberg correction for multiple tests.

| question | answer |
|---|---|
| Is a film's genre easier to predict when it grossed more? 6 per-film metrics × every route, in-domain and zero-shot: 42 tests | **No.** Nothing survives the correction; the largest correlation is ρ = +0.24 |
| Do the soundtrack's emotions track gross? | **Yes, at first sight**: anger ρ = +0.43, tension +0.36, fear +0.30, tenderness −0.37, valence −0.32, all significant after correction. **With the budget controlled, every one is within ±0.11.** |
| Does genre? | Action (median $475M vs $129M) and Sci-Fi ($615M vs $159M) gross more; Drama and Comedy less |
| Can gross be predicted from the soundtrack? | Partly (emotion R² = 0.28, VGGish 0.35), but the **budget alone does better (0.49)**, and adding the emotions to it does not help (0.45) |

**Reading:** blockbuster music mirrors the *scale and kind of production*. Expensive
action and sci-fi films are scored with tense, angry, energetic music and earn more;
budget and gross themselves correlate at ρ = +0.78. Once the budget is known, the
soundtrack adds nothing. Exploratory, not causal.

**A bug found on the way.** The Box Office Mojo parser read the navigation menu and
stored US-only figures. Fixed; on Eerola 7 grosses changed and every conclusion stayed
the same (quality vs gross ρ = +0.197, p = 0.246).

**Decision (7 Oct):** the box-office analysis is **not in the thesis**: it answers a
question about commercial success, not about genre. The text is kept in
`docs/latex/held_back/box_office.tex` and the analysis in the repository.

---

## 5. Note 4 — the chapters (now on Overleaf)

All chapters are written and **on Overleaf; the thesis compiles with 0 errors**.

| chapter | state |
|---|---|
| 1 Introduction | drafted; the list of contributions is **held back** for the author to write |
| 2 Fundamentals, 3 Related Work | written earlier; revised |
| 4 Methods | drafted (datasets, three label spaces, six representations, Stage 1 with the three derived features, Stage 2, baselines, protocol, implementation; TikZ pipeline figure) |
| 5 Experiments and Evaluation | drafted, with the result figures and the new section on metric and stability |
| 6 Discussion | drafted |
| 7 Conclusions and Future Work, abstract | **held back**: only headings and placeholders on Overleaf; drafts in `docs/latex/held_back/` |

What was done to the text after drafting:

1. **Missing references added** (librosa, scikit-learn, PyTorch, transformers, Lipton et
   al. 2014 on F1 thresholds; LAION-CLAP, see below).
2. **An examiner-style review** found factual slips, all corrected:
   - CLAP was cited to the wrong model (the checkpoint used is LAION-CLAP).
   - VGGish was said to be trained on AudioSet (it was trained on YouTube videos and is
     distributed with AudioSet).
   - The reliability ceiling 0.897 is an intraclass correlation, not r².
   - Clip length, film counts and dataset size were stated wrongly in places.
   - One argument was circular (the scaling ablation), one compared margins across label
     spaces that are not comparable.
   - A limitation was added: the five-genre subset was chosen partly by emotional
     signature.
3. **A consistency pass** over the whole thesis: about 25 wording and pointer fixes
   (numbers stated two ways, wrong cross-references, British spelling).
4. **Style revision** of the prose of every chapter (plainer sentences, fewer set phrases
   and dashes). A script checked that no number, reference, citation, table or figure
   changed.
5. **Captions shortened**: every figure and table has a one-line caption; the legend
   (colours, whiskers, columns) moved into the text after the first mention.
6. **Every reference and link checked**: all 36 bibliography entries against Crossref and
   OpenAlex (all real, all matching); the GitHub link and the three Hugging Face models
   resolve.
7. **AI-usage declaration filled in** from the repository and Overleaf history: models
   Claude Opus 5 and Claude Opus 5.5, July to October 2026, which parts were drafted,
   edited or generated. Three red notes remain that only the author can fill.

---

## 6. Notes 5–6 — figures

Eight figures, drawn by `experiments/make_figures.py` directly from the results files, so a
figure can never disagree with a table.

| figure | shows |
|---|---|
| `indomain_eerola5` | every representation and route on 5 genres, with 95 % intervals and both chance levels |
| `stability_repeats` | the 10 repeats per route, and the mean under each of 5 seeds |
| `metric_family` | the routes under 7 metrics, as distance above random guessing |
| `stage1_r2` | emotion-prediction R² per emotion × representation (heatmap) |
| `ablation` | removing each emotion: difference to all eight, with intervals |
| `signatures` | each genre's emotional signature, Eerola vs Blockbuster side by side |
| `zero_shot` | Eerola → Blockbuster: emotion 0.511 vs PCA-8 0.421 vs direct 0.407 |
| `cross_dataset` | both transfer directions and the pooled design |

![In-domain, five genres: every route with 95 % intervals](../figures/indomain_eerola5.png)

![Zero-shot transfer from Eerola to Blockbuster](../figures/zero_shot.png)

![Stability over repeats and seeds](../figures/stability_repeats.png)

![The routes under seven metrics](../figures/metric_family.png)

**Figures from other papers.** The circumplex figure in Fundamentals is redrawn "after
Russell (1980)"; the original could replace it with its source. Ma et al. (2021) is
published under CC BY 4.0, so its figures may be reused with attribution.

---

## 7. A correction since: the pooled cross-dataset design

The results file of the pooled design (both datasets in training) came from a 3×5 run,
while the script and the text said 5×5. It was re-run at 5×5 (7 Oct). The two transfer
directions reproduce exactly; the pooled numbers move slightly and the conclusion stays:

| pooled design | before (3×5) | now (5×5) |
|---|---:|---:|
| emotion route / direct / PCA-8 | 0.424 / 0.421 / 0.398 | **0.432 / 0.428 / 0.405** |
| emotion vs direct | +0.003, p = 0.906 | **+0.004, p = 0.850** |

Once both datasets are in training, the two routes tie: the emotion advantage is a
generalisation advantage.

---

## 8. Open items

1. **Write the held-back parts**: abstract, Chapter 7 (Conclusions and Future Work) and the
   list of contributions. Future Work must keep the experiment the Discussion points to
   (below).
2. **Zero-shot with a threshold tuned on the source dataset**: the one check the metric
   analysis raises that has not been run.
3. **AI-usage declaration**: three red notes (other tools, how the drafts were revised, the
   level for the held-back parts) and the date.
4. **Read the whole thesis once**, using `docs/verification_checklist.md`: what to check,
   the numbers that no results file stores, every correction made, and what to look at in
   the PDF (the two TikZ diagrams have not been checked visually).

---

## 9. Questions to expect, and the answers

**Why macro-F1 and not accuracy or Hamming loss?**
Because on this data those are won by a model that always says "Drama": exact match 0.283
and Hamming loss 0.218, better than every real model. Macro-F1 gives each genre the same
weight, so it cannot be won by ignoring the rare ones.

**Is the emotion route better in-domain or not?**
At the standard decision threshold, yes: ahead of every direct representation in all 10
repeats and all 5 seeds, significant under the film bootstrap (p = 0.042) and borderline
under Nadeau–Bengio (p = 0.151). With per-genre thresholds tuned on the training data, the
routes converge and the lead is not significant. So in-domain it "loses nothing and is
ahead at the default operating point", not "better under every metric".

**Then what is left of the main claim?**
The cross-dataset result: 0.511 vs 0.407 (p = 0.018) from Eerola to Blockbuster, the
reverse direction agreeing (p < 0.001), a tie once both datasets are in training
(p = 0.850), and interpretable predictions whose emotional signatures replicate on a second
dataset (r = 0.836). The advantage is a generalisation advantage; the thesis says so.

**Would threshold tuning also erase the zero-shot advantage?**
Unknown: not tested yet (open item 2).

**How stable are the numbers?**
Very: the 10-repeat mean moves by at most 0.008 between seeds, and the emotion lead over
VGGish stays between +0.037 and +0.059 under every seed. The main uncertainty is which
41 films the dataset contains (bootstrap SE about 0.02).

**Why is the box-office analysis not in the thesis?**
It was done (section 4) and its answer is clear: classification quality does not relate to
gross, and the emotions of the score relate to it only through the budget. It answers a
question about commercial success rather than about genre, so it was left out.

**How was AI used?**
As a programming and writing assistant (Claude Opus 5 and 5.5): code, documentation, chapter
drafts, figures and reference checking. The declaration at the end of the thesis lists
every part and its level; the research questions, datasets and decisions are the author's.

---

## Headline numbers as they stand

Each row is its own label space, compared only against its own chance level (base-rate
random guess):

| setting | emotion route | best direct audio | random guess |
|---|---:|---:|---:|
| Eerola, 5 genres, in-domain (default threshold) | 0.417 | 0.377 | 0.308 |
| Eerola, 5 genres, in-domain (tuned thresholds) | 0.440 | 0.429 | 0.308 |
| Eerola → Blockbuster, zero-shot (6 genres) | **0.511** | 0.407 | 0.292 |
| Blockbuster, in-domain (6 genres) | 0.616 | 0.593 | 0.295 |

In-domain significance (default threshold): emotion vs VGGish film bootstrap p = 0.042,
Nadeau–Bengio p = 0.151. Zero-shot: +0.106 [+0.015, +0.186], p = 0.018. On 8 genres the
pipeline scores 0.318 against 0.276 for AST (most-frequent baseline 0.102). The 5-genre
random guess is 0.308, the simulated value stored in `metrics_stability.json`.
