# Supervisor briefing — update since the last meeting

**Can a soundtrack's emotion predict a film's genre?** This page covers only what happened
since the last meeting: the six notes from that meeting, and one correction made since.
Earlier rounds, the definitions and every result table are in `briefing_full.md`; every
experiment in detail is in the technical log, `docs/README.md`.

---

## 1. Your notes from the last meeting

| # | Your note | What was done | Result |
|---|---|---|---|
| 1 | Fundamentals: deeper, for a CS student new to ML | Three new sections on Overleaf: audio representations, machine-learning background, how results are measured | Written. Three citation errors fixed along the way (see section 4) |
| 2 | Remove some emotions, test the difference | Every emotion left out in turn, plus theory-motivated subsets | **No difference.** No single emotion is needed (section 2.1) |
| 3 | Cross-validation across the datasets | Eerola → Blockbuster, Blockbuster → Eerola, and both pooled | **Emotion wins in both transfer directions**, ties when both corpora are in training (section 2.2) |
| 4 | Box office vs classification (Eerola) | Gross fetched by IMDb id for 37 films | **Exploratory; no link** between classification quality and gross (section 2.3) |
| 5 | Start the next chapters, generally | Every chapter now carries notes saying what it must establish and where the facts are | Methods, Evaluation, Introduction, Discussion, Conclusions: skeletons ready to write over |
| 6 | AI-usage and authorship declarations | Official FIN *Statement of Authorship* in the thesis; AI-usage page drafted | **The AI-usage page must be checked and completed by you** (section 5) |

---

## 2. Results for notes 2–4

### 2.1 Removing emotions (note 2)

5-genre subset, human ratings, macro-F1 (corrected protocol, section 3):

| emotions used | macro-F1 | vs all eight |
|---|---:|---|
| all eight | 0.426 | reference |
| fear only | 0.433 | +0.007, p = 0.53 |
| valence + energy only | 0.420 | −0.006, p = 0.52 |
| any one emotion left out | 0.423–0.431 | at most ±0.005 |

The ratings are strongly intercorrelated (fear rises with tension and falls with valence),
so the genre signal can be recovered from almost any pair. This **corrected a claim in the
Fundamentals chapter**, which had said the discrete emotions were *necessary* to tell
Action from Horror. Fear is the most diagnostic emotion, but it is not required. All eight
are kept for interpretability, not for accuracy.

### 2.2 Cross-validation across the datasets (note 3)

Shared 6-genre space, VGGish (the only representation both corpora have):

| design | direct audio | PCA-8 control | **emotion** | emotion vs direct |
|---|---:|---:|---:|---|
| train Eerola → test Blockbuster | 0.395 | 0.465 | **0.508** | +0.115, p = 0.016 |
| train Blockbuster → test Eerola | 0.320 | 0.352 | **0.370** | +0.050, p < 0.001 |
| both corpora in training | 0.421 | 0.398 | 0.424 | +0.003, p = 0.906 |

The advantage holds in **both** directions and disappears once both corpora are in the
training data. It is a *generalisation* advantage: an eight-dimensional representation
whose axes mean the same in every corpus survives a change of corpus; a 128-dimensional
embedding does not.

### 2.3 Box office (note 4, exploratory)

Gross was fetched by IMDb id from Wikidata and Box Office Mojo for 39 films, leaving
**37** after excluding two films whose IMDb id points at the wrong film and one with no
clips.

- Classification quality does **not** relate to gross (Spearman ρ = +0.20, p = 0.24).
- The only robust effect is that **Action films gross more** (median $267M vs $41M,
  p = 0.004). That is a fact about genre, not about the soundtrack.
- Anger correlates with gross (ρ = +0.34, p = 0.045), but that is Action again: it drops to
  +0.19 once Action is controlled for.

Caveats: grosses mix worldwide and domestic-only figures, nothing is inflation-adjusted,
and 37 films span four decades. One paragraph in the Discussion, not a result.

---

## 3. A correction since: the cross-validation

A check of the folds behind every in-domain Eerola number found two problems. Neither
leaks test data into training; both distort the numbers.

1. **Rare genres are missing from many test folds.** A fold contains whole films, and
   Comedy has only 4 films, so it is absent from 17 of the 50 test folds (Horror from
   15). Scoring each fold separately counts such a genre as zero there, however good the
   model is. **Fix:** score each repeat once, over all of its test predictions together.
2. **The genre classifier trained on in-sample emotion predictions** and was tested on
   out-of-sample ones. **Fix:** out-of-fold emotions, as the transfer experiments already
   did.

The corrected run reproduces the old numbers exactly under the old metric, so every change
comes from the two fixes:

| 5 genres (329 clips / 41 films) | before | **corrected** |
|---|---:|---:|
| human ratings (ceiling) | 0.397 | **0.428** |
| full pipeline: audio → predicted emotion → genre | 0.394 | **0.417** |
| best direct audio representation (VGGish) | 0.350 | 0.377 |
| AST / CLAP / MusiCNN / MIR, direct | 0.335–0.345 | 0.361–0.371 |
| chance (most frequent) | 0.166 | 0.168 |

On all 8 genres: pipeline **0.318** vs AST 0.276 (chance 0.102).

**What changed and what did not.**
- Every score rose by 0.02–0.03, and the ranking stayed the same.
- The *gap* between emotion and direct audio stayed the same (about +0.04–0.05).
- Fix 2 made no measurable difference.
- Significance is now reported under two tests:
  - The pipeline is ahead of every direct representation in **all 10 repeats**.
  - On 8 genres that is significant under both tests.
  - On 5 genres it is significant under the film bootstrap (the test used for the
    zero-shot result) but borderline under the stricter Nadeau–Bengio test (p = 0.07–0.15).
- **The honest statement:** in-domain, emotion is modestly and consistently ahead; across
  corpora it is clearly ahead.

---

## 4. The thesis document on Overleaf

- **It compiles cleanly**, corrected numbers included: 0 errors, and the 3 remaining
  warnings come from template packages.
- **Fixed:**
  - Every conference paper in the bibliography failed to render: a bug in the
    bibliography style file.
  - VGGish was cited to the wrong paper, and AST and CLAP were not cited at all.
  - A chance-level sentence in Fundamentals mixed figures from two corpora.
  - The template's placeholder pages were still in: a motto about *cluster analysis* and
    an "ABC — short for …" abbreviations list.
- **Typography:** words are no longer split across lines with a hyphen.

---

## 5. Open items

1. **Abstract** — still a placeholder, marked red on Overleaf with what it has to cover.
2. **AI-usage declaration** — the page is a draft describing the AI's own involvement. You
   must read it, correct anything that does not match what happened, and add anything it
   could not know (e.g. other tools you used). The regulation (Fakultätsratsbeschluss
   019/25) requires the system named, the affected sections marked, the level of use
   explained and the reason stated.
3. **Content chapters** — the writing task. Every chapter lists its facts and sources.
4. **Three ready-to-paste LaTeX sections still have the pre-correction numbers**
   (`statistical_power.tex`, `emotion_regression_bridge.tex`, the genre table of
   `waveform_vs_spectrogram.tex`). The Overleaf notes say so; update before pasting.
5. **More films** remain the only lever on the borderline 5-genre comparisons.

---

## Headline numbers as they stand

Each row is its own label space, compared only against its own chance level:

| setting | emotion route | best direct audio | chance |
|---|---:|---:|---:|
| Eerola, 5 genres, in-domain | 0.417 | 0.377 | 0.168 |
| Eerola, 8 genres, in-domain | 0.318 | 0.276 | 0.102 |
| Eerola → Blockbuster, zero-shot (6 genres) | **0.511** | 0.407 | 0.292 |
| Blockbuster, in-domain (6 genres) | 0.616 | 0.593 | 0.295 |

Zero-shot difference: +0.106 [+0.015, +0.186], p = 0.018; against the PCA-8 control +0.092,
p = 0.032. Our in-domain reproduction of Ma et al. (2021) scores 0.620, inside their
published 0.61–0.65.
