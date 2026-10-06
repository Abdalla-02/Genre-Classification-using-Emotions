# Ready-to-paste thesis sections — how to use them

The eight `.tex` files here are drafts of thesis sections, written from the saved results.
They were checked against the Overleaf project as it stood on 23 September 2026
(Overleaf commit `4ebbe0a`):

- **Numbers:** every figure traces to a file in `results/`, or was re-verified by
  re-running the script that prints it. The exceptions are listed at the end.
- **Citations:** every citation key exists in `bib/library.bib` on Overleaf.
- **Cross-references:** every `\ref` resolves, either to a label the thesis already has
  or to one defined by another section here. The paste order below takes care of that.
- **Terminology:** a *clip* (never "excerpt") and a *dataset* (never "corpus")
  throughout, and numbers are written 0.xxx.

**Nothing has been pasted into Overleaf.** These are for you to read, adapt and add.

## Held back from the thesis: `held_back/`

On 6 Oct 2026 three parts were taken **out** of the Overleaf build, to be rewritten by the
author. Their drafts are kept here, outside the Overleaf project, and are not compiled:

| file | was | on Overleaf now |
|---|---|---|
| `held_back/abstract.tex` | the abstract on the title page | a red placeholder |
| `held_back/4_conclusion.tex` | the whole Conclusions and Future Work chapter | the chapter heading, both section headings and their labels (other chapters refer to them), with a `\mytodo` note each |
| `held_back/intro_contributions.tex` | the end of "Aim of this thesis": the list of contributions | removed; the section ends after the research questions |
| `held_back/box_office.tex` | the box-office section of the Evaluation, its figure, and the box-office paragraph of the Limitations (removed 7 Oct 2026 as not relevant) | removed completely |

At the same time the prose of every other chapter was revised for style (shorter, plainer
sentences, fewer dashes and set phrases). No number, label, reference, citation, table or
figure changed; a script compared each chapter with its previous version.

## New since the fourth meeting: whole chapters in `chapters/` (now on Overleaf)

**Status, 5 Oct 2026:** all six files below are on Overleaf, with the figures, the
missing references and the fixes of an examiner-style review; the project compiles with
0 errors. The files here mirror the Overleaf versions.

`docs/latex/chapters/` holds a complete draft of every chapter that was still a skeleton.
Each file **replaces the Overleaf file of the same name as a whole**, so the ten-step paste
plan below is no longer needed: the drafts already contain the sections it lists,
adapted to each other.

| file | replaces on Overleaf | contents |
|---|---|---|
| `chapters/0_intro.tex` | `chapter/0_intro.tex` | motivation, the research questions, contributions, structure |
| `chapters/2_analysis.tex` | `chapter/2_analysis.tex` | Methods: datasets and the three label spaces, representations, Stage 1 (incl. the three derived features), Stage 2, baselines, the evaluation protocol, implementation; the pipeline figure in TikZ |
| `chapters/3_evaluation.tex` | `chapter/3_evaluation.tex` | every result, with the figures; **new section "Robustness: Choice of Metric and Stability"** (supervisor notes 1 and 2) and the Blockbuster box-office analysis (note 3) |
| `chapters/discussion.tex` | `chapter/discussion.tex` | interpretation, limitations, relation to prior work |
| `chapters/4_conclusion.tex` | `chapter/4_conclusion.tex` | now only the headings and labels; the draft is in `held_back/` |
| `held_back/abstract.tex` | the abstract on the title page | held back since 6 Oct (see above) |

**Figures.** The chapters include `figures/<name>.pdf`. Upload the repository's `figures/`
folder to the Overleaf project root (the PDFs only; the PNGs are previews). Every figure
is drawn by `experiments/make_figures.py` from `results/*.json`.

**Before pasting, search each file for `% CITATION NEEDED`.** Three places need a
reference that is not in `bib/library.bib` yet: librosa (McFee et al., 2015); PyTorch,
Hugging Face transformers and scikit-learn; and threshold tuning for F1 (Lipton et al.,
2014). Add the BibTeX entries, then the `\citep`.

Nothing has been compiled on Overleaf yet. Replace one chapter at a time and recompile.

---

## Three rules before pasting

1. **Keep every skeleton heading and its `\label` on Overleaf, and replace only the
   `\mytodo` note under it.** Four skeleton labels are referenced from elsewhere
   (`sec:eval_genre`, `sec:eval_direct`, `sec:eval_signatures`, `sec:eval_transfer`), and
   several sections here point to skeleton labels too. Deleting a skeleton heading
   leaves "??" in the PDF. Where a section below starts with its own `\section` or
   `\subsection` line, drop that line or demote it to fit under the skeleton heading.
   Keep its `\label` next to the skeleton's.
2. **Paste in the order of the table below.** A few sections refer to labels another one
   defines.
3. **Recompile after each paste.** Expected: 0 errors. The 3 warnings that come from
   template packages are normal.

---

## Where each section goes, in paste order

| # | file | goes to (Overleaf) | status | what to do |
|---|---|---|---|---|
| 1 | `evaluation_protocol.tex` | `2_analysis.tex` → "Evaluation Protocol" | **ready** | Replace the whole skeleton section, including its three `\mytodo` subsections. This file carries the same label, `sec:evaluation_protocol`, so delete the skeleton's `\section` and `\label` lines, or the label is defined twice. Consider shortening the grouped-CV/leakage and macro-F1 paragraphs to a back-reference: Fundamentals already explains those concepts. |
| 2 | `genre_subset.tex` | `2_analysis.tex` → "Genre Labels", point (b) | **ready** | One paragraph justifying the 5-genre subset. |
| 3 | `cross_dataset_transfer.tex`, first part | `2_analysis.tex` → "Genre Labels", point (c) | **ready** | Only the subsection "Constructing a Shared Genre Space" with its table. |
| 4 | `waveform_vs_spectrogram.tex`, first part | `2_analysis.tex` → "Audio Representations" | **ready** | The introduction (six representations, the wav2vec and MusiCNN rationale) and "Selecting the Layer to Pool". |
| 5 | `waveform_vs_spectrogram.tex`, stage-1 part | `3_evaluation.tex` → "Stage 1: Emotion Regression" | **ready** | The first two "Results" paragraphs and `tab:waveform_stage1`: the six-representation R² table the skeleton asks for. |
| 6 | `rating_reliability.tex` | `3_evaluation.tex` → "Stage 1: Emotion Regression" | **ready** | The R² ceiling of 0.897, i.e. how good emotion prediction could possibly be. Its Set 2 description could go to Methods, "The Eerola Film Soundtrack Corpus", instead. |
| 7 | `statistical_power.tex`, RESULTS part only | `3_evaluation.tex` → "Genre Classification from Emotion" (`tab:power_arms`) and "Emotion versus Direct Audio" (`tab:power_cmp`, the four findings, the 8-genre paragraph) | **ready** | **Skip its METHODS part** (marked in the file): step 1 already covers the same protocol. Keep `\label{subsec:statistical_power}`; two other sections refer to it. |
| 8 | `waveform_vs_spectrogram.tex`, genre part | `3_evaluation.tex` → "The waveform representation" | **ready** | `tab:waveform_stage2` and the two findings (wav2vec 0.326 direct → 0.419 through emotion). |
| 9 | `emotion_genre_relationship.tex`, "Evidence from the present dataset" only | `3_evaluation.tex` → "Emotional Signatures of Genres" | **ready** | The Cohen's d table per genre. **Skip the two parts before it** (marked in the file): Related Work already contains them. |
| 10 | `cross_dataset_transfer.tex`, the rest | `3_evaluation.tex` → "Cross-Corpus Transfer" | **ready** | Reproduction of Ma et al., zero-shot transfer, per-cue vs per-film. Keep this file's `\label{sec:zero_shot}` beside the skeleton's `sec:eval_transfer`, because step 1 refers to it. The subsection "Do the Emotion–Genre Signatures Replicate?" can move to "Emotional Signatures". |
| — | `emotion_regression_bridge.tex` | — | **do not paste as a whole** | Mostly superseded (marked in the file): its regression table is replaced by step 5, its Eerola results by step 7. Worth salvaging: the **regressor-choice paragraph** (Ridge 0.37 / SVR 0.54 / random forest 0.56) → Methods, "Stage 1", and the **face-validity paragraph** on Blockbuster → Evaluation, "Blockbuster in-domain". |

---

## Not covered by any section: write these from the `\mytodo` notes

The skeleton notes on Overleaf list the facts and where they are for each of these:

- **Introduction, Discussion, Conclusions.**
- **Evaluation → "Which emotions are needed? (ablation)"** — `results/cv_corrected.json`,
  ablation block.
- **Evaluation → "Error analysis"** — technical log §7c, §7e.
- **Evaluation → "Cross-Corpus Transfer", point (5)** — the reverse direction (Blockbuster →
  Eerola, +0.050, p < 0.001) and the pooled design (+0.003, p = 0.906), from
  `results/cross_dataset_cv.json`. `cross_dataset_transfer.tex` covers only Eerola →
  Blockbuster.
- **Evaluation → "Blockbuster in-domain"** — `results/blockbuster_deep.json`.
- **Evaluation → "Box Office"** — `results/box_office.json`, one short exploratory paragraph.
- **Methods → Datasets, Stage 1, Stage 2, Baselines, Implementation** — plain facts,
  listed in the notes.

---

## Also fix on Overleaf when you get to it

**Fundamentals, "Metrics", the macro-F1 paragraph** says a trivial classifier "scores
*better* than every real model on both exact-match accuracy and Hamming loss". Whether
that is true depends on the label space:

- **5 genres** (the headline protocol, `results/metrics_stability.json`): true as
  written. The most-frequent baseline wins outright on exact match (0.283 against at most
  0.170) and on Hamming loss (0.218 against at least 0.314).
- **8 genres** (`experiments/genre/exp_genre.py`, older protocol): it wins on Hamming
  loss (0.185 against 0.188 to 0.431) but only **ties** the best model on exact match
  (0.116).

Simplest fix: say which setting is meant, e.g. "*on the five-genre task, a trivial
classifier ... scores better than every real model on both exact-match accuracy and
Hamming loss*".

---

**"Corpus" → "dataset" on Overleaf.** These sections now say *dataset* everywhere.
The chapters already on Overleaf still say *corpus* 47 times: 10 in Discussion, 9 in
Fundamentals, 8 in Methods, 4 each in Related Work and Conclusions, 3 in Evaluation,
and 2 each in the Introduction and the abstract note. Change those too, or the thesis
mixes the two words. The skeleton heading "Cross-Corpus Transfer" (step 10 above)
then becomes "Cross-Dataset Transfer". A careful find-and-replace works:
"corpus" → "dataset", "corpora" → "datasets". Don't touch `\label`, `\ref` or
`\cite` keys; none of them contains the word today.

---

## Figures that come from a printed run rather than a results file

Each of these was re-verified today by re-running its script. None is a headline number,
but none is protected by the consistency audit either:

| figure | section | where it comes from |
|---|---|---|
| Cohen's d per genre (Horror fear +0.75, …) | `emotion_genre_relationship.tex` | `experiments/genre/exp_emotion_genre.py` |
| Ridge 0.37 / SVR 0.54 / random forest 0.56 | `emotion_regression_bridge.tex` | `experiments/emotion/exp_emotion_improve.py` |
| face validity: horror → fear +0.99, romance → tenderness +1.05 | `emotion_regression_bridge.tex` | `experiments/cross_dataset/exp_blockbuster_emotion.py` |
| per-emotion panel agreement (0.859 … 0.988) | `rating_reliability.tex` | `experiments/features/exp4_rating_reliability.py` |
| trivial classifier vs real models on exact match / Hamming | `evaluation_protocol.tex` | `experiments/genre/exp_genre.py` |
| median audio cross-correlation **0.964** between matched Set 1/Set 2 clips | `rating_reliability.tex` | a one-off check, recorded only in a comment in `src/features/loader.py`; **no script reproduces it** |

The last one is a data-alignment detail, not a result, but it is the single number here
that cannot be re-derived. If it is questioned, it should be reproduced properly.
