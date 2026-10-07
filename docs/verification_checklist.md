# Thesis verification checklist

Everything you should read again, decide, or check yourself before submission, collected
from the whole project (state: 7 October 2026, Overleaf commit `8658777`). Tick items off as
you go. Sections A and B are the ones that need you; C to E are the record of what was
found wrong and corrected, so you can confirm you agree with each correction.

Where a location is given as "§N" it refers to the technical log (`docs/README.md`).

---

## A. Only you can do these

- [ ] **AI-usage declaration** (last page). Written by the assistant from the repository
  and Overleaf history. Check every line, then fill the remaining red notes:
  - [ ] any other tool you used (translation, grammar checker, other chatbots), or delete
        the note;
  - [ ] how you read, revised and adapted the drafted chapters (Introduction, Methods,
        Evaluation, Discussion);
  - [x] the level for Chapter 7, the abstract and the list of contributions: filled in on
        Overleaf ("edited", 6 Oct);
  - [ ] the date next to the signature (its red box is also the cause of the one
        remaining "overfull box" warning; it disappears when you type the date).
  - Facts it states that you should confirm: models **Claude Opus 5** (9-22 Sep 2026) and
    **Claude Opus 5.5** (from 23 Sep 2026), both from the repository's commit trailers;
    period **July to October 2026**; the Claude.ai Overleaf connector was used from
    27 July; Fundamentals 2.1-2.3 and the Related Work text first came in through that
    connector. If you used another model in the Claude.ai chat before September, add it.
- [ ] **Write the held-back parts.** Drafts are in `docs/latex/held_back/` (not compiled):
  - [ ] Abstract (`abstract.tex`);
  - [ ] Chapter 7, Conclusions and Future Work (`4_conclusion.tex`). Keep the two section
        labels. Future Work must contain the **zero-shot comparison with per-genre
        thresholds tuned on the source dataset**, because the Discussion points to it;
  - [ ] the list of contributions at the end of Section 1.2 (`intro_contributions.tex`).
- [ ] **Box office** was removed on 7 Oct (`held_back/box_office.tex` keeps the text). If
  your supervisor asks about it, the answer is in `docs/current_state.md`, section 4.
- [ ] **Title page:** check names of the examiners, submission date, title.

## B. Read again: text produced by the assistant

Every chapter except the held-back parts was drafted or revised by the assistant, and on
6 October the wording of Chapters 1-6 was revised again for style (numbers, references
and tables were checked to be unchanged). Read each chapter once completely. What to look
at in particular:

- [ ] **Introduction.** The motivation makes general claims without citations (genre
  metadata organises archives and drives recommendation; listeners infer genre from a few
  seconds of music). Keep, cite, or soften.
- [ ] **Fundamentals.**
  - [ ] the descriptions of Gorbman (1987) and Juslin (2013): do they say what the text
        attributes to them?
  - [ ] the VGGish description: it now says VGGish was trained on a large YouTube video
        collection (a preliminary YouTube-8M) and is *distributed* with AudioSet. Check
        this against Hershey et al. (2017) yourself;
  - [ ] it quotes results before the Evaluation does (ablation outcome, ridge vs random
        forest, the 0.44/0.26 leakage numbers). An examiner may prefer forward
        references; optional restructuring.
- [ ] **Related Work.** Check each paper's description against the PDF: Austin et al.
  2010, Ma et al. 2021, Mangolin et al. 2020, Behrouzi et al. 2023, Kang & Herremans 2025,
  Eerola 2011 (genre-specific), Saari et al. 2016, Hu & Downie 2007, Laurier et al. 2009.
  Two descriptions were corrected earlier after checking the papers.
- [ ] **Methods.** Implementation facts that are easy to get wrong: which part of each
  clip each model sees (AST, MusiCNN, wav2vec: first 10.24 s; CLAP: a 10 s crop; VGGish
  and MIR: whole clip), sampling rates, checkpoints, the C grid, the three derived
  features (formulas in Stage 1), the 319 vs 360 clips used for the emotion regressor in
  the two transfer implementations.
- [ ] **Evaluation.** The robustness section (metric family, thresholds, stability) and
  the transfer section (PCA-8 claims) are the most delicate; make sure the wording says
  what you are willing to defend.
- [ ] **Discussion.** It interprets the results. Make the interpretation your own; in
  particular "Interpreting the Main Result" and "Why the Bottleneck Helps".
- [ ] **Film-level switch (7 Oct, Overleaf `d84fe44`, `8658777`).** Every genre number in
  Methods (evaluation protocol), Evaluation and Discussion was rewritten to film-level
  macro-F1, with clip level beside it, and six figures and the comparison table were
  redrawn. The Discussion's main result changed from "a generalisation advantage" to "an
  advantage when trained on few films or applied to another dataset". Read these parts
  again in full, and the new Limitations paragraph "Unit of evaluation". The held-back
  drafts still quote clip-level numbers (each has a note at the top).
- [ ] **Figures and table legends** were moved out of the captions into the text right
  after the first mention of each figure or table (7 Oct). Check that each legend reads
  naturally where it now stands.

## C. Claims an examiner is likely to challenge (know your answer)

- [ ] **The film-level unit was chosen after the clip-level results were known.** The
  thesis says so (Limitations, "Unit of evaluation") and keeps clip level beside every
  result. Know the four places where the two disagree: MusiCNN and PCA-8 (significant
  over clips, not over films), predicted vs rated emotion (level over clips, ratings
  ahead over films, p = 0.053), and the reverse transfer (significant only under a
  clip-resampling bootstrap).
- [ ] In-domain (5 genres, film level) the emotion route is ahead of VGGish by +0.095
  (p = 0.047) in every repeat and seed, but **not significantly ahead of MusiCNN
  (p = 0.079) or the PCA-8 control (p = 0.326)**, and the **PCA-8 control itself beats
  VGGish under every seed**. Over clips it is **borderline under Nadeau-Bengio**
  (p = 0.151 vs VGGish) and **not significant once per-genre thresholds are tuned**
  (+0.011, p = 0.439); thresholds were never tuned per film. The thesis says so.
- [ ] The **zero-shot** result (0.511 vs 0.407, p = 0.018) is one split, one
  representation (VGGish), and partly recall-driven (precision 0.488 vs 0.576 for direct
  audio). A source-tuned threshold was not tested.
- [ ] The win over the **PCA-8 control** across datasets is significant only in the
  original zero-shot run (p = 0.032), not in the re-implementation (+0.045, p = 0.258).
- [ ] The **5-genre subset** was chosen partly by emotional signature, which can favour
  the emotion route (stated as a limitation; the 8-genre results guard against it).
- [ ] **Dataset size:** 41/43 films; for the borderline comparisons no number of repeats
  could reach p < 0.05 (p_lim > 0.05).
- [ ] **Blockbuster emotions are predicted**, never rated; the signature replication
  (r = 0.836) cannot show independence from the Eerola listener panel.
- [ ] **Label noise:** genres are film-level IMDb labels applied to every clip.

## D. Numbers in the thesis that no results file stores

They come from scripts that only print their output, or from a one-off check. Re-run the
script before submission, or be ready to explain where the number comes from.

| number in the thesis | where | source |
|---|---|---|
| AST 0.44 ungrouped vs 0.26 grouped (leakage) | Fundamentals, Methods, Evaluation | `experiments/genre/exp_genre.py` (8 genres, single untuned five-fold run) |
| ridge 0.37 / SVR 0.54 / random forest 0.56 | Methods, Stage 1 | `experiments/emotion/exp_emotion_improve.py` |
| 3.49 vs 1.87 predicted vs true labels per clip | Evaluation, error analysis | `experiments/diagnostics/exp_error_analysis.py` (older protocol) |
| learning curve still rising | Evaluation, Discussion | `experiments/diagnostics/exp_learning_curve.py` |
| Cohen's d per genre (Horror-fear +0.75, the 8-genre signature table) | Fundamentals, Evaluation | `experiments/genre/exp_emotion_genre.py` |
| (removed) the 8-genre random-guess floor is now 0.22 at film level, simulated and stored in `film_level.json` | Fundamentals, Evaluation | `exp_film_level.py` |
| VGGish per-dimension means of the two datasets correlate at r = 0.97 | Methods, Evaluation | earlier zero-shot analysis (log §15) |
| wav2vec layer 2: R² 0.324 (sweep) vs 0.323 (main run) | Methods | two different runs, both stated |

## E. Errors found and corrected along the way (confirm you agree)

**Results and method**
- [ ] **Blockbuster → Eerola significance** (§23.3): the clip-level p < 0.001 came from a
  bootstrap that resampled clips, which are not independent. Resampling films gives
  +0.077, p = 0.127. The thesis now says the reverse direction agrees in sign but is not
  significant.
- [ ] **Cross-validation scoring** (§21): per-fold macro-F1 scored rare genres as 0 in
  folds that lack them; now scored once per repeat on pooled predictions. All in-domain
  levels rose by 0.02-0.03 (e.g. pipeline 0.394 -> 0.417); rankings unchanged.
- [ ] **In-domain emotion training** used in-sample predictions; now out-of-fold (no
  measurable difference, p = 0.37).
- [ ] **Regularisation:** sklearn's default C = 1 was the worst setting for every feature
  set; C is now tuned by nested CV.
- [ ] **VGGish vs hand-crafted features on Blockbuster** (§17): the earlier "VGGish wins"
  came from comparing against the MFCC subset only; against the full MIR set the
  difference is +0.009, p = 0.787.
- [ ] **Ablation** (§18): an earlier claim that specific emotions are necessary was
  wrong; no single emotion is needed. Fundamentals was corrected.
- [ ] **Metric analysis** (§22.1, 4th meeting): the in-domain lead depends on the decision
  threshold. Added to Evaluation, Discussion and the Q&A.
- [ ] **Pooled cross-dataset design** (§22.4): the results file held a 3x5 run while the
  script says 5x5. Re-run at 5x5 on 7 Oct; numbers updated everywhere (emotion 0.432,
  direct 0.428, PCA-8 0.405; +0.004, p = 0.850). Conclusion unchanged.
- [ ] **Box Office Mojo parser** (§22.2) read the navigation menu and stored US-only
  grosses; fixed and the Eerola analysis re-run. (Box office is no longer in the thesis.)
- [ ] **Data quality** in the enriched Eerola CSV: the IMDb ids of *Blanc* and *Pride and
  Prejudice* point at the wrong films (excluded from the box-office analysis).

**Text, found in the examiner-style review (5 Oct) and the consistency pass (6 Oct)**
- [ ] CLAP was cited to the Microsoft model; the checkpoint used is LAION-CLAP, now cited
  as Wu et al. (2023).
- [ ] The reliability ceiling 0.897 is an intraclass correlation (ICC(C,1)) on 102 matched
  clips, not r squared (0.91 squared would be 0.83).
- [ ] Clips are about 17 s long (not 10 s) for VGGish; the shared space has 41 films,
  not 43; the Eerola dataset has 360 clips from 46 films, 346 (43 films) with a modelled
  genre.
- [ ] A circular argument (the scaling ablation "supporting" the emotion route, although
  the emotion inputs are never re-scaled) was rewritten.
- [ ] "The margin roughly doubles across datasets" compared two label spaces; replaced by
  each space's lead over its own floor (1.9x vs 1.6x).
- [ ] The significance of the PCA-8 comparison was stated three different ways; unified.
- [ ] Adventure is the *frequent* genre the model cannot predict (Biography and
  Documentary fail for lack of films).
- [ ] The signature replication: the strongest emotion agrees for Horror and Sci-Fi only;
  Comedy changes from fear to happiness.
- [ ] The stability figure shows a tighter *cluster of points* for the emotion route, not
  a narrower box.
- [ ] A reported difference can differ by up to 0.002 from the difference of two rounded
  scores (it is a bootstrap mean); explained once in Methods.
- [ ] Earlier: "corpus" -> "dataset" and "excerpt" -> "clip" throughout; a chance-level
  sentence in Fundamentals mixed two datasets; the trivial-classifier sentence now refers
  to the five-genre task (exact match 0.283, Hamming 0.218); the sentence "an earlier
  version of this work drew an incorrect conclusion" in 2.5.3 was replaced.
- [ ] **Bibliography:** all 36 entries were checked against Crossref (DOIs) and OpenAlex
  (entries without DOI); all are real and match. Five entries are in the .bib but never
  cited (aljanaki2014computational, fan2017ranking, flexer2007closer, song2012evaluation,
  yang2004disambiguating); they do not appear in the PDF, so cite or delete them as you
  like. `elizalde2023clap` is no longer cited either.

## F. Look at the PDF (never checked visually by the assistant)

- [ ] The two TikZ diagrams: the circumplex model (Fundamentals) and the pipeline
  (Methods). They compile, but nobody has looked at them rendered.
- [ ] The eight result figures in Chapter 5: labels readable at print size, legends not
  covering data.
- [ ] The metric table in Section 5.4 is scaled to the text width; check that its font is
  still readable.
- [ ] The table in the AI declaration breaks across pages; check that it looks right.
- [ ] List of figures and tables, table of contents, page numbering.

## G. Repository

- `CLAUDE.md` is local only and git-ignored. Keep it out of the repository.
- Held-back drafts: `docs/latex/held_back/`. Chapter copies as on Overleaf:
  `docs/latex/chapters/`.
- After any change to a result: re-run the experiment, then
  `python experiments/make_figures.py` and `python experiments/audit_consistency.py`.
- GitHub reports that the repository has moved
  (`git@github.com:Abdalla-02/Genre-Classification-using-Emotions.git`); the link in
  Methods still works, but check whether you renamed it.
