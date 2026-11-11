# Thesis verification checklist

Everything you should read again, decide, or check yourself before submission, collected
from the whole project (state: 2 November 2026, Overleaf commit `240b538`; first written
7 October). Tick items off as you go. **Start with section 0**, the short list in order of
priority; sections A and B hold the details, C to E are the record of what was found wrong
and corrected, so you can confirm you agree with each correction.

Where a location is given as "§N" it refers to the technical log (`docs/README.md`).

---

## 0. Start here: what to do next (4 November 2026, evening)

**Decide** (the assistant cannot do these for you)
- [ ] **wav2vec 2.0 layer:** block 1 is nominally best after the fix, the thesis keeps
  block 2 (section B). Recommended: keep it, as the thesis says.
- Done on 5 November: your comments of 4 November (log section 33). Read the new Appendix
  section on how the ratings relate to each other: sadness correlates slightly positively
  with valence, and Figure 2.1 stays a schematic.
- Done on 4 November: the 0.434 floor and the zero-shot threshold test are in the thesis
  (Chapter 5, Discussion, new Appendix sections), and CLAP with ridge regression was
  tested post hoc (+0.032, p = 0.112; log section 32). Read how they are worded.

**Write** (section A)
- [ ] **Abstract, Chapter 7 and the contributions list.** The drafts in
  `docs/latex/held_back/` quote numbers from before the film-level switch and the fix, and
  know nothing of the work since 27 October (wav2vec diagnosis, the comparison over
  representations and regressors). Take the numbers from `docs/current_state.md`,
  sections 7 to 7c.
- [ ] **AI-usage declaration:** three red notes are still open: any other tools you used
  (or delete the note), how you read, revised and adapted the drafts, and the date next
  to the signature. The period of use (to November 2026), the wording row and the Figure
  2.1 row were updated on 3 November: check them. The row on Chapter 7, the abstract and
  the contributions says you wrote them and used AI only for minor edits, so it must stay
  true.
- [ ] **Title page:** already filled in. Check the title, both advisers and the
  submission date (04 December 2026).

**Read** (section B)
- [ ] The whole thesis once in full: every chapter and the appendix were reworded on
  3 and 4 November. Numbers and references were checked by script; the meaning of each
  sentence was not. Make sure it still says what you mean and sounds like you.
- [ ] New since 2 November: Figure 2.1 and its text, the Thayer (1989) reference (check it
  against the book), Figure 4.1, and the paragraph on VGGish and the random forest.

**Look at the PDF** (section F)
- [ ] Figure 2.1, Figure 4.1 (no overlapping boxes), the grid figure in Chapter 5, the
  9-column table in the Appendix, then the rest of section F.

**Before submission**
- [ ] Section C: know your answer to each likely challenge.
- [ ] Sections D and E: confirm each correction; re-run the print-only scripts of D.
- [ ] Merge the last version branch into `main` and keep the branches `feat/version-1.2`
  and `feat/version-1.3` (they hold the individual commits).

## A. Only you can do these

- [ ] **AI-usage declaration** (last page). Written by the assistant from the repository
  and Overleaf history. Check every line, then fill the remaining red notes:
  - [x] any other tool you used (translation, grammar checker, other chatbots): the red
        note is gone (Overleaf `4a77de0`);
  - [x] how you read, revised and adapted the drafted chapters: the red note is gone
        (Overleaf `4a77de0`);
  - [ ] the period of use: the text says "July 2026 to October 2026", but the assistant
        edited the thesis on 1 November 2026; extend it to the real end date;
  - [x] the level for Chapter 7, the abstract and the list of contributions: filled in on
        Overleaf ("edited", 6 Oct);
  - [ ] the date next to the signature (its red box is also the cause of the one
        remaining "overfull box" warning; it disappears when you type the date).
  - Facts it states that you should confirm: models **Claude Opus 5** (9-22 Sep 2026) and
    **Claude Opus 5.5** (from 23 Sep 2026), both from the repository's commit trailers;
    period **July to November 2026** (so far); the Claude.ai Overleaf connector was used from
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

- [ ] **Your comments of 21 Oct** (Overleaf `011fc8c`): the **four research questions**
  (RQ3 = comparison, RQ4 = generalisation) and the main question as how-questions; "pipeline"
  instead of "route" everywhere; the new precision/recall figure and the Horror paragraph in
  Section 5.4 with the new Appendix A section; the definitions under the Blockbuster chart;
  the abbreviations now spelled out at first use (read the affected sentences, some became
  longer). Overleaf `4ab939c`: the new table of pipelines and models in Methods, and the
  model names added at the first mention of each pipeline per section.
- [ ] **The emotion pipeline across representations and regressors** (Overleaf `5a4ca23`,
  1 Nov): the new Chapter 5 subsection "Other Representations and Emotion Regressors" with
  its table, the new Appendix section with the figure and the Stage-1 regressor table, the
  rewritten regressor paragraph in Methods, the sentence in Fundamentals, four sentences in
  the Discussion and the RQ3 summary. Decide whether you agree with the reading that CLAP
  with ridge regression (ahead of the VGGish pipeline under all five seeds, 0.502 vs 0.469)
  is reported but not adopted because it was picked after the results were known (§30.1).

- [ ] **Why wav2vec 2.0 works as the emotion input** (Overleaf `4c43946`, log §28): two new
  paragraphs at the end of Appendix A.5 (The Waveform Representation) and one sentence in
  Chapter 5, Stage 1, after the wav2vec R^2. Check that the explanation reads plainly and
  that you agree with "level with VGGish" rather than "best".

- [ ] **Your comments of 16 Oct** (Overleaf `352a922`, `497c6db`): check the new opening
  of the Introduction, the approach paragraph after the research questions, the rewritten
  R² ceiling, Cohen's d and significance parts of Fundamentals 2.6, and the **three
  research questions** (RQ2 = old RQ2a + RQ3, RQ3 = old RQ2b) wherever the Evaluation
  names them. The held-back Chapter 7 draft still uses the old numbering.

- [ ] **Fifth meeting (8 Oct, Overleaf `3b10594`).** Chapter 5 was rebuilt around the core
  results, the rest moved to the new Appendix A, and Chapter 6 was shortened and rewritten
  around one table. All Eerola numbers changed after the duplicate-film fix. Read Chapters 5
  and 6 and Appendix A in full; check in particular the per-genre paragraph (Section 5.4),
  the leakage paragraph (Section 5.6) and "Interpreting the Main Result".
- [ ] **Methods, wav2vec layer.** After the fix, blocks 1–3 are within 0.003 and block 1 is
  nominally best; the text says block 2 was selected earlier and kept. Decide whether that
  is acceptable to you.

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
  result. Know where the two disagree (after the 8 Oct fix): over clips the film
  bootstrap finds the lead significant against VGGish, CLAP, MusiCNN and PCA-8, over films
  it does not; against AST it is the other way round. Film level is the stricter unit, so
  the choice does not favour the method.
- [ ] In-domain (5 genres, film level) the emotion route is ahead of VGGish by +0.093
  in 9 of 10 repeats and under every seed, but **not significantly (p = 0.052)**, nor ahead
  of MusiCNN (p = 0.241) or the PCA-8 control (p = 0.089); significant only against AST,
  MIR and wav2vec 2.0. The **PCA-8 control itself beats VGGish under every seed**. Over
  clips it is **borderline under Nadeau-Bengio** (p = 0.086 vs VGGish) and **no longer
  ahead once per-genre thresholds are tuned** (-0.004, p = 0.813); thresholds were never
  tuned per film. The thesis says so.
- [ ] The **zero-shot** result (0.508 vs 0.403, p = 0.013) is one split, one
  representation (VGGish), and partly recall-driven (recall 0.651 vs 0.435, precision
  0.480 vs 0.705 for direct audio). With thresholds tuned on the source it is +0.041,
  p = 0.159 (§31); threshold-free the emotion pipeline still ranks films best.
- [ ] The win over the **PCA-8 control** across datasets is significant only in the
  original zero-shot run (p = 0.027), not in the re-implementation (+0.049, p = 0.239).
- [ ] The **5-genre subset** was chosen partly by emotional signature, which can favour
  the emotion route (stated as a limitation; the 8-genre results guard against it).
- [ ] **Dataset size:** 40/42 films; for the borderline comparisons no number of repeats
  could reach p < 0.05 (p_lim 0.069 to 0.200).
- [ ] **Blockbuster emotions are predicted**, never rated; the signature replication
  (r = 0.836) cannot show independence from the Eerola listener panel.
- [ ] **Label noise:** genres are film-level IMDb labels applied to every clip.

## D. Numbers in the thesis that no results file stores

They come from scripts that only print their output, or from a one-off check. Re-run the
script before submission, or be ready to explain where the number comes from.

| number in the thesis | where | source |
|---|---|---|
| AST 0.44 ungrouped vs 0.23 grouped (leakage) | Fundamentals, Methods, Evaluation | `experiments/genre/exp_genre.py` (8 genres, single untuned five-fold run) |
| ridge 0.39 / SVR 0.54 / random forest 0.55 | Methods, Stage 1 | `experiments/emotion/exp_emotion_improve.py` |
| 3.55 vs 1.87 predicted vs true labels per clip (8 genres) | Appendix A, error analysis | `experiments/diagnostics/exp_error_analysis.py` (older protocol) |
| learning curve still rising | Evaluation, Discussion | `experiments/diagnostics/exp_learning_curve.py` |
| Cohen's d per genre (Horror-fear +0.75, the 8-genre signature table) | Fundamentals, Evaluation | `experiments/genre/exp_emotion_genre.py` |
| (removed) the 8-genre random-guess floor is now 0.22 at film level, simulated and stored in `film_level.json` | Fundamentals, Evaluation | `exp_film_level.py` |
| VGGish per-dimension means of the two datasets correlate at r = 0.97 | Methods, Evaluation | earlier zero-shot analysis (log §15) |
| wav2vec layer 2: R² 0.294 (sweep) vs 0.299 (main run) | Methods | two different runs, both stated |

## E. Errors found and corrected along the way (confirm you agree)

**Results and method**
- [ ] **Duplicate film** (§24): "The Portait of a Lady" (clip 256) and "The Portrait of a
  Lady" are one film; merged in the loader, every Eerola experiment re-run. Film counts are
  42 (8 genres), 40 (5 genres, shared space) and 45 (all rated clips). Emotion vs VGGish at
  film level moved from p = 0.047 to p = 0.052.
- [ ] **Ridge regression in Stage 1** (§30.1): Methods said ridge reaches R^2 0.39 on
  VGGish against 0.55 for the random forest, and Fundamentals 37 % against 56 %; both came
  from a script that only prints, with an untuned ridge (alpha = 1). With alpha chosen by
  film-grouped cross-validation, ridge reaches 0.584, slightly above the forest (0.553).
  The text now says so; the random forest is kept and the genre result for VGGish hardly
  changes with the regressor. All four numbers are now in `emotion_pipeline_grid.json`.
- [ ] **Blockbuster → Eerola significance** (§23.3): the clip-level p < 0.001 came from a
  bootstrap that resampled clips, which are not independent. Resampling films gives
  +0.077, p = 0.127 (after the 8 Oct fix +0.089, p = 0.102). The thesis now says the
  reverse direction agrees in sign but is not significant.
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
  script says 5x5. Re-run at 5x5 on 7 Oct; numbers updated everywhere. Now (after the
  8 Oct fix) per film: emotion 0.531, direct 0.543, p = 0.688; per clip 0.424 vs 0.420,
  p = 0.816. Conclusion unchanged.
- [ ] **Box Office Mojo parser** (§22.2) read the navigation menu and stored US-only
  grosses; fixed and the Eerola analysis re-run. (Box office is no longer in the thesis.)
- [ ] **Data quality** in the enriched Eerola CSV: the IMDb ids of *Blanc* and *Pride and
  Prejudice* point at the wrong films (excluded from the box-office analysis).

**Text, found in the examiner-style review (5 Oct) and the consistency pass (6 Oct)**
- [ ] **Related Work:** LMTD was expanded as "Largest Movie Trailer Dataset"; it is the
  *Labeled* Movie Trailer Dataset (fixed 7 Oct). Check the "3,500 films, 9 genres" against
  Behrouzi et al. (2023) yourself.
- [ ] **Abbreviations and Notations** (appendix) rewritten on 7 Oct from the abbreviations
  the chapters actually use: about 25 added (e.g. SVM, SVR, kNN, CNN, LSTM, GRU, ICC, CI,
  AUC, AP, NB, RQ, MSD, LMTD, $p_{boot}$, $p_{NB}$, $p_{lim}$, $\Delta$), three removed that the
  text never uses (IMV, BR, CV). $p_{limit}$ in Methods is now written $p_{lim}$ as in the
  Evaluation.
- [ ] CLAP was cited to the Microsoft model; the checkpoint used is LAION-CLAP, now cited
  as Wu et al. (2023).
- [ ] The reliability ceiling 0.897 is an intraclass correlation (ICC(C,1)) on 102 matched
  clips, not r squared (0.91 squared would be 0.83).
- [ ] Clips are about 17 s long (not 10 s) for VGGish. Film counts, as corrected on
  6 Oct and again after the duplicate-film fix of 8 Oct: 360 rated clips from 45 films,
  346 clips (42 films) with a modelled genre, 329 (40) on five genres, 319 (40) in the
  shared space.
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
- [ ] The result figures in Chapter 5 and the new figure of the comparison over
  representations and regressors in the Appendix (1 Nov): labels readable at print size,
  legends not covering data.
- [ ] The 9-column table "The emotion pipeline for every representation and emotion
  regressor" in Chapter 5 (1 Nov): does it fit the text width, is the small font readable?
- [ ] The metric table in Section 5.4 is scaled to the text width; check that its font is
  still readable.
- [ ] The AI declaration is now two paragraphs (12 Nov); check that its wording matches how you
  worked, and fill in its three red notes.
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
