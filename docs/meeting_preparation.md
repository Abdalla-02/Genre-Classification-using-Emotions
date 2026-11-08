# Meeting preparation: changes since the version of 5 November

The version shared on Drive (`_Thesis_05.11.pdf`) is older than the current thesis on Overleaf
(9 November). Section numbers below refer to the current version. No result changed: every
number that was in the 5 November version is still the same.

## 1. What changed

### New paragraphs (read them in place)

| Where | Paragraph | What it adds |
|---|---|---|
| 2.5.3 | "Two alternatives for the emotion regressor" | How ridge regression and SVR with an RBF kernel work |
| 2.6.2 | "Without a threshold: ROC-AUC" | What ROC-AUC measures and how to read it |
| 2.6.4 | "Paired bootstrap over films" (extended) | A worked example: +0.093, 95 % interval [−0.001, +0.191], p = 0.052 |
| 2.6.4 | "Many tests at once" | The Holm correction, with an example (Holm 1979 added to the bibliography) |
| 4.6.3 | "Why macro-F1 and ROC-AUC are read together" | Each yes/no metric is fooled by a trivial classifier (macro-F1 by "all yes", Hamming loss by "only Drama"); ROC-AUC gives both 0.5 |
| 5.6.3 | After Figure 5.9 | Why there are two zero-shot implementations and why the first stays the main result |
| A.2 | After Table A.2 | How to read the new "All yes" column |

### Short additions (one or two sentences)

- **Chapter 5, first section:** a third convention. Where a lead could come from saying yes more often, ROC-AUC is given beside macro-F1.
- **5.2, "A harder reference":** ends with "This is why macro-F1 is read together with ROC-AUC."
- **5.3:** the threshold-free check. The emotion pipeline's lead over VGGish in film-level ROC-AUC is +0.071 (p = 0.093), about as large as its F1 lead, but not significant.
- **5.3 and 5.3.1:** what the Nadeau–Bengio test and the Holm method do, with pointers to Chapter 2.
- **4.3:** a pointer to 2.5.3 for ridge regression and SVR.
- **5.5.1:** what a cue is.
- **5.6:** the shared-composer example is now Hans Zimmer (*The Last Samurai* / *Interstellar*, *Dunkirk*) instead of Danny Elfman and Batman.
- **5.6.1:** the vote over cues, with an example (a film with 40 cues gets Action if at least 21 of them are classified as Action).
- **5.6.2:** why the emotion pipeline's F1 is 0.1 higher although direct VGGish is more precise. Macro-F1 is the mean of the F1 per genre, and direct VGGish scores only 0.14 on Romance and 0.17 on Horror.
- **A.7, under Table A.5:** what each model of Ma et al. means (average pooling, single attention, Simple-MI, the baselines).

### Figures and tables

- **Figure 5.8:** now has error bars (bootstrap standard deviation over the 110 films): emotion pipeline 0.508 ± 0.031, PCA-8 control 0.417 ± 0.040, direct VGGish 0.403 ± 0.039.
- **Figure 5.9:** now placed inside 5.6.3, not in the middle of 5.6.2.
- **Table A.2:** a new column "All yes" (the classifier that assigns every genre to every clip). It has the best weighted F1 in the table (0.607) and a macro-F1 of 0.430 over clips. It also has the worst Hamming loss (0.691) and a ROC-AUC of exactly 0.500.

### Removed or changed

- **4.7:** the GitHub link is removed.
- **Title page:** shows the date of compiling (`\today`).

## 2. Questions for the supervisor

1. **Length.** The main text is about 68 pages, plus 14 pages of appendix. Is there a page limit or a guideline? Which chapters feel too detailed? Chapters 2 and 4 together are 30 pages and partly overlap (metrics and tests are explained in both).
2. **Appendix.** It now has twelve sections. Is that acceptable, or should some go (for example the error analysis, A.9, or all eight genres, A.6)?
3. **Metrics.** Is it right to keep film-level macro-F1 as the headline and read ROC-AUC beside it as the check (4.6.3)? Is the "all yes" reference of 0.434 reported prominently enough, or too prominently (5.2)?
4. **Transfer claim.** The zero-shot result is now stated as "at the default threshold". With thresholds tuned on Eerola the lead shrinks to +0.041 (p = 0.159), while the emotion pipeline still ranks the films best. Is the wording in 5.6.2, 5.7 and 6.1 careful enough?
5. **Choice of pipeline.** CLAP with ridge regression scored highest, but the post-hoc test shows no clear lead (+0.032, p = 0.112). Is keeping VGGish with the random forest acceptable, given the reasons in 5.3.1?
6. **Figure 2.1.** It is a schematic adapted from Eerola and Vuoskoski. The real correlations of the ratings are in Figure A.4. Should the real data replace the schematic in Chapter 2?
7. **Code.** Should the code be handed in with the thesis or referenced? The link was removed, but the AI declaration still mentions "the accompanying repository".
8. **Abstract and Chapter 7.** What length and structure are expected?

## 3. Still open on my side

- Abstract, Chapter 7 and the list of contributions.
- Three red notes in the AI declaration: other tools, how the drafts were revised, and the date.
- The wav2vec 2.0 layer: keep block 2.
