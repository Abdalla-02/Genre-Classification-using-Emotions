# Thesis LaTeX: copies of the submitted text

The thesis was written on Overleaf and submitted on 23 November 2026 (Overleaf commit
`aac3f7a`, 108 pages). This folder holds plain copies of its text, so that every number in
it can be checked against `results/*.json` without access to Overleaf. The files are not
compiled from here; the preamble, title page, bibliography and figures stay on Overleaf
(the figures themselves are drawn by `experiments/make_figures.py` into `figures/`).

| file | part of the thesis |
|---|---|
| `chapters/abstract.tex` | Abstract |
| `chapters/0_intro.tex` | 1 Introduction and Motivation (research questions, contributions) |
| `chapters/fundamentals.tex` | 2 Fundamentals |
| `chapters/1_background.tex` | 3 Related Work |
| `chapters/2_analysis.tex` | 4 Methods |
| `chapters/3_evaluation.tex` | 5 Experiments and Evaluation |
| `chapters/discussion.tex` | 6 Discussion |
| `chapters/4_conclusion.tex` | 7 Conclusions and Future Work |
| `chapters/supplementary.tex` | Appendix A, Supplementary Results |
| `chapters/ai_declaration.tex` | Declaration on the Use of Generative AI |
| `not_in_thesis/box_office.tex` | the box-office analysis, removed from the thesis on 7 Oct 2026 as not about genre (results in `results/box_office.json`) |

`experiments/audit_consistency.py` checks the headline numbers in `chapters/3_evaluation.tex`
and `chapters/supplementary.tex` against the results files, and that no pre-fix film count
("41 / 43 films") survives in any chapter.

Earlier material in this folder (the paste-ready sections of September 2026, the drafts
held back from the build in October) was superseded by the chapters and removed after
submission; it remains in the git history.
