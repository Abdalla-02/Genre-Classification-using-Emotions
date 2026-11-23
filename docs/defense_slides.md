# Defense slides: material saved for later

Collected on 22 Nov 2026, the day before submission, for the colloquium deck. Every value comes
from Chapter 3 of the submitted thesis (Overleaf `38a6eb2`) or from `results/film_level.json`.

## Slide: where this thesis sits among the film-genre studies

| Study | Films | Genres | Input | Model | Emotion step | Reported result |
|---|---|---|---|---|---|---|
| Austin et al. 2010 | 98 | 4, single-label | score audio (222 features) | SVM | no | 53.9 % accuracy |
| Mangolin et al. 2020 | 10,594 | 18, multi-label | trailer, poster, subtitles, synopsis | LSTM + CNN | no | F-score 0.628 |
| Ma et al. 2021 | 110 | 6, multi-label | music cues (VGGish, MIR) | attention MIL | no | macro-F1 0.61-0.65 |
| Behrouzi et al. 2023 | 3,500 | 9, multi-label | trailer frames + audio | GRU + SVM | no | Hamming loss 0.133 |
| **This thesis** | 40 (+110 for transfer) | 5, multi-label | film-music clips (VGGish) | random forest -> logistic regression | **yes, 8 emotions** | macro-F1 0.460 (film level) |

Footnote on the slide: "Metrics differ between studies and are not directly comparable."

Notes for presenting it:
- The point of the slide is the emotion-step column: none of the four puts an interpretable step
  between audio and genre (the gap of Section 3.3).
- Expect the question why 0.460 is lower than Ma's 0.65: different data, genre set and metric;
  the scores cannot be ranked against each other. On Ma's own 110 films the direct pipeline of
  this thesis reaches 0.620, inside their range (Appendix A.9).
- If the slide is crowded, drop the "Model" column.
- Origin: suggested by NotebookLM as a table for Section 3.3 (with wrong values: "45 films",
  macro-AUC as Behrouzi's headline metric, "macro" F1 for Mangolin); corrected here and kept out of
  the thesis on purpose, since it would repeat Chapter 3 and add own results to it.
