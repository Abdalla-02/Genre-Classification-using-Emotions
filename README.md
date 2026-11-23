# Multi-Label Movie Genre Classification based on Emotional Features of Soundtracks

Code, results and documentation of a bachelor thesis at Otto-von-Guericke University
Magdeburg (Faculty of Computer Science, AG Data & Knowledge Engineering), **submitted on
23 November 2026**.

Film music is written to express emotions that suit a film's genre. The thesis asks how far
the emotions that listeners perceive in a soundtrack reveal the film's genre, and what
changes when genre is predicted through these emotions instead of directly from the audio:

```text
audio clip ──► VGGish embedding ──► random forest ──► 8 emotions (+3 derived) ──► logistic regression per genre ──► genres
                                └──────────────── logistic regression per genre (direct pipeline) ─────────────────┘
```

## Research questions

- **RQ1** How well do the emotions that listeners perceive in a soundtrack predict its film's genre?
- **RQ2** Which emotions characterise each genre?
- **RQ3** How well can genre be predicted from emotions, compared with predicting it directly
  from the same audio representation?
- **RQ4** How well do the two pipelines generalise when a model trained on one dataset is
  applied to another, without access to any of its labels?

## Main results

Genre is multi-label. The headline metric is **film-level macro-F1**: the clip probabilities
of a film are averaged, a genre is assigned at ≥ 0.5, and F1 is averaged over genres. Every
value below is read from `results/*.json`; each comes with the reference it should be read
against.

**Eerola dataset, five genres, 40 films** (`results/film_level.json`; ten repeats of
grouped 5-fold cross-validation):

| pipeline | macro-F1 (film) |
|---|---|
| ratings → genre (reference: Stage 1 replaced by the listener ratings) | 0.491 |
| **VGGish → emotions → genre (emotion pipeline)** | **0.460** (clip level 0.405) |
| best audio representation used directly (MusiCNN) | 0.396 |
| VGGish used directly | 0.367 |
| PCA of VGGish to 8 dimensions (control) | 0.390 |
| all-positive reference (every genre for every film) | 0.434 |
| random guess at the base rates / most-frequent dummy | 0.309 / 0.168 |

The emotion pipeline ranks films above chance (macro ROC-AUC 0.655). Its lead over VGGish
used directly (+0.093) is not significant on 40 films (paired film bootstrap, p = 0.052),
and with thresholds tuned per genre the pipelines converge over clips.

**Transfer to the Blockbuster dataset** (110 films, six shared genres; `results/zero_shot.json`,
strict regime): trained on Eerola and applied without any Blockbuster labels, the emotion
pipeline reaches 0.508 against 0.403 for VGGish used directly (random guess 0.292); the gap
shrinks and loses significance with tuned thresholds (`results/zero_shot_threshold.json`).
Trained on Blockbuster itself, the direct pipeline reaches 0.620 under leave-one-film-out,
inside the range of the VGGish models that Ma et al. (2021) published for these films
(0.61-0.65).

**In short:** the results suggest that the emotion step does not make genre prediction more
accurate in general. Its lead appears on the small dataset and under transfer at the default
threshold; what it offers by design is interpretability, such as fear as the signature of
Horror (`results/signatures_eerola.json`).

## Data

- **Eerola & Vuoskoski (2011)**: 360 film-music clips from 45 films, each rated by listeners
  on eight emotion scales. Genres from IMDb. Three label spaces are used and never compared
  with each other: eight genres (346 clips, 42 films), the five-genre subset (329 clips,
  40 films) and the six genres shared with Blockbuster (319 clips, 40 films).
- **Blockbuster (Ma et al., 2021)**: 110 films described by published VGGish and MIR
  features per music cue (4664 cues). There is no audio (copyright), so VGGish is the only
  representation available on both datasets.
- Raw data is **not redistributed** (`data/raw/` is git-ignored; see `data/raw/README.md`).
  The per-clip embeddings derived from it are committed in `data/processed/`, so every
  result reproduces from a clone without the audio.

## Repository layout

```text
├── src/                    library code: configuration, data loading, feature extractors,
│                           classifiers, cross-validation and metrics
├── experiments/            one runnable script per experiment, grouped by topic
│   ├── features/           data checks and embedding extraction
│   ├── emotion/            Stage 1: audio → emotions
│   ├── genre/              Stage 2: emotions → genre
│   ├── evaluation/         film-level scoring, repeated CV, significance tests
│   ├── cross_dataset/      transfer to and from Blockbuster
│   ├── diagnostics/        error analyses, thresholds, per-genre checks
│   ├── make_figures.py     draws every results figure of the thesis into figures/
│   ├── show_results.py     renders any results file as tables
│   └── audit_consistency.py  checks the documents against the results
├── results/                every headline experiment's per-fold scores (JSON)
├── figures/                the figures of the thesis (PDF + PNG)
├── data/processed/         cached embeddings and derived tables (committed)
└── docs/
    ├── README.md           the technical log: every experiment and decision, in order
    ├── evaluation_metrics.md  every metric, why it was chosen and what it showed
    ├── latex/chapters/     the text of the submitted thesis (abstract, Ch. 1-7, Appendix A)
    ├── defense_slides.md   material for the defense
    ├── literature/         two short literature notes
    └── archive/            supervisor briefings and review files from the writing phase
```

Where to start reading: `results/README.md` lists every results file, what it answers and
where it appears in the thesis; `experiments/README.md` describes every script.

## Reproducing the results

Python 3.12, CPU only. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
python experiments/features/verify_data.py              # load + clean, assert the dataset counts (needs data/raw)
python experiments/evaluation/exp_film_level.py         # the headline: every Eerola genre result per film, clip level beside it
python experiments/cross_dataset/exp_zero_shot.py       # train on Eerola, test on Blockbuster
python experiments/make_figures.py                      # redraw every results figure from results/*.json
python experiments/show_results.py film_level           # render a results file as tables
python experiments/audit_consistency.py                 # check every document against the results
```

The command for every experiment, grouped and with what each needs, is in
`results/README.md` ("Running the experiments"). Re-running all Eerola experiments takes
about two hours on a laptop CPU.

Reproducibility rests on a fixed seed (`SEED = 42`), pinned dependencies, committed
embedding caches, and cross-validation grouped by film, so that no film appears in both
training and test folds (a misspelt film name once split a film across folds and is now
merged in the loader, `config.SOUNDTRACK_NAME_FIXES`).

The stack is PyTorch-only (AST, CLAP and wav2vec 2.0 via `transformers`, VGGish via
`torchvggish`, the MIR baseline via `librosa`). The one exception, MusiCNN, needs
TensorFlow and runs in a separate environment (`.venv-musicnn`, Python 3.11); it talks to the
project only through the embedding cache.

## Note on terminology

The thesis says "pipeline"; the technical log and the names of some results fields still say
"route" (e.g. the emotion route = the emotion pipeline).

## License

The code is released under the MIT License (`LICENSE`). The datasets are not part of this
repository and keep their own terms.
