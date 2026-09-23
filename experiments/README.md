# Experiments

Runnable scripts, grouped by pipeline stage, mirroring the package layout in `src/`.
The sections below are in reading order: start with `features/`, then follow the chain
`audio -> emotion -> genre`. Every script is standalone, takes no required arguments,
and is run **from the repository root**:

```bash
.venv/Scripts/python.exe experiments/genre/exp_genre_subset.py
```

Section numbers in the right-hand column refer to `docs/README.md`, the progress log
where each result is recorded and discussed.

## show_results.py — turn saved results into Markdown tables

Not an experiment: a reader for `results/*.json`. Every headline experiment saves its
per-fold scores there, and this renders them as tables ready to paste into a report.

```bash
python experiments/show_results.py                        # list what is available
python experiments/show_results.py blockbuster_deep       # print its tables
python experiments/show_results.py --all --out report.md  # everything into one file
```

See `results/README.md` for what each file contains.

## audit_consistency.py — check the documents against the results

Also not an experiment. It re-reads every saved result and verifies that the numbers
quoted in `docs/current_state.md` and `docs/README.md` still match them, that every file
path named in the documents resolves, that the dataset still cleans to the expected
counts, that the embedding caches are complete, that no file has picked up CRLF endings,
and that every results file still renders. It exits non-zero, so it can be wired into a
pre-commit hook. **Run it before committing anything that touches a number.**

```bash
python experiments/audit_consistency.py
```

## features — dataset and features
| Script | Purpose | Log |
|---|---|---|
| `verify_data.py` | Integrity check: clip/soundtrack/label counts, audio present | §1 |
| `extract_features.py` | Extract + cache embeddings: `--model ast\|clap\|vggish\|mir\|wav2vec2` | §2 |
| `clip_length_analysis.py` | Clip-duration distribution of the corpus | §5 |
| `exp4_rating_reliability.py` | Exp 4: Set 1 vs Set 2 rating reliability + ceiling | §13 |
| `exp_waveform_vs_spectrogram.py` | Waveform (wav2vec 2.0) vs spectrogram representations, both stages | §16 |
| `exp_w2v_layer_sweep.py` | Control for the above: which wav2vec 2.0 layer to pool | §16 |
| `extract_musicnn.py` | MusiCNN embeddings -- runs in the separate `.venv-musicnn` (TensorFlow) | §16.4b |
| `fetch_box_office.py` | Box office per film from Wikidata / Box Office Mojo, by IMDb id | §20 |

## emotion — stage 1, audio -> emotion
| Script | Purpose | Log |
|---|---|---|
| `exp_emotion_regression.py` | Feature comparison for emotion regression (AST / VGGish / MIR) | §9 |
| `exp_emotion_improve.py` | Regressor comparison (Ridge / SVR / RF) + emotion->genre on the subset | §9 |

## genre — stage 2, emotion -> genre
| Script | Purpose | Log |
|---|---|---|
| `exp5_target.py` | Sanity check: embedding -> 12 discrete emotion categories | §3 |
| `exp_genre.py` | Genre classification with vs without emotion, 8 genres | §3 |
| `exp_genre_subset.py` | The same on the curated 5-genre subset | §7f |
| `exp_emotion_genre.py` | RQ3: which emotions predict which genres (Cohen's d) | §7b |
| `exp_emotion_ablation.py` | Leave-one-emotion-out and theory-motivated subsets: none is necessary | §18 |

## diagnostics — why the score is what it is
| Script | Purpose | Log |
|---|---|---|
| `exp_error_analysis.py` | Per-genre false positives / negatives | §7c |
| `exp_threshold_fix.py` | Does curbing over-prediction help? (no) | §7c |
| `exp_model_search.py` | Does any other model/feature combination beat the baseline? (no) | §7d |
| `exp_learning_curve.py` | Would more data help? (yes, still rising) | §7e |
| `exp_clip_length.py` | Full-clip windowed pooling vs the 10.24 s window | §6 |
| `exp_box_office.py` | Box office vs classification quality, emotions and genre (exploratory) | §20 |

## cross_dataset — transfer to Blockbuster
| Script | Purpose | Log |
|---|---|---|
| `exp_blockbuster.py` | VGGish vs MFCC on Ma et al. (2021) | §3 |
| `exp_blockbuster_emotion.py` | Cross-dataset emotion bridge + PCA-8 control | §10 |
| `exp_zero_shot.py` | Shared 6-genre space; train on Eerola, test on Blockbuster without training on it; replicates Ma et al. in-domain | §15 |
| `exp_blockbuster_deep.py` | Blockbuster under the primary protocol: repeated CV, tuned C, cue-level (MIL) arms, full 140-feature MIR | §17 |
| `exp_signature_replication.py` | Do the Eerola emotion-genre signatures reappear on Blockbuster? (RQ3 external validity) | §17.5 |
| `exp_cross_dataset_cv.py` | Both transfer directions plus a pooled design, same three arms | §19 |

## evaluation — statistical reliability
| Script | Purpose | Log |
|---|---|---|
| `exp_statistical_power.py` | Repeated GroupKFold, confidence intervals, corrected significance tests, nested-CV tuning | §11 |
| `exp_cv_corrected.py` | The same folds, re-scored with the two CV corrections (pooled per-repeat macro-F1; out-of-fold emotion training); all six representations and the ablation on one protocol; reproduces §11's old-metric numbers as a check | §21 |

**`evaluation/exp_statistical_power.py` supersedes the single-run numbers** printed by
the scripts in `genre/` and `cross_dataset/`: it re-scores the same arms over 50
leakage-safe folds instead of 5 and reports confidence intervals. Where the two disagree,
quote §11. It is also the only script that writes machine-readable output
(`results/statistical_power.json`). `diagnostics/exp_model_search.py`,
`cross_dataset/exp_zero_shot.py`, `features/exp_waveform_vs_spectrogram.py` and
`features/exp_w2v_layer_sweep.py` likewise write `results/model_search.json`,
`results/zero_shot.json`, `results/waveform_vs_spectrogram.json` and
`results/w2v_layer_sweep.json`.
