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

## features — dataset and features
| Script | Purpose | Log |
|---|---|---|
| `verify_data.py` | Integrity check: clip/soundtrack/label counts, audio present | §1 |
| `extract_features.py` | Extract + cache AST / CLAP embeddings (`--model`, `--set2`) | §2 |
| `clip_length_analysis.py` | Clip-duration distribution of the corpus | §5 |

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

## diagnostics — why the score is what it is
| Script | Purpose | Log |
|---|---|---|
| `exp_error_analysis.py` | Per-genre false positives / negatives | §7c |
| `exp_threshold_fix.py` | Does curbing over-prediction help? (no) | §7c |
| `exp_learning_curve.py` | Would more data help? (yes, still rising) | §7e |
| `exp_clip_length.py` | Full-clip windowed pooling vs the 10.24 s window | §6 |

## cross_dataset — transfer to Blockbuster
| Script | Purpose | Log |
|---|---|---|
| `exp_blockbuster.py` | VGGish vs MFCC on Ma et al. (2021) | §3 |
| `exp_blockbuster_emotion.py` | Cross-dataset emotion bridge + PCA-8 control | §10 |

## evaluation — statistical reliability
| Script | Purpose | Log |
|---|---|---|
| `exp_statistical_power.py` | Repeated GroupKFold, confidence intervals, corrected significance tests, nested-CV tuning | §11 |

**`evaluation/exp_statistical_power.py` supersedes the single-run numbers** printed by
the scripts in `genre/` and `cross_dataset/`: it re-scores the same arms over 50
leakage-safe folds instead of 5 and reports confidence intervals. Where the two disagree,
quote §11. It is also the only script that writes machine-readable output
(`results/statistical_power.json`, containing every per-fold score).
