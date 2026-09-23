"""Which wav2vec 2.0 layer should be pooled? (a required control, not a tuning run)

Pooling wav2vec 2.0's FINAL layer gives a mean R^2 of 0.165 on emotion regression, far
below every spectrogram representation (~0.56). Reporting that as "the waveform front-end
is weaker" would be premature: it is well established that the last layers of a
self-supervised speech model specialise toward its pretraining objective, and that middle
layers transfer better to tasks the model was not trained for. If the deficit disappears
at layer 6, the finding is "we pooled the wrong layer"; if it survives across all 13
layers, the finding is about the representation.

This sweep therefore extracts all 13 hidden-state layers in ONE forward pass per clip and
scores each layer with the same protocol used everywhere else (RandomForest, GroupKFold by
film, mean R^2 over the 8 emotions). The best layer is then what the main comparison
should use.

Run:  python experiments/features/exp_w2v_layer_sweep.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.features import load_set1  # noqa: E402
from src.features.wav2vec_extractor import W2V_CHECKPOINT, W2V_WINDOW_S  # noqa: E402
from src.utils import set_seed  # noqa: E402

CACHE = config.W2V_EMBEDDINGS_DIR / "layers_set1"   # <number>.npy holding (13, 768)
RESULTS = config.RESULTS_DIR / "w2v_layer_sweep.json"


def extract_all_layers(df) -> np.ndarray:
    """Return (n_clips, 13, 768): every transformer layer, mean-pooled over time."""
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = [(int(n), p) for n, p in zip(df["number"], df["audio_path"])
            if not (CACHE / f"{int(n):03d}.npy").is_file()]
    if todo:
        import librosa
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
        fe = Wav2Vec2FeatureExtractor.from_pretrained(W2V_CHECKPOINT)
        model = Wav2Vec2Model.from_pretrained(W2V_CHECKPOINT, output_hidden_states=True)
        model.eval()
        sr = fe.sampling_rate
        with torch.no_grad():
            for number, path in tqdm(todo, desc="wav2vec2 layers"):
                y, _ = librosa.load(path, sr=sr, mono=True)
                y = y[:int(round(W2V_WINDOW_S * sr))]
                if len(y) < sr:
                    y = np.pad(y, (0, sr - len(y)))
                out = model(**fe(y, sampling_rate=sr, return_tensors="pt"))
                stack = torch.stack(out.hidden_states)          # (13, 1, T, 768)
                np.save(CACHE / f"{number:03d}.npy",
                        stack.mean(dim=2).squeeze(1).cpu().numpy().astype(np.float32))
    return np.stack([np.load(CACHE / f"{int(n):03d}.npy") for n in df["number"]])


def main() -> None:
    set_seed()
    df = load_set1(clean=False)
    Y = df[config.EMOTIONS].to_numpy(float)
    groups = df["soundtrack"].to_numpy()
    X = extract_all_layers(df)
    print(f"\nlayer stack: {X.shape}  (clips, layers, dims)\n")

    print("=" * 60)
    print("wav2vec 2.0 layer sweep -- emotion regression, GroupKFold, mean R^2")
    print("  reference: AST 0.560 | VGGish 0.558 | CLAP 0.561 | MIR 0.490")
    print("=" * 60)
    print(f"{'layer':>6}{'mean R^2':>10}   {'':<20}")
    rows = {}
    cv = GroupKFold(5)
    for layer in range(X.shape[1]):
        Xl = X[:, layer, :]
        model = RandomForestRegressor(n_estimators=200, random_state=config.SEED, n_jobs=-1)
        pred = cross_val_predict(model, Xl, Y, cv=cv.split(Xl, Y, groups))
        r2 = float(np.mean([r2_score(Y[:, j], pred[:, j])
                            for j in range(len(config.EMOTIONS))]))
        rows[layer] = r2
        bar = "#" * int(max(r2, 0) * 60)
        tag = "  <- CNN encoder output" if layer == 0 else ""
        print(f"{layer:>6}{r2:>10.3f}   {bar}{tag}")

    best = max(rows, key=rows.get)
    print(f"\nbest layer: {best} (R^2={rows[best]:.3f}); "
          f"final layer 12 gives {rows[12]:.3f}")
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({"mean_r2_by_layer": rows, "best_layer": best},
                                  indent=2), encoding="utf-8")
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
