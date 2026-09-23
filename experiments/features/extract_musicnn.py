"""MusiCNN embedding extraction -- runs in the SEPARATE TensorFlow environment.

Why this script is different from every other extractor
-------------------------------------------------------
The rest of the project is deliberately PyTorch-only, so the environment stays
reproducible. MusiCNN is TensorFlow-only, so it gets its own virtual environment and its
own script, and communicates with the rest of the project the only way it needs to: by
writing per-clip ``.npy`` files into the usual embedding cache. Nothing in ``src/`` imports
TensorFlow, and this file deliberately imports nothing from ``src`` either -- it uses only
the standard library, numpy, librosa and tensorflow, so it runs in the small env.

    py -3.11 -m venv .venv-musicnn
    .venv-musicnn/Scripts/python -m pip install "tensorflow==2.15.1" "numpy<2" librosa
    .venv-musicnn/Scripts/python -m pip install --no-deps musicnn-keras
    .venv-musicnn/Scripts/python experiments/features/extract_musicnn.py

(``--no-deps`` is required: musicnn-keras declares ``numpy<1.17``, a stale pin inherited
from the original TF-1 package, which contradicts TensorFlow 2's own ``numpy>=1.22``. The
package's Python code works fine with a modern numpy; only the metadata is wrong.)

Why MusiCNN is worth the trouble: it is the only representation here pretrained on MUSIC
tagging (Million Song Dataset / MagnaTagATune) rather than general audio events (AST,
VGGish: AudioSet) or speech (wav2vec 2.0). Several of its output tags are mood words, so
its domain match to emotion regression is the best of any model tested.

Preprocessing replicates musicnn exactly (from musicnn_keras.configuration): 16 kHz,
96-band mel, FFT 512 / hop 256, ``log10(10000*mel + 1)``, patches of 187 frames (~3 s).
The penultimate layer ``bn_dense`` (200-d) is the standard feature-extraction tap. Patches
are mean-pooled to one vector per clip, matching how every other extractor here pools.

Clip handling matches AST and wav2vec 2.0: the first 10.24 s of each clip.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

import librosa
import numpy as np

# musicnn's own constants (musicnn_keras/configuration.py) -- replicated, not imported,
# so this script does not depend on the broken package layout.
SR = 16000
FFT_HOP = 256
FFT_SIZE = 512
N_MELS = 96
PATCH_FRAMES = 187          # ~3 s, the length the models were trained on
WINDOW_S = 10.24            # same excerpt as AST / wav2vec 2.0
PENULTIMATE_LAYER = "bn_dense"
EMBED_DIM = 200

REPO_ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = REPO_ROOT / "data" / "processed" / "musicnn_checkpoints"
CKPT_URL = ("https://raw.githubusercontent.com/Quint-e/musicnn_keras/master/"
            "musicnn_keras/keras_checkpoints/{}.h5")
AUDIO_DIR = REPO_ROOT / "data" / "raw" / "Eerola_DB" / "audio" / "set1"
OUT_ROOT = REPO_ROOT / "data" / "processed" / "Eerola_DB" / "embeddings"


def ensure_checkpoint(model: str) -> Path:
    """Download the .h5 weights once (3.3 MB) and cache them."""
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    path = CKPT_DIR / f"{model}.h5"
    if not path.is_file():
        print(f"downloading {model}.h5 ...")
        urllib.request.urlretrieve(CKPT_URL.format(model), path)
    return path


def log_mel(waveform: np.ndarray) -> np.ndarray:
    """musicnn's spectrogram: (time, 96) log-compressed mel."""
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=SR, hop_length=FFT_HOP, n_fft=FFT_SIZE, n_mels=N_MELS).T
    return np.log10(10000 * mel.astype(np.float32) + 1)


def patches(spec: np.ndarray) -> np.ndarray:
    """Split (time, 96) into non-overlapping (n, 187, 96) patches, padding if too short."""
    if spec.shape[0] < PATCH_FRAMES:
        spec = np.pad(spec, ((0, PATCH_FRAMES - spec.shape[0]), (0, 0)))
    n = spec.shape[0] // PATCH_FRAMES
    return np.stack([spec[i * PATCH_FRAMES:(i + 1) * PATCH_FRAMES] for i in range(n)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="MSD_musicnn",
                    choices=["MSD_musicnn", "MTT_musicnn"],
                    help="MSD = Million Song Dataset, MTT = MagnaTagATune")
    ap.add_argument("--tag", default=None,
                    help="cache sub-directory name (default: derived from --model)")
    args = ap.parse_args()

    import tensorflow as tf  # imported late so --help works without TF

    ckpt = ensure_checkpoint(args.model)
    full = tf.keras.models.load_model(ckpt)
    penult = tf.keras.Model(full.input, full.get_layer(PENULTIMATE_LAYER).output)
    print(f"{args.model}: input {full.input_shape} -> penultimate "
          f"{penult.output_shape}")

    tag = args.tag or ("musicnn" if args.model == "MSD_musicnn" else "musicnn_mtt")
    out_dir = OUT_ROOT / tag / "set1"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(AUDIO_DIR.glob("*.mp3"))
    if not files:
        sys.exit(f"no audio found in {AUDIO_DIR}")
    print(f"extracting {len(files)} clips -> {out_dir}")

    n_new = 0
    for i, f in enumerate(files, 1):
        dest = out_dir / f"{int(f.stem):03d}.npy"
        if dest.is_file():
            continue
        y, _ = librosa.load(f, sr=SR, mono=True)
        y = y[:int(round(WINDOW_S * SR))]
        batch = patches(log_mel(y))[..., np.newaxis]        # (n, 187, 96, 1)
        emb = penult.predict_on_batch(batch)                # (n, 200)
        np.save(dest, np.asarray(emb, dtype=np.float32).mean(axis=0))
        n_new += 1
        if i % 25 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    cached = len(list(out_dir.glob("*.npy")))
    print(f"done: {n_new} newly extracted, {cached} cached in total "
          f"({EMBED_DIM}-d each)")


if __name__ == "__main__":
    main()
