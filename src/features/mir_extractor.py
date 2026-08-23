"""Hand-crafted MIR feature extraction via librosa -- a classic-MER baseline.

A standard set of music-information-retrieval features (MFCC + deltas, chroma, spectral
descriptors, ZCR, RMS, tempo), each summarised by mean and standard deviation over frames
into one fixed vector per clip. This mirrors the *kind* of features in Blockbuster's MIR
set, but is computed with librosa, not MATLAB MIRtoolbox -- so it is a comparison point
for emotion regression, NOT a cross-dataset bridge to Blockbuster.

``MirEmbedder`` exposes the same interface as ``AstEmbedder`` (``load_audio``,
``embed_waveform``, ``sampling_rate``) so it drops into ``extract_embeddings``.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


def _mean_std(mat: np.ndarray) -> np.ndarray:
    """Concatenate per-row mean and std of a (features, frames) matrix."""
    mat = np.atleast_2d(mat)
    return np.concatenate([mat.mean(axis=1), mat.std(axis=1)])


class MirEmbedder:
    """Compute a fixed-length hand-crafted MIR feature vector per clip (librosa)."""

    def __init__(self, sr: int = 16000):
        self.sampling_rate = sr

    def load_audio(self, path: str | Path) -> np.ndarray:
        waveform, _ = librosa.load(path, sr=self.sampling_rate, mono=True)
        return waveform

    def embed_waveform(self, y: np.ndarray) -> np.ndarray:
        sr = self.sampling_rate
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        blocks = [
            mfcc,
            librosa.feature.delta(mfcc),
            librosa.feature.chroma_stft(y=y, sr=sr),
            librosa.feature.spectral_centroid(y=y, sr=sr),
            librosa.feature.spectral_bandwidth(y=y, sr=sr),
            librosa.feature.spectral_rolloff(y=y, sr=sr),
            librosa.feature.spectral_flatness(y=y),
            librosa.feature.spectral_contrast(y=y, sr=sr),
            librosa.feature.zero_crossing_rate(y),
            librosa.feature.rms(y=y),
        ]
        vec = np.concatenate([_mean_std(b) for b in blocks])
        try:
            tempo = float(np.atleast_1d(librosa.feature.tempo(y=y, sr=sr))[0])
        except Exception:
            tempo = 0.0
        return np.concatenate([vec, [tempo]]).astype(np.float32)
