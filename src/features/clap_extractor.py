"""CLAP (HTS-AT audio encoder) embedding extraction -- baseline for AST.

Same *approach* as AST (a pretrained, AudioSet-trained spectrogram transformer), which
makes it a fair architecture/pretraining baseline: swapping AST -> CLAP while keeping
clips, cross-validation and downstream models identical isolates the effect of the
embedding model. Uses the projected audio embedding from ``get_audio_features`` (the
standard CLAP audio vector). NB: CLAP expects 48 kHz audio (AST uses 16 kHz).

``ClapEmbedder`` exposes the same interface as ``AstEmbedder`` (``load_audio``,
``embed_waveform``, ``sampling_rate``) so it drops into ``extract_embeddings`` /
``assemble_from_cache`` unchanged.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import torch

CLAP_CHECKPOINT = "laion/clap-htsat-unfused"
CLAP_EMBED_DIM = 512


class ClapEmbedder:
    """Lazy wrapper around the CLAP model + processor (eval, no-grad)."""

    def __init__(self, checkpoint: str = CLAP_CHECKPOINT):
        from transformers import ClapModel, ClapProcessor

        self.processor = ClapProcessor.from_pretrained(checkpoint)
        self.model = ClapModel.from_pretrained(checkpoint)
        self.model.eval()
        self.sampling_rate = self.processor.feature_extractor.sampling_rate

    @torch.no_grad()
    def embed_waveform(self, waveform: np.ndarray) -> np.ndarray:
        inputs = self.processor(
            audios=waveform, sampling_rate=self.sampling_rate, return_tensors="pt"
        )
        # projected audio embedding (batch, 512) -> the standard CLAP audio vector
        return self.model.get_audio_features(**inputs).squeeze(0).cpu().numpy()

    def load_audio(self, path: str | Path) -> np.ndarray:
        waveform, _ = librosa.load(path, sr=self.sampling_rate, mono=True)
        return waveform
