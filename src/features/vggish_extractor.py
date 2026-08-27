"""VGGish embedding extraction (torchvggish), postprocessed to match Blockbuster.

VGGish is the one feature both datasets can produce identically: Blockbuster ships
pre-extracted (postprocessed, uint8) VGGish, and we extract the same 128-d postprocessed
space for the Eerola clips here. It is therefore the bridge for cross-dataset emotion
prediction (train an emotion regressor on Eerola VGGish, apply to Blockbuster VGGish).

``VggishEmbedder`` exposes the same interface as ``AstEmbedder`` (``load_audio``,
``embed_waveform``, ``sampling_rate``) so it drops into ``extract_embeddings`` unchanged.
Per clip: VGGish yields ~1 frame/0.96 s; frames are mean-pooled to one 128-d vector.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import torch

VGGISH_EMBED_DIM = 128


class VggishEmbedder:
    """Lazy wrapper around torchvggish (eval, no-grad), postprocessed to uint8 range."""

    def __init__(self):
        from torchvggish import vggish, vggish_input

        self.model = vggish(postprocess=True)  # PCA + quantize -> Blockbuster's space
        self.model.eval()
        self._to_examples = vggish_input.waveform_to_examples
        self.sampling_rate = 16000

    @torch.no_grad()
    def embed_waveform(self, waveform: np.ndarray) -> np.ndarray:
        examples = self._to_examples(waveform, self.sampling_rate)
        frames = self.model(examples).detach().cpu().numpy()  # (n_frames, 128)
        return frames.astype(np.float32).mean(axis=0)         # mean-pool -> (128,)

    def load_audio(self, path: str | Path) -> np.ndarray:
        waveform, _ = librosa.load(path, sr=self.sampling_rate, mono=True)
        return waveform
