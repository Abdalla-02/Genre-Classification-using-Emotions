"""wav2vec 2.0 embedding extraction -- the WAVEFORM-domain representation.

Why this exists (supervisor request #3: "try a waveform extractor, not just
spectrogram"). Every other representation in this project starts by turning the audio
into a time-frequency image:

  * AST      -- 128-band log-mel spectrogram -> Vision-Transformer patches
  * VGGish   -- 96x64 log-mel patches -> VGG-style CNN
  * CLAP     -- log-mel spectrogram -> HTS-AT
  * MIR      -- MFCC/chroma/spectral descriptors, all computed from the STFT

So "does emotion beat direct audio features?" had so far only ever been asked of
*spectrogram* features. A negative result could always have been blamed on the
front-end rather than on the representation. wav2vec 2.0 removes that confound: its
feature encoder is a stack of 1-D convolutions applied to the raw 16 kHz sample
sequence, with no spectral transform at any point. If the emotion bottleneck still
holds up against it, the finding is about learned audio embeddings in general, not
about mel spectrograms.

Checkpoint: ``facebook/wav2vec2-base`` -- the self-supervised pretrained model (NOT an
ASR fine-tune, whose representations are specialised toward phonetic content). 768-dim
hidden states, mean-pooled over time exactly as for AST.

WHICH LAYER: hidden-state index 2 -- the output of the SECOND transformer block -- not the
final one. (``hidden_states[0]`` is the convolutional encoder's projected output, so index
i is transformer block i.) Pooling
the final layer is the obvious default and is wrong here: a self-supervised model's last
layers specialise toward its own pretraining objective, so they transfer poorly to a task
it was never trained for. Measured on emotion regression
(experiments/features/exp_w2v_layer_sweep.py), mean R^2 falls monotonically from 0.324 at
layer 2 to 0.161 at layer 12 -- the layer choice alone is worth a factor of two, so
reporting the final layer would have understated the waveform front-end by more than the
effect the thesis is trying to measure. Layer 2 is used everywhere as a result.

Honest caveat for the thesis:
wav2vec 2.0 was pretrained on read speech (LibriSpeech), not music, so it carries a
domain mismatch that AST (AudioSet, which includes music) and VGGish do not -- it tests
the *input domain* question cleanly while confounding pretraining corpus. It is reported
as an additional baseline, not as a replacement for AST.

Clip handling matches AST: the first 10.24 s of each clip (see the progress log, section
6 -- full-clip windowed pooling was tested and changes nothing), so the comparison
between the two isolates the front-end.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import torch

W2V_CHECKPOINT = "facebook/wav2vec2-base"
W2V_EMBED_DIM = 768
W2V_WINDOW_S = 10.24  # same excerpt length as AST, so only the front-end differs
# Transformer layer to pool. Chosen by the layer sweep, NOT tuned against any genre
# result: it is selected on the Stage-1 emotion-regression task and then held fixed.
W2V_LAYER = 2


class Wav2VecEmbedder:
    """Lazy wrapper around wav2vec 2.0 (eval, no-grad); same interface as AstEmbedder."""

    def __init__(self, checkpoint: str = W2V_CHECKPOINT, layer: int = W2V_LAYER):
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(checkpoint)
        self.model = Wav2Vec2Model.from_pretrained(checkpoint, output_hidden_states=True)
        self.model.eval()
        self.layer = layer
        self.sampling_rate = self.feature_extractor.sampling_rate  # 16000

    @torch.no_grad()
    def embed_waveform(self, waveform: np.ndarray) -> np.ndarray:
        n = int(round(W2V_WINDOW_S * self.sampling_rate))
        waveform = waveform[:n]
        if len(waveform) < self.sampling_rate:  # guard: conv stack needs some length
            waveform = np.pad(waveform, (0, self.sampling_rate - len(waveform)))
        # zero-mean/unit-variance normalisation is part of the model's contract
        inputs = self.feature_extractor(
            waveform, sampling_rate=self.sampling_rate, return_tensors="pt"
        )
        out = self.model(**inputs)
        # hidden_states[0] is the CNN encoder output, [i] the i-th transformer layer
        return out.hidden_states[self.layer].mean(dim=1).squeeze(0).cpu().numpy()

    def load_audio(self, path: str | Path) -> np.ndarray:
        waveform, _ = librosa.load(path, sr=self.sampling_rate, mono=True)
        return waveform
