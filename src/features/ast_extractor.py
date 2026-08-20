"""AST (Audio Spectrogram Transformer) embedding extraction.

Each clip is decoded to 16 kHz mono, passed through
``MIT/ast-finetuned-audioset-10-10-0.4593`` and mean-pooled over the time/sequence
axis to a single 768-dim embedding. Embeddings are cached to disk per clip so the
(one-time, CPU-bound) extraction is never repeated.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src import config

AST_CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"
EMBED_DIM = 768


class AstEmbedder:
    """Lazy wrapper around the AST feature extractor + model (eval, no-grad)."""

    def __init__(self, checkpoint: str = AST_CHECKPOINT):
        from transformers import ASTFeatureExtractor, ASTModel

        self.feature_extractor = ASTFeatureExtractor.from_pretrained(checkpoint)
        self.model = ASTModel.from_pretrained(checkpoint)
        self.model.eval()
        self.sampling_rate = self.feature_extractor.sampling_rate

    @torch.no_grad()
    def embed_waveform(self, waveform: np.ndarray) -> np.ndarray:
        inputs = self.feature_extractor(
            waveform, sampling_rate=self.sampling_rate, return_tensors="pt"
        )
        out = self.model(**inputs)
        # mean-pool the last hidden state over the sequence (time) axis
        return out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()

    def load_audio(self, path: str | Path) -> np.ndarray:
        waveform, _ = librosa.load(path, sr=self.sampling_rate, mono=True)
        return waveform

    def embed_windows(self, waveform: np.ndarray, window_s: float = 10.24,
                      hop_s: float = 5.12) -> np.ndarray:
        """Tile the clip into ``window_s`` windows (``hop_s`` stride) and embed each,
        returning an ``(n_windows, 768)`` stack. Clips <= one window give one row
        (AST pads internally). The last window is shifted to cover the clip end, so no
        audio is dropped. Pooling (first/center/mean/max) happens downstream."""
        w = int(round(window_s * self.sampling_rate))
        h = int(round(hop_s * self.sampling_rate))
        if len(waveform) <= w:
            return self.embed_waveform(waveform)[None, :]
        starts = list(range(0, len(waveform) - w + 1, h))
        if starts[-1] + w < len(waveform):
            starts.append(len(waveform) - w)  # tail window covers the end
        return np.stack([self.embed_waveform(waveform[s:s + w]) for s in starts])


def extract_embeddings(
    df: pd.DataFrame,
    set_name: str,
    embedder=None,
    cache_dir: Path = config.EMBEDDINGS_DIR,
    number_col: str = "number",
    path_col: str = "audio_path",
    make_embedder=None,
    label: str = "AST",
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return an ``(n_clips, embed_dim)`` embedding matrix aligned to ``df`` row order,
    plus a durations DataFrame (``number``, ``duration_sec``).

    Model-agnostic: ``embedder`` only needs ``load_audio(path)``, ``embed_waveform(y)``
    and ``sampling_rate`` (see AstEmbedder / ClapEmbedder). Per-clip embeddings are
    cached at ``cache_dir/<set_name>/<number:03d>.npy`` and reused on later runs
    (safe to interrupt/resume). ``make_embedder`` lazily builds the model on the first
    cache miss (defaults to AstEmbedder) so a fully-cached re-run loads nothing.
    """
    clip_dir = Path(cache_dir) / set_name
    clip_dir.mkdir(parents=True, exist_ok=True)

    embeddings: list[np.ndarray] = []
    durations: list[dict] = []

    for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"{label} {set_name}"):
        number = getattr(row, number_col)
        path = getattr(row, path_col)
        cache_file = clip_dir / f"{int(number):03d}.npy"

        if cache_file.is_file():
            embeddings.append(np.load(cache_file))
            dur = librosa.get_duration(path=path)  # cheap header read; keep CSV complete
        else:
            if embedder is None:
                embedder = (make_embedder or AstEmbedder)()
            waveform = embedder.load_audio(path)
            dur = len(waveform) / embedder.sampling_rate
            emb = embedder.embed_waveform(waveform).astype(np.float32)
            np.save(cache_file, emb)
            embeddings.append(emb)
        durations.append({"number": int(number), "duration_sec": dur})

    return np.vstack(embeddings).astype(np.float32), pd.DataFrame(durations)


def assemble_from_cache(
    df: pd.DataFrame,
    set_name: str,
    cache_dir: Path = config.EMBEDDINGS_DIR,
    number_col: str = "number",
) -> np.ndarray:
    """Assemble an ``(n_clips, 768)`` matrix from the per-clip cache, in ``df`` order.

    Raises ``FileNotFoundError`` listing any clips whose embedding is not cached, so
    experiments never silently train on a partial matrix.
    """
    clip_dir = Path(cache_dir) / set_name
    numbers = df[number_col].astype(int).tolist()
    missing = [n for n in numbers if not (clip_dir / f"{n:03d}.npy").is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} cached embeddings missing for '{set_name}' "
            f"(run notebooks/extract_features.py): {missing[:10]}"
            + (" ..." if len(missing) > 10 else "")
        )
    return np.vstack([np.load(clip_dir / f"{n:03d}.npy") for n in numbers]).astype(
        np.float32
    )


def extract_window_embeddings(df, set_name: str, embedder: AstEmbedder | None = None,
                              cache_dir: Path = config.WINDOW_EMBEDDINGS_DIR,
                              number_col: str = "number", path_col: str = "audio_path"):
    """Cache an ``(n_windows, 768)`` per-clip window stack at
    ``cache_dir/<set_name>/<n>.npy`` (resumable). Returns nothing; use pool_windows()."""
    clip_dir = Path(cache_dir) / set_name
    clip_dir.mkdir(parents=True, exist_ok=True)
    n_windows = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"AST-win {set_name}"):
        number = int(getattr(row, number_col))
        cache_file = clip_dir / f"{number:03d}.npy"
        if cache_file.is_file():
            n_windows.append(len(np.load(cache_file)))
            continue
        if embedder is None:
            embedder = AstEmbedder()
        stack = embedder.embed_windows(embedder.load_audio(getattr(row, path_col)))
        np.save(cache_file, stack.astype(np.float32))
        n_windows.append(len(stack))
    return np.asarray(n_windows)


def pool_windows(df, set_name: str, mode: str = "mean",
                 cache_dir: Path = config.WINDOW_EMBEDDINGS_DIR,
                 number_col: str = "number") -> np.ndarray:
    """Assemble an ``(n_clips, 768)`` matrix from cached window stacks, pooled per clip.

    mode: 'first' (window 0 = the current 10.24 s baseline), 'center' (middle window),
    'mean' (average all windows = full clip), 'max' (per-dim max over windows).
    """
    clip_dir = Path(cache_dir) / set_name
    rows = []
    for number in df[number_col].astype(int):
        stack = np.load(clip_dir / f"{number:03d}.npy")  # (n_win, 768)
        if mode == "first":
            rows.append(stack[0])
        elif mode == "center":
            rows.append(stack[len(stack) // 2])
        elif mode == "mean":
            rows.append(stack.mean(axis=0))
        elif mode == "max":
            rows.append(stack.max(axis=0))
        else:
            raise ValueError(mode)
    return np.vstack(rows).astype(np.float32)


def durations_from_audio(
    df: pd.DataFrame, sr: int = 16000, path_col: str = "audio_path",
    number_col: str = "number",
) -> pd.DataFrame:
    """Compute clip durations (seconds) directly from the audio files."""
    rows = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="durations"):
        path = getattr(row, path_col)
        dur = librosa.get_duration(path=path)
        rows.append({"number": int(getattr(row, number_col)), "duration_sec": dur})
    return pd.DataFrame(rows)
