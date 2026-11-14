"""Figure for Fundamentals 2.4.1: waveform and log-mel spectrogram of one Eerola clip.

Conceptual, not a result: it shows what the audio front-end of VGGish (and of three other
models) sees. The clip is Set 1 no. 304 from Hellraiser, a Horror film; with 7.33 of 9 it shares the
highest fear rating among the Horror clips. The log-mel settings follow the VGGish front-end:
16 kHz mono, 25 ms windows, 10 ms hop, 64 mel bands between 125 and 7500 Hz, log energy.

Needs the clip audio in data/raw (git-ignored), so unlike make_figures.py it does not run
from a fresh clone; the figure itself is committed.

Run:  python experiments/make_spectrogram.py
Out:  figures/spectrogram.pdf, figures/spectrogram.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.features import load_set1  # noqa: E402

CLIP = 304
SR, WIN, HOP, MELS = 16000, 400, 160, 64

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def main() -> None:
    df = load_set1()
    row = df[df["number"] == CLIP].iloc[0]
    y, _ = librosa.load(row["audio_path"], sr=SR, mono=True)
    y = y[: 10 * SR]                                   # the first ten seconds
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=512, win_length=WIN, hop_length=HOP,
                                       n_mels=MELS, fmin=125, fmax=7500, power=2.0)
    L = librosa.power_to_db(S, ref=np.max, top_db=80)
    t = np.arange(len(y)) / SR
    tf = np.arange(L.shape[1] + 1) * HOP / SR

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.9, 3.4), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 2.2], "hspace": 0.25})
    a1.plot(t, y, color="0.25", lw=0.3)
    a1.set_ylabel("amplitude")
    a1.set_yticks([])
    a1.text(0.0, 1.02, f"Waveform: {len(y):,} sample values (10 s at 16 kHz)",
            transform=a1.transAxes, fontsize=8, va="bottom")
    mel_lo, mel_hi = librosa.hz_to_mel(125), librosa.hz_to_mel(7500)
    mel_edges = np.linspace(mel_lo, mel_hi, MELS + 1)      # equal steps on the mel scale
    m = a2.pcolormesh(tf, mel_edges, L, cmap="magma", shading="flat", rasterized=True)
    ticks = [125, 500, 1000, 2000, 4000, 7500]
    a2.set_yticks(librosa.hz_to_mel(ticks), ["125", "500", "1k", "2k", "4k", "7.5k"])
    a2.set_ylim(mel_lo, mel_hi)
    a2.set_ylabel("frequency (Hz,\nmel scale)")
    a2.set_xlabel("time (s)")
    a2.text(0.0, 1.02, f"Log-mel spectrogram: {MELS} mel bands, one 25 ms window every 10 ms",
            transform=a2.transAxes, fontsize=8, va="bottom")
    a2.spines["top"].set_visible(False)
    cb = fig.colorbar(m, ax=[a1, a2], fraction=0.03, pad=0.02)
    cb.set_label("energy (dB)")
    out = REPO / "figures"
    fig.savefig(out / "spectrogram.pdf", dpi=300)
    fig.savefig(out / "spectrogram.png", dpi=200)
    print(f"clip {CLIP} ({row['soundtrack']}): fear {row['fear']}, tension {row['tension']}, "
          f"valence {row['valence']}; {L.shape} -> figures/spectrogram.pdf")


if __name__ == "__main__":
    main()
