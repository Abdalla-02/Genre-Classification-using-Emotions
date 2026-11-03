"""Draw Figure 2.1, the circumplex model of affect, redrawn after Russell (1980).

Unlike the results figures in ``make_figures.py``, this one is conceptual: it shows
Russell's 28 affect words in his circular order around the valence-arousal plane, with
his eight affect concepts at the ends of the axes and diagonals, and marks where the
five discrete emotions of the Eerola dataset sit. The angles are those of Russell's
direct circular scaling (degrees, 0 = pleasure, counter-clockwise); the radii are only
staggered so that the labels of close words do not overlap. The figure is a redrawing,
not a copy, and the caption names the source.

Output: ``figures/circumplex.pdf`` and ``figures/circumplex.png``.
Run:    python experiments/make_circumplex.py
"""

from __future__ import annotations

import numpy as np

from make_figures import COL, WIDTH, plt, save

# word: (angle in degrees, radius, label offset in points (dx, dy))
WORDS = {
    "happy": (7.8, 1.0, (5, 0)), "delighted": (24.9, 1.0, (4, 2)),
    "excited": (48.6, 1.0, (4, 3)), "astonished": (69.8, 1.0, (4, 3)),
    "aroused": (73.8, 0.86, (4, -1)), "tense": (92.8, 1.0, (4, 3)),
    "alarmed": (96.5, 1.0, (-4, 5)), "angry": (99.0, 0.72, (-4, 0)),
    "afraid": (116.0, 1.0, (-4, 2)), "annoyed": (123.0, 0.72, (4, 0)),
    "distressed": (137.6, 1.0, (-4, 3)), "frustrated": (141.2, 0.86, (-4, -2)),
    "miserable": (188.7, 1.0, (-5, 0)), "sad": (207.5, 1.0, (-4, -4)),
    "gloomy": (209.0, 0.79, (-4, 4)), "depressed": (210.0, 0.58, (4, -2)),
    "bored": (241.4, 1.0, (-4, -4)), "droopy": (256.7, 1.0, (-3, -6)),
    "tired": (267.7, 0.86, (-4, -1)), "sleepy": (271.9, 1.0, (4, -5)),
    "serene": (328.6, 0.62, (4, 0)), "content": (323.0, 0.71, (-4, 0)),
    "at ease": (321.0, 0.80, (4, 0)), "satisfied": (319.8, 0.89, (-4, 0)),
    "relaxed": (318.0, 0.98, (4, 0)), "calm": (316.2, 1.07, (-4, 0)),
    "glad": (348.0, 1.0, (5, 0)), "pleased": (353.2, 0.9, (4, 0)),
}

# Russell's eight affect concepts, at 45-degree steps from pleasure
CONCEPTS = ["pleasure", "excitement", "arousal", "distress",
            "misery", "depression", "sleepiness", "contentment"]

# the five discrete emotions of Eerola et al. (2011): placed at the angle of the
# closest Russell word, inside the ring (tenderness has no word of its own: it sits with
# the calm, content words); the last entry is the label offset in points
EEROLA = {"happiness": (10.0, 0.45, (0, 8)), "anger": (99.0, 0.45, (5, 0)),
          "fear": (130.0, 0.42, (-5, 0)), "sadness": (207.5, 0.30, (-5, 0)),
          "tenderness": (345.0, 0.40, (0, -8))}


def xy(angle, r):
    a = np.deg2rad(angle)
    return r * np.cos(a), r * np.sin(a)


def main() -> None:
    fig, ax = plt.subplots(figsize=(WIDTH * 0.78, WIDTH * 0.62))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="0.75", lw=0.6, ls="--"))
    for k in range(4):                       # axes and diagonals
        x, y = xy(45 * k, 1.32)
        ax.plot([-x, x], [-y, y], color="0.55" if k % 2 == 0 else "0.85",
                lw=0.9 if k % 2 == 0 else 0.5, zorder=0)
    for k, c in enumerate(CONCEPTS):
        x, y = xy(45 * k, 1.42)
        ha = "center" if abs(x) < 0.1 else ("left" if x > 0 else "right")
        va = "center" if abs(y) < 0.1 else ("bottom" if y > 0 else "top")
        ax.text(x, y, c.upper(), ha=ha, va=va, fontsize=7.5, fontweight="bold",
                color="0.2")
    for w, (ang, r, (dx, dy)) in WORDS.items():
        x, y = xy(ang, r)
        ax.plot(x, y, "o", ms=2.2, color="0.35", zorder=3)
        ax.annotate(w, (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha="left" if dx > 0 else "right", va="center", fontsize=6.5,
                    color="0.3", style="italic")
    for e, (ang, r, (dx, dy)) in EEROLA.items():
        x, y = xy(ang, r)
        ax.plot(x, y, "D", ms=4.5, color=COL["emotion"], zorder=4)
        ax.annotate(e, (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha="center" if dx == 0 else ("left" if dx > 0 else "right"),
                    va="center", fontsize=7.5, color=COL["emotion"], fontweight="bold")
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.5, 1.5)
    save(fig, "circumplex")


if __name__ == "__main__":
    main()
