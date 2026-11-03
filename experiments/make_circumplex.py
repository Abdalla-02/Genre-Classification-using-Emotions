"""Draw Figure 2.1, the circumplex model of affect, redrawn after Russell (1980).

Unlike the results figures in ``make_figures.py``, this one is conceptual: it shows
Russell's eight affect concepts in his circular order (his Figure 1) and marks where the
five discrete emotions of the Eerola dataset sit. Each emotion is placed at the angle of
the closest word in Russell's direct circular scaling (degrees, 0 = pleasure,
counter-clockwise); tenderness has no word of its own there and sits with the calm,
content words. The figure is a redrawing, not a copy, and the caption names the source.

Output: ``figures/circumplex.pdf`` and ``figures/circumplex.png``.
Run:    python experiments/make_circumplex.py
"""

from __future__ import annotations

import numpy as np

from make_figures import COL, WIDTH, plt, save

# Russell's eight affect concepts, at 45-degree steps from pleasure
CONCEPTS = ["pleasure", "excitement", "arousal", "distress",
            "misery", "depression", "sleepiness", "contentment"]

# the five discrete emotions of Eerola et al. (2011): angle, radius, label offset (points)
EEROLA = {"happiness": (7.8, 0.62, (0, 8)), "anger": (99.0, 0.62, (6, 0)),
          "fear": (116.0, 0.62, (-6, 0)), "sadness": (207.5, 0.62, (0, -8)),
          "tenderness": (320.0, 0.62, (0, -8))}


def xy(angle, r):
    a = np.deg2rad(angle)
    return r * np.cos(a), r * np.sin(a)


def main() -> None:
    fig, ax = plt.subplots(figsize=(WIDTH * 0.62, WIDTH * 0.5))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="0.6", lw=0.8))
    for k in range(4):                       # axes and diagonals
        x, y = xy(45 * k, 1.0)
        ax.plot([-x, x], [-y, y], color="0.45" if k % 2 == 0 else "0.8",
                lw=0.9 if k % 2 == 0 else 0.6, ls="-" if k % 2 == 0 else "--", zorder=0)
    for k, c in enumerate(CONCEPTS):
        x, y = xy(45 * k, 1.0)
        ax.plot(x, y, "o", ms=3, color="0.25", zorder=3)
        tx, ty = xy(45 * k, 1.08)
        ha = "center" if abs(tx) < 0.1 else ("left" if tx > 0 else "right")
        va = "center" if abs(ty) < 0.1 else ("bottom" if ty > 0 else "top")
        ax.text(tx, ty, c.capitalize(), ha=ha, va=va, fontsize=8, color="0.15")
    for e, (ang, r, (dx, dy)) in EEROLA.items():
        x, y = xy(ang, r)
        ax.plot(x, y, "D", ms=5, color=COL["emotion"], zorder=4)
        ax.annotate(e, (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha="center" if dx == 0 else ("left" if dx > 0 else "right"),
                    va="center", fontsize=8, color=COL["emotion"], fontweight="bold")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.25, 1.25)
    save(fig, "circumplex")


if __name__ == "__main__":
    main()
