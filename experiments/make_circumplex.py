"""Draw Figure 2.1: valence and arousal with the energy and tension diagonals.

Unlike the results figures in ``make_figures.py``, this one is conceptual. It is adapted
from Figure 1 of Eerola and Vuoskoski (2011), "A comparison of the discrete and
dimensional models of emotion in music": the solid axes are valence and arousal (the
circumplex of Russell, 1980), the dashed diagonals energy arousal (tiredness to energy)
and tension arousal (calmness to tension). The grey squares are the five discrete
emotions the dataset rates. The original also shows surprise and disgust, which the
dataset does not rate, and has no happiness; happiness is added here in the pleasant,
energetic quarter. The other positions follow the original drawing.

Output: ``figures/circumplex.pdf`` and ``figures/circumplex.png``.
Run:    python experiments/make_circumplex.py
"""

from __future__ import annotations

import numpy as np

from make_figures import WIDTH, plt, save

# the five discrete emotions of Eerola et al. (2011): angle (degrees, 0 = valence+,
# counter-clockwise) and radius (circle = 1)
EMOTIONS = {"fear": (143, 0.83), "anger": (170, 0.79), "sadness": (241, 0.66),
            "tenderness": (-18, 0.77), "happiness": (40, 0.74)}

# the ends of the two arousal diagonals
DIAGONALS = {"Energy": 45, "Tension": 135, "Tiredness": 225, "Calmness": 315}


def xy(angle, r):
    a = np.deg2rad(angle)
    return r * np.cos(a), r * np.sin(a)


def main() -> None:
    fig, ax = plt.subplots(figsize=(WIDTH * 0.62, WIDTH * 0.55))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="black", lw=0.9))
    arrow = dict(arrowstyle="-|>", color="black", lw=0.9, mutation_scale=8)
    for (x0, y0), (x1, y1) in [((-1.12, 0), (1.12, 0)), ((0, -1.12), (0, 1.12))]:
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=arrow)
        ax.annotate("", (x0, y0), (x1, y1), arrowprops=arrow)   # both ends, as in the original
    solid = dict(boxstyle="square,pad=0.25", fc="white", ec="black", lw=0.8)
    ax.text(0, 1.2, "Arousal+", ha="center", va="bottom", fontsize=10, bbox=solid)
    ax.text(1.2, 0, "Valence+", ha="left", va="center", fontsize=10, bbox=solid)
    dashed = dict(boxstyle="square,pad=0.25", fc="white", ec="black", lw=0.7, ls="--")
    for name, ang in DIAGONALS.items():
        x, y = xy(ang, 1.12)
        ax.annotate("", (x, y), (0, 0), arrowprops=dict(arrowstyle="-|>", color="black",
                    lw=0.8, ls="--", mutation_scale=7))
        tx, ty = xy(ang, 1.42)
        ax.text(tx, ty, name, ha="center", va="center", fontsize=8.5, bbox=dashed)
    for e, (ang, r) in EMOTIONS.items():
        x, y = xy(ang, r)
        ax.plot(x, y, "s", ms=8, mfc="0.6", mec="black", mew=0.8, zorder=4)
        ax.annotate(e, (x, y), xytext=(7, 0), textcoords="offset points", ha="left",
                    va="center", fontsize=9, style="italic", zorder=5)
    # the key of the original figure
    from matplotlib.lines import Line2D
    key = [Line2D([], [], color="black", lw=0.9, label="Russell's circumplex model"),
           Line2D([], [], color="black", lw=0.8, ls="--", label="Thayer's model"),
           Line2D([], [], ls="", marker="s", ms=8, mfc="0.6", mec="black", mew=0.8,
                  label="basic emotion terms")]
    ax.legend(handles=key, loc="upper center", bbox_to_anchor=(0.45, 0.0), ncol=1,
              frameon=False, fontsize=8, handlelength=2.2, columnspacing=1.4)
    ax.set_xlim(-1.45, 1.95)
    ax.set_ylim(-1.3, 1.4)
    plt.rcParams["savefig.pad_inches"] = 0.06   # the boxed labels sit at the edge
    save(fig, "circumplex")


if __name__ == "__main__":
    main()
