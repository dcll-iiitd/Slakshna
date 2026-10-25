"""Shared figure style for the RL paper figures.

Colours are the first slots of the validated categorical palette (blue, orange, aqua); the first
three pass colour-blind separation as a set. Colour is never the only cue: every series also gets a
line style or marker. Output is vector PDF plus a PNG preview.
"""
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1a1a19", "#52514e", "#e3e2de"

COL1, COL2 = 3.4, 7.0   # single / double column width in inches (typical two-column venue)


def apply_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 1.5, "lines.markersize": 4.5,
        "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,     # embed fonts as TrueType (venue requirement)
    })


def wilson(p, n, z=1.96):
    """95% Wilson interval for a proportion p measured on n items -> (low, high)."""
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def save(fig, path_no_ext):
    fig.savefig(path_no_ext + ".pdf", bbox_inches="tight")
    fig.savefig(path_no_ext + ".png", bbox_inches="tight")
    plt.close(fig)
