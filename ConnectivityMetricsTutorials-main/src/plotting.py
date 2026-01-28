"""Plotting utilities for the Hyperscanning Workshop."""

import matplotlib.pyplot as plt


def configure_plots() -> None:
    """
    Configure matplotlib defaults for consistent workshop visualizations.

    Sets font sizes, figure aesthetics, and other defaults according
    to the workshop style guide.
    """
    plt.rcParams.update({
        # Figure
        "figure.facecolor": "white",
        "figure.dpi": 150,
        # Axes
        "axes.facecolor": "white",
        "axes.edgecolor": "#2C3E50",
        "axes.labelcolor": "#2C3E50",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Ticks
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.color": "#2C3E50",
        "ytick.color": "#2C3E50",
        # Legend
        "legend.fontsize": 10,
        "legend.frameon": False,
        # Grid
        "grid.alpha": 0.3,
        "grid.color": "#CCCCCC",
        # Lines
        "lines.linewidth": 1.5,
        # Font
        "font.family": "sans-serif",
    })
