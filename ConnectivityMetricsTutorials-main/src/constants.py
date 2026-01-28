"""Constants for EEG signal analysis.

This module provides standard definitions for EEG frequency bands
and associated colors for visualization.
"""

# Standard EEG frequency bands (Hz)
EEG_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 100.0),
}

# Colors for visualization of each band
# NOTE: These colors are synchronized with src/colors.py and docs/STYLE_GUIDE.md
BAND_COLORS: dict[str, str] = {
    "delta": "#6C5B7B",   # Plum
    "theta": "#C06C84",   # Rose
    "alpha": "#F8B500",   # Gold
    "beta": "#00CEC9",    # Teal
    "gamma": "#6DD47E",   # Mint
}

# Extended band definitions (for specific research needs)
EEG_BANDS_EXTENDED: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "low_alpha": (8.0, 10.0),
    "high_alpha": (10.0, 13.0),
    "low_beta": (13.0, 20.0),
    "high_beta": (20.0, 30.0),
    "low_gamma": (30.0, 50.0),
    "high_gamma": (50.0, 100.0),
}
