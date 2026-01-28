"""Color palette for the Hyperscanning Workshop visualizations."""

from typing import Dict

# Signal colors (soft pastels)
COLORS: Dict[str, str] = {
    # Primary signals
    "signal_1": "#7EB8DA",  # Sky Blue - Subject 1
    "signal_2": "#F4A4B8",  # Rose Pink - Subject 2
    "signal_3": "#B8D4A8",  # Sage Green - Reference
    "signal_4": "#E8C87A",  # Golden - Highlight
    "signal_5": "#C4A8D4",  # Lavender
    "signal_6": "#A8D4D0",  # Soft Teal
    # Connectivity
    "low_sync": "#ECF0F1",  # Low connectivity
    "high_sync": "#9B59B6",  # High connectivity
    # Diverging (correlations)
    "negative": "#E17055",  # Negative values
    "zero": "#FFFFFF",  # Zero
    "positive": "#00CEC9",  # Positive values
    # Frequency bands
    "delta": "#6C5B7B",  # 1-4 Hz
    "theta": "#C06C84",  # 4-8 Hz
    "alpha": "#F8B500",  # 8-13 Hz
    "beta": "#00CEC9",  # 13-30 Hz
    "gamma": "#6DD47E",  # 30+ Hz
    # Utility
    "grid": "#CCCCCC",
    "text": "#2C3E50",
    "background": "#FFFFFF",
}
