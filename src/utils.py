"""
utils.py
--------
Utility functions for metrics formatting, reporting, and file management.
"""

import os
import pandas as pd


def print_section_header(title: str, character: str = "="):
    """Print visually distinct terminal banner."""
    line = character * 72
    print(f"\n{line}\n{title}\n{line}")


def save_metrics_summary(df: pd.DataFrame, filepath: str):
    """Export summary dataframe to CSV with directory creation."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"[OK] Saved summary report to: {filepath}")
