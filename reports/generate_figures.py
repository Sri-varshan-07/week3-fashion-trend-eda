"""
generate_figures.py
-------------------
Standalone utility to generate all 4 publication-ready figures for Week 3 report.
"""

import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.visualization import generate_all_figures

if __name__ == "__main__":
    output_directory = os.path.join(os.path.dirname(__file__), "figures")
    print(f"Generating Week 3 analytical figures in: {output_directory}")
    paths = generate_all_figures(output_dir=output_directory)
    for fig_id, path in paths.items():
        print(f"  [OK] {fig_id}: {path}")
    print("All figures successfully rendered.")
