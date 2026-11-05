"""
Week 3: Exploratory Data Analysis & Visualization in Fashion Trends
Core Package Initialization
"""

from .data_loader import generate_fashion_dataset, load_fashion_data
from .feature_engineering import engineer_fashion_features, SmoothedTargetEncoder
from .eda_analysis import run_statistical_profiling, compute_lead_lag_correlation, analyze_sustainability_margins
from .visualization import (
    plot_cross_channel_lead_lag,
    plot_category_commercial_viability,
    plot_textile_sustainability_transition,
    plot_correlation_matrix,
    generate_all_figures
)

__version__ = "1.0.0"
__all__ = [
    "generate_fashion_dataset",
    "load_fashion_data",
    "engineer_fashion_features",
    "SmoothedTargetEncoder",
    "run_statistical_profiling",
    "compute_lead_lag_correlation",
    "analyze_sustainability_margins",
    "plot_cross_channel_lead_lag",
    "plot_category_commercial_viability",
    "plot_textile_sustainability_transition",
    "plot_correlation_matrix",
    "generate_all_figures",
]
