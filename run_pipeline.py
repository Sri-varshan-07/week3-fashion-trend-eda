"""
run_pipeline.py
---------------
End-to-end execution pipeline for Week 3:
1. Sourcing multi-modal transactional and trend dataset
2. Executing production feature engineering & transformations
3. Running statistical EDA, lead-lag analysis, and margin profiling
4. Generating publication-ready visual analytical figures
"""

import os
import sys

# Ensure src is discoverable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import generate_fashion_dataset
from src.feature_engineering import engineer_fashion_features
from src.eda_analysis import (
    run_statistical_profiling,
    compute_lead_lag_correlation,
    analyze_sustainability_margins,
    analyze_markdown_elasticity_cliff
)
from src.visualization import generate_all_figures
from src.utils import print_section_header, save_metrics_summary


def main():
    print_section_header("WEEK 3: EXPLORATORY DATA ANALYSIS & VISUALIZATION PIPELINE")
    
    # 1. Dataset Generation
    print("\n[1/4] Ingesting multi-modal dataset (POS, Search, Social, Textile)...")
    raw_df = generate_fashion_dataset(n_samples=15000, n_skus=240, random_seed=42)
    raw_path = os.path.join("data", "raw", "sample_fashion_raw.csv")
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    raw_df.to_csv(raw_path, index=False)
    print(f"      Ingested {len(raw_df):,} transaction records across {raw_df['SKU_Code'].nunique()} SKUs.")
    print(f"      Saved raw data to: {raw_path}")

    # 2. Comprehensive Feature Engineering
    print("\n[2/4] Executing comprehensive feature engineering transformations...")
    eng_df = engineer_fashion_features(raw_df, fit_encoders=True)
    proc_path = os.path.join("data", "processed", "fashion_trends_engineered.csv")
    os.makedirs(os.path.dirname(proc_path), exist_ok=True)
    eng_df.to_csv(proc_path, index=False)
    print(f"      Engineered {eng_df.shape[1]} features (cyclical, lagged momentum, sentiment, target encoded).")
    print(f"      Saved processed data to: {proc_path}")

    # 3. Statistical Profiling & Domain Analytics
    print("\n[3/4] Running statistical profiling and empirical analyses...")
    
    # Lead-lag correlation
    lead_lag_df = compute_lead_lag_correlation(eng_df)
    print("\n--- Cross-Channel Lead-Lag Predictive Horizon ---")
    print(lead_lag_df.to_string(index=False))
    
    # Sustainability margin paradox
    sust_df = analyze_sustainability_margins(eng_df)
    print("\n--- Textile Sustainability Margin Breakdown ---")
    print(sust_df[['Fabric_Category', 'Full_Price_SellThrough_Pct', 'Return_Rate_Pct', 'Net_Operating_Margin_Pct']].to_string(index=False))

    # Markdown elasticity cliff
    cliff_df = analyze_markdown_elasticity_cliff(eng_df)
    print("\n--- Markdown Elasticity Cliff Profiling ---")
    print(cliff_df.to_string(index=False))

    # 4. Generate Visual Analytics
    print("\n[4/4] Rendering publication-ready visual analytical models...")
    fig_paths = generate_all_figures(output_dir=os.path.join("reports", "figures"))
    for fig_id, path in fig_paths.items():
        print(f"      Generated {fig_id.upper()}: {path}")

    print_section_header("PIPELINE COMPLETED SUCCESSFULLY", character="*")


if __name__ == "__main__":
    main()
