"""
data_loader.py
--------------
Multi-modal fashion dataset generator and loader simulating:
1. POS Retail Transactions (H&M / Zara style)
2. Algorithmic Web Search Telemetry (Google Trends API / PyTrends)
3. Social Media Velocity Signals (TikTok & Instagram Public Scrapes)
4. Textile & Sustainable Material Standards (Textile Exchange)
"""

import os
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

CATEGORIES_AND_SILHOUETTES = {
    "Trousers": ["Linen Wide-Leg Trouser", "Tailored Pleated Pant", "Cargo Utility Pant", "Relaxed Denim"],
    "Outerwear": ["Oversized Wool Blazer", "Relaxed Trench Coat", "Cropped Quilted Jacket", "Minimalist Car Coat"],
    "Knitwear": ["Chunky Knit Cardigan", "Fine Gauge Merino Crew", "Cashmere Polo Sweater", "Cable Knit Vest"],
    "Dresses": ["Slip Dress Bias Cut", "Pleated Midi Dress", "Tiered Linen Maxi", "Bodycon Ribbed Dress"],
    "Tops": ["Ribbed Cotton Tank Top", "Oversized Poplin Shirt", "Bandeau Linen Tube Top", "Silk Blend Blouse"],
    "Skirts": ["Pleated Midi Skirt", "Denim Maxi Skirt", "A-Line Linen Mini", "Bias Cut Satin Skirt"]
}

FABRIC_BLENDS = [
    ("85% French Flax Linen / 15% Organic Cotton", "Sustainable Natural", 92.0),
    ("100% GOTS Certified Organic Cotton", "Sustainable Natural", 88.0),
    ("70% Recycled PET / 30% ECONYL Nylon", "Recycled Synthetic", 78.0),
    ("100% Virgin Polyester", "Virgin Synthetic", 32.0),
    ("100% Conventional Cotton", "Conventional Natural", 45.0),
    ("80% Wool / 20% Cashmere", "Premium Natural", 85.0),
]

SAMPLE_REVIEWS = {
    "positive": [
        "Incredibly breathable and flattering silhouette, love the drape and quality.",
        "Durable fabric, lightweight, feels luxurious and chic for summer.",
        "Perfect tailored fit, clean luxury aesthetic, very high quality stitching.",
        "Super comfortable natural linen feel, fits true to size, highly recommend.",
        "High end feel, soft touch, sustainable fabric that washes beautifully."
    ],
    "negative": [
        "Cheap poly feel, scratchy and frayed after only one gentle wash.",
        "Horrible synthetic hand-feel, translucent, runs very small.",
        "Disappointed by quality, synthetic wrinkles easily and static cling.",
        "Very itchy material, flawed cut, returned immediately.",
        "Low quality fast fashion item, uncomfortable fabric and stiff seams."
    ]
}


def generate_fashion_dataset(n_samples: int = 15000, n_skus: int = 240, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic representative multi-modal transactional & trend dataset.
    Default produces 15,000 samples across 240 SKUs for responsive local execution;
    can scale up to N=142,500.
    """
    np.random.seed(random_seed)
    
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2026, 6, 30)
    date_range_days = (end_date - start_date).days

    # Pre-generate SKU catalog master
    sku_master = []
    for i in range(n_skus):
        cat = np.random.choice(list(CATEGORIES_AND_SILHOUETTES.keys()))
        sil = np.random.choice(CATEGORIES_AND_SILHOUETTES[cat])
        fabric, fabric_type, eco_score = FABRIC_BLENDS[np.random.choice(len(FABRIC_BLENDS))]
        
        # Base pricing by category and fabric
        base_price = {
            "Trousers": 79.0, "Outerwear": 189.0, "Knitwear": 119.0,
            "Dresses": 99.0, "Tops": 39.0, "Skirts": 69.0
        }[cat]
        if "Sustainable" in fabric_type or "Premium" in fabric_type:
            base_price *= np.random.uniform(1.15, 1.35)
        elif "Virgin" in fabric_type:
            base_price *= np.random.uniform(0.70, 0.90)
            
        sku_master.append({
            "SKU_Code": f"SKU-{1000 + i}",
            "Product_Category": cat,
            "Silhouette": sil,
            "Fabric_Composition": fabric,
            "Fabric_Type": fabric_type,
            "Eco_Score": eco_score,
            "Base_Price": round(base_price + np.random.normal(0, 5), 2)
        })
    sku_df = pd.DataFrame(sku_master)

    # Generate transaction records
    records = []
    for _ in range(n_samples):
        sku_row = sku_df.sample(1).iloc[0]
        
        # Temporal assignment
        rand_days = np.random.randint(0, date_range_days)
        trans_date = start_date + timedelta(days=rand_days)
        month = trans_date.month
        
        # Seasonal wave factors
        is_summer = 1.0 if month in [5, 6, 7, 8] else 0.3
        is_winter = 1.0 if month in [11, 12, 1, 2] else 0.3
        
        # Trend affinity: Linen/Summer peaks in summer; Wool in winter
        if "Linen" in sku_row["Silhouette"] or "Tank" in sku_row["Silhouette"]:
            seasonal_affinity = is_summer
        elif "Wool" in sku_row["Silhouette"] or "Knit" in sku_row["Silhouette"] or "Trench" in sku_row["Silhouette"]:
            seasonal_affinity = is_winter
        else:
            seasonal_affinity = 0.6
            
        # Social Virality Index [0, 100]
        base_virality = np.random.beta(2, 5) * 70
        if "Linen Wide-Leg" in sku_row["Silhouette"] or "Oversized Wool" in sku_row["Silhouette"]:
            base_virality += 25 * seasonal_affinity
        virality_index = float(np.clip(base_virality + np.random.normal(0, 5), 0.0, 98.4))
        
        # Search Interest Index [0, 100] correlated with virality
        search_index = int(np.clip(virality_index * 0.85 + np.random.normal(5, 8), 4, 100))
        
        # Pricing & Markdown
        orig_price = sku_row["Base_Price"]
        if sku_row["Fabric_Type"] == "Sustainable Natural":
            # Sustainable items require very little markdown (4.2% average)
            discount_pct = float(np.clip(np.random.exponential(0.04), 0.0, 0.20))
        elif sku_row["Fabric_Type"] == "Virgin Synthetic":
            # Synthetics face heavy end-of-season liquidation (32.8% average)
            discount_pct = float(np.clip(np.random.normal(0.32, 0.15), 0.0, 0.65))
        else:
            discount_pct = float(np.clip(np.random.exponential(0.12), 0.0, 0.45))
            
        sale_price = round(orig_price * (1.0 - discount_pct), 2)
        
        # Units sold (power-law distribution with virality leverage)
        unit_power = np.random.exponential(scale=2.5) * (1.0 + (virality_index / 40.0))
        units_sold = int(max(1, round(unit_power)))
        
        # Return flag (8.4% for sustainable, 21.6% for synthetic)
        if sku_row["Fabric_Type"] == "Sustainable Natural":
            p_return = 0.084
        elif sku_row["Fabric_Type"] == "Virgin Synthetic":
            p_return = 0.216
        else:
            p_return = 0.125
        return_flag = int(np.random.random() < p_return)
        
        # Customer review text simulation
        if return_flag == 1 or discount_pct > 0.40:
            review_text = np.random.choice(SAMPLE_REVIEWS["negative"])
        else:
            review_text = np.random.choice(SAMPLE_REVIEWS["positive"])
            
        records.append({
            "Transaction_ID": f"TXN-{uuid.uuid4().hex[:10].upper()}",
            "Transaction_Date": trans_date.strftime("%Y-%m-%d"),
            "SKU_Code": sku_row["SKU_Code"],
            "Product_Category": sku_row["Product_Category"],
            "Silhouette": sku_row["Silhouette"],
            "Fabric_Composition": sku_row["Fabric_Composition"],
            "Fabric_Type": sku_row["Fabric_Type"],
            "Eco_Score": sku_row["Eco_Score"],
            "Unit_Retail_Price": orig_price,
            "Discount_Applied": round(discount_pct, 4),
            "Sale_Price": sale_price,
            "Social_Virality_Index": round(virality_index, 2),
            "Search_Interest_Index": search_index,
            "Units_Sold": units_sold,
            "Return_Flag": return_flag,
            "Customer_Review": review_text
        })
        
    df = pd.DataFrame(records)
    return df


def load_fashion_data(filepath: str = None) -> pd.DataFrame:
    """Load or generate the fashion dataset."""
    if filepath and os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"])
        return df
    return generate_fashion_dataset()
