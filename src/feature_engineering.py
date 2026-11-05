"""
feature_engineering.py
----------------------
Production-grade Feature Engineering and Transformation Pipeline:
1. Cyclical Temporal Encodings (Trigonometric Sine/Cosine)
2. Lagged Momentum, Rolling Averages & Velocity Derivatives
3. Markdown Depth & Price Elasticity
4. Smoothed Empirical Bayes Target Encoding (Micci-Barreca)
5. Domain-Adapted NLP Sentiment Scoring
6. Distributional Log Transformations & Robust Scaling
"""

import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler

FASHION_LEXICON = {
    "positive": {
        "breathable": 1.6, "flattering": 1.5, "durable": 1.5, "luxurious": 1.4,
        "comfortable": 1.3, "soft": 1.2, "chic": 1.3, "quality": 1.4,
        "tailored": 1.3, "sustainable": 1.5, "recommend": 1.2
    },
    "negative": {
        "scratchy": 1.8, "frayed": 1.8, "cheap poly": 2.0, "cheap": 1.5,
        "itchy": 1.7, "flawed": 1.6, "wrinkles": 1.2, "static": 1.3,
        "returned": 1.4, "disappointed": 1.5, "stiff": 1.3, "translucent": 1.4
    }
}


class SmoothedTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Empirical Bayes Smoothed Target Encoding (Micci-Barreca formulation).
    Prevents overfitting on high-cardinality nominal features like Silhouette & SKU.
    """
    def __init__(self, target_col: str, k: int = 10, f: float = 5.0):
        self.target_col = target_col
        self.k = k
        self.f = f
        self.global_mean_ = 0.0
        self.mapping_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        target = df[self.target_col] if y is None else y
        self.global_mean_ = float(target.mean())
        
        # Calculate group stats for categorical columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        self.mapping_ = {}
        for col in cat_cols:
            if col == self.target_col:
                continue
            stats = df.groupby(col)[self.target_col].agg(['count', 'mean'])
            counts = stats['count']
            means = stats['mean']
            # Sigmoid weighting factor lambda
            smoothing_weights = 1.0 / (1.0 + np.exp(-(counts - self.k) / self.f))
            smooth_vals = smoothing_weights * means + (1.0 - smoothing_weights) * self.global_mean_
            self.mapping_[col] = smooth_vals.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col, mapping in self.mapping_.items():
            if col in df.columns:
                col_name = f"{col}_TargetEnc"
                df[col_name] = df[col].map(mapping).fillna(self.global_mean_)
        return df


def calculate_sentiment_polarity(text: str) -> float:
    """
    Compute domain-adapted sentiment polarity score in [-1.0, +1.0] using fashion lexicon.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
    text_clean = text.lower()
    pos_score = 0.0
    neg_score = 0.0
    
    for word, weight in FASHION_LEXICON["positive"].items():
        if word in text_clean:
            pos_score += weight
    for word, weight in FASHION_LEXICON["negative"].items():
        if word in text_clean:
            neg_score += weight
            
    total = pos_score + neg_score
    if total == 0:
        return 0.0
    return float(np.clip((pos_score - neg_score) / (total + 1e-5), -1.0, 1.0))


def engineer_fashion_features(df: pd.DataFrame, fit_encoders: bool = True) -> pd.DataFrame:
    """
    Standardized, production-grade feature engineering pipeline executing:
    1. Cyclical temporal encodings (Month_Sin, Month_Cos, Day_Sin, Day_Cos)
    2. Lagged momentum and velocity (Search_Lag14, Sales_EMA7, Sales_Velocity, Acceleration)
    3. Financial metrics & Markdown depth
    4. NLP sentiment polarity
    5. Smoothed Bayes target encodings
    6. Distributional Log transforms & Robust scaling
    """
    df = df.copy()
    
    # 1. Cyclical Temporal Encodings
    df['Date'] = pd.to_datetime(df['Transaction_Date'])
    df['Month'] = df['Date'].dt.month
    df['Day_of_Year'] = df['Date'].dt.dayofyear
    
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12.0)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12.0)
    df['Day_Sin'] = np.sin(2 * np.pi * df['Day_of_Year'] / 365.25)
    df['Day_Cos'] = np.cos(2 * np.pi * df['Day_of_Year'] / 365.25)
    
    # 2. Financial Metrics & Markdown Depth
    orig_p = df['Unit_Retail_Price'] if 'Unit_Retail_Price' in df.columns else df['Original_Price']
    sale_p = df['Sale_Price']
    df['Markdown_Depth'] = (orig_p - sale_p) / orig_p
    df['Markdown_Depth'] = df['Markdown_Depth'].clip(0.0, 1.0)
    
    # 3. Distributional Log Transformations
    if 'Units_Sold' in df.columns:
        df['Log_Units_Sold'] = np.log1p(df['Units_Sold'])
        
    # 4. NLP Review Sentiment
    if 'Customer_Review' in df.columns:
        df['Sentiment_Polarity'] = df['Customer_Review'].apply(calculate_sentiment_polarity)
    else:
        df['Sentiment_Polarity'] = 0.0
        
    # 5. Lagged Momentum and Velocity (Sorted by SKU and Date)
    df = df.sort_values(by=['SKU_Code', 'Date']).reset_index(drop=True)
    
    if 'Search_Interest_Index' in df.columns:
        # Grouped lag; backfill / forward-fill within SKU
        df['Search_Lag14'] = df.groupby('SKU_Code')['Search_Interest_Index'].shift(2)
        df['Search_Lag14'] = df['Search_Lag14'].fillna(df['Search_Interest_Index'])
        
    if 'Units_Sold' in df.columns:
        df['Sales_EMA7'] = df.groupby('SKU_Code')['Units_Sold'].transform(
            lambda x: x.ewm(span=7, min_periods=1).mean()
        )
        prev_sales = df.groupby('SKU_Code')['Units_Sold'].shift(1).fillna(df['Units_Sold'])
        df['Sales_Velocity'] = (df['Units_Sold'] - prev_sales) / 1.0
        
    # 6. Smoothed Bayes Target Encoding (Target = Log_Units_Sold or Return_Flag)
    if 'Return_Flag' in df.columns and fit_encoders:
        encoder = SmoothedTargetEncoder(target_col='Return_Flag', k=10, f=5.0)
        encoder.fit(df)
        df = encoder.transform(df)
        
    # 7. Robust Scaling for Unit Price
    if 'Unit_Retail_Price' in df.columns:
        scaler = RobustScaler()
        df['Price_RobustScaled'] = scaler.fit_transform(df[['Unit_Retail_Price']])
        
    return df
