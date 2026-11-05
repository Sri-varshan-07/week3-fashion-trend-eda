# Multi-Modal Data Architecture & Dictionary

## 1. Overview & Data Ingestion Channels

The Week 3 analytical framework is built upon an enterprise multi-modal architecture reconciling four heterogeneous data streams:

```
+---------------------------------------------------------------------------------------+
|                              MULTI-MODAL DATA ARCHITECTURE                            |
+---------------------------------------------------------------------------------------+
|  1. POS Retail Transactions (142,500 checkouts, 2,400 SKUs, H&M/Zara open schemas)    |
|  2. Web Search Telemetry (Google Trends API / PyTrends normalized indices [0-100])     |
|  3. Cultural Social Media Stream (TikTok & Instagram scrape metadata, virality index) |
|  4. Textile & Sustainability Specs (Textile Exchange standards, fiber mixes, GSM)     |
+---------------------------------------------------------------------------------------+
```

---

## 2. Complete Data Schema & Attribute Dictionary

| Attribute Name | Source Domain | Data Type | Range / Distinct | Business Description |
| :--- | :--- | :--- | :--- | :--- |
| `Transaction_ID` | POS Retail | String / UUID | 142,500 Unique IDs | Omnichannel retail checkout identifier |
| `Transaction_Date` | POS Retail | Date (YYYY-MM-DD) | Jan 2025 - Jun 2026 | Date timestamp of purchase transaction |
| `SKU_Code` | Catalog Master | Categorical | 2,400 Active SKUs | Unique stock keeping unit code |
| `Product_Category` | Catalog Master | Categorical | 6 Super-Categories | Trousers, Outerwear, Knitwear, Dresses, Tops, Skirts |
| `Silhouette` | Catalog Master | Categorical | 24 Trending Cuts | Garment silhouette cut (e.g. Linen Wide-Leg Trouser) |
| `Unit_Retail_Price` | POS Retail | Float ($ USD) | $14.99 - $289.00 | Gross catalog retail selling price |
| `Discount_Applied` | POS Retail | Float (%) | 0.0% to 65.0% | Markdown applied at point-of-sale |
| `Sale_Price` | POS Retail | Float ($ USD) | $11.24 - $289.00 | Net price paid by consumer post-discount |
| `Fabric_Composition` | Textile Spec | String / Blend | 6 Standard Blends | Exact fiber composition (e.g., 85% Linen / 15% Cotton) |
| `Fabric_Type` | Textile Spec | Categorical | 5 Primary Classes | Sustainable Natural, Recycled Synthetic, Virgin Synthetic, etc. |
| `Eco_Score` | Textile Spec | Float (0 - 100) | 32.0 to 92.0 | Third-party sustainability compliance score |
| `Social_Virality_Index` | Social Stream | Float (0 - 100) | 0.00 to 98.40 | Normalized TikTok & IG engagement momentum index |
| `Search_Interest_Index` | Google Trends | Integer (0 - 100) | 4 to 100 | Weekly consumer search interest telemetry |
| `Units_Sold` | POS Retail | Integer | 1 to 45 units/txn | Transaction unit volume |
| `Return_Flag` | POS Returns | Binary (0 / 1) | 0 (90.2%), 1 (9.8%) | Customer post-sale return indicator |
| `Customer_Review` | Omnichannel CRM | String / Text | 38,400+ reviews | Verbatim customer feedback text snippet |

---

## 3. Data Hygiene & Quality Rules

1. **Missing Values**: Any missing `Discount_Applied` defaults to `0.0` (full price).
2. **Outlier Boundaries**: Prices constrained between `$10.00` and `$500.00` to prevent runway sample distortion.
3. **Temporal Sorting**: All lagged momentum operations require sorting by `(SKU_Code, Transaction_Date)` prior to rolling calculations.
