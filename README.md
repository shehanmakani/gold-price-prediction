# Gold Price Prediction: Genuine Next-Day Forecasting with Walk-Forward Validation

> Contributors: Sathwika Garlapati · Shehan Makani · Prarthna Rana

---

## What This Project Does (Precisely)

This repository implements a **strictly defined next-day gold price forecasting task**:

> **Given all information available at the close of trading day *t*, predict the closing price on day *t+1*.**

Every design decision follows from that single sentence.

---

## Dataset

| Property | Value |
|---|---|
| **File** | `Gold Price.csv` |
| **Source** | MCX India gold spot price (Multi Commodity Exchange) |
| **Shape** | 3,104 rows × 7 columns |
| **Date range** | 2014-01-01 → 2026-01-02 |
| **Price units** | **INR per 10 grams** (MCX convention) |
| **Price range** | ₹24,545 – ₹1,37,789 |

### Why prices look large
The Kaggle dataset label says "2015–2021" but the file contains data from 2014 through January 2026 (3,104 rows). Prices are in **Indian Rupees per 10 grams**, the standard MCX format — not USD/oz. A price of ₹1,00,000/10g ≈ $3,700/oz at ₹84/USD — consistent with global gold prices.

---

## Prediction Task & Feature Engineering

### Strict No-Leakage Rule
All features use `.shift(1)` or greater — **no same-day Open, High, or Low values** are ever used to predict that same day's close. The HL_Spread feature uses *yesterday's* High-Low range.

### Final Feature Set (19 features)

| Group | Features | Minimum Lag |
|---|---|---|
| Price lags | Lag_1, Lag_5, Lag_10, Lag_21, Lag_63 | 1 day |
| Moving averages | MA_7, MA_21, MA_63 | shift(1) applied |
| Rolling volatility | Std_7, Std_21 | shift(1) applied |
| Momentum | ROC_5, ROC_21 | shift(1) applied |
| Range spread | HL_Spread (prev. day High − Low) | shift(1) applied |
| Log return | Log_Return (prev. day) | shift(1) applied |
| Calendar | DayOfWeek, Month | current day (no leakage) |
| Trend ratios | MA7_vs_MA21, MA21_vs_MA63 | shift(1) applied |
| Z-score | ZScore_21 | shift(1) applied |

---

## Methodology

### Walk-Forward Cross-Validation

```
Training: [t=0 ... t=499] → Test: [t=500]
Training: [t=0 ... t=520] → Test: [t=521]
Training: [t=0 ... t=541] → Test: [t=542]
...
```

- **Initial training size:** 500 days  
- **Step size:** 21 days (≈ 1 trading month)  
- **Nested hyperparameter tuning:** `GridSearchCV` with `TimeSeriesSplit(n_splits=3)` inside each training window only  
- **No random train/test splits** — temporal integrity is strictly maintained

### Directional Accuracy Definition

```
DA = fraction of folds where:
     sign(predicted_close - previous_close) == sign(actual_close - previous_close)
```

This measures whether the model correctly predicts the *direction* of next-day price movement.

---

## Models

| Model | Hyperparameter Grid |
|---|---|
| **Ridge Regression** | alpha ∈ {0.01, 0.1, 1, 10, 100, 1000} |
| **XGBoost** | n_estimators ∈ {100,300}, max_depth ∈ {3,5}, lr ∈ {0.01,0.05,0.1} |
| **SVR** | C ∈ {0.1,1,10,100}, ε ∈ {0.01,0.1}, kernel=rbf |
| **Random Forest** | n_estimators ∈ {100,300}, max_depth ∈ {5,10,None} |

---

## Baselines (Minimum Bar for Any ML Model)

| Baseline | Description |
|---|---|
| **Persistence (naïve)** | Predict t+1 = today's closing price |
| **7-day MA** | Predict t+1 = 7-day rolling mean of past prices |

---

## Repository Structure

```
gold-price-prediction/
├── Gold Price.csv                                    # Raw dataset (MCX India, INR/10g)
├── gold_price_prediction_v2.ipynb                    # ✅ Main notebook (v2 — all fixes)
├── ML_FinalProject_GoldPrice_(Walk_Forward).ipynb    # Original submission notebook
├── ML_FinalProject_GoldPrice (Initial Testing).ipynb # Initial exploration
├── requirements.txt                                  # Dependencies
├── README.md                                         # This file
└── LICENSE
```

---

## Installation

```bash
git clone https://github.com/shehanmakani/gold-price-prediction.git
cd gold-price-prediction
pip install -r requirements.txt
jupyter notebook gold_price_prediction_v2.ipynb
```

**Google Colab:** Upload `Gold Price.csv` to session storage and execute all cells.

---

## Changelog (v2 — addressing reviewer feedback)

| Issue | Fix Applied |
|---|---|
| Ambiguous prediction task | Explicitly defined as **next-day (t+1) close** |
| Same-day feature leakage | All features shifted ≥1 day; HL_Spread uses *previous* day's range |
| Inconsistent dataset description | Documented as 3,104 rows, 2014-01-01 → 2026-01-02 |
| Price scale unexplained | Documented as INR/10g (MCX India) with USD equivalence |
| No baselines | Added Persistence and 7-day MA baselines |
| Directional accuracy undefined | Precisely defined (sign of movement vs previous close) |
| Inconsistent feature list | Single canonical 19-feature list used throughout |
| Random Forest omitted | All 4 models in full results table |
| No regime analysis | Section 8: calm vs 2020-volatile performance |
| Overconfident conclusion | Conclusions qualified against baselines |

---

## License

MIT — see `LICENSE`.
