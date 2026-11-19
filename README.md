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

### No-Look-Ahead Rule
Row *t* holds only what is known at the **close of day *t*** (the forecast origin); the target is `Price.shift(-1)`, the close on day *t+1*. Nothing from day *t+1* is ever used as an input. `Lag_k` is the close *k−1* trading days before the origin, so `Lag_1` is today's close.

> An earlier version added an extra `.shift(1)` to every feature, so the model never saw today's close and was effectively forecasting two days ahead. That has been fixed (see Changelog).

### Final Feature Set (19 features)

| Group | Features | Data used |
|---|---|---|
| Price lags | Lag_1, Lag_5, Lag_10, Lag_21, Lag_63 | close on day t, t−4, t−9, t−20, t−62 |
| Moving averages | MA_7, MA_21, MA_63 | windows ending on day t |
| Rolling volatility | Std_7, Std_21 | windows ending on day t |
| Momentum | ROC_5, ROC_21 | ending on day t |
| Range spread | HL_Spread (day t High − Low) | day t |
| Log return | Log_Return (day t) | day t |
| Calendar | DayOfWeek, Month | day t |
| Trend ratios | MA7_vs_MA21, MA21_vs_MA63 | day t |
| Z-score | ZScore_21 | day t |

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
     sign(predicted_close - today_close) == sign(actual_close - today_close)
```

`today_close` is the close at the forecast origin (day *t*) — the last price known when the forecast is made. This measures whether the model correctly predicts the *direction* of the next-day move. Persistence predicts "no change", so its directional accuracy is 0% by construction.

### RMSE
Each walk-forward fold produces a single forecast. All forecasts are collected first and **RMSE is computed once over the pooled errors**. (Averaging a per-fold RMSE over one-point folds just reproduces MAE.) Baselines are scored on exactly the same test days as the models.

---

## Results (corrected, 121 folds, next-day forecast)

| Model | MAPE | RMSE (₹/10g, pooled) | Dir. Acc. |
|---|---|---|---|
| Ridge | 0.54% | 422 | 55.4% |
| XGBoost | 0.71% | 727 | 50.4% |
| Random Forest | 0.78% | 902 | 40.5% |
| SVR | 1.92% | 3,249 | 51.2% |
| **Persistence** (tomorrow = today) | **0.52%** | **421** | 0% (no-change forecast) |
| 7-day MA | 0.97% | 835 | 57.0% |

**Takeaway:** no model beats persistence. Ridge essentially matches it; tree models do worse because they cannot extrapolate past the training price range. Directional accuracy is close to a coin flip on 121 forecasts.

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
├── gold_price_prediction_v1.ipynb                    # ✅ Next-day evaluation: 4 models + baselines (results above)
├── gold_price_prediction_v2.ipynb                    # Extensions: macro features, regime switching, t+5/t+21
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
jupyter notebook gold_price_prediction_v1.ipynb
```

**Google Colab:** Upload `Gold Price.csv` to session storage and execute all cells.

---

## Changelog — evaluation fixes (Nov 2026)

| Bug | Fix Applied |
|---|---|
| Features shifted one day too many (model never saw today's close) | Extra `.shift(1)` removed; features use data up to day *t*'s close, target is day *t+1* |
| Directional accuracy measured against yesterday's close (`Lag_1`) | Now measured against today's close (`Price` at the origin) |
| RMSE averaged per one-point fold (= MAE) and compared with a full-sample persistence RMSE | All forecasts collected first, RMSE computed once; baselines scored on the same test days |
| v2 multi-step training used targets beyond the forecast origin for h>1 | Training rows limited to targets dated on or before the origin |

Earlier conclusions (Ridge beating persistence on RMSE; Random Forest 75% directional accuracy in 2020) came from these bugs and have been withdrawn.

---

## Changelog (v2 — addressing reviewer feedback)

| Issue | Fix Applied |
|---|---|
| Ambiguous prediction task | Explicitly defined as **next-day (t+1) close** |
| Same-day feature leakage | Features restricted to data known at the forecast origin (see Nov 2026 fix) |
| Inconsistent dataset description | Documented as 3,104 rows, 2014-01-01 → 2026-01-02 |
| Price scale unexplained | Documented as INR/10g (MCX India) with USD equivalence |
| No baselines | Added Persistence and 7-day MA baselines |
| Directional accuracy undefined | Precisely defined (sign of movement vs the close at the forecast origin) |
| Inconsistent feature list | Single canonical 19-feature list used throughout |
| Random Forest omitted | All 4 models in full results table |
| No regime analysis | Section 8: calm vs 2020-volatile performance |
| Overconfident conclusion | Conclusions qualified against baselines |

---

## License

MIT — see `LICENSE`.
