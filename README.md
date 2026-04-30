# Gold Price Prediction: Time-Aware Machine Learning Evaluation

This repository contains the implementation of a robust forecasting framework for daily gold prices (2015–2021). Unlike standard regression models that suffer from data leakage, this project utilizes **Walk-Forward Cross-Validation** to ensure realistic performance estimates.

## 🚀 Key Features
- **Leak-Free Engineering:** Strictly uses $t-n$ data to predict $t$.
- **Stationarity Analysis:** Includes Augmented Dickey-Fuller (ADF) testing and Log-Return transformations.
- **Nested Hyperparameter Tuning:** Model parameters are optimized within each walk-forward fold to prevent look-ahead bias.
- **Directional Accuracy:** Evaluation focused on trading utility (sign of movement).

## 📊 Performance Summary
| Model | MAPE (Mean ± Std) | Directional Accuracy |
| :--- | :--- | :--- |
| **Ridge Regression** | **0.35% ± 0.17%** | **93.45%** |
| XGBoost | 2.14% ± 2.12% | 61.31% |

## 🛠️ Installation & Usage
1. Clone the repo: `https://github.com/shehanmakani/gold-price-prediction'
2. Install dependencies: `pip install -r requirements.txt`
3. Run the analysis: Open `ML_FinalProject_GoldPrice.ipynb` in Jupyter or Google Colab.

## 📝 Project Context
This project was developed for **DS675 Machine Learning** as a Milestone 4 Final Submission. 
**Contributors:** Sathwika Garlapati, Shehan Makani, and Prarthna Rana.
