# Crypto Market Regime Detection

Detects Bitcoin market regimes (Bull, Bear, Sideways) using Hidden Markov Models, then trains a supervised classifier to predict the current regime from observable technical indicators.

---

## The Problem

Bitcoin's price behaves very differently depending on the market context. A strategy that works in a bull market can be catastrophic in a bear market. The challenge: **market regimes are not directly observable** — you can't read them from a single price or indicator.

This project tackles that problem in two stages:
1. **Unsupervised discovery** — use a Hidden Markov Model to identify hidden market states from 8 years of BTC data
2. **Supervised prediction** — train a classifier that learns to predict the current regime from technical indicators, enabling real-time inference

---

## Pipeline

```
BTC Historical Data (2018–2026)
        ↓
Feature Engineering (6 technical indicators)
        ↓
Hidden Markov Model (unsupervised)
        ↓
Regime Labels: Bull / Bear / Sideways
        ↓
XGBoost Classifier (supervised)
        ↓
Real-time Regime Prediction
```

---

## Key Results

| Metric | Value |
|--------|-------|
| HMM States | 3 (Bull / Bear / Sideways) |
| Regime persistence | >92% probability of staying in the same state |
| Classifier (XGBoost) Accuracy | **85.5%** |
| Macro F1 Score | **0.817** |
| Bear detection (Recall) | 77% |
| Max Drawdown reduction vs Buy & Hold | **-40.2% vs -49.7%** |

The classifier was trained on 2018–2024 data and evaluated on 2024–2026 (a historically strong bull run), demonstrating generalization across different market conditions.

---

## Project Structure

```
notebooks/
  01_data_acquisition.ipynb          Data download and feature engineering
  02_eda_feature_engineering.ipynb   EDA and feature standardization
  03_hmm_regime_detection.ipynb      HMM training and regime labeling
  04_supervised_classification.ipynb Classifier training and backtesting

data/raw/
  btc_usd_raw.csv                    Raw OHLCV data from Yahoo Finance
  btc_features_clean.csv             Engineered features (unscaled)
  btc_features_scaled.csv            Standardized features (HMM input)
  btc_regimes.csv                    Price data + HMM regime labels
  hmm_model.pkl                      Trained HMM model
  rf_model.pkl                       Trained Random Forest
  xgb_model.pkl                      Trained XGBoost (final model)
  label_encoder.pkl                  Label encoder for regime classes
```

---

## Features Used

| Feature | Description |
|---------|-------------|
| `log_returns` | Daily logarithmic returns |
| `volatility_7d` | 7-day rolling volatility |
| `volatility_21d` | 21-day rolling volatility |
| `rsi` | Relative Strength Index (14-day) |
| `macd_diff` | MACD histogram (momentum acceleration) |
| `volume_norm` | Volume relative to 21-day average |

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/Sebasduarte14/crypto-market-regime-detection
cd crypto-market-regime-detection
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run notebooks in order**
```
01 → 02 → 03 → 04
```

---

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Data | `pandas`, `numpy`, `yfinance` |
| Features | `ta` (technical analysis) |
| Modeling | `hmmlearn`, `scikit-learn`, `xgboost` |
| Visualization | `matplotlib`, `seaborn` |

---

## Limitations & Future Work

- The backtesting strategy (invest only in Bull regimes) does not outperform buy & hold in the test period (2024–2026 bull run). The model's value lies in **regime identification**, not direct trading signals.
- Future improvements: walk-forward validation, additional features (Fear & Greed Index, BTC dominance), LSTM-based sequence modeling.

---

*Data Science Bootcamp Final Project*
