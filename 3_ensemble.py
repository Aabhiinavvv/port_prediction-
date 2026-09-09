"""
Module 1c — Ensemble combiner
Combines ARIMA + XGBoost + LSTM predictions using inverse-RMSE weighting.

LSTM predictions come from lstm_colab.py (run in Google Colab, since torch
isn't installable in this sandbox — disk space). After running it in Colab,
download lstm_predictions.csv and drop it in this same folder before
running this script. If it's not found, the script falls back to a
2-model ARIMA+XGBoost ensemble (same as before) so nothing breaks.
"""
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error

arima = pd.read_csv("step1_arima_predictions.csv", parse_dates=["date"])
xgbp = pd.read_csv("step2_xgb_predictions.csv", parse_dates=["date"])
merged = arima.merge(xgbp[["date", "xgb_pred"]], on="date")

has_lstm = os.path.exists("lstm_predictions.csv")
if has_lstm:
    lstm = pd.read_csv("lstm_predictions.csv")
    # lstm_colab.py doesn't carry dates through (Colab session is separate),
    # so we align by position on the last N rows - both scripts use the same
    # TEST_DAYS window on the same route, so row order matches.
    if len(lstm) == len(merged):
        merged["lstm_pred"] = lstm["lstm_pred"].values
    else:
        print(f"WARNING: lstm_predictions.csv has {len(lstm)} rows, expected {len(merged)}. Skipping LSTM.")
        has_lstm = False

# ---- inverse-RMSE weighting across whichever models are available ----
model_cols = ["arima_pred", "xgb_pred"] + (["lstm_pred"] if has_lstm else [])
rmses = {col: np.sqrt(mean_squared_error(merged.actual, merged[col])) for col in model_cols}
inv = {col: 1 / r for col, r in rmses.items()}
total_inv = sum(inv.values())
weights = {col: v / total_inv for col, v in inv.items()}

print("Models used:", model_cols)
print("Individual RMSE:", {k: round(v, 3) for k, v in rmses.items()})
print("Weights:", {k: round(v, 3) for k, v in weights.items()})

merged["ensemble_pred"] = sum(weights[col] * merged[col] for col in model_cols)

rmse_ens = np.sqrt(mean_squared_error(merged.actual, merged.ensemble_pred))
mape_ens = np.mean(np.abs((merged.actual - merged.ensemble_pred) / merged.actual)) * 100
print(f"\nEnsemble RMSE: {rmse_ens:.3f}   MAPE: {mape_ens:.2f}%")

merged.to_csv("step3_ensemble_predictions.csv", index=False)
