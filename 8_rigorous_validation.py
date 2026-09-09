"""
Module 8 — Rigorous validation
FIX 1: Ensemble weights computed on a VALIDATION window, evaluated on a
       separate held-out TEST window (previously both used the same window,
       which quietly inflates the reported ensemble accuracy).
FIX 2: Walk-forward cross-validation - instead of one lucky/unlucky 90-day
       test window, we roll the window backward 5 times and report the
       mean +/- std of MAPE. This is what actually tells you whether 3.48%
       MAPE is a reliable number or a fluke.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

ts = pd.read_csv(Path(__file__).resolve().parent / "data" / "freight_rates_timeseries.csv", parse_dates=["date"])
ROUTE, VESSEL = "AUNTL_INPAR", "Handysize"
TARGET = "freight_rate_usd_per_tonne"
FEATURES = ["lag_1","lag_7","lag_14","lag_30","lag_90","roll_mean_7","roll_mean_30",
            "roll_std_30","bdi_index","coal_price_index","port_congestion_index","dow","month"]

def add_features(d):
    d = d.copy()
    for lag in [1, 7, 14, 30, 90]:
        d[f"lag_{lag}"] = d[TARGET].shift(lag)
    d["roll_mean_7"] = d[TARGET].shift(1).rolling(7).mean()
    d["roll_mean_30"] = d[TARGET].shift(1).rolling(30).mean()
    d["roll_std_30"] = d[TARGET].shift(1).rolling(30).std()
    d["dow"] = pd.to_datetime(d["date"]).dt.dayofweek
    d["month"] = pd.to_datetime(d["date"]).dt.month
    return d

df = ts[(ts.route_id == ROUTE) & (ts.vessel_type == VESSEL)].sort_values("date").reset_index(drop=True)
df = add_features(df).dropna().reset_index(drop=True)

# ---------------------------------------------------------------------
# FIX 2: Walk-forward validation - 5 folds, each a 90-day test window,
# rolled back 90 days each time, all trained only on data BEFORE that window
# ---------------------------------------------------------------------
N_FOLDS = 5
TEST_DAYS = 90
fold_results = []

for fold in range(N_FOLDS):
    end_idx = len(df) - fold * TEST_DAYS
    start_test_idx = end_idx - TEST_DAYS
    if start_test_idx - 400 < 0:   # need enough history to train on
        break

    train = df.iloc[:start_test_idx]
    test = df.iloc[start_test_idx:end_idx]

    model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    model.fit(train[FEATURES], train[TARGET])
    pred = model.predict(test[FEATURES])

    mape = np.mean(np.abs((test[TARGET].values - pred) / test[TARGET].values)) * 100
    fold_results.append({
        "fold": fold + 1,
        "test_start": test["date"].min().date(),
        "test_end": test["date"].max().date(),
        "mape_pct": round(mape, 2),
    })

fold_df = pd.DataFrame(fold_results)
print("=== Walk-forward validation (XGBoost, 5 folds) ===")
print(fold_df.to_string(index=False))
print(f"\nMean MAPE : {fold_df.mape_pct.mean():.2f}%")
print(f"Std  MAPE : {fold_df.mape_pct.std():.2f}%   <- if this is large relative to the mean, the single-window number was NOT reliable")
print(f"Worst fold: {fold_df.mape_pct.max():.2f}%")

# ---------------------------------------------------------------------
# FIX 1: Proper 3-way split for ensemble weighting - train / validation
# (used ONLY to compute ensemble weights) / test (used ONLY to report
# final accuracy, never touched before that)
# ---------------------------------------------------------------------
VAL_DAYS, TEST_DAYS2 = 90, 90
train2 = df.iloc[:-(VAL_DAYS + TEST_DAYS2)]
val2 = df.iloc[-(VAL_DAYS + TEST_DAYS2):-TEST_DAYS2]
test2 = df.iloc[-TEST_DAYS2:]

# ARIMA on train2, predict over val2 and test2 horizon together then split
arima_model = SARIMAX(train2.set_index("date")[TARGET], order=(2,1,2), seasonal_order=(1,1,1,7),
                       enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
arima_full_fc = arima_model.forecast(steps=VAL_DAYS + TEST_DAYS2)
arima_val_pred = arima_full_fc.values[:VAL_DAYS]
arima_test_pred = arima_full_fc.values[VAL_DAYS:]

xgb_model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
xgb_model.fit(train2[FEATURES], train2[TARGET])
xgb_val_pred = xgb_model.predict(val2[FEATURES])
xgb_test_pred = xgb_model.predict(test2[FEATURES])

# weights computed ONLY on validation set
rmse_arima_val = np.sqrt(mean_squared_error(val2[TARGET], arima_val_pred))
rmse_xgb_val = np.sqrt(mean_squared_error(val2[TARGET], xgb_val_pred))
w_arima = (1/rmse_arima_val) / ((1/rmse_arima_val) + (1/rmse_xgb_val))
w_xgb = 1 - w_arima
print(f"\n=== Fixed ensemble (weights from VALIDATION set, evaluated on separate TEST set) ===")
print(f"Weights (from validation): ARIMA={w_arima:.3f}  XGBoost={w_xgb:.3f}")

# apply those weights to the UNSEEN test set
ensemble_test_pred = w_arima * arima_test_pred + w_xgb * xgb_test_pred
mape_arima_test = np.mean(np.abs((test2[TARGET].values - arima_test_pred) / test2[TARGET].values)) * 100
mape_xgb_test = np.mean(np.abs((test2[TARGET].values - xgb_test_pred) / test2[TARGET].values)) * 100
mape_ens_test = np.mean(np.abs((test2[TARGET].values - ensemble_test_pred) / test2[TARGET].values)) * 100

print(f"\nOn held-out TEST set (never used to pick weights):")
print(f"  ARIMA    MAPE: {mape_arima_test:.2f}%")
print(f"  XGBoost  MAPE: {mape_xgb_test:.2f}%")
print(f"  Ensemble MAPE: {mape_ens_test:.2f}%")
