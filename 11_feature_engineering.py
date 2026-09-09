"""
Module 11 — Feature Engineering Experiment
Tests whether additional features improve on the baseline feature set,
specifically on Capesize (our hardest-to-predict vessel type).
New features tried:
  - Cyclical (sin/cos) day-of-year encoding instead of raw month/dow
    (raw month treats Dec(12) and Jan(1) as far apart - cyclical fixes this)
  - Rate-of-change momentum (5-day and 14-day % change)
  - Distance-normalized rate (rate / distance_nm) - removes route-length
    effect so the model can focus purely on market-driven movement
  - Congestion momentum (is congestion rising or falling)
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

TARGET = "freight_rate_usd_per_tonne"
BASELINE_FEATURES = ["lag_1","lag_7","lag_14","lag_30","lag_90","roll_mean_7","roll_mean_30",
                      "roll_std_30","bdi_index","coal_price_index","port_congestion_index","dow","month"]

def add_baseline_features(d):
    d = d.copy()
    for lag in [1, 7, 14, 30, 90]:
        d[f"lag_{lag}"] = d[TARGET].shift(lag)
    d["roll_mean_7"] = d[TARGET].shift(1).rolling(7).mean()
    d["roll_mean_30"] = d[TARGET].shift(1).rolling(30).mean()
    d["roll_std_30"] = d[TARGET].shift(1).rolling(30).std()
    d["dow"] = pd.to_datetime(d["date"]).dt.dayofweek
    d["month"] = pd.to_datetime(d["date"]).dt.month
    return d

def add_new_features(d):
    d = d.copy()
    doy = pd.to_datetime(d["date"]).dt.dayofyear
    d["doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
    d["doy_cos"] = np.cos(2 * np.pi * doy / 365.0)
    d["momentum_5"] = d[TARGET].shift(1).pct_change(5)
    d["momentum_14"] = d[TARGET].shift(1).pct_change(14)
    d["rate_per_nm"] = d[TARGET].shift(1) / d["distance_nm"]
    d["congestion_momentum"] = d["port_congestion_index"].shift(1).diff(7)
    return d

ts = pd.read_csv(Path(__file__).resolve().parent / "data" / "freight_rates_timeseries.csv", parse_dates=["date"])
ROUTE, VESSEL, TEST_DAYS = "USHRV_INGAN", "Capesize", 90

df = ts[(ts.route_id == ROUTE) & (ts.vessel_type == VESSEL)].sort_values("date").reset_index(drop=True)
df = add_baseline_features(df)
df = add_new_features(df)
df = df.dropna().reset_index(drop=True)

NEW_FEATURES = ["doy_sin", "doy_cos", "momentum_5", "momentum_14", "rate_per_nm", "congestion_momentum"]

train, test = df.iloc[:-TEST_DAYS], df.iloc[-TEST_DAYS:]

def evaluate(feature_list, label):
    model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    model.fit(train[feature_list], train[TARGET])
    pred = model.predict(test[feature_list])
    mape = np.mean(np.abs((test[TARGET].values - pred) / test[TARGET].values)) * 100
    print(f"{label}: MAPE = {mape:.2f}%  ({len(feature_list)} features)")
    return mape, model

mape_baseline, _ = evaluate(BASELINE_FEATURES, "Baseline features only")
mape_all, model_all = evaluate(BASELINE_FEATURES + NEW_FEATURES, "Baseline + ALL new features")

# also test dropping raw month/dow in favor of cyclical encoding only
cyclical_swap = [f for f in BASELINE_FEATURES if f not in ["dow", "month"]] + ["doy_sin", "doy_cos"]
mape_cyclical, _ = evaluate(cyclical_swap, "Baseline with cyclical encoding swapped in")

print(f"\nImprovement (all new features): {mape_baseline - mape_all:+.2f} pp")
print(f"Improvement (cyclical swap only): {mape_baseline - mape_cyclical:+.2f} pp")

print("\nFeature importance with all features (top 8):")
imp = pd.Series(model_all.feature_importances_, index=BASELINE_FEATURES + NEW_FEATURES).sort_values(ascending=False)
print(imp.head(8).to_string())
