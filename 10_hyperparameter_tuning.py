"""
Module 10 — Hyperparameter Optimization
=========================================
Uses RandomizedSearchCV with TimeSeriesSplit (NOT regular KFold - regular
KFold would shuffle time order into train/val, leaking future info into
training, exactly the kind of mistake we caught earlier with the ensemble).

TimeSeriesSplit always trains on the past and validates on the future,
consistent with how the model will actually be used.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

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

ts = pd.read_csv(Path(__file__).resolve().parent / "data" / "freight_rates_timeseries.csv", parse_dates=["date"])
ROUTE, VESSEL, TEST_DAYS = "USHRV_INGAN", "Capesize", 90  # the worst-performing combo from module 7 (10.50% MAPE)

df = ts[(ts.route_id == ROUTE) & (ts.vessel_type == VESSEL)].sort_values("date").reset_index(drop=True)
df = add_features(df).dropna().reset_index(drop=True)
train, test = df.iloc[:-TEST_DAYS], df.iloc[-TEST_DAYS:]

# ---- BEFORE: default-ish params (what we've been using all along) ----
default_model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
default_model.fit(train[FEATURES], train[TARGET])
default_pred = default_model.predict(test[FEATURES])
default_mape = np.mean(np.abs((test[TARGET].values - default_pred) / test[TARGET].values)) * 100
print(f"BEFORE tuning -> MAPE: {default_mape:.2f}%")

# ---- Hyperparameter search space ----
param_dist = {
    "n_estimators": [100, 200, 300, 500, 800],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "reg_alpha": [0, 0.01, 0.1, 1],
    "reg_lambda": [0.5, 1, 1.5, 2],
}

tscv = TimeSeriesSplit(n_splits=5, test_size=90)  # 5 forward-chaining folds, always past->future

search = RandomizedSearchCV(
    estimator=xgb.XGBRegressor(random_state=42, verbosity=0),
    param_distributions=param_dist,
    n_iter=40,                          # 40 random combinations tried
    scoring="neg_root_mean_squared_error",
    cv=tscv,
    random_state=42,
    n_jobs=-1,
)
search.fit(train[FEATURES], train[TARGET])

print(f"\nBest params found: {search.best_params_}")

# ---- AFTER: tuned params, evaluated on the SAME held-out test set ----
tuned_model = xgb.XGBRegressor(**search.best_params_, random_state=42, verbosity=0)
tuned_model.fit(train[FEATURES], train[TARGET])
tuned_pred = tuned_model.predict(test[FEATURES])
tuned_mape = np.mean(np.abs((test[TARGET].values - tuned_pred) / test[TARGET].values)) * 100
print(f"\nAFTER tuning  -> MAPE: {tuned_mape:.2f}%")
print(f"Improvement: {default_mape - tuned_mape:+.2f} percentage points")

pd.Series(search.best_params_).to_json("best_xgb_params.json")
print("\nSaved best_xgb_params.json - reuse these across all 132 route/vessel models")
