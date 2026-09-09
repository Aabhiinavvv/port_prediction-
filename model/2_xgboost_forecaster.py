"""
Module 1b — XGBoost forecaster
Uses lagged freight_rate + exogenous features (bdi, coal, congestion, calendar).
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from pathlib import Path

ts = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "freight_rates_timeseries.csv", parse_dates=["date"])
ROUTE, VESSEL, TEST_DAYS = "AUNTL_INPAR", "Handysize", 90

df = ts[(ts.route_id == ROUTE) & (ts.vessel_type == VESSEL)].sort_values("date").reset_index(drop=True)

# ---- feature engineering ----
def add_features(d):
    d = d.copy()
    for lag in [1, 7, 14, 30, 90]:
        d[f"lag_{lag}"] = d["freight_rate_usd_per_tonne"].shift(lag)
    d["roll_mean_7"] = d["freight_rate_usd_per_tonne"].shift(1).rolling(7).mean()
    d["roll_mean_30"] = d["freight_rate_usd_per_tonne"].shift(1).rolling(30).mean()
    d["roll_std_30"] = d["freight_rate_usd_per_tonne"].shift(1).rolling(30).std()
    d["dow"] = pd.to_datetime(d["date"]).dt.dayofweek
    d["month"] = pd.to_datetime(d["date"]).dt.month
    return d

df = add_features(df).dropna().reset_index(drop=True)

FEATURES = ["lag_1","lag_7","lag_14","lag_30","lag_90","roll_mean_7","roll_mean_30",
            "roll_std_30","bdi_index","coal_price_index","port_congestion_index","dow","month"]
TARGET = "freight_rate_usd_per_tonne"

train, test = df.iloc[:-TEST_DAYS], df.iloc[-TEST_DAYS:]

model = xgb.XGBRegressor(
    n_estimators=400, max_depth=5, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)
model.fit(train[FEATURES], train[TARGET])
xgb_pred = model.predict(test[FEATURES])

rmse = np.sqrt(mean_squared_error(test[TARGET], xgb_pred))
mape = np.mean(np.abs((test[TARGET].values - xgb_pred) / test[TARGET].values)) * 100
print(f"XGBoost -> RMSE: {rmse:.3f}   MAPE: {mape:.2f}%")

# feature importance (good talking point for panel: "which factors drive freight rate")
imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nTop feature importances:")
print(imp.head(6).to_string())

pd.DataFrame({"date": test["date"].values, "actual": test[TARGET].values, "xgb_pred": xgb_pred}) \
  .to_csv("step2_xgb_predictions.csv", index=False)
