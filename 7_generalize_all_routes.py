"""
Module 1d — Generalized forecasting across ALL route x vessel_type combos
Uses XGBoost only (it was the best single model in the pilot test on
AUNTL_INPAR/Handysize, and it's far faster to loop 132 times than SARIMAX).
Produces a summary table you can show the panel: "we trained and evaluated
132 independent models, here's the aggregate accuracy."
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

ts = pd.read_csv(Path(__file__).resolve().parent / "data" / "freight_rates_timeseries.csv", parse_dates=["date"])
TEST_DAYS = 90
FEATURES = ["lag_1","lag_7","lag_14","lag_30","lag_90","roll_mean_7","roll_mean_30",
            "roll_std_30","bdi_index","coal_price_index","port_congestion_index","dow","month"]
TARGET = "freight_rate_usd_per_tonne"

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

results = []
combos = ts[["route_id", "vessel_type"]].drop_duplicates().values.tolist()
print(f"Training {len(combos)} independent models (one per route x vessel_type)...\n")

for route_id, vessel_type in combos:
    df = ts[(ts.route_id == route_id) & (ts.vessel_type == vessel_type)].sort_values("date").reset_index(drop=True)
    df = add_features(df).dropna().reset_index(drop=True)
    if len(df) < TEST_DAYS + 200:
        continue

    train, test = df.iloc[:-TEST_DAYS], df.iloc[-TEST_DAYS:]
    model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    model.fit(train[FEATURES], train[TARGET])
    pred = model.predict(test[FEATURES])

    rmse = np.sqrt(mean_squared_error(test[TARGET], pred))
    mape = np.mean(np.abs((test[TARGET].values - pred) / test[TARGET].values)) * 100
    nrmse = (rmse / test[TARGET].mean()) * 100  # RMSE normalized by mean rate - fairer across vessel types than raw RMSE or MAPE alone
    results.append({"route_id": route_id, "vessel_type": vessel_type, "rmse": round(rmse, 3),
                     "mape_pct": round(mape, 2), "mean_rate": round(test[TARGET].mean(), 2), "nrmse_pct": round(nrmse, 2)})

summary = pd.DataFrame(results)
summary.to_csv("all_routes_forecast_summary.csv", index=False)

print(f"Trained {len(summary)} models successfully.\n")
print("Overall accuracy:")
print(f"  Mean MAPE : {summary.mape_pct.mean():.2f}%")
print(f"  Median MAPE: {summary.mape_pct.median():.2f}%")
print(f"  Worst MAPE: {summary.mape_pct.max():.2f}%  (route: {summary.loc[summary.mape_pct.idxmax(), 'route_id']}, {summary.loc[summary.mape_pct.idxmax(), 'vessel_type']})")
print(f"  Best MAPE : {summary.mape_pct.min():.2f}%  (route: {summary.loc[summary.mape_pct.idxmin(), 'route_id']}, {summary.loc[summary.mape_pct.idxmin(), 'vessel_type']})")

print("\nMean MAPE by vessel type:")
print(summary.groupby("vessel_type")["mape_pct"].mean().round(2).to_string())

print("\nMean normalized-RMSE (RMSE/mean_rate) by vessel type - fairer comparison since it's on the same scale as MAPE but derived from RMSE:")
print(summary.groupby("vessel_type")["nrmse_pct"].mean().round(2).to_string())

print("\nMAPE vs nRMSE side by side - if they tell different stories for a vessel type, MAPE is being distorted by a low base rate:")
print(summary.groupby("vessel_type")[["mape_pct", "nrmse_pct"]].mean().round(2).to_string())
