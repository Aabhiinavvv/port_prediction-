"""
Module 9 — Production Forecaster (FINAL)
=========================================
Decision: XGBoost only. Walk-forward validation (8_rigorous_validation.py)
showed a fixed-order ARIMA is unstable across time windows (MAPE ranged
3.45% to 17.62% depending on the window) and DEGRADED the ensemble instead
of helping it. XGBoost was stable across all 5 folds (mean 2.83%, std 1.85%).
Keep ARIMA only as a comparison baseline in diagnostics, not in production.

Adds: 80% prediction interval via XGBoost quantile regression (10th/90th
percentile models), not just a point estimate - so the dashboard and the
vessel-cost optimizer can show a realistic range instead of false precision.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from pathlib import Path

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

def train_and_predict_with_interval(df, features=FEATURES, target=TARGET, test_days=90):
    """Trains 3 XGBoost models (point/p10/p90) and returns predictions + interval on the test window."""
    df = add_features(df).dropna().reset_index(drop=True)
    train, test = df.iloc[:-test_days], df.iloc[-test_days:]

    point_model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    point_model.fit(train[features], train[target])

    p10_model = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.10,
                                  n_estimators=300, max_depth=5, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    p10_model.fit(train[features], train[target])

    p90_model = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.90,
                                  n_estimators=300, max_depth=5, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    p90_model.fit(train[features], train[target])

    result = test[["date", target]].copy()
    result["point_pred"] = point_model.predict(test[features])
    result["p10"] = p10_model.predict(test[features])
    result["p90"] = p90_model.predict(test[features])
    # guard against quantile crossing (rare but possible with independently trained models)
    result["p10"], result["p90"] = np.minimum(result.p10, result.p90), np.maximum(result.p10, result.p90)

    coverage = ((result[target] >= result.p10) & (result[target] <= result.p90)).mean()
    return result, point_model, p10_model, p90_model, coverage


if __name__ == "__main__":
    ts = pd.read_csv(Path(__file__).resolve().parent / "data" / "freight_rates_timeseries.csv", parse_dates=["date"])
    ROUTE, VESSEL = "AUNTL_INPAR", "Handysize"
    df = ts[(ts.route_id == ROUTE) & (ts.vessel_type == VESSEL)].sort_values("date").reset_index(drop=True)

    result, point_model, p10_model, p90_model, coverage = train_and_predict_with_interval(df)

    mape = np.mean(np.abs((result[TARGET] - result.point_pred) / result[TARGET])) * 100
    print(f"Point forecast MAPE: {mape:.2f}%")
    print(f"80% interval empirical coverage: {coverage*100:.1f}%  (target: 80% - if far off, the quantile models need recalibration)")
    print(f"\nAverage interval width: ${(result.p90 - result.p10).mean():.2f}/tonne")
    print("\nSample predictions:")
    print(result[["date", TARGET, "p10", "point_pred", "p90"]].head(8).to_string(index=False))

    result.to_csv("step9_production_forecast_with_interval.csv", index=False)
