"""Retrain the freight model using observed public BDI inputs.

The freight-rate target in the current prototype remains the existing benchmark
series. This script therefore measures whether real public market information
improves the model without pretending the benchmark target is observed fixture
pricing.

Usage:
    python data_pipeline/public_data_ingestion.py --start 2019-01-01 --end 2025-12-31
    python data_pipeline/train_public_xgb.py

Outputs:
    data/public_processed/public_xgb_predictions.csv
    data/public_processed/public_xgb_metrics.csv
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TARGET = "freight_rate_usd_per_tonne"


def add_public_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["route_id", "vessel_type", "date"]).copy()
    # Actual public BDI is merged by date before these features are constructed.
    for lag in (1, 7, 14, 30, 90):
        out[f"lag_{lag}"] = out.groupby(["route_id", "vessel_type"])[TARGET].shift(lag)
    g = out.groupby(["route_id", "vessel_type"])[TARGET]
    out["roll_mean_7"] = g.transform(lambda s: s.shift(1).rolling(7).mean())
    out["roll_mean_30"] = g.transform(lambda s: s.shift(1).rolling(30).mean())
    out["roll_std_30"] = g.transform(lambda s: s.shift(1).rolling(30).std())
    out["dow"] = out.date.dt.dayofweek
    out["month"] = out.date.dt.month
    out["bdi_ma7"] = out["bdi_index"].rolling(7, min_periods=7).mean()
    out["bdi_ma30"] = out["bdi_index"].rolling(30, min_periods=30).mean()
    out["bdi_return_7d"] = out["bdi_index"].pct_change(7)
    out["bdi_return_30d"] = out["bdi_index"].pct_change(30)
    out["bdi_volatility_30d"] = out["bdi_index"].pct_change().rolling(30).std()
    return out


def metrics(y, p):
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    mape = np.mean(np.abs((y - p) / np.maximum(np.abs(y), 1e-6))) * 100
    return {
        "MAE": mean_absolute_error(y, p),
        "RMSE": mean_squared_error(y, p) ** 0.5,
        "MAPE_pct": mape,
    }


def main():
    freight_path = DATA / "freight_rates_timeseries.csv"
    bdi_path = DATA / "public_raw" / "bdi_public.csv"
    if not freight_path.exists():
        raise FileNotFoundError(f"Missing {freight_path}")
    if not bdi_path.exists():
        raise FileNotFoundError(
            "Missing public BDI data. Run: python data_pipeline/public_data_ingestion.py"
        )

    freight = pd.read_csv(freight_path, parse_dates=["date"])
    bdi = pd.read_csv(bdi_path, parse_dates=["date"])[["date", "bdi"]]
    bdi = bdi.rename(columns={"bdi": "bdi_index"}).drop_duplicates("date")

    df = freight.merge(bdi, on="date", how="left", validate="many_to_one")
    coverage = df["bdi_index"].notna().mean()
    if coverage < .90:
        raise RuntimeError(f"Only {coverage:.1%} of freight rows have public BDI coverage.")

    df = add_public_features(df)
    features = [
        "lag_1", "lag_7", "lag_14", "lag_30", "lag_90",
        "roll_mean_7", "roll_mean_30", "roll_std_30",
        "bdi_index", "bdi_ma7", "bdi_ma30", "bdi_return_7d",
        "bdi_return_30d", "bdi_volatility_30d",
        "port_congestion_index", "dow", "month",
    ]
    df = df.dropna(subset=features + [TARGET]).reset_index(drop=True)

    # Chronological holdout: never train on future observations.
    split_date = df.date.quantile(.80)
    train = df[df.date <= split_date]
    test = df[df.date > split_date]

    model = xgb.XGBRegressor(
        n_estimators=350, max_depth=5, learning_rate=.05,
        subsample=.8, colsample_bytree=.8, random_state=42,
        objective="reg:squarederror", verbosity=0,
    )
    model.fit(train[features], train[TARGET])
    pred = model.predict(test[features])

    result = test[["date", "route_id", "vessel_type", TARGET]].copy()
    result["prediction"] = pred
    result["absolute_error"] = abs(result[TARGET] - result.prediction)
    result["source_note"] = "Observed public BDI input; existing prototype benchmark target"

    metric_rows = [{"model":"public_bdi_xgb", "split_date":str(split_date.date()), **metrics(test[TARGET], pred)}]
    metrics_df = pd.DataFrame(metric_rows)

    outdir = DATA / "public_processed"; outdir.mkdir(parents=True, exist_ok=True)
    result.to_csv(outdir / "public_xgb_predictions.csv", index=False)
    metrics_df.to_csv(outdir / "public_xgb_metrics.csv", index=False)
    (outdir / "public_xgb_feature_list.json").write_text(json.dumps({"features":features}, indent=2))

    print("Public-input XGBoost training complete")
    print(f"BDI coverage: {coverage:.2%}")
    print(metrics_df.to_string(index=False))
    print(f"Predictions -> {outdir / 'public_xgb_predictions.csv'}")


if __name__ == "__main__":
    main()
