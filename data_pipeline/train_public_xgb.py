"""Train the prototype XGBoost model with public exogenous signals.

The freight target remains the repository's prototype benchmark target. Public
BDI/weather/commodity variables are observed or derived public inputs; they do
not turn the target into proprietary fixture-rate observations.
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


def metrics(y, p):
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    return {
        "MAE": mean_absolute_error(y, p),
        "RMSE": mean_squared_error(y, p) ** 0.5,
        "MAPE_pct": np.mean(np.abs((y-p) / np.maximum(np.abs(y), 1e-6))) * 100,
    }


def add_target_history(df):
    out = df.sort_values(["route_id", "vessel_type", "date"]).copy()
    grp = out.groupby(["route_id", "vessel_type"])[TARGET]
    for lag in (1, 7, 14, 30, 90):
        out[f"lag_{lag}"] = grp.shift(lag)
    out["roll_mean_7"] = grp.transform(lambda s: s.shift(1).rolling(7).mean())
    out["roll_mean_30"] = grp.transform(lambda s: s.shift(1).rolling(30).mean())
    out["roll_std_30"] = grp.transform(lambda s: s.shift(1).rolling(30).std())
    out["dow"] = out.date.dt.dayofweek
    out["month"] = out.date.dt.month
    return out


def main():
    freight_path = DATA / "freight_rates_timeseries.csv"
    public_path = DATA / "public_processed" / "public_exogenous_features.csv"
    if not freight_path.exists():
        raise FileNotFoundError(f"Missing {freight_path}")
    if not public_path.exists():
        raise FileNotFoundError("Run public_data_ingestion.py and public_market_adapter.py first.")

    freight = pd.read_csv(freight_path, parse_dates=["date"])
    public = pd.read_csv(public_path, parse_dates=["date"])
    public = public.drop_duplicates("date")
    df = freight.merge(public, on="date", how="left", validate="many_to_one")
    df = add_target_history(df)

    public_features = [
        "bdi", "bdi_ma7", "bdi_ma30", "bdi_ma90", "bdi_return_7d",
        "bdi_return_30d", "bdi_volatility_30d",
        "weather_disruption_mean", "weather_disruption_max",
        "extreme_weather_port_count", "precipitation_mean_mm",
        "wind_max_kn", "gust_max_kn",
        "coal_australian", "coal_south_african", "iron_ore", "crude_oil",
    ]
    base_features = [
        "lag_1", "lag_7", "lag_14", "lag_30", "lag_90",
        "roll_mean_7", "roll_mean_30", "roll_std_30",
        "port_congestion_index", "dow", "month",
    ]
    features = base_features + [c for c in public_features if c in df.columns]
    # Public commodity columns are optional; only require signals with coverage.
    df = df.dropna(subset=base_features + ["bdi"] + [TARGET]).reset_index(drop=True)
    split_date = df.date.quantile(.80)
    train, test = df[df.date <= split_date], df[df.date > split_date]

    model = xgb.XGBRegressor(
        n_estimators=350, max_depth=5, learning_rate=.05,
        subsample=.8, colsample_bytree=.8, random_state=42,
        objective="reg:squarederror", verbosity=0,
    )
    model.fit(train[features], train[TARGET])
    pred = model.predict(test[features])

    outdir = DATA / "public_processed"; outdir.mkdir(parents=True, exist_ok=True)
    predictions = test[["date", "route_id", "vessel_type", TARGET]].copy()
    predictions["prediction"] = pred
    predictions["absolute_error"] = abs(predictions[TARGET] - predictions.prediction)
    predictions["target_provenance"] = "prototype_synthetic_benchmark"
    predictions["input_provenance"] = "public_bdi_weather_plus_optional_world_bank"
    predictions.to_csv(outdir / "public_xgb_predictions.csv", index=False)

    metrics_df = pd.DataFrame([{
        "model":"public_exogenous_xgb",
        "split_date":str(split_date.date()),
        "feature_count":len(features),
        "commodity_features_loaded":int(any(c in features for c in ["coal_australian","coal_south_african","iron_ore","crude_oil"])),
        **metrics(test[TARGET], pred),
    }])
    metrics_df.to_csv(outdir / "public_xgb_metrics.csv", index=False)
    (outdir / "public_xgb_feature_list.json").write_text(json.dumps({"features":features}, indent=2))
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
