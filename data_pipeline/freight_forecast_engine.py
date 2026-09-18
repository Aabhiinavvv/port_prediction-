"""Production-style freight forecasting engine for the chartering decision layer.

Design:
- leakage-safe lag/rolling features
- direct multi-horizon XGBoost models (1/7/30/60 days)
- categorical route/vessel encoding
- chronological validation only
- conformal-style residual bands calibrated on a held-out validation window
- optional future exogenous scenarios
- machine-readable forecast output for Streamlit / optimization

Important: the repository's current freight target is a prototype benchmark. This
engine does not turn it into observed fixture-rate data.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TARGET = "freight_rate_usd_per_tonne"

HORIZONS = (1, 7, 30, 60)
LAGS = (1, 7, 14, 30, 90)
BASE_NUMERIC = (
    "lag_1", "lag_7", "lag_14", "lag_30", "lag_90",
    "roll_mean_7", "roll_mean_30", "roll_std_30",
    "port_congestion_index", "dow", "month",
)
PUBLIC_NUMERIC = (
    "bdi", "bdi_ma7", "bdi_ma30", "bdi_ma90",
    "bdi_return_7d", "bdi_return_30d", "bdi_volatility_30d",
    "weather_disruption_mean", "weather_disruption_max",
    "extreme_weather_port_count", "precipitation_mean_mm",
    "wind_max_kn", "gust_max_kn",
    "coal_australian", "coal_south_african", "iron_ore", "crude_oil",
)
CATEGORICAL = ("route_id", "vessel_type")


@dataclass
class HorizonModel:
    horizon: int
    model: object
    transformer: ColumnTransformer
    features: list[str]
    residual_abs_p90: float
    residual_abs_p80: float
    validation_mae: float
    validation_rmse: float
    validation_mape_pct: float
    train_end: str


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(mean_squared_error(y, p) ** 0.5),
        "mape_pct": float(
            np.mean(np.abs((y - p) / np.maximum(np.abs(y), 1e-6))) * 100
        ),
    }


def add_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["route_id", "vessel_type", "date"]).copy()
    grp = out.groupby(["route_id", "vessel_type"], sort=False)[TARGET]
    for lag in LAGS:
        out[f"lag_{lag}"] = grp.shift(lag)
    out["roll_mean_7"] = grp.transform(lambda s: s.shift(1).rolling(7).mean())
    out["roll_mean_30"] = grp.transform(lambda s: s.shift(1).rolling(30).mean())
    out["roll_std_30"] = grp.transform(lambda s: s.shift(1).rolling(30).std())
    out["dow"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month
    return out


def add_direct_targets(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = frame.copy()
    out["target_h"] = (
        out.groupby(["route_id", "vessel_type"], sort=False)[TARGET]
        .shift(-horizon)
    )
    return out


def _available_features(frame: pd.DataFrame) -> list[str]:
    return [
        c for c in BASE_NUMERIC + PUBLIC_NUMERIC + CATEGORICAL
        if c in frame.columns
    ]


class FreightForecastEngine:
    """Train and serve a leakage-safe global direct multi-horizon forecaster."""

    def __init__(
        self,
        horizons: Iterable[int] = HORIZONS,
        validation_days: int = 90,
        min_train_rows: int = 5000,
        random_state: int = 42,
    ):
        self.horizons = tuple(int(h) for h in horizons)
        self.validation_days = int(validation_days)
        self.min_train_rows = int(min_train_rows)
        self.random_state = int(random_state)
        self.models: dict[int, HorizonModel] = {}
        self.feature_columns: list[str] = []
        self.provenance = {
            "target": "prototype_synthetic_benchmark",
            "forecast_method": "direct_multi_horizon_xgboost",
            "interval_method": "held_out_absolute_residual_quantiles",
        }

    @staticmethod
    def _transformer(features: list[str]) -> ColumnTransformer:
        cats = [c for c in CATEGORICAL if c in features]
        nums = [c for c in features if c not in cats]
        return ColumnTransformer(
            [
                ("num", "passthrough", nums),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
            ],
            remainder="drop",
        )

    def _fit_one(self, data: pd.DataFrame, horizon: int) -> HorizonModel:
        work = add_direct_targets(data, horizon)
        features = _available_features(work)
        required = features + ["target_h"]
        work = work.dropna(subset=required).copy()
        if len(work) < self.min_train_rows:
            raise ValueError(
                f"H{horizon}: only {len(work):,} usable rows; "
                f"need at least {self.min_train_rows:,}."
            )

        dates = np.sort(work["date"].unique())
        if len(dates) <= self.validation_days:
            raise ValueError(f"H{horizon}: not enough dates for validation.")

        val_start = pd.Timestamp(dates[-self.validation_days])
        train = work[work.date < val_start]
        valid = work[work.date >= val_start]
        if train.empty or valid.empty:
            raise ValueError(f"H{horizon}: invalid chronological split.")

        transformer = self._transformer(features)
        x_train = transformer.fit_transform(train[features])
        x_valid = transformer.transform(valid[features])

        model = xgb.XGBRegressor(
            n_estimators=450,
            max_depth=5,
            learning_rate=0.04,
            min_child_weight=5,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            objective="reg:squarederror",
            eval_metric="mae",
            random_state=self.random_state,
            tree_method="hist",
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(
            x_train,
            train["target_h"].to_numpy(),
            eval_set=[(x_valid, valid["target_h"].to_numpy())],
            verbose=False,
        )
        pred = model.predict(x_valid)
        m = _metrics(valid["target_h"].to_numpy(), pred)
        residual_abs = np.abs(valid["target_h"].to_numpy() - pred)

        return HorizonModel(
            horizon=horizon,
            model=model,
            transformer=transformer,
            features=features,
            residual_abs_p90=float(np.quantile(residual_abs, 0.90)),
            residual_abs_p80=float(np.quantile(residual_abs, 0.80)),
            validation_mae=m["mae"],
            validation_rmse=m["rmse"],
            validation_mape_pct=m["mape_pct"],
            train_end=str(train.date.max().date()),
        )

    def fit(self, data: pd.DataFrame) -> "FreightForecastEngine":
        required = {"date", "route_id", "vessel_type", TARGET}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        data = data.copy()
        data["date"] = pd.to_datetime(data["date"])
        data = data.sort_values(["route_id", "vessel_type", "date"])
        self.feature_columns = _available_features(add_history_features(data))
        self.models = {h: self._fit_one(data, h) for h in self.horizons}
        return self

    def _latest_row(
        self,
        data: pd.DataFrame,
        route_id: str,
        vessel_type: str,
        as_of: pd.Timestamp | None = None,
        future_exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        data = data.copy()
        data["date"] = pd.to_datetime(data["date"])
        if future_exog is not None and not future_exog.empty:
            future_exog = future_exog.copy()
            future_exog["date"] = pd.to_datetime(future_exog["date"])
            # Future exogenous values are merged only by date. Missing future
            # values are intentionally left as NaN and handled by XGBoost.
            data = data.merge(future_exog, on="date", how="left", suffixes=("", "_future"))
            for c in PUBLIC_NUMERIC:
                if f"{c}_future" in data.columns:
                    data[c] = data[f"{c}_future"].combine_first(data.get(c))
            data = data.drop(columns=[c for c in data.columns if c.endswith("_future")])
        sub = data[(data.route_id == route_id) & (data.vessel_type == vessel_type)].copy()
        if as_of is not None:
            sub = sub[sub.date <= pd.Timestamp(as_of)]
        if sub.empty:
            raise ValueError(f"No history for route={route_id}, vessel={vessel_type}.")
        sub = add_history_features(sub)
        return sub.tail(1)

    def forecast(
        self,
        data: pd.DataFrame,
        route_id: str,
        vessel_type: str,
        as_of: str | pd.Timestamp | None = None,
        future_exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if not self.models:
            raise RuntimeError("Call fit() before forecast().")
        as_of_ts = pd.Timestamp(as_of) if as_of is not None else None
        latest = self._latest_row(data, route_id, vessel_type, as_of_ts, future_exog)
        if latest.empty:
            raise ValueError("Could not construct a latest feature row.")

        rows = []
        for h in self.horizons:
            hm = self.models[h]
            x = hm.transformer.transform(latest[hm.features])
            point = float(hm.model.predict(x)[0])
            rows.append({
                "route_id": route_id,
                "vessel_type": vessel_type,
                "as_of": str(pd.Timestamp(latest.date.iloc[0]).date()),
                "horizon_days": h,
                "forecast_date": str((pd.Timestamp(latest.date.iloc[0]) + pd.Timedelta(days=h)).date()),
                "forecast_usd_per_tonne": max(0.0, point),
                "p10_usd_per_tonne": max(0.0, point - hm.residual_abs_p90),
                "p50_usd_per_tonne": max(0.0, point),
                "p90_usd_per_tonne": max(0.0, point + hm.residual_abs_p90),
                "interval_width_usd_per_tonne": 2 * hm.residual_abs_p90,
                "validation_mae": hm.validation_mae,
                "validation_rmse": hm.validation_rmse,
                "validation_mape_pct": hm.validation_mape_pct,
                "model_train_end": hm.train_end,
                "target_provenance": self.provenance["target"],
                "interval_provenance": self.provenance["interval_method"],
            })
        return pd.DataFrame(rows)

    def validation_summary(self) -> pd.DataFrame:
        rows = []
        for h, hm in sorted(self.models.items()):
            rows.append({
                "horizon_days": h,
                "MAE": hm.validation_mae,
                "RMSE": hm.validation_rmse,
                "MAPE_pct": hm.validation_mape_pct,
                "P80_abs_residual": hm.residual_abs_p80,
                "P90_abs_residual": hm.residual_abs_p90,
                "train_end": hm.train_end,
            })
        return pd.DataFrame(rows)

    def save_metadata(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "horizons": list(self.horizons),
            "validation_days": self.validation_days,
            "features": self.feature_columns,
            "provenance": self.provenance,
            "validation": self.validation_summary().to_dict(orient="records"),
        }
        path.write_text(json.dumps(payload, indent=2))


def load_training_data() -> pd.DataFrame:
    freight_path = DATA / "freight_rates_timeseries.csv"
    public_path = DATA / "public_processed" / "public_exogenous_features.csv"
    if not freight_path.exists():
        raise FileNotFoundError(f"Missing {freight_path}")
    freight = pd.read_csv(freight_path, parse_dates=["date"])
    if public_path.exists():
        public = pd.read_csv(public_path, parse_dates=["date"]).drop_duplicates("date")
        freight = freight.merge(public, on="date", how="left", validate="many_to_one")
    return freight


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the freight forecast engine.")
    parser.add_argument("--route", default=None, help="Optional route_id for a demo forecast.")
    parser.add_argument("--vessel", default=None, help="Optional vessel type for a demo forecast.")
    parser.add_argument("--as-of", default=None, help="Forecast cutoff date YYYY-MM-DD.")
    args = parser.parse_args()

    data = load_training_data()
    engine = FreightForecastEngine().fit(data)

    outdir = DATA / "public_processed"
    outdir.mkdir(parents=True, exist_ok=True)
    engine.validation_summary().to_csv(outdir / "freight_forecast_engine_validation.csv", index=False)
    engine.save_metadata(outdir / "freight_forecast_engine_metadata.json")

    print("\n=== Freight Forecast Engine validation ===")
    print(engine.validation_summary().round(4).to_string(index=False))

    if args.route and args.vessel:
        result = engine.forecast(data, args.route, args.vessel, args.as_of)
        result.to_csv(outdir / "freight_forecast_latest.csv", index=False)
        print("\n=== Forecast ===")
        print(result.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
