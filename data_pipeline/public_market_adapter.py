"""Build a date-level public market/weather feature table for ML joins.

Observed inputs:
- Public indicative BDI series produced by public_data_ingestion.py
- Open-Meteo historical weather at Indian port coordinates
- World Bank Pink Sheet commodity prices (optional downloaded CSV)

The adapter never overwrites the prototype freight target. It adds public
exogenous features and provenance columns so the model can be evaluated
without misrepresenting synthetic freight observations as fixture data.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "public_raw"
PROCESSED = DATA / "public_processed"


def load_bdi() -> pd.DataFrame:
    p = RAW / "bdi_public.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run public_data_ingestion.py first.")
    x = pd.read_csv(p, parse_dates=["date"])[["date", "bdi"]]
    x = x.drop_duplicates("date").sort_values("date")
    x["bdi_ma7"] = x.bdi.rolling(7, min_periods=7).mean()
    x["bdi_ma30"] = x.bdi.rolling(30, min_periods=30).mean()
    x["bdi_ma90"] = x.bdi.rolling(90, min_periods=90).mean()
    x["bdi_return_7d"] = x.bdi.pct_change(7)
    x["bdi_return_30d"] = x.bdi.pct_change(30)
    x["bdi_volatility_30d"] = x.bdi.pct_change().rolling(30).std()
    return x


def load_weather() -> pd.DataFrame:
    p = PROCESSED / "weather_public_features.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run public_data_ingestion.py first.")
    x = pd.read_csv(p, parse_dates=["date"])
    # Aggregate port-level weather into a date-level disruption envelope.
    return x.groupby("date", as_index=False).agg(
        weather_disruption_mean=("weather_disruption_score", "mean"),
        weather_disruption_max=("weather_disruption_score", "max"),
        extreme_weather_port_count=("extreme_weather_flag", "sum"),
        precipitation_mean_mm=("precipitation_sum", "mean"),
        wind_max_kn=("wind_speed_10m_max", "max"),
        gust_max_kn=("wind_gusts_10m_max", "max"),
    )


def load_world_bank() -> pd.DataFrame:
    """Read a locally downloaded World Bank Pink Sheet CSV if available.

    The public workbook layout changes occasionally, so parsing is deliberately
    conservative. No commodity column is fabricated if the file is absent.
    """
    candidates = [RAW / "world_bank_commodities.csv", RAW / "world_bank_pink_sheet.csv"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return pd.DataFrame(columns=["date"])
    x = pd.read_csv(path)
    date_col = next((c for c in x.columns if c.lower() in {"date", "month"}), None)
    if date_col is None:
        return pd.DataFrame(columns=["date"])
    x["date"] = pd.to_datetime(x[date_col], errors="coerce").dt.to_period("D").dt.to_timestamp()
    x = x.dropna(subset=["date"]).copy()
    # Keep recognized numeric commodity columns; prefix to make provenance clear.
    keep = ["date"]
    aliases = {
        "coal_australian": ["Coal, Australian"],
        "coal_south_african": ["Coal, South African"],
        "crude_oil": ["Crude oil, average"],
        "iron_ore": ["Iron ore, cfr spot"],
    }
    lower = {str(c).strip().lower(): c for c in x.columns}
    for out_name, names in aliases.items():
        found = next((lower.get(n.lower()) for n in names if n.lower() in lower), None)
        if found:
            x[out_name] = pd.to_numeric(x[found], errors="coerce")
            keep.append(out_name)
    return x[keep].groupby("date", as_index=False).mean(numeric_only=True)


def build(start: str, end: str) -> pd.DataFrame:
    bdi = load_bdi()
    weather = load_weather()
    wb = load_world_bank()
    dates = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    x = dates.merge(bdi, on="date", how="left").merge(weather, on="date", how="left").merge(wb, on="date", how="left")
    public_cols = [c for c in x.columns if c != "date"]
    x["public_observation_coverage"] = x[public_cols].notna().mean(axis=1)
    x["bdi_source"] = "public_indicative"
    x["weather_source"] = "Open-Meteo historical reanalysis"
    x["commodity_source"] = np.where(wb.shape[1] > 1, "World Bank Pink Sheet", "not_loaded")
    x["synthetic_flag"] = 0
    x["observed_or_derived"] = "observed+derived"
    PROCESSED.mkdir(parents=True, exist_ok=True)
    x.to_csv(PROCESSED / "public_exogenous_features.csv", index=False)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2025-12-31")
    args = ap.parse_args()
    x = build(args.start, args.end)
    print(f"Built {len(x):,} daily public feature rows")
    print("Columns:", ", ".join(x.columns))
    print("World Bank commodity data loaded:", any(c.startswith("coal_") or c in {"iron_ore", "crude_oil"} for c in x.columns))

if __name__ == "__main__":
    main()
