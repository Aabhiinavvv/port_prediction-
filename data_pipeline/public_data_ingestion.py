"""Free public-data ingestion for the freight intelligence prototype.

Sources:
- Open-Meteo historical weather API
- Yahoo Finance public chart endpoint for indicative BDI series, with Stooq fallback
- World Bank commodity workbook (optional local input)

No paid/licensed feed is required by this module. BDI is explicitly treated as
an indicative public series, not as an official Baltic Exchange redistribution.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from io import StringIO
import time
import requests
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "public_raw"
PROCESSED = ROOT / "data" / "public_processed"
RAW.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

PORTS = {
    "INPAR": (20.27, 86.67), "INVIZ": (17.69, 83.29), "INHAL": (22.04, 88.06),
    "INKOL": (22.57, 88.35), "INCHE": (13.08, 80.27), "INKAM": (13.26, 80.34),
    "INTUT": (8.76, 78.14), "INNMA": (12.91, 74.86), "INMOR": (15.42, 73.80),
}


def fetch_weather(start: str, end: str) -> pd.DataFrame:
    rows = []
    url = "https://archive-api.open-meteo.com/v1/archive"
    for port_id, (lat, lon) in PORTS.items():
        params = {
            "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
            "daily": "temperature_2m_mean,precipitation_sum,wind_speed_10m_max,wind_gusts_10m_max,weather_code",
            "timezone": "UTC", "wind_speed_unit": "kn", "precipitation_unit": "mm",
        }
        r = requests.get(url, params=params, timeout=60, headers={"User-Agent":"freight-intelligence/1.0"})
        r.raise_for_status()
        d = pd.DataFrame(r.json()["daily"])
        d["port_id"] = port_id
        d["source"] = "Open-Meteo Historical Weather API"
        d["synthetic_flag"] = 0
        rows.append(d)
        time.sleep(.2)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(RAW / "weather_open_meteo.csv", index=False)
    return out


def fetch_bdi(start: str, end: str) -> pd.DataFrame:
    p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    p2 = int((pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)).timestamp())
    yahoo = f"https://query1.finance.yahoo.com/v8/finance/chart/%5EBDIY?period1={p1}&period2={p2}&interval=1d&events=history"
    try:
        r = requests.get(yahoo, headers={"User-Agent":"Mozilla/5.0"}, timeout=30)
        r.raise_for_status()
        j = r.json()["chart"]["result"][0]
        q = j["indicators"]["quote"][0]
        out = pd.DataFrame({"date": pd.to_datetime(j["timestamp"], unit="s", utc=True).date, "bdi": q["close"]})
        out["source"] = "Yahoo Finance public chart"
    except Exception:
        url = "https://stooq.com/q/d/l/?s=%5Ebdi&d1=" + pd.Timestamp(start).strftime("%Y%m%d") + "&d2=" + pd.Timestamp(end).strftime("%Y%m%d") + "&i=d"
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=30)
        r.raise_for_status()
        out = pd.read_csv(StringIO(r.text)).rename(columns={"Date":"date","Close":"bdi"})[["date","bdi"]]
        out["source"] = "Stooq public CSV"
    out = out.dropna(subset=["bdi"])
    out["date"] = pd.to_datetime(out["date"])
    out["synthetic_flag"] = 0
    out["source_type"] = "public_indicative"
    out.to_csv(RAW / "bdi_public.csv", index=False)
    return out


def build_market_features() -> pd.DataFrame:
    df = pd.read_csv(RAW / "bdi_public.csv", parse_dates=["date"]).sort_values("date")
    for w in (7, 30, 90):
        df[f"bdi_ma{w}"] = df["bdi"].rolling(w, min_periods=w).mean()
    df["bdi_return_7d"] = df["bdi"].pct_change(7)
    df["bdi_return_30d"] = df["bdi"].pct_change(30)
    df["bdi_volatility_30d"] = df["bdi"].pct_change().rolling(30).std()
    df["market_regime"] = np.select(
        [df.bdi_return_30d > .12, df.bdi_return_30d < -.12, df.bdi_volatility_30d > .035],
        ["RISING", "FALLING", "HIGH_VOLATILITY"], default="STABLE")
    df["observed_or_derived"] = "observed+derived"
    df["synthetic_flag"] = 0
    df.to_csv(PROCESSED / "market_public_features.csv", index=False)
    return df


def build_weather_features() -> pd.DataFrame:
    df = pd.read_csv(RAW / "weather_open_meteo.csv", parse_dates=["time"]).rename(columns={"time":"date"})
    wind = df["wind_speed_10m_max"].fillna(0)
    gust = df["wind_gusts_10m_max"].fillna(wind)
    rain = df["precipitation_sum"].fillna(0)
    df["wind_risk"] = np.clip((wind - 20) / 25, 0, 1)
    df["gust_risk"] = np.clip((gust - 30) / 35, 0, 1)
    df["rain_risk"] = np.clip(rain / 80, 0, 1)
    df["weather_disruption_score"] = .50*df.wind_risk + .30*df.gust_risk + .20*df.rain_risk
    df["extreme_weather_flag"] = (df.weather_disruption_score >= .70).astype(int)
    df["observed_or_derived"] = "observed+derived"
    df["synthetic_flag"] = 0
    df.to_csv(PROCESSED / "weather_public_features.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2025-12-31")
    args = ap.parse_args()
    w = fetch_weather(args.start, args.end)
    b = fetch_bdi(args.start, args.end)
    build_weather_features(); build_market_features()
    print(f"Weather: {len(w):,} rows | BDI: {len(b):,} rows")
    print(f"Output: {PROCESSED}")


if __name__ == "__main__":
    main()
