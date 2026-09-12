"""Small, fault-tolerant adapters for free public context data.

These sources are explanatory features, not substitutes for licensed dry-bulk
fixture or AIS data. All callers must preserve the returned source/status label.
"""
from __future__ import annotations

from datetime import datetime, timezone
import requests

PORT_COORDS = {
    "AUNTL": (-32.93, 151.78), "AUGLT": (-23.84, 151.26), "AUHPT": (-21.29, 149.30),
    "AUABT": (-19.88, 148.08), "USHRV": (36.95, -76.33), "USBLT": (39.25, -76.58),
    "USNOR": (29.95, -90.07), "MZBEW": (-19.84, 34.84), "MZNAC": (-14.54, 40.67),
    "IDSMR": (-0.50, 117.15), "IDTBN": (-3.30, 114.58), "IDBPN": (-1.27, 116.83),
    "RUVOS": (43.12, 131.90), "RUNAK": (42.82, 132.88), "INPAR": (20.27, 86.67),
    "INVIZ": (17.69, 83.29), "INGAN": (17.63, 83.22), "INGOP": (19.27, 84.91),
    "INDHM": (20.78, 86.98), "INHAL": (22.04, 88.06), "INKOL": (22.57, 88.35),
}

def _get_json(url: str, params: dict) -> dict | list:
    response = requests.get(url, params=params, timeout=8, headers={"User-Agent": "charter-prototype/1.0"})
    response.raise_for_status()
    return response.json()

def world_bank_growth(country_code: str) -> dict:
    """Return most recent annual GDP growth from the keyless World Bank API."""
    try:
        data = _get_json(
            f"https://api.worldbank.org/v2/country/{country_code}/indicator/NY.GDP.MKTP.KD.ZG",
            {"format": "json", "per_page": 10},
        )
        record = next(row for row in data[1] if row.get("value") is not None)
        return {"value": round(float(record["value"]), 2), "year": record["date"], "source": "World Bank API", "status": "live"}
    except Exception as exc:
        return {"value": None, "year": None, "source": "World Bank API", "status": f"unavailable: {type(exc).__name__}"}

def marine_risk(port_code: str) -> dict:
    """Return a next-24-hour marine condition signal from keyless Open-Meteo."""
    try:
        lat, lon = PORT_COORDS[port_code]
        data = _get_json(
            "https://marine-api.open-meteo.com/v1/marine",
            {"latitude": lat, "longitude": lon, "hourly": "wave_height,wind_wave_height", "forecast_days": 2, "timezone": "UTC"},
        )
        wave = [v for v in data["hourly"]["wave_height"][:24] if v is not None]
        max_wave = max(wave) if wave else None
        level = "High" if max_wave and max_wave >= 3 else ("Moderate" if max_wave and max_wave >= 1.5 else "Low")
        return {"max_wave_m": max_wave, "risk": level, "source": "Open-Meteo Marine API", "status": "live"}
    except Exception as exc:
        return {"max_wave_m": None, "risk": "Unknown", "source": "Open-Meteo Marine API", "status": f"unavailable: {type(exc).__name__}"}

def live_context(origin_code: str, destination_code: str, origin_country: str) -> dict:
    country = {"Australia": "AUS", "USA": "USA", "Mozambique": "MOZ", "Indonesia": "IDN", "Russia": "RUS"}.get(origin_country, "IND")
    return {
        "retrieved_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "origin_gdp_growth": world_bank_growth(country),
        "origin_marine": marine_risk(origin_code),
        "destination_marine": marine_risk(destination_code),
    }
