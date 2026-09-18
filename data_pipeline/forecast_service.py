"""Small service layer for the freight forecast engine.

Use this module from Streamlit/API code so model training is not repeated for
every page interaction. In production, persist trained models and refresh them
on a scheduled cadence.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from freight_forecast_engine import FreightForecastEngine, load_training_data

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

_ENGINE = None
_DATA = None


def get_engine(force_refresh: bool = False) -> tuple[FreightForecastEngine, pd.DataFrame]:
    global _ENGINE, _DATA
    if _ENGINE is None or _DATA is None or force_refresh:
        _DATA = load_training_data()
        _ENGINE = FreightForecastEngine().fit(_DATA)
    return _ENGINE, _DATA


def forecast_route(route_id: str, vessel_type: str, as_of: str | None = None) -> pd.DataFrame:
    engine, data = get_engine()
    return engine.forecast(data, route_id, vessel_type, as_of=as_of)


def validation() -> pd.DataFrame:
    engine, _ = get_engine()
    return engine.validation_summary()
