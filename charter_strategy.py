"""Explainable commercial strategy rules for a bulk-chartering decision.

This module intentionally turns model output into a recommendation with explicit
assumptions. It is a decision-support layer, never an autonomous fixture engine.
"""
from __future__ import annotations

import pandas as pd


def recommend_strategy(
    scenarios: pd.DataFrame,
    outlook: pd.DataFrame,
    laycan: object,
    risk_index: int,
    cargo_tonnes: float,
    selected_vessel: str,
    cycle_days: float,
) -> dict:
    """Choose the commercial posture that best balances savings and resilience."""
    today_rate = float(outlook.forecast_usd_t.iloc[0])
    permitted = outlook[outlook.date.dt.date <= laycan] if laycan else outlook
    permitted = permitted if not permitted.empty else outlook
    trough = permitted.loc[permitted.forecast_usd_t.idxmin()]
    rate_gap_pct = (today_rate - trough.forecast_usd_t) / today_rate * 100
    volatility_pct = (outlook.p90.iloc[0] - outlook.p10.iloc[0]) / today_rate * 100
    preferred = scenarios.loc[scenarios.saving_vs_repeated_spot_usd.idxmax()]

    if risk_index >= 70 or volatility_pct >= 18:
        posture = "Staged 3-voyage cover"
        strategy_voyages = 3
        action = "Fix 40% of committed volume now; retain 60% for the preferred entry window with a freight cap."
        rationale = "High market uncertainty makes a full long commitment less attractive than protected, staged coverage."
    elif rate_gap_pct >= 3 and trough.date.date() <= laycan:
        posture = "Wait for forecast entry window"
        strategy_voyages = 3 if risk_index < 55 else 1
        action = f"Defer the main fixture until {trough.date:%d %b}; monitor daily and set a maximum acceptable rate."
        rationale = f"The model projects a {rate_gap_pct:.1f}% lower rate before laycan, creating a measurable wait-versus-fix opportunity."
    elif preferred.voyages == 6 and risk_index < 55:
        posture = "Medium-term 6-voyage contract"
        strategy_voyages = 6
        action = "Negotiate a 6-voyage index-linked agreement with a cap/collar and port-performance clauses."
        rationale = "Stable risk conditions and the highest forecast saving support securing capacity beyond isolated spot fixtures."
    else:
        posture = "Short-term 3-voyage contract"
        strategy_voyages = 3
        action = "Secure three voyages with a re-opener after voyage two and defined congestion/alternate-port triggers."
        rationale = "This balances committed capacity with flexibility while the market remains uncertain."

    selected = scenarios[scenarios.voyages == strategy_voyages].iloc[0]
    protections = [
        "Freight cap/collar or index-linked adjustment clause",
        "Alternate East Coast discharge-port option subject to berth acceptance",
        "Congestion trigger with agreed demurrage and laycan extension treatment",
        "Next-employment / ballast-positioning option to limit idle exposure",
    ]
    return {
        "posture": posture,
        "action": action,
        "rationale": rationale,
        "recommended_voyages": int(strategy_voyages),
        "recommended_vessel": selected_vessel,
        "entry_date": trough.date.date(),
        "entry_rate": float(trough.forecast_usd_t),
        "today_rate": today_rate,
        "rate_gap_pct": rate_gap_pct,
        "volatility_pct": volatility_pct,
        "expected_saving": float(selected.saving_vs_repeated_spot_usd),
        "expected_cost": float(selected.all_in_cost_usd),
        "cycle_days": cycle_days,
        "cargo_tonnes": cargo_tonnes,
        "protections": protections,
    }
