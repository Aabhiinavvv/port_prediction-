
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd


FUEL_TPD = {"Handysize": 18, "Supramax": 26, "Panamax": 32, "Capesize": 55}


def fleet_availability(vessels: pd.DataFrame) -> pd.DataFrame:
    """Prototype fleet roster; replace with TMS/fleet position events in production."""
    profile = {
        "Handysize": (3, 2), "Supramax": (2, 5), "Panamax": (1, 11), "Capesize": (1, 16),
    }
    rows = []
    for vessel_type in vessels.vessel_type:
        count, days = profile.get(vessel_type, (0, 99))
        rows.append({
            "vessel_type": vessel_type, "available_hulls": count,
            "next_available_days": days,
            "availability_status": "Available for laycan" if days <= 21 and count else "Check repositioning",
            "data_status": "Prototype fleet assumption",
        })
    return pd.DataFrame(rows)


def optimize_vessels(candidates: pd.DataFrame, distance_nm: float, congestion: int, priority: str) -> pd.DataFrame:
    """Rank feasible vessel classes by cost, carbon, reliability and fleet availability."""
    result = candidates.copy()
    sailing_days = distance_nm / (12 * 24)
    result["voyage_fuel_tonnes"] = result.vessel_type.map(FUEL_TPD).fillna(26) * sailing_days
    result["voyage_co2_tonnes"] = result["voyage_fuel_tonnes"] * 3.114
    result["port_delay_days"] = 1.0 + congestion / 100
    result["availability_penalty"] = result["next_available_days"].clip(upper=30) / 30
    result["cost_component"] = (result["total_freight"] - result["total_freight"].min()) / max(1, result["total_freight"].max() - result["total_freight"].min())
    result["carbon_component"] = (result["voyage_co2_tonnes"] - result["voyage_co2_tonnes"].min()) / max(1, result["voyage_co2_tonnes"].max() - result["voyage_co2_tonnes"].min())
    result["fit_component"] = 1 - result["utilisation"].sub(.85).abs().clip(upper=.85) / .85
    weights = {
        "Lowest cost": (.55, .10, .20, .15),
        "Lowest CO₂": (.15, .55, .15, .15),
        "Highest reliability": (.20, .10, .30, .40),
        "Balanced": (.35, .20, .25, .20),
    }.get(priority, (.35, .20, .25, .20))
    cost_w, carbon_w, fit_w, availability_w = weights
    result["multi_objective_score"] = 100 * (
        1 - cost_w * result["cost_component"] - carbon_w * result["carbon_component"]
        + fit_w * result["fit_component"] - availability_w * result["availability_penalty"]
    )
    return result.sort_values("multi_objective_score", ascending=False)


def risk_cockpit(congestion: int, uncertainty: float, rate: float, port_days: float, fleet_days: int, jit_risk: int) -> pd.DataFrame:
    drivers = [
        ("Port & berth", min(100, congestion * .8 + port_days * 5), "Confirm berth window; retain alternate-port and demurrage clauses."),
        ("Market / freight", min(100, uncertainty / max(rate, .1) * 300), "Use a rate cap, index linkage, or staged fixture."),
        ("Fleet availability", min(100, fleet_days * 4), "Hold the vessel option or line up a repositioning alternative."),
        ("JIT arrival execution", jit_risk, "Reconfirm terminal readiness before issuing the speed instruction."),
    ]
    rows = []
    for driver, score, action in drivers:
        level = "High" if score >= 70 else "Medium" if score >= 40 else "Low"
        rows.append({"risk_driver": driver, "score": round(score), "level": level, "recommended_mitigation": action})
    return pd.DataFrame(rows).sort_values("score", ascending=False)


def freight_scenarios(outlook: pd.DataFrame, congestion: int) -> pd.DataFrame:
    """Scenario paths remain derived from the model forecast, not claimed market quotes."""
    factors = {
        "Base model": 1.00,
        "Congestion escalation": 1 + min(.18, congestion / 600),
        "Bull market / disruption": 1.15,
        "Bear market / soft demand": .88,
    }
    frames = []
    for name, factor in factors.items():
        frame = outlook[["date", "forecast_usd_t"]].copy()
        frame["scenario"] = name
        frame["rate_usd_mt"] = frame.pop("forecast_usd_t") * factor
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def port_scorecard(ports: pd.DataFrame, cargo: int) -> pd.DataFrame:
    result = ports.copy()
    result["estimated_port_days"] = cargo / result["cargo_handling_rate_tpd"] + result["avg_pre_berthing_delay_days"]
    result["throughput_score"] = 100 * result["cargo_handling_rate_tpd"] / result["cargo_handling_rate_tpd"].max()
    result["delay_score"] = 100 * (1 - result["avg_pre_berthing_delay_days"] / result["avg_pre_berthing_delay_days"].max())
    result["port_performance_score"] = (.6 * result["throughput_score"] + .4 * result["delay_score"]).round(1)
    return result[["port_code", "port_name", "dry_bulk_berths", "cargo_handling_rate_tpd", "avg_pre_berthing_delay_days", "estimated_port_days", "port_performance_score"]].sort_values("port_performance_score", ascending=False)


def decision_audit(route_id: str, vessel: str, cargo: int, priority: str, risk_index: int, jit_action: str, source_status: str) -> pd.DataFrame:
    rows = [
        ("Decision timestamp", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")),
        ("Route / parcel", f"{route_id} · {cargo:,.0f} MT"),
        ("Selected vessel", vessel),
        ("Optimization priority", priority),
        ("Risk index", f"{risk_index}/100"),
        ("JIT action", jit_action),
        ("Data provenance", source_status),
        ("Standards profile", "DCSA Port Call / IMO Maritime Single Window mapping"),
    ]
    return pd.DataFrame(rows, columns=["audit_field", "recorded_value"])


def standards_mapping() -> pd.DataFrame:
    return pd.DataFrame([
        ("route_id", "Port Call / operational schedule reference", "Carrier / voyage-planning system"),
        ("berth_ready_time", "DCSA Estimated / Requested / Planned berth event", "Terminal / Port Authority"),
        ("congestion_index", "Port operational status enrichment", "Port Community System / terminal"),
        ("vessel availability", "Operational vessel schedule / fleet position", "Carrier / AIS / fleet system"),
        ("cargo and vessel details", "IMO Maritime Single Window declaration context", "Ship agent / authority"),
        ("freight forecast", "Commercial decision-support enrichment", "Licensed broker / internal model"),
    ], columns=["dashboard_field", "standards-ready representation", "production source"])
