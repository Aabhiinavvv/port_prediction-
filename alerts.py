"""Action-oriented early warning rules for the charter strategy dashboard."""
from __future__ import annotations

from datetime import date


def build_alerts(
    *, risk_index: int, congestion: int, port_days: float, cycle_days: float,
    utilisation: float, volatility_pct: float, entry_date: date, laycan: date,
    weather_context: dict | None = None,
) -> list[dict]:
    alerts: list[dict] = []
    if risk_index >= 70 or volatility_pct >= 18:
        alerts.append({"severity": "Critical", "alert": "Freight-market volatility", "evidence": f"Risk {risk_index}/100; forecast spread {volatility_pct:.1f}%.", "action": "Use staged cover, a freight cap/collar and daily re-forecasting before fixture approval."})
    elif risk_index >= 55:
        alerts.append({"severity": "Warning", "alert": "Elevated freight uncertainty", "evidence": f"Risk {risk_index}/100.", "action": "Avoid an unprotected long commitment; retain a rate re-opener or index linkage."})

    if congestion >= 65:
        alerts.append({"severity": "Critical", "alert": "Port congestion exposure", "evidence": f"Congestion scenario {congestion}/100; estimated port time {port_days:.1f} days.", "action": "Activate alternate-port option, agree laycan extension and demurrage treatment before fixing."})
    elif congestion >= 45:
        alerts.append({"severity": "Warning", "alert": "Port wait-time watch", "evidence": f"Congestion scenario {congestion}/100.", "action": "Refresh port status before nomination and hold berth-window contingency."})

    if port_days / cycle_days >= .35:
        alerts.append({"severity": "Warning", "alert": "Idle-time concentration", "evidence": f"Port activity is {port_days / cycle_days:.0%} of the {cycle_days:.1f}-day voyage cycle.", "action": "Include next-employment/ballast option and performance clauses to reduce idle exposure."})
    if utilisation < .70:
        alerts.append({"severity": "Warning", "alert": "Low vessel utilisation", "evidence": f"Selected parcel fills only {utilisation:.0%} of nominal capacity.", "action": "Consider aggregation, split parcel or a smaller vessel before accepting the freight offer."})
    if entry_date > laycan:
        alerts.append({"severity": "Critical", "alert": "Laycan versus entry-window conflict", "evidence": f"Forecast trough {entry_date:%d %b} falls after laycan {laycan:%d %b}.", "action": "Fix protected capacity now or renegotiate laycan; do not wait for the full forecast trough."})

    if weather_context:
        for location, signal in weather_context.items():
            if signal and signal.get("risk") in {"High", "Moderate"}:
                alerts.append({"severity": "Warning" if signal["risk"] == "Moderate" else "Critical", "alert": f"Marine-weather disruption at {location}", "evidence": f"{signal['risk']} risk; maximum wave {signal.get('max_wave_m')} m.", "action": "Validate sailing and berth windows; retain weather-delay allowance in the fixture plan."})
    if not alerts:
        alerts.append({"severity": "Info", "alert": "No critical trigger", "evidence": "Current scenario is within configured thresholds.", "action": "Keep daily market monitoring and re-run the strategy if laycan, congestion or cargo readiness changes."})
    return alerts
