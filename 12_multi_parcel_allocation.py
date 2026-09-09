"""
Module 12 — Multi-Parcel Vessel Allocation (real optimization problem)
=========================================================================
Module 4 (4_vessel_optimizer.py) answered: "for ONE cargo parcel, which
vessel type is cheapest?" - a single decision.

This module answers the actual fleet-planning question: "I have SEVERAL
cargo parcels to move this month across different routes - which vessel
type should serve which parcel, to minimize TOTAL cost across all of them,
given port eligibility constraints?" This is a proper assignment problem,
solved with PuLP.
"""
import pandas as pd
import pulp
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
ports = pd.read_csv(DATA_DIR / "ports_east_coast_india.csv")
vessels = pd.read_csv(DATA_DIR / "vessel_types.csv")

def is_eligible(vessel_row, port_row, cargo_qty):
    return (vessel_row.typical_draft_m <= port_row.max_draft_m and
            vessel_row.typical_loa_m <= port_row.max_loa_m and
            vessel_row.typical_beam_m <= port_row.max_beam_m and
            vessel_row.dwt_max >= cargo_qty * 0.9)

def optimize_multi_parcel(parcels, predicted_rates_lookup):
    """
    parcels: list of dicts, e.g.
      [{"parcel_id": "P1", "dest_port_code": "INPAR", "cargo_qty_tonnes": 55000},
       {"parcel_id": "P2", "dest_port_code": "INDHM", "cargo_qty_tonnes": 160000}, ...]
    predicted_rates_lookup: dict {(dest_port_code, vessel_type): predicted_rate_per_tonne}
       (in production this comes straight from Module 1's forecasting ensemble,
        called once per destination port)

    Returns the cost-minimizing vessel assignment across ALL parcels simultaneously.
    NOTE: this version assumes unlimited vessel availability of each type (a
    simplifying assumption for the prototype). A production version would add
    a vessel-count constraint per type to reflect fleet/charter-market supply.
    """
    prob = pulp.LpProblem("multi_parcel_allocation", pulp.LpMinimize)

    # decision variables: assign[parcel_id, vessel_type] = 1 if that vessel type serves that parcel
    assign = {}
    eligibility = {}
    for p in parcels:
        port_row = ports[ports.port_code == p["dest_port_code"]].iloc[0]
        for _, v in vessels.iterrows():
            key = (p["parcel_id"], v.vessel_type)
            eligibility[key] = is_eligible(v, port_row, p["cargo_qty_tonnes"])
            assign[key] = pulp.LpVariable(f"assign_{p['parcel_id']}_{v.vessel_type}", cat="Binary")

    # objective: minimize total cost across all parcels (only eligible assignments contribute; ineligible get a huge penalty so they're never chosen)
    prob += pulp.lpSum(
        assign[(p["parcel_id"], v)] * (
            predicted_rates_lookup.get((p["dest_port_code"], v), 1e9) * p["cargo_qty_tonnes"]
            if eligibility[(p["parcel_id"], v)] else 1e12
        )
        for p in parcels for v in vessels.vessel_type
    )

    # each parcel must be served by exactly one vessel type
    for p in parcels:
        prob += pulp.lpSum(assign[(p["parcel_id"], v)] for v in vessels.vessel_type) == 1

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    results = []
    total_cost = 0
    for p in parcels:
        chosen = [v for v in vessels.vessel_type if assign[(p["parcel_id"], v)].value() == 1][0]
        rate = predicted_rates_lookup.get((p["dest_port_code"], chosen), None)
        cost = rate * p["cargo_qty_tonnes"] if rate else None
        total_cost += cost or 0
        results.append({
            "parcel_id": p["parcel_id"], "dest_port_code": p["dest_port_code"],
            "cargo_qty_tonnes": p["cargo_qty_tonnes"], "assigned_vessel": chosen,
            "rate_per_tonne": rate, "cost_usd": round(cost, 2) if cost else None,
            "was_eligible_check": eligibility[(p["parcel_id"], chosen)],
        })

    return pd.DataFrame(results), round(total_cost, 2)


if __name__ == "__main__":
    # Example: a month's worth of parcels across 4 different routes
    parcels = [
        {"parcel_id": "P1", "dest_port_code": "INPAR", "cargo_qty_tonnes": 55000},
        {"parcel_id": "P2", "dest_port_code": "INDHM", "cargo_qty_tonnes": 160000},
        {"parcel_id": "P3", "dest_port_code": "INVIZ", "cargo_qty_tonnes": 30000},
        {"parcel_id": "P4", "dest_port_code": "INGAN", "cargo_qty_tonnes": 75000},
        {"parcel_id": "P5", "dest_port_code": "INGOP", "cargo_qty_tonnes": 32000},
    ]

    # dummy predicted rates per (port, vessel_type) - in production, call
    # Module 1's forecaster once per (route, vessel_type) to fill this in
    predicted_rates_lookup = {
        ("INPAR", "Handysize"): 21.2, ("INPAR", "Supramax"): 17.3, ("INPAR", "Panamax"): 13.9, ("INPAR", "Capesize"): 9.2,
        ("INDHM", "Handysize"): 21.5, ("INDHM", "Supramax"): 17.6, ("INDHM", "Panamax"): 14.1, ("INDHM", "Capesize"): 9.4,
        ("INVIZ", "Handysize"): 20.8, ("INVIZ", "Supramax"): 17.0, ("INVIZ", "Panamax"): 13.7, ("INVIZ", "Capesize"): 9.0,
        ("INGAN", "Handysize"): 20.5, ("INGAN", "Supramax"): 16.8, ("INGAN", "Panamax"): 13.5, ("INGAN", "Capesize"): 8.8,
        ("INGOP", "Handysize"): 20.9, ("INGOP", "Supramax"): 17.1, ("INGOP", "Panamax"): 13.8,
    }

    result, total_cost = optimize_multi_parcel(parcels, predicted_rates_lookup)
    print(result.to_string(index=False))
    print(f"\nTotal fleet cost across all 5 parcels: ${total_cost:,.2f}")
