"""
Module 2 — Vessel Type Optimizer
Step A: hard-constraint filter (draft/LOA/beam) -> eligible vessel types
Step B: among eligible types, pick lowest total freight cost via LP
         (here it's a single-decision-variable choice, so PuLP is used to
         demonstrate the pattern; for multi-parcel/multi-period allocation
         this scales to a real assignment problem)
"""
import pandas as pd
import pulp
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
ports = pd.read_csv(DATA_DIR / "ports_east_coast_india.csv")
vessels = pd.read_csv(DATA_DIR / "vessel_types.csv")

def optimize_vessel(dest_port_code, cargo_qty_tonnes, predicted_rate_per_tonne_by_vessel):
    """
    predicted_rate_per_tonne_by_vessel: dict like {"Handysize": 23.1, "Supramax": 19.4, ...}
    (these come straight from Module 1's forecasting ensemble)
    """
    port = ports[ports.port_code == dest_port_code].iloc[0]

    # ---- Step A: hard constraint filter ----
    eligible = vessels[
        (vessels.typical_draft_m <= port.max_draft_m) &
        (vessels.typical_loa_m <= port.max_loa_m) &
        (vessels.typical_beam_m <= port.max_beam_m) &
        (vessels.dwt_max >= cargo_qty_tonnes * 0.9)   # vessel must be able to carry most of the parcel
    ].copy()

    if eligible.empty:
        return {"status": "infeasible", "reason": "No vessel type satisfies port draft/LOA/beam constraints"}

    # ---- Step B: LP cost minimization over eligible types ----
    prob = pulp.LpProblem("vessel_choice", pulp.LpMinimize)
    choice_vars = {v: pulp.LpVariable(f"choose_{v}", cat="Binary") for v in eligible.vessel_type}

    # objective: total voyage cost = predicted rate * cargo qty (only for eligible types)
    prob += pulp.lpSum(
        choice_vars[v] * predicted_rate_per_tonne_by_vessel.get(v, 1e9) * cargo_qty_tonnes
        for v in eligible.vessel_type
    )
    prob += pulp.lpSum(choice_vars.values()) == 1  # exactly one vessel type chosen

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    chosen = [v for v, var in choice_vars.items() if var.value() == 1][0]
    cost = predicted_rate_per_tonne_by_vessel[chosen] * cargo_qty_tonnes

    return {
        "status": "optimal",
        "eligible_types": eligible.vessel_type.tolist(),
        "recommended_vessel": chosen,
        "predicted_rate_per_tonne": predicted_rate_per_tonne_by_vessel[chosen],
        "total_voyage_cost_usd": round(cost, 2),
    }


if __name__ == "__main__":
    # Example: 55,000 tonnes coal to Haldia (shallow draft port -> Capesize/Panamax auto-excluded)
    predicted_rates = {"Handysize": 23.1, "Supramax": 19.4, "Panamax": 15.8, "Capesize": 11.2}
    result = optimize_vessel("INHAL", 55000, predicted_rates)
    print("Example 1 — Haldia, 55,000t cargo:")
    print(result)

    print()
    # Example: 160,000 tonnes to Dhamra (deep-water port -> Capesize eligible)
    result2 = optimize_vessel("INDHM", 160000, predicted_rates)
    print("Example 2 — Dhamra, 160,000t cargo:")
    print(result2)
