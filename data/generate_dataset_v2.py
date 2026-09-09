"""
SIH Freight Forecasting Dataset Generator — v2
================================================
Changes from v1:
  - Daily granularity instead of weekly (2019-01-01 to 2025-12-31, ~2557 days/series)
  - Routes expanded from 17 to 32 (4 new origin ports added: Abbot Point AU,
    New Orleans US, Balikpapan ID, Nakhodka RU)
  - idle_risk_flag column: rolling z-score based label (route+vessel demand
    below normal -> idle risk period)
  - anomaly_flag + anomaly_type columns: GROUND-TRUTH injected anomaly events
    (port congestion spikes e.g. strikes/weather, and freight-rate market
    shocks e.g. geopolitical/canal disruption) so you can benchmark Isolation
    Forest / Z-score detectors against a known answer, not just eyeball it.

Still SYNTHETIC — same disclaimer as v1: real freight panel data is licensed
(Baltic Exchange/Clarksons/Drewry). This is a statistically realistic stand-in
built for an end-to-end pipeline demo, matching a fixed schema you can later
point at real data.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. EAST COAST INDIA DISCHARGE PORTS (unchanged from v1 — real, public specs)
# ---------------------------------------------------------------------------
ports_india = pd.DataFrame([
    ["INPAR", "Paradip",     "Odisha",         18.0, 300, 48, 6, 45000, 1.5],
    ["INVIZ", "Visakhapatnam","Andhra Pradesh", 17.0, 280, 45, 5, 40000, 2.0],
    ["INGAN", "Gangavaram",  "Andhra Pradesh",  20.0, 320, 50, 3, 50000, 1.0],
    ["INGOP", "Gopalpur",    "Odisha",         17.5, 290, 45, 2, 35000, 1.2],
    ["INDHM", "Dhamra",      "Odisha",         20.5, 330, 50, 3, 55000, 0.8],
    ["INHAL", "Haldia",      "West Bengal",    9.0,  186, 30, 4, 20000, 3.5],
    ["INKOL", "Kolkata (Sagar anchorage transfer)", "West Bengal", 8.5, 180, 28, 3, 15000, 4.0],
], columns=["port_code","port_name","state","max_draft_m","max_loa_m","max_beam_m",
            "dry_bulk_berths","cargo_handling_rate_tpd","avg_pre_berthing_delay_days"])
ports_india.to_csv("ports_east_coast_india.csv", index=False)

# ---------------------------------------------------------------------------
# 2. ORIGIN / LOADING PORTS — 10 original + 4 new (Abbot Point, New Orleans,
#    Balikpapan, Nakhodka) to support the expanded route list
# ---------------------------------------------------------------------------
ports_origin = pd.DataFrame([
    ["AUNTL", "Newcastle",   "Australia",  17.5, 350, 50, 60000],
    ["AUGLT", "Gladstone",   "Australia",  16.3, 300, 48, 50000],
    ["AUHPT", "Hay Point",   "Australia",  17.0, 320, 48, 55000],
    ["AUABT", "Abbot Point", "Australia",  17.5, 300, 48, 45000],
    ["USHRV", "Hampton Roads/Norfolk", "USA", 18.0, 330, 48, 40000],
    ["USBLT", "Baltimore",   "USA",        13.7, 290, 45, 30000],
    ["USNOR", "New Orleans", "USA",        13.5, 280, 44, 32000],
    ["MZBEW", "Beira",       "Mozambique", 8.5,  200, 32, 15000],
    ["MZNAC", "Nacala",      "Mozambique", 16.0, 280, 45, 25000],
    ["IDSMR", "Samarinda/Muara Berau anchorage", "Indonesia", 13.0, 250, 40, 20000],
    ["IDTBN", "Taboneo anchorage (Banjarmasin)", "Indonesia", 12.5, 240, 38, 18000],
    ["IDBPN", "Balikpapan",  "Indonesia",  14.0, 260, 40, 22000],
    ["RUVOS", "Vostochny",   "Russia",     16.5, 300, 47, 35000],
    ["RUNAK", "Nakhodka",    "Russia",     14.5, 260, 40, 28000],
], columns=["port_code","port_name","country","max_draft_m","max_loa_m","max_beam_m","cargo_handling_rate_tpd"])
ports_origin.to_csv("loading_ports_origin.csv", index=False)

# ---------------------------------------------------------------------------
# 3. VESSEL TYPES (unchanged, real industry-standard classes)
# ---------------------------------------------------------------------------
vessel_types = pd.DataFrame([
    ["Handysize", 28000, 38000, 180, 30, 10.5],
    ["Supramax",  50000, 60000, 200, 32, 12.5],
    ["Panamax",   65000, 80000, 225, 32.3, 14.0],
    ["Capesize", 150000, 180000, 290, 45, 18.0],
], columns=["vessel_type","dwt_min","dwt_max","typical_loa_m","typical_beam_m","typical_draft_m"])
vessel_types.to_csv("vessel_types.csv", index=False)

# ---------------------------------------------------------------------------
# 4. ROUTES — expanded from 17 to 32 (approximate nm, refine with a
#    sea-distance calculator for production use)
# ---------------------------------------------------------------------------
route_distances = {
    ("AUNTL","INPAR"): 5900, ("AUNTL","INVIZ"): 5750, ("AUNTL","INDHM"): 5950, ("AUNTL","INGAN"): 5800,
    ("AUGLT","INPAR"): 6100, ("AUGLT","INVIZ"): 5950, ("AUGLT","INDHM"): 6150,
    ("AUHPT","INGAN"): 5850, ("AUHPT","INPAR"): 5950,
    ("AUABT","INVIZ"): 5700, ("AUABT","INGAN"): 5780,
    ("USHRV","INPAR"): 9700, ("USHRV","INVIZ"): 9550, ("USHRV","INGAN"): 9600,
    ("USBLT","INHAL"): 9800, ("USBLT","INVIZ"): 9650,
    ("USNOR","INPAR"): 11200, ("USNOR","INVIZ"): 11050,
    ("MZBEW","INPAR"): 3550, ("MZBEW","INVIZ"): 3350, ("MZBEW","INGOP"): 3450,
    ("MZNAC","INGAN"): 3100, ("MZNAC","INPAR"): 3300,
    ("IDSMR","INVIZ"): 2450, ("IDSMR","INPAR"): 2650, ("IDSMR","INDHM"): 2700,
    ("IDTBN","INGOP"): 2500, ("IDTBN","INPAR"): 2600,
    ("IDBPN","INVIZ"): 2500, ("IDBPN","INGAN"): 2550,
    ("RUVOS","INPAR"): 5300, ("RUVOS","INVIZ"): 5150,
    ("RUNAK","INGAN"): 6350,
}
routes = pd.DataFrame(
    [(o, d, dist) for (o, d), dist in route_distances.items()],
    columns=["origin_port_code","dest_port_code","distance_nm"]
)
routes = routes.merge(ports_origin[["port_code","country"]], left_on="origin_port_code", right_on="port_code", how="left") \
                .drop(columns="port_code").rename(columns={"country":"origin_country"})
routes = routes.merge(ports_india[["port_code","port_name"]], left_on="dest_port_code", right_on="port_code", how="left") \
                .drop(columns="port_code").rename(columns={"port_name":"dest_port_name"})
routes["route_id"] = routes["origin_port_code"] + "_" + routes["dest_port_code"]
routes.to_csv("routes.csv", index=False)

# ---------------------------------------------------------------------------
# 5. SYNTHETIC DAILY TIME SERIES (2019-01-01 to 2025-12-31)
# ---------------------------------------------------------------------------
dates = pd.date_range("2019-01-01", "2025-12-31", freq="D")
n_days = len(dates)
doy = dates.dayofyear.values

# --- global coal price index & BDI-like index (daily, same construction as v1) ---
coal = np.zeros(n_days); coal[0] = 100
for t in range(1, n_days):
    coal[t] = coal[t-1] + 0.02*(100-coal[t-1]) + rng.normal(0, 0.7)
spike_s, spike_e = int(n_days*0.155), int(n_days*0.205)   # ~2021 energy crisis window
coal[spike_s:spike_e] += np.linspace(0, 90, spike_e-spike_s)
coal[spike_e:spike_e+200] += np.linspace(90, 15, 200)
coal = pd.Series(coal, index=dates)

bdi = np.zeros(n_days); bdi[0] = 100
vol = 3.0
for t in range(1, n_days):
    vol = 0.95*vol + 0.05*abs(rng.normal(2, 1.5))
    coal_effect = 0.15*(coal.iloc[t]-coal.iloc[t-1])
    seasonal = 8*np.sin(2*np.pi*(doy[t]/365.0)+1.2)
    bdi[t] = max(30, bdi[t-1] + coal_effect + seasonal*0.02 + 0.01 + rng.normal(0, vol))
bdi[spike_s:spike_e] += np.linspace(0, 140, spike_e-spike_s)
bdi[spike_e:spike_e+250] += np.linspace(140, 20, 250)
bdi = pd.Series(bdi, index=dates)

vessel_base_rate = {"Handysize": 2.3, "Supramax": 1.9, "Panamax": 1.5, "Capesize": 1.0}
congestion_base = {"INPAR":18,"INVIZ":22,"INGAN":10,"INGOP":15,"INDHM":8,"INHAL":30,"INKOL":32}

def inject_events(n, n_events, dur_range, rng):
    """Return boolean mask + list of (start, end) for randomly placed events."""
    mask = np.zeros(n, dtype=bool)
    spans = []
    for _ in range(n_events):
        start = rng.integers(0, n - dur_range[1])
        dur = rng.integers(dur_range[0], dur_range[1]+1)
        end = min(n, start+dur)
        mask[start:end] = True
        spans.append((start, end))
    return mask, spans

rows = []
for _, r in routes.iterrows():
    dest = r["dest_port_code"]

    # --- port congestion base series (route/port-level, shared across vessel types) ---
    congestion = np.clip(
        congestion_base.get(dest, 15) + 10*np.sin(2*np.pi*(doy/365.0)+0.5) + rng.normal(0, 4, n_days),
        0, 100
    )
    # ground-truth congestion anomaly events (strikes / weather / breakdowns)
    n_cong_events = max(4, n_days // 350)
    cong_mask, cong_spans = inject_events(n_days, n_cong_events, (3, 12), rng)
    for s, e in cong_spans:
        congestion[s:e] += rng.uniform(30, 60)
    congestion = np.clip(congestion, 0, 100)

    # ground-truth freight-market shock events (route-wide: geopolitical, canal disruption, demand collapse)
    n_freight_events = max(3, n_days // 450)
    freight_mask, freight_spans = inject_events(n_days, n_freight_events, (5, 18), rng)
    freight_multiplier = np.ones(n_days)
    for s, e in freight_spans:
        mult = rng.uniform(1.4, 2.2) if rng.random() > 0.25 else rng.uniform(0.5, 0.65)
        freight_multiplier[s:e] *= mult

    distance_factor = r["distance_nm"] / 1000.0
    route_noise = rng.normal(0, 0.15, n_days).cumsum() * 0.01

    for vtype, base in vessel_base_rate.items():
        freight_rate = (
            base * distance_factor * (bdi.values/100.0) * (1 + 0.15*(congestion/100.0))
            + route_noise + rng.normal(0, 0.35, n_days)
        )
        freight_rate = np.clip(freight_rate, 2, None) * freight_multiplier

        # idle-risk label: rolling 60-day z-score of freight_rate below -1 -> low-demand/idle risk period
        fr_series = pd.Series(freight_rate)
        roll_mean = fr_series.rolling(60, min_periods=20).mean()
        roll_std = fr_series.rolling(60, min_periods=20).std()
        zscore = (fr_series - roll_mean) / roll_std.replace(0, np.nan)
        idle_flag = (zscore < -1.3).fillna(False).astype(int).values

        anomaly_flag = (freight_mask | cong_mask).astype(int)
        anomaly_type = np.where(freight_mask & cong_mask, "both",
                        np.where(freight_mask, "freight_rate_shock",
                        np.where(cong_mask, "port_congestion_spike", "none")))

        cargo_range = ([28000,38000] if vtype=="Handysize" else [50000,60000] if vtype=="Supramax"
                        else [65000,80000] if vtype=="Panamax" else [150000,180000])
        cargo_qty = rng.integers(*cargo_range, size=n_days)

        df = pd.DataFrame({
            "date": dates.date,
            "route_id": r["route_id"],
            "origin_port_code": r["origin_port_code"],
            "dest_port_code": dest,
            "vessel_type": vtype,
            "distance_nm": r["distance_nm"],
            "freight_rate_usd_per_tonne": np.round(freight_rate, 2),
            "bdi_index": np.round(bdi.values, 2),
            "coal_price_index": np.round(coal.values, 2),
            "port_congestion_index": np.round(congestion, 1),
            "cargo_qty_tonnes": cargo_qty,
            "idle_risk_flag": idle_flag,
            "anomaly_flag": anomaly_flag,
            "anomaly_type": anomaly_type,
        })
        rows.append(df)

ts = pd.concat(rows, ignore_index=True)
ts.to_csv("freight_rates_timeseries.csv", index=False)

print("Rows generated:")
print(" ports_east_coast_india:", len(ports_india))
print(" loading_ports_origin  :", len(ports_origin))
print(" vessel_types          :", len(vessel_types))
print(" routes                :", len(routes))
print(" freight_rates_timeseries:", len(ts))
print("\nAnomaly rate:", ts["anomaly_flag"].mean().round(4))
print("Idle-risk rate:", ts["idle_risk_flag"].mean().round(4))
print("\nAnomaly type counts:\n", ts.loc[ts.anomaly_flag==1, "anomaly_type"].value_counts())
print("\nFreight rate stats by vessel type:\n", ts.groupby("vessel_type")["freight_rate_usd_per_tonne"].describe()[["mean","min","max"]])
