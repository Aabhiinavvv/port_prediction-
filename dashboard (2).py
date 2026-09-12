
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pulp
from sklearn.ensemble import IsolationForest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

st.set_page_config(page_title="Freight Forecasting Dashboard", layout="wide")

@st.cache_data
def load_data():
    ts = pd.read_csv(DATA_DIR / "freight_rates_timeseries.csv", parse_dates=["date"])
    ports = pd.read_csv(DATA_DIR / "ports_east_coast_india.csv")
    vessels = pd.read_csv(DATA_DIR / "vessel_types.csv")
    routes = pd.read_csv(DATA_DIR / "routes.csv")
    try:
        accuracy = pd.read_csv(PROJECT_DIR / "all_routes_forecast_summary.csv")
    except FileNotFoundError:
        accuracy = pd.DataFrame(columns=["route_id", "vessel_type", "rmse", "mape_pct"])
    return ts, ports, vessels, routes, accuracy

def render_table(dataframe):
    st.markdown(dataframe.to_html(index=False, classes="dataframe", border=0), unsafe_allow_html=True)

ts, ports, vessels, routes, accuracy = load_data()

st.title("🚢 Freight Forecasting & Chartering Dashboard")
st.caption("SIH Prototype — East Coast India dry-bulk coal procurement")

# ---------------- Sidebar inputs ----------------
st.sidebar.header("Cargo & Route Inputs")
dest_port = st.sidebar.selectbox("Destination port (India)", ports.port_code.tolist(),
                                   format_func=lambda x: ports.loc[ports.port_code==x, "port_name"].values[0])
available_routes = routes[routes.dest_port_code == dest_port]
if available_routes.empty:
    st.sidebar.warning("No routes to this port in the dataset.")
    st.stop()
route_id = st.sidebar.selectbox("Origin -> Route", available_routes.route_id.tolist())
cargo_qty = st.sidebar.number_input("Cargo quantity (tonnes)", min_value=5000, max_value=200000, value=55000, step=5000)

with st.expander("Decision brief", expanded=True):
    brief_cols = st.columns(4)
    brief_cols[0].metric("Destination", dest_port)
    brief_cols[1].metric("Route", route_id)
    brief_cols[2].metric("Cargo", f"{cargo_qty:,.0f} t")
    brief_cols[3].metric("Vessel options", "Calculating...")

st.sidebar.markdown("---")
st.sidebar.caption("Dataset: synthetic, calibrated to realistic freight-market statistics (see DATA_DICTIONARY.md)")

# ---------------- Module 1: Forecast for each vessel type on this route ----------------
st.header("1. Freight Rate Forecast")

FEATURES = ["lag_1","lag_7","lag_14","lag_30","lag_90","roll_mean_7","roll_mean_30",
            "roll_std_30","bdi_index","coal_price_index","port_congestion_index","dow","month"]
TARGET = "freight_rate_usd_per_tonne"

def add_features(d):
    d = d.copy()
    for lag in [1, 7, 14, 30, 90]:
        d[f"lag_{lag}"] = d[TARGET].shift(lag)
    d["roll_mean_7"] = d[TARGET].shift(1).rolling(7).mean()
    d["roll_mean_30"] = d[TARGET].shift(1).rolling(30).mean()
    d["roll_std_30"] = d[TARGET].shift(1).rolling(30).std()
    d["dow"] = pd.to_datetime(d["date"]).dt.dayofweek
    d["month"] = pd.to_datetime(d["date"]).dt.month
    return d

@st.cache_data
def forecast_all_vessel_types(route_id):
    """Returns point forecast + 80% prediction interval (p10-p90) per vessel type.
    XGBoost only - a fixed-order ARIMA was tested via walk-forward validation
    and found unstable across time windows (see 8_rigorous_validation.py),
    so it was dropped from production rather than risk a misleading ensemble."""
    predictions = {}
    for vtype in vessels.vessel_type:
        df = ts[(ts.route_id == route_id) & (ts.vessel_type == vtype)].sort_values("date").reset_index(drop=True)
        if df.empty:
            continue
        df = add_features(df).dropna().reset_index(drop=True)
        train = df.iloc[:-1]
        latest = df.iloc[[-1]]

        point_model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
        point_model.fit(train[FEATURES], train[TARGET])
        point_pred = float(point_model.predict(latest[FEATURES])[0])

        p10_model = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.10,
                                      n_estimators=300, max_depth=5, learning_rate=0.05,
                                      subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
        p10_model.fit(train[FEATURES], train[TARGET])
        p10_pred = float(p10_model.predict(latest[FEATURES])[0])

        p90_model = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.90,
                                      n_estimators=300, max_depth=5, learning_rate=0.05,
                                      subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
        p90_model.fit(train[FEATURES], train[TARGET])
        p90_pred = float(p90_model.predict(latest[FEATURES])[0])

        p10_pred, p90_pred = min(p10_pred, p90_pred), max(p10_pred, p90_pred)
        predictions[vtype] = {"point": round(point_pred, 2), "p10": round(p10_pred, 2), "p90": round(p90_pred, 2)}
    return predictions

predicted_full = forecast_all_vessel_types(route_id)
predicted_rates = {v: d["point"] for v, d in predicted_full.items()}

if predicted_full:
    brief_cols[3].metric("Vessel options", len(predicted_full))

    trend = (ts[(ts.route_id == route_id) & (ts.vessel_type == list(predicted_full)[0])]
             .sort_values("date")
             .set_index("date")[[TARGET]])
    st.subheader("Recent market trend")
    st.line_chart(trend.tail(90), y_label="USD per tonne", x_label="Date")

cols = st.columns(len(predicted_full))
for c, (vtype, vals) in zip(cols, predicted_full.items()):
    match = accuracy[(accuracy.route_id == route_id) & (accuracy.vessel_type == vtype)]
    if not match.empty:
        mape = match.iloc[0]["mape_pct"]
        confidence = "🟢 High" if mape < 3 else ("🟡 Medium" if mape < 6 else "🔴 Low")
        c.metric(vtype, f"${vals['point']}/t", help=f"Model confidence: {confidence} (backtested MAPE {mape}%)")
        c.caption(f"{confidence} ({mape}% MAPE)")
    else:
        c.metric(vtype, f"${vals['point']}/t")
    c.caption(f"80% range: ${vals['p10']}–${vals['p90']}")

# ---------------- Module 2: Vessel optimizer ----------------
st.header("2. Recommended Vessel Type")

port = ports[ports.port_code == dest_port].iloc[0]
eligible = vessels[
    (vessels.typical_draft_m <= port.max_draft_m) &
    (vessels.typical_loa_m <= port.max_loa_m) &
    (vessels.typical_beam_m <= port.max_beam_m) &
    (vessels.dwt_max >= cargo_qty * 0.9)
].copy()

if eligible.empty:
    st.error(f"⚠️ No standard vessel class fits {port.port_name}'s draft/LOA/beam constraints for this cargo size. "
              f"This port likely needs lighterage or a smaller feeder vessel.")
else:
    eligible["predicted_rate"] = eligible.vessel_type.map(predicted_rates)
    eligible["cost_low"] = eligible.vessel_type.map(lambda v: predicted_full[v]["p10"]) * cargo_qty
    eligible["cost_high"] = eligible.vessel_type.map(lambda v: predicted_full[v]["p90"]) * cargo_qty
    eligible["total_cost_usd"] = eligible["predicted_rate"] * cargo_qty
    best = eligible.loc[eligible.total_cost_usd.idxmin()]
    st.success(f"✅ Recommended: **{best.vessel_type}** — ${best.predicted_rate}/t × {cargo_qty:,}t = "
               f"**${best.total_cost_usd:,.0f}** (80% likely range: ${best.cost_low:,.0f}–${best.cost_high:,.0f})")
    st.caption("Cost range reflects genuine forecast uncertainty, not a guaranteed price — confirm against live market rates before committing to a charter.")
    render_table(eligible[["vessel_type","predicted_rate","total_cost_usd","cost_low","cost_high"]].sort_values("total_cost_usd"))

# ---------------- Module 3: Port time intelligence ----------------
st.header("3. Port Time Intelligence")
st.caption("Estimate anchorage waiting time, berth stay, and voyage arrival from current congestion and port capacity.")

time_vessels = list(predicted_full.keys())
default_time_vessel = (best.vessel_type if not eligible.empty else time_vessels[0])
time_vessel = st.selectbox("Vessel for time estimates", time_vessels,
                           index=time_vessels.index(default_time_vessel))
latest_port_row = (ts[(ts.route_id == route_id) & (ts.vessel_type == time_vessel)]
                   .sort_values("date").iloc[-1])
congestion = float(latest_port_row["port_congestion_index"])
base_wait_days = float(port["avg_pre_berthing_delay_days"])
handling_rate = float(port["cargo_handling_rate_tpd"])
distance_nm = float(routes.loc[routes.route_id == route_id, "distance_nm"].iloc[0])

# The dataset has no observed turnaround or AIS/weather labels. These calibrated
# estimates expose their assumptions so they can be replaced by live feeds later.
queue_factor = 0.65 + 0.70 * congestion / 100
wait_days = base_wait_days * queue_factor
wait_low = max(0.0, wait_days * 0.70)
wait_high = wait_days * 1.35
handling_days = cargo_qty / handling_rate
admin_days = 0.50
turnaround_days = handling_days + admin_days + 1.50 * congestion / 100

time_cols = st.columns(3)
time_cols[0].metric("Port waiting time", f"{wait_days:.1f} days",
                    help=f"Estimated range: {wait_low:.1f}-{wait_high:.1f} days")
time_cols[1].metric("Berth stay / turnaround", f"{turnaround_days:.1f} days",
                    help=f"Cargo handling: {handling_days:.1f} days + admin/congestion allowance")
time_cols[2].metric("Current congestion", f"{congestion:.0f}/100")

eta_col1, eta_col2, eta_col3 = st.columns(3)
departure_date = eta_col1.date_input("Planned departure", value=pd.Timestamp.today().date())
speed_knots = eta_col2.number_input("Cruising speed (knots)", min_value=8.0, max_value=20.0,
                                    value=12.0, step=0.5)
weather_state = eta_col3.selectbox("Weather scenario", ["Normal", "Adverse"])
weather_buffer_hours = 0.0 if weather_state == "Normal" else 18.0
sailing_hours = distance_nm / speed_knots
eta = (pd.Timestamp(departure_date) + pd.Timedelta(hours=sailing_hours + wait_days * 24 + weather_buffer_hours))
st.info(f"Estimated voyage ETA at **{eta.strftime('%d %b %Y, %H:%M')}**. "
        f"Sailing: {sailing_hours / 24:.1f} days + anchorage queue: {wait_days:.1f} days"
        f"{f' + weather buffer: {weather_buffer_hours:.0f}h' if weather_buffer_hours else ''}.")
st.caption("Operational estimate, not a live AIS or weather prediction. Replace with AIS, berth queue, and weather feeds for production use.")

# ---------------- Module 4: Multi-Parcel Fleet Planning ----------------
st.header("4. Multi-Parcel Fleet Planning")
st.caption("Optimize vessel assignment across several cargo parcels at once, not just the single route above.")

def is_eligible(vessel_row, port_row, cargo_qty):
    return (vessel_row.typical_draft_m <= port_row.max_draft_m and
            vessel_row.typical_loa_m <= port_row.max_loa_m and
            vessel_row.typical_beam_m <= port_row.max_beam_m and
            vessel_row.dwt_max >= cargo_qty * 0.9)

default_parcels = pd.DataFrame([
    {"parcel_id": "P1", "route_id": route_id, "cargo_qty_tonnes": cargo_qty},
])
st.caption(f"Planning parcel: {route_id}, {cargo_qty:,.0f} tonnes")
parcel_input = default_parcels

if st.button("Optimize fleet across all parcels"):
    parcel_rows = parcel_input.dropna().to_dict("records")
    if not parcel_rows:
        st.warning("Add at least one parcel above.")
    else:
        with st.spinner(f"Forecasting rates and solving assignment for {len(parcel_rows)} parcel(s)..."):
            unique_routes = {p["route_id"] for p in parcel_rows}
            rates_by_route = {rid: forecast_all_vessel_types(rid) for rid in unique_routes}

            prob = pulp.LpProblem("dashboard_multi_parcel", pulp.LpMinimize)
            assign, eligibility = {}, {}
            for p in parcel_rows:
                route_row = routes[routes.route_id == p["route_id"]].iloc[0]
                port_row = ports[ports.port_code == route_row.dest_port_code].iloc[0]
                for _, v in vessels.iterrows():
                    key = (p["parcel_id"], v.vessel_type)
                    eligibility[key] = is_eligible(v, port_row, p["cargo_qty_tonnes"])
                    assign[key] = pulp.LpVariable(f"assign_{p['parcel_id']}_{v.vessel_type}", cat="Binary")

            prob += pulp.lpSum(
                assign[(p["parcel_id"], v)] * (
                    rates_by_route[p["route_id"]][v]["point"] * p["cargo_qty_tonnes"]
                    if eligibility[(p["parcel_id"], v)] else 1e12
                )
                for p in parcel_rows for v in vessels.vessel_type
            )
            for p in parcel_rows:
                prob += pulp.lpSum(assign[(p["parcel_id"], v)] for v in vessels.vessel_type) == 1

            prob.solve(pulp.PULP_CBC_CMD(msg=0))

            results, total_cost, infeasible_parcels = [], 0, []
            for p in parcel_rows:
                chosen_list = [v for v in vessels.vessel_type if assign[(p["parcel_id"], v)].value() == 1]
                chosen = chosen_list[0] if chosen_list else None
                if chosen is None or not eligibility[(p["parcel_id"], chosen)]:
                    infeasible_parcels.append(p["parcel_id"])
                    continue
                rate = rates_by_route[p["route_id"]][chosen]["point"]
                cost = rate * p["cargo_qty_tonnes"]
                total_cost += cost
                dest = routes[routes.route_id == p["route_id"]].iloc[0].dest_port_name
                results.append({"parcel_id": p["parcel_id"], "route_id": p["route_id"], "destination": dest,
                                 "cargo_qty_tonnes": p["cargo_qty_tonnes"], "assigned_vessel": chosen,
                                 "rate_per_tonne": round(rate, 2), "cost_usd": round(cost, 2)})

        if infeasible_parcels:
            st.error(f"⚠️ No eligible vessel found for parcel(s): {', '.join(infeasible_parcels)} — port constraints too tight for the cargo size.")
        if results:
            st.success(f"✅ Optimal fleet cost across {len(results)} parcel(s): **${total_cost:,.0f}**")
            render_table(pd.DataFrame(results))

st.divider()

# ---------------- Module 5: Idle risk (K-Means) + anomaly flag ----------------
st.header("5. Idle-Risk & Early-Warning Status")

ref_vtype = "Handysize" if "Handysize" in predicted_rates else list(predicted_rates.keys())[0]
df_ref = ts[(ts.route_id == route_id) & (ts.vessel_type == ref_vtype)].sort_values("date").reset_index(drop=True)

# --- Idle risk via K-Means on DETRENDED features (see 5_idle_clustering.py
#     for why raw-level clustering fails - it just splits pre/post the 2021
#     market spike instead of catching genuine local demand dips) ---
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df_ref["freight_roll_mean_60"] = df_ref[TARGET].rolling(60, min_periods=20).mean()
df_ref["freight_dev_pct"] = (df_ref[TARGET] - df_ref["freight_roll_mean_60"]) / df_ref["freight_roll_mean_60"]
df_ref["bdi_roll_mean_60"] = df_ref["bdi_index"].rolling(60, min_periods=20).mean()
df_ref["bdi_dev_pct"] = (df_ref["bdi_index"] - df_ref["bdi_roll_mean_60"]) / df_ref["bdi_roll_mean_60"]
cluster_df = df_ref.dropna(subset=["freight_dev_pct", "bdi_dev_pct"]).reset_index(drop=True)

X_cluster = StandardScaler().fit_transform(cluster_df[["freight_dev_pct", "bdi_dev_pct", "port_congestion_index"]])
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
cluster_df["cluster"] = kmeans.fit_predict(X_cluster)
low_demand_cluster = cluster_df.groupby("cluster")["freight_dev_pct"].mean().idxmin()
cluster_df["idle_risk_pred"] = (cluster_df["cluster"] == low_demand_cluster).astype(int)

current_idle_risk = bool(cluster_df["idle_risk_pred"].iloc[-1])
idle_rate_recent = cluster_df["idle_risk_pred"].tail(90).mean() * 100

df_ref["freight_pct_change"] = df_ref[TARGET].pct_change()
df_ref["congestion_pct_change"] = df_ref["port_congestion_index"].pct_change()
df_ref["freight_roll_mean_14"] = df_ref[TARGET].rolling(14, min_periods=5).mean()
df_ref["freight_dev_pct"] = (df_ref[TARGET] - df_ref["freight_roll_mean_14"]) / df_ref["freight_roll_mean_14"]
anom_feat = df_ref.dropna(subset=["freight_pct_change","congestion_pct_change","freight_dev_pct"])
iso = IsolationForest(contamination=0.045, random_state=42, n_estimators=200)
anom_pred = iso.fit_predict(anom_feat[["freight_pct_change","congestion_pct_change","port_congestion_index","freight_dev_pct"]])
current_anomaly = bool(anom_pred[-1] == -1)

c1, c2 = st.columns(2)
with c1:
    if current_idle_risk:
        st.warning(f"🟡 Idle-risk period detected (K-Means demand clustering). "
                    f"{idle_rate_recent:.0f}% of the last 90 days on this route were in the low-demand cluster.")
    else:
        st.info(f"🟢 Normal demand — no idle risk flagged currently. "
                 f"{idle_rate_recent:.0f}% of the last 90 days were low-demand.")
with c2:
    if current_anomaly:
        st.error("🔴 Anomaly flagged — unusual freight rate / congestion pattern on this route right now.")
    else:
        st.info("🟢 No anomaly detected currently.")

recent_window = df_ref[["date", TARGET, "port_congestion_index"]].tail(30).copy()
recent_window["date"] = recent_window["date"].dt.strftime("%Y-%m-%d")
st.caption("Latest 30-day freight and congestion readings")
render_table(recent_window)

# ---------------- Module 6: Multi-Parcel Fleet Allocation ----------------
st.header("6. Multi-Parcel Fleet Allocation")
st.caption("Plan several cargo parcels across different routes at once — the optimizer assigns "
           "the cost-minimizing vessel type to each, jointly, rather than one parcel at a time.")

import pulp

if "parcel_df" not in st.session_state:
    st.session_state.parcel_df = pd.DataFrame([
        {"route_id": route_id, "cargo_qty_tonnes": 55000},
        {"route_id": route_id, "cargo_qty_tonnes": 30000},
    ])

st.caption("Add, remove, or edit parcels before running the allocation.")
edited_parcels = st.data_editor(
    st.session_state.parcel_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "route_id": st.column_config.SelectboxColumn("Route", options=routes.route_id.tolist(), required=True),
        "cargo_qty_tonnes": st.column_config.NumberColumn("Cargo (tonnes)", min_value=5000, step=5000, required=True),
    },
    key="parcel_editor",
)

if st.button("🚢 Optimize fleet allocation across all parcels"):
    parcels = [
        {"parcel_id": f"P{i+1}", "route_id": row.route_id, "cargo_qty_tonnes": row.cargo_qty_tonnes}
        for i, row in edited_parcels.reset_index(drop=True).iterrows()
    ]

    # get predicted rates for every unique route involved (reuses the same cached forecaster as Module 1)
    rate_lookup = {}
    for r in {p["route_id"] for p in parcels}:
        route_preds = forecast_all_vessel_types(r)
        for vtype, vals in route_preds.items():
            rate_lookup[(r, vtype)] = vals["point"]

    prob = pulp.LpProblem("multi_parcel_allocation", pulp.LpMinimize)
    assign, eligibility = {}, {}
    for p in parcels:
        dest_code = routes.loc[routes.route_id == p["route_id"], "dest_port_code"].values[0]
        port_row = ports[ports.port_code == dest_code].iloc[0]
        for _, v in vessels.iterrows():
            key = (p["parcel_id"], v.vessel_type)
            eligibility[key] = (v.typical_draft_m <= port_row.max_draft_m and
                                v.typical_loa_m <= port_row.max_loa_m and
                                v.typical_beam_m <= port_row.max_beam_m and
                                v.dwt_max >= p["cargo_qty_tonnes"] * 0.9)
            assign[key] = pulp.LpVariable(f"assign_{p['parcel_id']}_{v.vessel_type}", cat="Binary")

    prob += pulp.lpSum(
        assign[(p["parcel_id"], v)] * (
            rate_lookup.get((p["route_id"], v), 1e9) * p["cargo_qty_tonnes"]
            if eligibility[(p["parcel_id"], v)] else 1e12
        )
        for p in parcels for v in vessels.vessel_type
    )
    for p in parcels:
        prob += pulp.lpSum(assign[(p["parcel_id"], v)] for v in vessels.vessel_type) == 1

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    rows_out, total_cost, infeasible_parcels = [], 0, []
    for p in parcels:
        chosen_list = [v for v in vessels.vessel_type if assign[(p["parcel_id"], v)].value() == 1]
        chosen = chosen_list[0] if chosen_list else None
        if chosen is None or not eligibility.get((p["parcel_id"], chosen), False):
            infeasible_parcels.append(p["parcel_id"])
            continue
        rate = rate_lookup.get((p["route_id"], chosen))
        cost = rate * p["cargo_qty_tonnes"]
        total_cost += cost
        rows_out.append({"parcel_id": p["parcel_id"], "route_id": p["route_id"],
                          "cargo_qty_tonnes": p["cargo_qty_tonnes"], "assigned_vessel": chosen,
                          "rate_per_tonne": rate, "cost_usd": round(cost, 2)})

    if infeasible_parcels:
        st.warning(f"⚠️ No eligible vessel type for parcel(s): {', '.join(infeasible_parcels)} — "
                   f"likely a shallow-draft destination port. These were excluded from the plan.")

    if rows_out:
        render_table(pd.DataFrame(rows_out))
        st.success(f"✅ Total fleet cost across {len(rows_out)} parcel(s): **${total_cost:,.0f}**")

st.caption("Model accuracy reference: mean MAPE 3.48% across 132 route/vessel combinations "
           "(see all_routes_forecast_summary.csv). Not a source of financial advice — validate against live market data before committing to a charter.")
