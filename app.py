"""Decision dashboard for proactive dry-bulk chartering into India's East Coast."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
from live_data import live_context
from charter_strategy import recommend_strategy
from alerts import build_alerts
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TARGET = "freight_rate_usd_per_tonne"
FEATURES = ["lag_1", "lag_7", "lag_14", "lag_30", "lag_90", "roll_mean_7", "roll_mean_30", "roll_std_30", "bdi_index", "coal_price_index", "port_congestion_index", "dow", "month"]

st.set_page_config(page_title="Charter Intelligence", page_icon="⚓", layout="wide")
st.markdown("""
<style>
    .stApp { background: radial-gradient(circle at 15% 5%, #173b58 0, #071827 38%, #06121e 100%); }
    h1, h2, h3 { color: #edf5fb !important; font-family: Georgia, serif; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #061521, #0e2f46); }
    [data-testid="stSidebar"] * { color: #eef6fc; }
    .hero { background: linear-gradient(105deg, #103b56, #0c263b 55%, #174d68); padding: 1.4rem 1.7rem; border-radius: 16px; border: 1px solid #3d7895; margin: .4rem 0 1.1rem 0; }
    .hero-kicker { color: #8ed6e9; letter-spacing: .12em; font-size: .73rem; font-weight: 700; }
    .hero-title { color: #ffffff; font-size: 1.55rem; font-weight: 700; margin: .2rem 0; }
    .hero-copy { color: #c8dbe6; margin: 0; }
    .strategy-card { background: linear-gradient(135deg, #0b3148, #104b59); padding: 1.25rem 1.45rem; border-radius: 14px; border-left: 5px solid #e4bd65; margin-bottom: 1rem; }
    .strategy-label { color: #f4cf77; font-size: .78rem; font-weight: 700; letter-spacing: .1em; }
    .strategy-name { color: #fff; font-size: 1.45rem; font-weight: 700; margin: .25rem 0; }
    .strategy-copy { color: #d4e4eb; }
    [data-testid="stMetric"] { background: rgba(15, 54, 75, .78); border: 1px solid rgba(124, 190, 210, .28); border-radius: 12px; padding: .8rem; }
    [data-testid="stMetricLabel"] { color: #b9d8e5 !important; }
    [data-testid="stMetricValue"] { color: #f5fbff !important; }
    .stTabs [data-baseweb="tab-list"] { gap: .35rem; }
    .stTabs [data-baseweb="tab"] { color: #b8d2df; background: #0b2638; border-radius: 8px 8px 0 0; }
    .stTabs [aria-selected="true"] { color: #fff !important; background: #1a5874 !important; }
    .stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return (
        pd.read_csv(DATA / "freight_rates_timeseries.csv", parse_dates=["date"]),
        pd.read_csv(DATA / "ports_east_coast_india.csv"),
        pd.read_csv(DATA / "loading_ports_origin.csv"),
        pd.read_csv(DATA / "vessel_types.csv"),
        pd.read_csv(DATA / "routes.csv"),
        pd.read_csv(ROOT / "all_routes_forecast_summary.csv"),
    )

def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for lag in [1, 7, 14, 30, 90]: out[f"lag_{lag}"] = out[TARGET].shift(lag)
    out["roll_mean_7"] = out[TARGET].shift(1).rolling(7).mean()
    out["roll_mean_30"] = out[TARGET].shift(1).rolling(30).mean()
    out["roll_std_30"] = out[TARGET].shift(1).rolling(30).std()
    out["dow"] = out.date.dt.dayofweek
    out["month"] = out.date.dt.month
    return out.dropna().reset_index(drop=True)

@st.cache_data(show_spinner=False)
def forecast_route(ts: pd.DataFrame, route_id: str, vessel: str, horizon: int = 60):
    df = add_features(ts[(ts.route_id == route_id) & (ts.vessel_type == vessel)].sort_values("date").reset_index(drop=True))
    train, latest = df.iloc[:-1], df.iloc[[-1]]
    model = xgb.XGBRegressor(n_estimators=220, max_depth=4, learning_rate=.05, subsample=.85, colsample_bytree=.85, random_state=42, verbosity=0)
    model.fit(train[FEATURES], train[TARGET])
    point = float(model.predict(latest[FEATURES])[0])
    residual = train[TARGET] - model.predict(train[FEATURES])
    uncertainty = float(np.quantile(abs(residual), .8))
    slope = float(np.polyfit(df.index[-30:], df[TARGET].tail(30), 1)[0])
    days = np.arange(1, horizon + 1)
    seasonal = .55 * np.sin((df.index[-1] + days) * 2 * np.pi / 30)
    projected = np.maximum(1, point + slope * days + seasonal)
    outlook = pd.DataFrame({"date": pd.date_range(df.date.iloc[-1] + pd.Timedelta(days=1), periods=horizon), "forecast_usd_t": projected, "p10": np.maximum(1, projected - uncertainty), "p90": projected + uncertainty})
    if HAS_SHAP:
        explainer = shap.TreeExplainer(model)
        values = np.asarray(explainer.shap_values(latest[FEATURES])).reshape(-1)
        explanation = "SHAP"
    else:
        # Safe fallback enables panel demos where optional SHAP is not installed.
        values = model.feature_importances_ * (latest[FEATURES].iloc[0] - train[FEATURES].mean())
        explanation = "feature-contribution fallback"
    impacts = pd.DataFrame({"driver": FEATURES, "impact": values}).sort_values("impact", key=abs, ascending=False).head(6)
    return point, uncertainty, outlook, impacts, explanation

def feasible(vessels, origin, dest, cargo):
    out = vessels.copy()
    for column in ["typical_draft_m", "typical_loa_m", "typical_beam_m"]:
        limit = min(origin[column.replace("typical_", "max_")], dest[column.replace("typical_", "max_")])
        out[f"{column}_ok"] = out[column] <= limit
    out["cargo_ok"] = out.dwt_max >= cargo * .90
    checks = ["typical_draft_m_ok", "typical_loa_m_ok", "typical_beam_m_ok", "cargo_ok"]
    out["eligible"] = out[checks].all(axis=1)
    out["utilisation"] = (cargo / out.dwt_max).clip(upper=1)
    return out

def just_in_time_plan(route, dest, vessel_type: str, congestion: int, fuel_price: int, berth_adjustment_hours: int):
    """Compare an early 12-knot arrival with a berth-aligned slow-steaming plan.

    These are transparent planning assumptions for the prototype; production
    should replace them with vessel noon reports and terminal berth events.
    """
    standard_speed_kn, minimum_speed_kn = 12.0, 9.0
    standard_hours = float(route.distance_nm) / standard_speed_kn
    baseline_wait_hours = float(dest.avg_pre_berthing_delay_days) * 24 * (1 + congestion / 100)
    berth_ready_hours = max(1.0, standard_hours + baseline_wait_hours + berth_adjustment_hours)
    required_speed_kn = float(route.distance_nm) / berth_ready_hours
    jit_speed_kn = min(standard_speed_kn, max(minimum_speed_kn, required_speed_kn))
    jit_hours = float(route.distance_nm) / jit_speed_kn
    early_wait_hours = max(0.0, berth_ready_hours - standard_hours)
    jit_wait_hours = max(0.0, berth_ready_hours - jit_hours)
    daily_fuel = {"Handysize": 18, "Supramax": 26, "Panamax": 32, "Capesize": 55}.get(vessel_type, 26)

    def fuel(speed_kn: float, sailing_hours: float, waiting_hours: float) -> float:
        sailing = daily_fuel * (speed_kn / standard_speed_kn) ** 3 * sailing_hours / 24
        anchorage = 3.5 * waiting_hours / 24
        return sailing + anchorage

    standard_fuel = fuel(standard_speed_kn, standard_hours, early_wait_hours)
    jit_fuel = fuel(jit_speed_kn, jit_hours, jit_wait_hours)
    saved_fuel = standard_fuel - jit_fuel
    risk = min(100, round(congestion * .55 + min(40, early_wait_hours / 2) + (15 if required_speed_kn < minimum_speed_kn else 0)))
    if required_speed_kn > standard_speed_kn:
        action = "Maintain 12 kn — the requested berth window is earlier than the standard sailing plan."
    elif required_speed_kn < minimum_speed_kn:
        action = f"Slow steam at {minimum_speed_kn:.1f} kn, then plan for {jit_wait_hours:.0f} hours at anchorage."
    else:
        action = f"Slow steam at {jit_speed_kn:.1f} kn to meet the predicted berth window Just-in-Time."
    return {
        "action": action, "berth_ready_days": berth_ready_hours / 24,
        "standard_speed": standard_speed_kn, "jit_speed": jit_speed_kn,
        "standard_sailing_days": standard_hours / 24, "jit_sailing_days": jit_hours / 24,
        "early_wait_hours": early_wait_hours, "jit_wait_hours": jit_wait_hours,
        "standard_fuel": standard_fuel, "jit_fuel": jit_fuel,
        "fuel_saved": saved_fuel, "cost_saved": saved_fuel * fuel_price,
        "co2_saved": saved_fuel * 3.114, "risk": risk,
    }

def main():
    ts, india_ports, origin_ports, vessels, routes, accuracy = load_data()
    st.title("⚓ Freight Chartering Decision Intelligence")
    st.caption("A decision cockpit for moving East Coast India bulk procurement from reactive spot fixtures to proactive multi-voyage chartering.")
    st.info("Freight target data is the project’s historical/synthetic series. Free APIs below enrich macro and weather risk only; they do not claim to be live fixture or AIS data.")

    with st.sidebar:
        st.header("Cargo mandate")
        dest_code = st.selectbox("Discharge port", india_ports.port_code, format_func=lambda x: india_ports.loc[india_ports.port_code == x, "port_name"].iat[0])
        route_options = routes[routes.dest_port_code == dest_code]
        route_id = st.selectbox("Origin → destination route", route_options.route_id)
        cargo = st.number_input("Cargo quantity (MT)", 5_000, 200_000, 55_000, 1_000)
        commodity = st.selectbox("Commodity", ["Thermal coal", "Coking coal", "Iron ore", "Limestone"])
        voyages = st.selectbox("Charter strategy", [1, 3, 6], format_func=lambda n: "Spot / one voyage" if n == 1 else f"{n}-voyage {'short' if n == 3 else 'medium'} term")
        laycan = st.date_input("Laycan start", date.today() + timedelta(days=21))
        congestion = st.slider("Congestion scenario (0–100)", 0, 100, 45)
        with st.expander("Just-in-Time port-call assumptions"):
            fuel_price = st.number_input("Bunker fuel price (USD/tonne)", 300, 1200, 650, 25)
            berth_adjustment = st.slider("Berth-window adjustment (hours)", -72, 168, 0,
                                         help="Model a berth delay or early slot without claiming live terminal data.")
        with st.expander("Commercial assumptions"):
            discount_3 = st.slider("3-voyage commitment discount", 0.0, 10.0, 2.5, .5) / 100
            discount_6 = st.slider("6-voyage commitment discount", 0.0, 12.0, 5.0, .5) / 100
            st.caption("Change these assumptions during the panel discussion to show sensitivity.")

    route = routes[routes.route_id == route_id].iloc[0]
    origin = origin_ports[origin_ports.port_code == route.origin_port_code].iloc[0]
    dest = india_ports[india_ports.port_code == dest_code].iloc[0]
    eligibility = feasible(vessels, origin, dest, cargo)
    candidates = eligibility[eligibility.eligible].copy()
    if candidates.empty:
        st.error("No vessel class fits both ports and this parcel. Split cargo, select another port, or validate lighterage/terminal approval.")
        st.dataframe(eligibility, hide_index=True)
        st.stop()

    predictions = {}
    for vessel in candidates.vessel_type:
        predictions[vessel] = forecast_route(ts, route_id, vessel)
    candidates["forecast_rate"] = candidates.vessel_type.map(lambda v: predictions[v][0])
    candidates["total_freight"] = candidates.forecast_rate * cargo
    candidates["fit_score"] = 100 - abs(candidates.utilisation - .85) * 65
    candidates["decision_score"] = candidates.fit_score - candidates.forecast_rate.rank(pct=True) * 10
    best = candidates.sort_values("decision_score", ascending=False).iloc[0]
    rate, uncertainty, outlook, impacts, explanation = predictions[best.vessel_type]
    trough = outlook.loc[outlook.forecast_usd_t.idxmin()]
    load_days = cargo / origin.cargo_handling_rate_tpd + 1.2 * (1 + congestion / 100)
    discharge_days = cargo / dest.cargo_handling_rate_tpd + dest.avg_pre_berthing_delay_days * (1 + congestion / 100)
    sailing_days = route.distance_nm / (12 * 24)
    cycle = sailing_days + load_days + discharge_days
    risk = min(100, round(20 + congestion * .5 + (uncertainty / rate) * 100))
    spot_rate = float(outlook.forecast_usd_t.iloc[0])
    scenario_rows = []
    for n, discount, reserve in [(1, 0, 0), (3, discount_3, .015), (6, discount_6, .025)]:
        contract_rate = rate * (1 - discount)
        all_in = contract_rate * cargo * n * (1 + reserve)
        repeated_spot = spot_rate * cargo * n
        scenario_rows.append([n, contract_rate, all_in, repeated_spot - all_in, cycle * n])
    scenarios = pd.DataFrame(scenario_rows, columns=["voyages", "contract_rate_usd_mt", "all_in_cost_usd", "saving_vs_repeated_spot_usd", "planned_cycle_days"])
    selected_scenario = scenarios[scenarios.voyages == voyages].iloc[0]
    validation = accuracy[(accuracy.route_id == route_id) & (accuracy.vessel_type == best.vessel_type)]
    mape = float(validation.iloc[0].mape_pct) if not validation.empty else np.nan
    strategy = recommend_strategy(scenarios, outlook, laycan, risk, cargo, best.vessel_type, cycle)
    jit = just_in_time_plan(route, dest, best.vessel_type, congestion, fuel_price, berth_adjustment)
    weather_context = None
    if "live" in st.session_state:
        weather_context = {"loading port": st.session_state["live"].get("origin_marine"), "discharge port": st.session_state["live"].get("destination_marine")}
    alerts = build_alerts(
        risk_index=risk, congestion=congestion, port_days=load_days + discharge_days,
        cycle_days=cycle, utilisation=float(best.utilisation), volatility_pct=strategy["volatility_pct"],
        entry_date=strategy["entry_date"], laycan=laycan, weather_context=weather_context,
    )
    alert_count = sum(a["severity"] in {"Critical", "Warning"} for a in alerts)

    st.markdown(f"""
    <div class="hero">
      <div class="hero-kicker">CHARTERING DECISION SYSTEM · {origin.country.upper()} → INDIA EAST COAST</div>
      <div class="hero-title">Turn one cargo mandate into a protected charter strategy.</div>
      <p class="hero-copy">Route: {origin.port_name} → {dest.port_name} &nbsp;|&nbsp; Cargo: {cargo:,.0f} MT {commodity} &nbsp;|&nbsp; Laycan: {laycan:%d %b %Y}</p>
    </div>
    """, unsafe_allow_html=True)

    a, b, c, d = st.columns(4)
    a.metric("Recommended vessel", best.vessel_type, f"{best.utilisation:.0%} parcel utilisation")
    b.metric("Expected freight", f"${rate:.2f}/MT", f"± ${uncertainty:.2f} model residual band")
    c.metric("Voyage cycle", f"{cycle:.1f} days", f"Sailing {sailing_days:.1f} days")
    d.metric("Active alerts", alert_count, "Review required" if alert_count else "No critical trigger")

    tabs = st.tabs(["Executive brief", "Strategy engine", "JIT port call", "Forecast & XAI", "Port feasibility", "Multi-voyage plan", "Risk & resilience", "Live data & governance"])
    with tabs[0]:
        st.subheader("Decision for approval")
        st.markdown(f"""
        <div class="strategy-card">
          <div class="strategy-label">RECOMMENDED COMMERCIAL POSTURE</div>
          <div class="strategy-name">{strategy['posture']}</div>
          <div class="strategy-copy">{strategy['action']}<br><br><b>Why:</b> {strategy['rationale']}</div>
        </div>
        """, unsafe_allow_html=True)
        e1, e2, e3 = st.columns(3)
        e1.metric("Strategy contract exposure", f"${strategy['expected_cost']:,.0f}")
        e2.metric("Avoided repeated-spot cost", f"${strategy['expected_saving']:,.0f}")
        e3.metric("Forecast validation", f"{mape:.2f}% MAPE" if not np.isnan(mape) else "Not available")
        st.markdown("**Why this stands out:** one screen connects a market forecast to a vessel choice, a port-feasibility gate, a multi-voyage commercial decision and an explainable risk narrative.")
        st.write(f"Estimated port time is **{load_days + discharge_days:.1f} days** and total cycle time is **{cycle:.1f} days**. This makes idle exposure visible before the fixture—not after demurrage starts accruing.")
        st.markdown("**Panel talking point:** the dashboard is not a price-guessing tool. It is a governed decision workflow that quantifies the trade-off between rate, vessel fit, port constraints, reliability and contract duration.")
        st.subheader("Early-warning alerts")
        for alert in alerts[:3]:
            message = f"**{alert['alert']}** — {alert['evidence']}  \n**Recommended action:** {alert['action']}"
            if alert["severity"] == "Critical": st.error(message)
            elif alert["severity"] == "Warning": st.warning(message)
            else: st.info(message)

    with tabs[1]:
        st.subheader("Commercial strategy engine")
        st.write("The engine converts forecast, port fit, market uncertainty and laycan feasibility into an explicit charter posture. It prioritizes procurement savings while protecting supply reliability.")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Recommended cover", f"{strategy['recommended_voyages']} voyages")
        s2.metric("Preferred entry", strategy["entry_date"].strftime("%d %b"), f"${strategy['entry_rate']:.2f}/MT")
        s3.metric("Wait-versus-fix opportunity", f"{strategy['rate_gap_pct']:.1f}%")
        s4.metric("Forecast spread", f"{strategy['volatility_pct']:.1f}%")
        st.subheader("Strategy safeguards to negotiate")
        safeguard = pd.DataFrame({"contract protection": strategy["protections"], "business purpose": ["Limits freight-market upside while retaining fair-market linkage", "Protects supply continuity during berth/terminal disruption", "Prevents avoidable demurrage and idle-time disputes", "Reduces deadheading and vessel idle cost"]})
        st.dataframe(safeguard, hide_index=True, use_container_width=True)
        st.caption("Recommendation is decision support only. Chartering desk approval, vessel vetting, sanctions/compliance checks and terminal confirmation remain mandatory.")

    with tabs[2]:
        st.subheader("Just-in-Time arrival recommendation")
        st.success(f"⚓ **{jit['action']}**")
        j1, j2, j3, j4 = st.columns(4)
        j1.metric("Predicted berth-ready", f"{jit['berth_ready_days']:.1f} days")
        j2.metric("Anchorage time avoided", f"{max(0, jit['early_wait_hours'] - jit['jit_wait_hours']):.0f} h")
        j3.metric("Estimated fuel saving", f"{max(0, jit['fuel_saved']):.1f} t", f"${max(0, jit['cost_saved']):,.0f}")
        risk_label = "High" if jit["risk"] >= 70 else ("Medium" if jit["risk"] >= 40 else "Low")
        j4.metric("CO₂ avoided", f"{max(0, jit['co2_saved']):.1f} t", f"{risk_label} port-call risk")
        jit_table = pd.DataFrame([
            ["Arrive at 12 kn", jit["standard_speed"], jit["standard_sailing_days"], jit["early_wait_hours"], jit["standard_fuel"], jit["standard_fuel"] * fuel_price],
            ["Just-in-Time plan", jit["jit_speed"], jit["jit_sailing_days"], jit["jit_wait_hours"], jit["jit_fuel"], jit["jit_fuel"] * fuel_price],
        ], columns=["plan", "speed_kn", "sailing_days", "anchorage_hours", "fuel_tonnes", "fuel_cost_usd"])
        st.dataframe(jit_table.round(1), hide_index=True, use_container_width=True)
        st.info(
            f"**Why this action?** Congestion is {congestion}/100 and {dest.port_name}'s baseline pre-berthing delay is "
            f"{dest.avg_pre_berthing_delay_days:.1f} days. Fuel uses transparent vessel-class assumptions and cubic speed–fuel scaling. "
            "In production, connect terminal berth events and vessel noon reports."
        )

    with tabs[3]:
        st.subheader("60-day rate outlook")
        recent = ts[(ts.route_id == route_id) & (ts.vessel_type == best.vessel_type)].sort_values("date").tail(90)[["date", TARGET]].rename(columns={TARGET: "observed_history"})
        chart = recent.set_index("date").join(outlook.set_index("date")[["p10", "forecast_usd_t", "p90"]], how="outer")
        st.line_chart(chart, use_container_width=True)
        st.caption("Band is an empirical 80% residual interval from the current XGBoost model—not a guaranteed market quote.")
        weekly = outlook.assign(week=outlook.date.dt.to_period("W").astype(str)).groupby("week", as_index=False).agg(min_rate=("forecast_usd_t", "min"), average_rate=("forecast_usd_t", "mean"), earliest_date=("date", "min")).head(6)
        weekly["action"] = np.where(weekly.min_rate <= outlook.forecast_usd_t.quantile(.30), "Preferred entry window", "Monitor / defer")
        st.subheader("Market-entry windows")
        st.dataframe(weekly, hide_index=True, use_container_width=True)
        st.subheader("Explainable AI — local forecast drivers")
        impacts = impacts.set_index("driver")
        st.bar_chart(impacts, use_container_width=True)
        st.caption(f"Local {explanation} values for the selected XGBoost forecast: positive values raise the predicted freight rate; negative values reduce it. Install the declared `shap` package for exact SHAP values.")
        if not validation.empty: st.caption(f"Historical validation reference for this route/vessel: MAPE {validation.iloc[0].mape_pct}%, RMSE {validation.iloc[0].rmse}.")

    with tabs[4]:
        st.subheader("Two-port constraint gate")
        st.dataframe(pd.DataFrame([origin, dest])[ ["port_name", "max_draft_m", "max_loa_m", "max_beam_m", "cargo_handling_rate_tpd"] ], hide_index=True, use_container_width=True)
        show = eligibility[["vessel_type", "dwt_max", "typical_draft_m", "typical_loa_m", "typical_beam_m", "utilisation", "eligible"]].copy()
        show.utilisation = show.utilisation.map("{:.0%}".format)
        st.dataframe(show, hide_index=True, use_container_width=True)
        st.warning("Planning limits only: confirm berth nomination, tidal/seasonal draft, terminal acceptance, cargo density and stowage with the port/chartering desk before fixing.")

    with tabs[5]:
        st.subheader("Spot versus multiple-voyage contract strategy")
        st.dataframe(scenarios, hide_index=True, use_container_width=True)
        st.bar_chart(scenarios.set_index("voyages")[["all_in_cost_usd"]], use_container_width=True)
        st.success(f"Selected {voyages}-voyage scenario: indicative saving vs repeated spot entry **${selected_scenario.saving_vs_repeated_spot_usd:,.0f}**. This is a planning scenario, not a price guarantee.")
        st.write("Idle-management clause: define congestion trigger, alternate-port option, and next-employment/ballast plan before signing the multi-voyage agreement.")

    with tabs[6]:
        st.subheader("Early-warning and resilience plan")
        st.dataframe(pd.DataFrame(alerts), hide_index=True, use_container_width=True)
        risk_rows = pd.DataFrame([
            ["Freight volatility", f"± ${uncertainty:.2f}/MT residual band", "Use rate cap, index-linked clause or staggered fixtures"],
            ["Port congestion", f"Scenario score {congestion}/100; {load_days + discharge_days:.1f} port days", "Agree congestion trigger, alternate port and demurrage allocation"],
            ["Vessel/parcel mismatch", f"{best.utilisation:.0%} utilisation in selected class", "Aggregate/split parcels if utilisation moves outside the commercial band"],
            ["Weather disruption", "Optional live marine signal available", "Refresh API context before fixture and monitor sailing/berthing windows"],
        ], columns=["risk", "current signal", "pre-agreed action"])
        st.dataframe(risk_rows, hide_index=True, use_container_width=True)
        if risk >= 60:
            st.error("Risk threshold breached: do not rely on a single port or a single fixing date. Use a staged fixing plan and alternate employment/positioning clause.")
        else:
            st.success("Risk is within the selected appetite. Re-run the scenario when congestion or expected cargo readiness changes materially.")

    with tabs[7]:
        st.subheader("Free API context — optional enrichment")
        if st.button("Refresh World Bank + Open-Meteo context"):
            st.session_state["live"] = live_context(route.origin_port_code, dest_code, route.origin_country)
            st.rerun()
        if "live" in st.session_state:
            live = st.session_state["live"]
            st.caption(f"Retrieved {live['retrieved_at']}")
            st.json(live)
        else:
            st.caption("Click refresh to retrieve keyless public context. The model remains runnable without internet.")
        st.markdown("**Production governance:** retain source, timestamp, licence, quality flag and owner for every feature. Licensed broker/Baltic and AIS data are required for operational freight quotes and live congestion; free sources must not be presented as substitutes.")
        st.download_button("Download selected multi-voyage scenario", scenarios.to_csv(index=False).encode(), "charter_scenario.csv", "text/csv")

if __name__ == "__main__":
    main()
