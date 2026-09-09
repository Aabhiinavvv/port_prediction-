# Freight Forecasting Dataset — Data Dictionary

5 linked CSV files. Join key across all of them is `port_code` / `route_id`.

## 1. ports_east_coast_india.csv (REAL — verify against live Port Trust notices)
| Column | Description |
|---|---|
| port_code | Short code (e.g. INPAR = Paradip) |
| port_name | Full port name |
| state | Indian state |
| max_draft_m | Max permissible draft in metres |
| max_loa_m | Max Length Overall a vessel can have |
| max_beam_m | Max beam (width) |
| dry_bulk_berths | Number of dry-bulk-capable berths |
| cargo_handling_rate_tpd | Tonnes per day discharge capacity |
| avg_pre_berthing_delay_days | Typical waiting time before berthing |

## 2. loading_ports_origin.csv (REAL, approximate)
Same structure as above but for origin-country loading ports (Australia, US, Mozambique, Indonesia, Russia).

## 3. vessel_types.csv (REAL, industry-standard classes)
| Column | Description |
|---|---|
| vessel_type | Handysize / Supramax / Panamax / Capesize |
| dwt_min, dwt_max | Deadweight tonnage range |
| typical_loa_m, typical_beam_m, typical_draft_m | Standard dimensions for that class |

## 4. routes.csv (distances approximate — refine with a sea-distance calculator)
| Column | Description |
|---|---|
| route_id | `{origin_port_code}_{dest_port_code}` |
| origin_port_code, dest_port_code | Join keys |
| distance_nm | Sailing distance in nautical miles |
| origin_country, dest_port_name | Convenience labels |

## 5. freight_rates_timeseries.csv (SYNTHETIC — this is what you train on)
**v2 — daily granularity**, 2019-01-01 to 2025-12-31, one row per (route × vessel_type × day) = **337,524 rows** across 33 routes.

| Column | Description |
|---|---|
| date | ISO date |
| route_id | Join to routes.csv |
| origin_port_code, dest_port_code | Join keys |
| vessel_type | Handysize / Supramax / Panamax / Capesize |
| distance_nm | Sailing distance |
| freight_rate_usd_per_tonne | **Target variable** for forecasting models |
| bdi_index | Simulated Baltic Dry Index-like market index (base 100) |
| coal_price_index | Simulated coal commodity price index (base 100) |
| port_congestion_index | 0-100 simulated congestion level at destination port |
| cargo_qty_tonnes | Simulated cargo parcel size for that vessel type |
| idle_risk_flag | 1 if this route+vessel is in a low-demand/idle-risk period (rolling 60-day z-score of freight rate < -1.3). Feeds the idle scenario management module. ~17% positive rate. |
| anomaly_flag | 1 if a ground-truth injected anomaly event is active on this day (see below). ~4.2% positive rate. Use this to *evaluate* your Isolation Forest / Z-score detector against a known answer, not just eyeball results. |
| anomaly_type | `freight_rate_shock` (market disruption — geopolitical, canal blockage, demand collapse), `port_congestion_spike` (strike, weather, equipment breakdown at destination), `both`, or `none` |

### Ground-truth anomaly injection logic
- **Port congestion spikes**: ~4-8 random events per route, each lasting 3-12 days, congestion index bumped up 30-60 points — mimics strikes/weather/breakdowns.
- **Freight-rate shocks**: ~3-6 random events per route, each lasting 5-18 days, freight rate multiplied 1.4x-2.2x (spike, ~75% of events) or 0.5x-0.65x (demand collapse, ~25% of events) — mimics geopolitical disruption, canal blockages, or sudden demand drops.
- These are separate from the *unconditional* 2021-style market-wide BDI/coal spike baked into the whole series — that one is not flagged as an "anomaly" since it's a real macro regime shift, not a point event. Point out this distinction to judges: your model should treat regime shifts and point anomalies differently.

### How it was generated (so you can defend it to judges)
- `coal_price_index`: mean-reverting random walk + an injected 2021-style energy-crisis spike (weeks ~115-155), mimicking the real global coal price surge.
- `bdi_index`: correlated with coal price changes, has GARCH-style volatility clustering (calm periods vs turbulent periods), plus the same 2021 spike carried through — mimicking real BDI behaviour.
- `freight_rate_usd_per_tonne` = `vessel_base_rate × (distance/1000) × (bdi/100) × (1 + 0.15×congestion/100) + slow route drift + weekly noise`.
- `port_congestion_index`: port-specific baseline (Haldia/Kolkata highest — shallow draft, more waiting) + mild yearly seasonality + noise.

### Known limitation — be upfront about this in your SIH pitch
This is a **calibrated synthetic dataset**, not scraped real freight data (Baltic Exchange/Clarksons/Drewry panel data is paid and licensed). The generation logic is built to reproduce the *statistical properties* real freight markets have — trend, seasonality, volatility clustering, commodity co-movement, occasional shocks — so your ARIMA/LSTM/XGBoost pipeline, optimizer, and anomaly detector all have something realistic to learn from and can be demoed end-to-end. For production, swap this file only, keeping the same schema, with:
- Baltic Exchange / Clarksons subscription data, or
- DGCIS India coal import records + DG Shipping vessel movement data, or
- DGCIS + DG Shipping / AIS congestion feed if you get access during the hackathon mentorship round.

## Suggested next steps for modeling
- Forecasting target: `freight_rate_usd_per_tonne`, grouped by `route_id` + `vessel_type` (337k daily rows across 33 routes — enough for LSTM per-series or a pooled/global model)
- Features: lags of freight_rate (1,7,30,90 days), bdi_index, coal_price_index, port_congestion_index, distance_nm, month/seasonality dummies
- Vessel optimization: join `routes.csv` + `vessel_types.csv` + `ports_east_coast_india.csv` on draft/LOA/beam to build the eligibility filter
- Idle scenario management: `idle_risk_flag` is your training/evaluation label — treat it as a classification target, or as validation for a clustering approach
- Risk/anomaly detection: `anomaly_flag` + `anomaly_type` are ground-truth labels — train a supervised classifier (or evaluate an unsupervised Isolation Forest/Z-score detector) against them, report precision/recall
