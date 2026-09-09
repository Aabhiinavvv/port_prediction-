"""
Module 3 — Idle Scenario Management
K-Means clusters days into "high-demand" vs "low-demand/idle-risk" states
using freight_rate + bdi + congestion. We then check how well the
unsupervised clusters line up with the ground-truth idle_risk_flag label
(built with a rolling z-score rule) as a sanity check.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path

ts = pd.read_csv(Path(__file__).resolve().parent / "data" / "freight_rates_timeseries.csv", parse_dates=["date"])
ROUTE, VESSEL = "AUNTL_INPAR", "Handysize"
df = ts[(ts.route_id == ROUTE) & (ts.vessel_type == VESSEL)].sort_values("date").reset_index(drop=True)

FEATURES = ["freight_rate_usd_per_tonne", "bdi_index", "port_congestion_index"]

# ---- KEY FIX: cluster on DETRENDED/RELATIVE features, not raw levels.
#      Raw freight_rate carries the multi-year upward trend + 2021 spike, so
#      raw-level clustering just separates "pre/post spike era" instead of
#      "locally weak demand" - which is what idle-risk actually means. ----
df["freight_roll_mean_60"] = df["freight_rate_usd_per_tonne"].rolling(60, min_periods=20).mean()
df["freight_dev_pct"] = (df["freight_rate_usd_per_tonne"] - df["freight_roll_mean_60"]) / df["freight_roll_mean_60"]
df["bdi_roll_mean_60"] = df["bdi_index"].rolling(60, min_periods=20).mean()
df["bdi_dev_pct"] = (df["bdi_index"] - df["bdi_roll_mean_60"]) / df["bdi_roll_mean_60"]
df = df.dropna(subset=["freight_dev_pct", "bdi_dev_pct"]).reset_index(drop=True)

FEATURES = ["freight_dev_pct", "bdi_dev_pct", "port_congestion_index"]
X = StandardScaler().fit_transform(df[FEATURES])

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X)

# Which cluster corresponds to LOW demand? -> the one with lower mean freight_rate
cluster_means = df.groupby("cluster")["freight_dev_pct"].mean()
low_demand_cluster = cluster_means.idxmin()
df["predicted_idle_risk"] = (df["cluster"] == low_demand_cluster).astype(int)

# ---- Validate against ground-truth idle_risk_flag ----
agreement = (df["predicted_idle_risk"] == df["idle_risk_flag"]).mean()
print(f"Cluster-vs-ground-truth agreement: {agreement*100:.1f}%")
print(f"\nGround-truth idle-risk rate : {df['idle_risk_flag'].mean()*100:.1f}%")
print(f"Cluster-predicted idle rate : {df['predicted_idle_risk'].mean()*100:.1f}%")

print("\nCross-tab (rows=ground truth, cols=cluster prediction):")
print(pd.crosstab(df["idle_risk_flag"], df["predicted_idle_risk"]))

# ---- What to actually DO during a predicted idle period ----
# Recommend the historically best-paying alternate route the same vessel type
# could have served during past low-demand windows (a simple lookup demo).
idle_days = df[df["predicted_idle_risk"] == 1]
print(f"\n{len(idle_days)} idle-risk days identified.")
print("Sample idle-risk dates:", idle_days["date"].dt.date.head(5).tolist())
