"""Walk-forward ablation: A baseline, B + public BDI, C + weather, D + commodities.

The freight target is still the repository's prototype benchmark target. The
public signals are evaluated for incremental predictive value; this script
never represents the synthetic target as proprietary fixture observations.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TARGET = "freight_rate_usd_per_tonne"
BASE = ["lag_1","lag_7","lag_14","lag_30","lag_90","roll_mean_7","roll_mean_30","roll_std_30","port_congestion_index","dow","month"]
BDI = ["bdi","bdi_ma7","bdi_ma30","bdi_ma90","bdi_return_7d","bdi_return_30d","bdi_volatility_30d"]
WEATHER = ["weather_disruption_mean","weather_disruption_max","extreme_weather_port_count","precipitation_mean_mm","wind_max_kn","gust_max_kn"]
COMMODITIES = ["coal_australian","coal_south_african","iron_ore","crude_oil"]


def score(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"MAE": float(mean_absolute_error(y,p)), "RMSE": float(mean_squared_error(y,p)**0.5), "MAPE_pct": float(np.mean(np.abs((y-p)/np.maximum(np.abs(y),1e-6)))*100)}


def prepare():
    f = pd.read_csv(DATA/"freight_rates_timeseries.csv", parse_dates=["date"])
    p = pd.read_csv(DATA/"public_processed"/"public_exogenous_features.csv", parse_dates=["date"]).drop_duplicates("date")
    df = f.merge(p, on="date", how="left", validate="many_to_one").sort_values(["route_id","vessel_type","date"]).copy()
    g = df.groupby(["route_id","vessel_type"])[TARGET]
    for lag in (1,7,14,30,90): df[f"lag_{lag}"] = g.shift(lag)
    df["roll_mean_7"] = g.transform(lambda s:s.shift(1).rolling(7).mean())
    df["roll_mean_30"] = g.transform(lambda s:s.shift(1).rolling(30).mean())
    df["roll_std_30"] = g.transform(lambda s:s.shift(1).rolling(30).std())
    df["dow"], df["month"] = df.date.dt.dayofweek, df.date.dt.month
    return df


def run_model(train, test, features):
    # Same model, rows and chronological split for each ablation where possible.
    tr = train.dropna(subset=features+[TARGET]); te = test.dropna(subset=features+[TARGET])
    if len(te)==0 or len(tr)==0: return None, len(tr), len(te)
    model=xgb.XGBRegressor(n_estimators=350,max_depth=5,learning_rate=.05,subsample=.8,colsample_bytree=.8,random_state=42,objective="reg:squarederror",verbosity=0)
    model.fit(tr[features],tr[TARGET]); pred=model.predict(te[features])
    return score(te[TARGET],pred),len(tr),len(te)


def main():
    df=prepare()
    dates=np.sort(df.date.unique())
    # Five chronological folds; each test window is the next 90 calendar days.
    cut_dates=[]
    for q in [0.60,0.68,0.76,0.84,0.92]:
        i=min(len(dates)-91,max(1,int(len(dates)*q)))
        cut_dates.append(pd.Timestamp(dates[i]))
    groups={"A_baseline":BASE,"B_baseline_plus_BDI":BASE+BDI,"C_plus_weather":BASE+BDI+WEATHER,"D_plus_commodities":BASE+BDI+WEATHER+COMMODITIES}
    rows=[]
    for fold,start in enumerate(cut_dates,1):
        end=start+pd.Timedelta(days=89)
        train=df[df.date<start]; test=df[(df.date>=start)&(df.date<=end)]
        for name,features in groups.items():
            available=[c for c in features if c in df.columns]
            missing=[c for c in features if c not in df.columns]
            # Require all requested features. This prevents silently turning D into C.
            if missing:
                rows.append({"fold":fold,"model":name,"status":"missing_columns","missing":",".join(missing),"MAE":np.nan,"RMSE":np.nan,"MAPE_pct":np.nan,"train_rows":0,"test_rows":0})
                continue
            r,ntr,nte=run_model(train,test,available)
            rows.append({"fold":fold,"model":name,"status":"ok" if r else "no_rows","missing":"",**(r or {"MAE":np.nan,"RMSE":np.nan,"MAPE_pct":np.nan}),"train_rows":ntr,"test_rows":nte})
    out=pd.DataFrame(rows)
    outdir=DATA/"public_processed"; outdir.mkdir(exist_ok=True)
    out.to_csv(outdir/"public_ablation_fold_metrics.csv",index=False)
    summary=out[out.status=="ok"].groupby("model").agg(folds=("fold","count"),MAE_mean=("MAE","mean"),RMSE_mean=("RMSE","mean"),MAPE_mean=("MAPE_pct","mean"),MAPE_median=("MAPE_pct","median"),MAPE_worst=("MAPE_pct","max")).reset_index()
    summary.to_csv(outdir/"public_ablation_summary.csv",index=False)
    coverage={c: int(df[c].notna().sum()) if c in df.columns else 0 for c in BDI+WEATHER+COMMODITIES}
    meta={"target_provenance":"prototype_synthetic_benchmark","folds":5,"test_window_days":90,"coverage_non_null_rows":coverage,"note":"A/B/C/D use identical chronological folds. D is only valid when all commodity columns are actually present."}
    (outdir/"public_ablation_metadata.json").write_text(json.dumps(meta,indent=2))
    print(summary.to_string(index=False))
    print("\nCoverage:", json.dumps(coverage,indent=2))

if __name__=="__main__": main()
