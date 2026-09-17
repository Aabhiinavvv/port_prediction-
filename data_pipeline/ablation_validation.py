"""Ablation study for public-data freight forecasting inputs.

Compares four feature sets on the same chronological holdout:
A. Existing prototype inputs
B. + public BDI
C. + public BDI + Open-Meteo weather
D. + public BDI + weather + World Bank commodities (when available)

Important: the current freight target is the repository's prototype benchmark,
not observed fixture pricing. Results measure feature usefulness against that
benchmark and must not be presented as real-market forecasting accuracy.
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
BASE = ["lag_1","lag_7","lag_14","lag_30","lag_90","roll_mean_7","roll_mean_30","roll_std_30","bdi_index","port_congestion_index","dow","month"]
PUBLIC_BDI = ["bdi","bdi_ma7","bdi_ma30","bdi_ma90","bdi_return_7d","bdi_return_30d","bdi_volatility_30d"]
WEATHER = ["weather_disruption_mean","weather_disruption_max","extreme_weather_port_count","precipitation_mean_mm","wind_max_kn","gust_max_kn"]
COMMODITIES = ["coal_australian","coal_south_african","iron_ore","crude_oil"]


def add_target_features(x):
    x=x.sort_values(["route_id","vessel_type","date"]).copy()
    g=x.groupby(["route_id","vessel_type"])[TARGET]
    for lag in (1,7,14,30,90): x[f"lag_{lag}"]=g.shift(lag)
    x["roll_mean_7"]=g.transform(lambda s:s.shift(1).rolling(7).mean())
    x["roll_mean_30"]=g.transform(lambda s:s.shift(1).rolling(30).mean())
    x["roll_std_30"]=g.transform(lambda s:s.shift(1).rolling(30).std())
    x["dow"]=x.date.dt.dayofweek; x["month"]=x.date.dt.month
    return x


def load_public():
    b=pd.read_csv(DATA/"public_raw"/"bdi_public.csv",parse_dates=["date"])[["date","bdi"]].drop_duplicates("date")
    b=b.sort_values("date")
    for w in (7,30,90): b[f"bdi_ma{w}"]=b.bdi.rolling(w,min_periods=w).mean()
    b["bdi_return_7d"]=b.bdi.pct_change(7); b["bdi_return_30d"]=b.bdi.pct_change(30)
    b["bdi_volatility_30d"]=b.bdi.pct_change().rolling(30).std()
    w=pd.read_csv(DATA/"public_processed"/"weather_public_features.csv",parse_dates=["date"])
    w=w.groupby("date",as_index=False).agg(weather_disruption_mean=("weather_disruption_score","mean"),weather_disruption_max=("weather_disruption_score","max"),extreme_weather_port_count=("extreme_weather_flag","sum"),precipitation_mean_mm=("precipitation_sum","mean"),wind_max_kn=("wind_speed_10m_max","max"),gust_max_kn=("wind_gusts_10m_max","max"))
    c=pd.DataFrame(columns=["date"])
    for name in ("world_bank_commodities.csv","world_bank_pink_sheet.csv"):
        p=DATA/"public_raw"/name
        if p.exists():
            raw=pd.read_csv(p); dc=next((z for z in raw.columns if z.lower() in {"date","month"}),None)
            if dc:
                raw["date"]=pd.to_datetime(raw[dc],errors="coerce").dt.to_period("D").dt.to_timestamp()
                aliases={"coal_australian":["Coal, Australian"],"coal_south_african":["Coal, South African"],"iron_ore":["Iron ore, cfr spot"],"crude_oil":["Crude oil, average"]}
                low={str(z).strip().lower():z for z in raw.columns}; cols=["date"]
                for out,names in aliases.items():
                    hit=next((low.get(n.lower()) for n in names if n.lower() in low),None)
                    if hit: raw[out]=pd.to_numeric(raw[hit],errors="coerce"); cols.append(out)
                c=raw[cols].dropna(subset=["date"]).groupby("date",as_index=False).mean(numeric_only=True); break
    return b,w,c


def score(y,p):
    y=np.asarray(y);p=np.asarray(p)
    return mean_absolute_error(y,p),mean_squared_error(y,p)**.5,np.mean(np.abs((y-p)/np.maximum(np.abs(y),1e-6)))*100


def main():
    freight=pd.read_csv(DATA/"freight_rates_timeseries.csv",parse_dates=["date"])
    b,w,c=load_public(); x=freight.merge(b,on="date",how="left").merge(w,on="date",how="left").merge(c,on="date",how="left")
    x=add_target_features(x)
    # Public features must be available at prediction time. Do not forward-fill.
    sets={"A_existing":["lag_1","lag_7","lag_14","lag_30","lag_90","roll_mean_7","roll_mean_30","roll_std_30","bdi_index","port_congestion_index","dow","month"],
          "B_public_BDI": [z for z in BASE if z!="bdi_index"]+["bdi","bdi_ma7","bdi_ma30","bdi_ma90","bdi_return_7d","bdi_return_30d","bdi_volatility_30d","port_congestion_index"],
          "C_BDI_weather": [z for z in BASE if z!="bdi_index"]+PUBLIC_BDI+WEATHER+["port_congestion_index"],
          "D_all_public": [z for z in BASE if z!="bdi_index"]+PUBLIC_BDI+WEATHER+COMMODITIES+["port_congestion_index"]}
    # The baseline's bdi_index is the synthetic input; B-D deliberately use public BDI.
    rows=[]; predictions=[]
    split=pd.Timestamp("2025-01-01")
    for name,features in sets.items():
        available=[z for z in features if z in x.columns]
        model_df=x.dropna(subset=available+[TARGET]).copy()
        train=model_df[model_df.date<split]; test=model_df[model_df.date>=split]
        if len(train)==0 or len(test)==0: continue
        model=xgb.XGBRegressor(n_estimators=350,max_depth=5,learning_rate=.05,subsample=.8,colsample_bytree=.8,random_state=42,objective="reg:squarederror",verbosity=0)
        model.fit(train[available],train[TARGET]); p=model.predict(test[available])
        mae,rmse,mape=score(test[TARGET],p)
        rows.append({"model":name,"features":len(available),"train_rows":len(train),"test_rows":len(test),"MAE":mae,"RMSE":rmse,"MAPE_pct":mape,"split_date":"2025-01-01","target_note":"prototype benchmark; not observed fixture pricing"})
        predictions.append(pd.DataFrame({"date":test.date,"route_id":test.route_id,"vessel_type":test.vessel_type,"model":name,"actual":test[TARGET],"prediction":p}))
    out=DATA/"public_processed";out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/"ablation_metrics.csv",index=False)
    if predictions: pd.concat(predictions,ignore_index=True).to_csv(out/"ablation_predictions.csv",index=False)
    (out/"ablation_config.json").write_text(json.dumps({"split":"2025-01-01","models":sets,"target_note":"Prototype benchmark target"},indent=2,default=str))
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=="__main__": main()
