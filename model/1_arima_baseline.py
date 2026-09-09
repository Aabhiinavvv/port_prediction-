"""
Module 1a — Data prep + ARIMA baseline
Run this on ONE route+vessel first to prove the pipeline, then loop over all
route/vessel combos for the real system.
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

ts = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "freight_rates_timeseries.csv", parse_dates=["date"])

ROUTE, VESSEL = "AUNTL_INPAR", "Handysize"
df = ts[(ts.route_id == ROUTE) & (ts.vessel_type == VESSEL)].sort_values("date").reset_index(drop=True)
df = df.set_index("date")
y = df["freight_rate_usd_per_tonne"]

# ---- TIME-BASED split: never shuffle time series data ----
TEST_DAYS = 90
train, test = y.iloc[:-TEST_DAYS], y.iloc[-TEST_DAYS:]
print(f"Train: {len(train)} days ({train.index.min().date()} to {train.index.max().date()})")
print(f"Test : {len(test)} days ({test.index.min().date()} to {test.index.max().date()})")

# ---- ARIMA / SARIMAX with weekly seasonality (period=7) ----
model = SARIMAX(train, order=(2, 1, 2), seasonal_order=(1, 1, 1, 7),
                 enforce_stationarity=False, enforce_invertibility=False)
fit = model.fit(disp=False)
arima_pred = fit.forecast(steps=TEST_DAYS)
arima_pred.index = test.index

rmse = np.sqrt(mean_squared_error(test, arima_pred))
mape = np.mean(np.abs((test - arima_pred) / test)) * 100
print(f"\nARIMA  -> RMSE: {rmse:.3f}   MAPE: {mape:.2f}%")

# save predictions for the ensemble step later
pd.DataFrame({"date": test.index, "actual": test.values, "arima_pred": arima_pred.values}) \
  .to_csv("step1_arima_predictions.csv", index=False)
