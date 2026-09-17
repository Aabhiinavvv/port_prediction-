# Public Data Integration

Run from the repository root:

```bash
python data_pipeline/public_data_ingestion.py --start 2019-01-01 --end 2025-12-31
```

This downloads actual public/reanalysis inputs and writes:

- `data/public_raw/weather_open_meteo.csv`
- `data/public_raw/bdi_public.csv`
- `data/public_processed/weather_public_features.csv`
- `data/public_processed/market_public_features.csv`

Weather uses the Open-Meteo Historical Weather API. BDI uses a public indicative Yahoo Finance series with a Stooq fallback. The BDI field is explicitly not represented as a licensed Baltic Exchange redistribution.

The pipeline keeps `synthetic_flag=0` for observed public inputs and deterministic derived features. It does not overwrite the existing `data/freight_rates_timeseries.csv` synthetic benchmark.
