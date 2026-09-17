# Free Public Data Integration

This branch adds a public-data layer without replacing the prototype benchmark target.

## Pipeline

```text
Open-Meteo historical weather ─┐
                               ├─> public_exogenous_features.csv ─> XGBoost
Public indicative BDI ─────────┤
                               │
World Bank Pink Sheet (optional)┘
```

## Run

```bash
python data_pipeline/public_data_ingestion.py --start 2019-01-01 --end 2025-12-31
python data_pipeline/public_market_adapter.py --start 2019-01-01 --end 2025-12-31
python data_pipeline/train_public_xgb.py
```

## Data honesty

- Open-Meteo weather fields are observed public data; disruption scores are derived.
- BDI is an indicative public series from the configured public source and is not presented as official Baltic Exchange redistribution.
- World Bank commodity fields are optional and are loaded only when a compatible public Pink Sheet CSV is present locally.
- The existing freight-rate target is a prototype benchmark/synthetic target. Adding public inputs does **not** make that target observed fixture pricing.
- Provenance fields are retained so downstream dashboards can distinguish observed, derived and synthetic information.

## Outputs

- `data/public_raw/bdi_public.csv`
- `data/public_raw/weather_open_meteo.csv`
- `data/public_processed/weather_public_features.csv`
- `data/public_processed/market_public_features.csv`
- `data/public_processed/public_exogenous_features.csv`
- `data/public_processed/public_xgb_predictions.csv`
- `data/public_processed/public_xgb_metrics.csv`
- `data/public_processed/public_xgb_feature_list.json`
