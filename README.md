# port_prediction-

## Run the decision dashboard

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

`app.py` is the consolidated executive dashboard for East Coast India chartering.
It adds a commercial charter-strategy engine, two-port vessel feasibility, a
60-day market-entry outlook, explainable forecast drivers, voyage-cycle and idle
exposure, and one/three/six-voyage contract scenarios. The optional refresh
button uses keyless World Bank and Open-Meteo APIs for contextual macro and
marine-risk signals, with a safe offline fallback.
