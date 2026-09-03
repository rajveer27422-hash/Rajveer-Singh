# Cp–T Explorer

Interactive dashboard for exploring heat-capacity (Cp) vs. temperature (T)
curves, built for the Interactive Presentation Requirements brief.

## Run it locally (gives you a live link at http://localhost:8501)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a URL like `http://localhost:8501` and open it in your
browser automatically. Keep the terminal window open while you use it.

`app.py` and `Materials.csv` must stay in the same folder.

## Get a real public link (free, no local install needed)

1. Push this folder to a GitHub repo (just `app.py`, `Materials.csv`,
   `requirements.txt`).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, click "New app", point it at the repo and `app.py`.
3. Deploy — you'll get a public `https://<your-app>.streamlit.app` link you
   can share with anyone (e.g. your instructor).

## What it covers from the requirements brief

- Searchable material dropdown + category filter (sidebar)
- Multiple materials plotted simultaneously, with a legend
- Cp comparison table and a ranking bar chart at a chosen reference temperature
- User-defined temperature range (slider) and curve resolution
- Material name, formula, category, phase(s), and data source shown per curve
- Interactive zoom/pan/hover (Plotly) with exact Cp/T values on hover
- Clear axis titles, units (J/mol·K), and legends
- Automatic warning + dashed extrapolated line when a curve is plotted
  outside its validated fitted temperature range
- Extras: phase-transition markers (e.g. solid→liquid), a ΔCp
  difference-curve tab (pick exactly 2 materials), CSV/HTML export, and a
  per-material data-source/segment table

## Data notes

`Materials.csv` contains 364 fitted segments across 190+ materials. Some
materials (e.g. Iron, Calcium, Boron) have more than one independent fit from
different sources (NIST Chemistry WebBook, NIST-JANAF, Gaskell) — these are
disambiguated in the dropdown by appending the source name. Two equation
forms are used:

- **Shomate**: Cp° = A + B·t + C·t² + D·t³ + E/t², where t = T(K)/1000
- **Polynomial**: Cp = A + B·T + C/T² (+ D·T²), T in Kelvin directly

Both are implemented in `cp_value()` in `app.py`.
