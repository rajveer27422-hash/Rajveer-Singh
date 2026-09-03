"""
Cp–T Explorer
=============
Interactive dashboard for exploring heat-capacity (Cp) vs. temperature (T)
curves for materials, built to satisfy the "Interactive Presentation
Requirements" brief: searchable material/category selection, multi-material
plotting & comparison, user-defined temperature ranges, material metadata,
interactive zoom/pan/hover, clear axis labelling, out-of-range warnings,
ranking, curve comparison, and exportable graphs.

Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

DATA_PATH = Path(__file__).parent / "Materials.csv"

# --------------------------------------------------------------------------
# Page config + visual identity
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Cp\u2013T Explorer | Material Heat Capacity Dashboard",
    page_icon="\U0001F321\uFE0F",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#B8722A"      # warm bronze/ember - thermal theme
ACCENT_COOL = "#4C7A93"  # cool slate blue - contrast series
BG = "#14171A"
PANEL = "#1D2126"
TEXT = "#E8E6E1"
MUTED = "#8A8F96"

CB_PALETTE = [
    "#B8722A", "#4C7A93", "#9C6ADE", "#5FA777", "#D14D4D",
    "#C9A227", "#3E8E9E", "#B25D9C", "#7A8C3C", "#5C6BC0",
]

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'IBM Plex Sans', sans-serif;
        color: {TEXT};
    }}
    h1, h2, h3, .app-title {{
        font-family: 'Space Grotesk', sans-serif;
    }}
    .stApp {{ background-color: {BG}; }}
    section[data-testid="stSidebar"] {{ background-color: {PANEL}; }}
    .metric-card {{
        background-color: {PANEL};
        border: 1px solid #2A2F35;
        border-left: 3px solid {ACCENT};
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.6rem;
    }}
    .metric-card .label {{ color: {MUTED}; font-size: 0.78rem; }}
    .metric-card .value {{ font-size: 1.15rem; font-weight: 600; }}
    .warn-box {{
        background-color: #3A2A16;
        border: 1px solid {ACCENT};
        border-radius: 6px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.6rem;
        font-size: 0.92rem;
    }}
    hr {{ border-color: #2A2F35; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
@st.cache_data
def load_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    for c in ["Category", "Material", "Formula", "Phase", "Equation", "Unit", "Source"]:
        df[c] = df[c].astype(str).str.strip()
    for c in ["A", "B", "C", "D", "E"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["Tmin"] = pd.to_numeric(df["Tmin"], errors="coerce")
    df["Tmax"] = pd.to_numeric(df["Tmax"], errors="coerce")
    df = df.dropna(subset=["Tmin", "Tmax"]).reset_index(drop=True)

    # Build a unique, human-readable label per data ENTRY (an ID may repeat
    # across several rows -- one row per fitted temperature segment).
    counts = df.groupby("Material")["ID"].nunique()
    dup_materials = set(counts[counts > 1].index)
    id_info = df.drop_duplicates("ID")[["ID", "Material", "Formula", "Category", "Source"]].copy()

    def make_label(row):
        if row["Material"] in dup_materials:
            return f"{row['Material']} ({row['Formula']}) \u2014 {row['Source']}"
        return f"{row['Material']} ({row['Formula']})"

    id_info["Label"] = id_info.apply(make_label, axis=1)
    df = df.merge(id_info[["ID", "Label"]], on="ID", how="left")
    id_info = id_info.sort_values("Label").reset_index(drop=True)
    return df, id_info


df, id_info = load_data(DATA_PATH)

# --------------------------------------------------------------------------
# Cp math
# --------------------------------------------------------------------------
def cp_value(T: np.ndarray, seg: pd.Series) -> np.ndarray:
    """Evaluate Cp(T) in J/mol.K for a single fitted segment."""
    A, B, C, D, E = seg["A"], seg["B"], seg["C"], seg["D"], seg["E"]
    if seg["Equation"] == "Shomate":
        t = T / 1000.0
        return A + B * t + C * t**2 + D * t**3 + E / (t**2)
    # Covers: Cp=A+BT+C/T^2 , Cp=A+B*T+C/T^2 , Cp=A+BT+C/T^2+DT^2
    with np.errstate(divide="ignore", invalid="ignore"):
        return A + B * T + np.where(T != 0, C / (T**2), np.nan) + D * T**2


def compute_curve(entry_df: pd.DataFrame, T_grid: np.ndarray):
    """Piecewise Cp(T) across all fitted segments of one material entry.

    Returns (cp, in_range) where in_range[i] is False if T_grid[i] falls
    outside every fitted segment for this entry (value is still computed,
    by extrapolating the nearest segment's fit, but flagged as unreliable).
    """
    seg_df = entry_df.sort_values("Tmin").reset_index(drop=True)
    cp = np.full_like(T_grid, np.nan, dtype=float)
    in_range = np.zeros_like(T_grid, dtype=bool)
    for _, seg in seg_df.iterrows():
        mask = (T_grid >= seg["Tmin"]) & (T_grid <= seg["Tmax"])
        if mask.any():
            cp[mask] = cp_value(T_grid[mask], seg)
            in_range[mask] = True
    overall_min, overall_max = seg_df["Tmin"].min(), seg_df["Tmax"].max()
    below, above = T_grid < overall_min, T_grid > overall_max
    if below.any():
        cp[below] = cp_value(T_grid[below], seg_df.iloc[0])
    if above.any():
        cp[above] = cp_value(T_grid[above], seg_df.iloc[-1])
    return cp, in_range


def phase_boundaries(entry_df: pd.DataFrame):
    """Internal temperatures where the tabulated Phase actually changes."""
    seg_df = entry_df.sort_values("Tmin").reset_index(drop=True)
    out = []
    for i in range(len(seg_df) - 1):
        if seg_df.loc[i, "Phase"] != seg_df.loc[i + 1, "Phase"]:
            out.append((seg_df.loc[i, "Tmax"], seg_df.loc[i, "Phase"], seg_df.loc[i + 1, "Phase"]))
    return out


def cp_at_T(entry_df: pd.DataFrame, T: float):
    cp, in_range = compute_curve(entry_df, np.array([float(T)]))
    return float(cp[0]), bool(in_range[0])


# --------------------------------------------------------------------------
# Sidebar - filtering & selection
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## \U0001F321\uFE0F Cp\u2013T Explorer")
    st.caption("Filter, search and plot heat-capacity curves for 190+ materials.")
    st.divider()

    st.markdown("**1. Filter by category**")
    all_categories = sorted(id_info["Category"].unique())
    selected_categories = st.multiselect(
        "Material category", all_categories, default=[],
        help="Leave empty to search across all categories.",
        label_visibility="collapsed",
    )

    pool = id_info if not selected_categories else id_info[id_info["Category"].isin(selected_categories)]

    st.markdown("**2. Select material(s)**")
    default_labels = [l for l in pool["Label"] if l.startswith(("Aluminum", "Copper", "Iron ("))][:3]
    if not default_labels:
        default_labels = list(pool["Label"].iloc[:2])
    selected_labels = st.multiselect(
        "Search materials",
        options=list(pool["Label"]),
        default=default_labels,
        help="Start typing to search by material name or formula.",
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**3. Temperature range**")
    if selected_labels:
        sel_ids = id_info[id_info["Label"].isin(selected_labels)]["ID"].tolist()
        data_lo = float(df[df["ID"].isin(sel_ids)]["Tmin"].min())
        data_hi = float(df[df["ID"].isin(sel_ids)]["Tmax"].max())
    else:
        data_lo, data_hi = float(df["Tmin"].min()), float(df["Tmax"].max())

    global_lo, global_hi = float(df["Tmin"].min()), float(df["Tmax"].max())
    t_range = st.slider(
        "Temperature (K)",
        min_value=float(np.floor(global_lo)),
        max_value=float(np.ceil(global_hi)),
        value=(float(np.floor(data_lo)), float(np.ceil(data_hi))),
        step=1.0,
    )
    n_points = st.slider("Curve resolution (points)", 50, 600, 250, step=50)

    st.divider()
    st.markdown("**4. Display options**")
    show_phase_lines = st.checkbox("Show phase-transition markers", value=True)
    show_extrapolation = st.checkbox("Show dashed extrapolation outside fit range", value=True)
    ref_T = st.number_input(
        "Reference temperature for comparison & ranking (K)",
        min_value=float(t_range[0]), max_value=float(t_range[1]),
        value=float(np.clip(298.0, t_range[0], t_range[1])), step=1.0,
    )

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown("# Heat Capacity (C\u209a) vs. Temperature Explorer")
st.caption(
    "Interactive C\u209a\u2013T curves fitted from NIST / Gaskell / NIST-JANAF data. "
    "Select materials in the sidebar to plot, compare, and rank them."
)

if not selected_labels:
    st.info("\U0001F448 Select one or more materials in the sidebar to get started.")
    st.stop()

sel_ids = id_info[id_info["Label"].isin(selected_labels)]["ID"].tolist()
T_grid = np.linspace(t_range[0], t_range[1], int(n_points))

# --------------------------------------------------------------------------
# Compute curves + collect warnings
# --------------------------------------------------------------------------
curves = {}
out_of_range_msgs = []
for _id in sel_ids:
    entry_df = df[df["ID"] == _id]
    label = id_info.loc[id_info["ID"] == _id, "Label"].iloc[0]
    cp, in_range = compute_curve(entry_df, T_grid)
    curves[_id] = {"label": label, "entry": entry_df, "cp": cp, "in_range": in_range}
    if not in_range.all():
        vmin, vmax = entry_df["Tmin"].min(), entry_df["Tmax"].max()
        out_of_range_msgs.append(
            f"**{label}** is only validated for **{vmin:.0f}\u2013{vmax:.0f} K**; "
            f"the shaded/dashed portion outside that range is extrapolated from the nearest fit and may be inaccurate."
        )

if out_of_range_msgs:
    with st.container():
        st.markdown(
            f"<div class='warn-box'>\u26A0\uFE0F <b>Out-of-range warning</b><br>"
            + "<br>".join(out_of_range_msgs) + "</div>",
            unsafe_allow_html=True,
        )

# --------------------------------------------------------------------------
# Main Cp-T chart
# --------------------------------------------------------------------------
fig = go.Figure()

for i, (_id, c) in enumerate(curves.items()):
    color = CB_PALETTE[i % len(CB_PALETTE)]
    entry_df = c["entry"]
    meta = entry_df.iloc[0]
    cp, in_range = c["cp"], c["in_range"]

    solid = np.where(in_range, cp, np.nan)
    fig.add_trace(go.Scatter(
        x=T_grid, y=solid, mode="lines", name=c["label"],
        line=dict(color=color, width=2.6),
        hovertemplate=(
            f"<b>{meta['Material']} ({meta['Formula']})</b><br>"
            f"Category: {meta['Category']}<br>Source: {meta['Source']}<br>"
            "T = %{x:.1f} K<br>Cp = %{y:.3f} J/mol\u00B7K<extra></extra>"
        ),
    ))

    if show_extrapolation and not in_range.all():
        dashed = np.where(~in_range, cp, np.nan)
        fig.add_trace(go.Scatter(
            x=T_grid, y=dashed, mode="lines", name=f"{c['label']} (extrapolated)",
            line=dict(color=color, width=1.6, dash="dot"),
            showlegend=False,
            hovertemplate=(
                f"<b>{meta['Material']} \u2013 extrapolated, outside validated range</b><br>"
                "T = %{x:.1f} K<br>Cp = %{y:.3f} J/mol\u00B7K<extra></extra>"
            ),
        ))

    if show_phase_lines:
        for (Tb, ph1, ph2) in phase_boundaries(entry_df):
            if t_range[0] <= Tb <= t_range[1]:
                fig.add_vline(
                    x=Tb, line_width=1, line_dash="dash", line_color=color, opacity=0.55,
                )
                fig.add_annotation(
                    x=Tb, y=1.0, yref="paper", showarrow=False, yanchor="bottom",
                    text=f"{meta['Formula']}: {ph1}\u2192{ph2} @ {Tb:.0f}K",
                    font=dict(size=10, color=color), textangle=-90, xanchor="left",
                )

fig.update_layout(
    height=560,
    template="plotly_dark",
    paper_bgcolor=BG, plot_bgcolor=PANEL,
    font=dict(family="IBM Plex Sans", color=TEXT),
    xaxis_title="Temperature, T (K)",
    yaxis_title="Heat Capacity, C\u209a (J/mol\u00B7K)",
    legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0),
    margin=dict(t=60, l=10, r=10, b=10),
    hovermode="closest",
)
fig.update_xaxes(showgrid=True, gridcolor="#2A2F35", zeroline=False)
fig.update_yaxes(showgrid=True, gridcolor="#2A2F35", zeroline=False)

st.plotly_chart(fig, use_container_width=True, config={
    "displaylogo": False,
    "toImageButtonOptions": {"format": "png", "filename": "cp_T_curves", "scale": 2},
})
st.caption(
    "Drag to zoom, double-click to reset, hover for exact values. "
    "Use the camera icon in the chart toolbar to export a PNG."
)

# --------------------------------------------------------------------------
# Tabs: comparison table / ranking / details / delta curve / export
# --------------------------------------------------------------------------
tab_compare, tab_rank, tab_details, tab_delta, tab_export = st.tabs(
    ["\U0001F4CA Comparison", "\U0001F3C6 Ranking", "\U0001F4C4 Material details", "\u0394 Curve difference", "\u2B07\uFE0F Export"]
)

with tab_compare:
    st.markdown(f"#### C\u209a comparison at T = {ref_T:.1f} K")
    rows = []
    for _id, c in curves.items():
        meta = c["entry"].iloc[0]
        cp_ref, in_rng = cp_at_T(c["entry"], ref_T)
        phases = " / ".join(sorted(set(c["entry"]["Phase"])))
        rows.append({
            "Material": meta["Material"], "Formula": meta["Formula"],
            "Category": meta["Category"], "Phase(s)": phases,
            f"Cp @ {ref_T:.0f}K (J/mol\u00B7K)": round(cp_ref, 3),
            "In validated range?": "Yes" if in_rng else "No (extrapolated)",
            "Valid range (K)": f"{c['entry']['Tmin'].min():.0f}\u2013{c['entry']['Tmax'].max():.0f}",
            "Source": meta["Source"],
        })
    compare_df = pd.DataFrame(rows).sort_values(f"Cp @ {ref_T:.0f}K (J/mol\u00B7K)", ascending=False)
    st.dataframe(compare_df, use_container_width=True, hide_index=True)

with tab_rank:
    st.markdown(f"#### Ranked by C\u209a at T = {ref_T:.1f} K")
    rank_rows = []
    for _id, c in curves.items():
        meta = c["entry"].iloc[0]
        cp_ref, in_rng = cp_at_T(c["entry"], ref_T)
        rank_rows.append((c["label"], cp_ref, in_rng))
    rank_df = pd.DataFrame(rank_rows, columns=["Material", "Cp", "InRange"]).sort_values("Cp", ascending=True)
    bar_colors = [ACCENT if r else MUTED for r in rank_df["InRange"]]
    rank_fig = go.Figure(go.Bar(
        x=rank_df["Cp"], y=rank_df["Material"], orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.2f}" for v in rank_df["Cp"]], textposition="outside",
    ))
    rank_fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=PANEL,
        font=dict(family="IBM Plex Sans", color=TEXT),
        xaxis_title="Cp (J/mol\u00B7K)", height=120 + 40 * len(rank_df),
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(rank_fig, use_container_width=True)
    st.caption(f"Bronze bars are within the validated fit range at {ref_T:.0f} K; grey bars are extrapolated.")

with tab_details:
    for _id, c in curves.items():
        meta = c["entry"].iloc[0]
        with st.expander(f"{meta['Material']} ({meta['Formula']})", expanded=len(curves) == 1):
            c1, c2, c3, c4 = st.columns(4)
            for col, label, value in zip(
                (c1, c2, c3, c4),
                ("Category", "Phase(s)", "Data source", "Equation form"),
                (
                    meta["Category"],
                    " / ".join(sorted(set(c["entry"]["Phase"]))),
                    meta["Source"],
                    ", ".join(sorted(set(c["entry"]["Equation"]))),
                ),
            ):
                col.markdown(
                    f"<div class='metric-card'><div class='label'>{label}</div>"
                    f"<div class='value'>{value}</div></div>", unsafe_allow_html=True,
                )
            st.markdown("**Fitted segments**")
            seg_show = c["entry"][["Phase", "Equation", "A", "B", "C", "D", "E", "Tmin", "Tmax"]].sort_values("Tmin")
            st.dataframe(seg_show, use_container_width=True, hide_index=True)
            cp_now, in_rng = cp_at_T(c["entry"], ref_T)
            st.markdown(
                f"Cp at **{ref_T:.0f} K** \u2248 **{cp_now:.3f} J/mol\u00B7K** "
                f"({'within' if in_rng else 'outside \u2014 extrapolated'} validated range "
                f"{c['entry']['Tmin'].min():.0f}\u2013{c['entry']['Tmax'].max():.0f} K)."
            )

with tab_delta:
    if len(curves) != 2:
        st.info("Select exactly **two** materials in the sidebar to see a \u0394Cp (difference) curve.")
    else:
        (id_a, c_a), (id_b, c_b) = list(curves.items())
        delta = c_a["cp"] - c_b["cp"]
        both_valid = c_a["in_range"] & c_b["in_range"]
        d_fig = go.Figure()
        d_fig.add_trace(go.Scatter(
            x=T_grid, y=np.where(both_valid, delta, np.nan), mode="lines",
            line=dict(color=ACCENT, width=2.5),
            name=f"{c_a['label']} \u2212 {c_b['label']}",
        ))
        d_fig.add_trace(go.Scatter(
            x=T_grid, y=np.where(~both_valid, delta, np.nan), mode="lines",
            line=dict(color=ACCENT, width=1.4, dash="dot"), showlegend=False,
        ))
        d_fig.add_hline(y=0, line_color=MUTED, line_width=1)
        d_fig.update_layout(
            template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=PANEL,
            font=dict(family="IBM Plex Sans", color=TEXT),
            xaxis_title="Temperature, T (K)",
            yaxis_title=f"\u0394Cp = Cp({c_a['label']}) \u2212 Cp({c_b['label']}) (J/mol\u00B7K)",
            height=420, margin=dict(t=30, l=10, r=10, b=10),
        )
        st.plotly_chart(d_fig, use_container_width=True)
        st.caption("Dotted portions fall outside the validated range of at least one of the two materials.")

with tab_export:
    st.markdown("#### Download the plotted data")
    export_rows = []
    for _id, c in curves.items():
        export_rows.append(pd.DataFrame({
            "Material": c["label"], "T_K": T_grid, "Cp_J_per_mol_K": c["cp"], "In_validated_range": c["in_range"],
        }))
    export_df = pd.concat(export_rows, ignore_index=True)
    st.download_button(
        "Download curve data as CSV", data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="cp_T_curve_data.csv", mime="text/csv",
    )
    st.download_button(
        "Download chart as standalone HTML", data=fig.to_html(include_plotlyjs="cdn"),
        file_name="cp_T_chart.html", mime="text/html",
    )
    st.caption("For a PNG, use the camera icon in the chart's toolbar above.")

st.divider()
st.caption(
    "Data: NIST Chemistry WebBook, NIST-JANAF Thermochemical Tables, and Gaskell's "
    "*Introduction to the Thermodynamics of Materials*. Shomate equation: "
    "Cp\u00B0 = A + B\u00B7t + C\u00B7t\u00B2 + D\u00B7t\u00B3 + E/t\u00B2, t = T(K)/1000. "
    "Polynomial forms: Cp = A + B\u00B7T + C/T\u00B2 (+ D\u00B7T\u00B2)."
)
