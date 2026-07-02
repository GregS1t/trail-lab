"""trail_analysis_v2.py — Trail running analysis with Plotly figures.

Rewrite of trail_analysis.py replacing Matplotlib with Plotly.
All plot_* functions return a plotly.graph_objects.Figure.
Non-plotting functions (load_fit, clean_df, compute_*, ...) are
imported unchanged from trail_analysis.py.

Usage in a notebook:
    from trail_analysis import (
        load_fit, clean_df, compute_slope, compute_dplus_dminus,
        compute_gap, compute_hr_zones, classify_walk_run,
        compute_cardiac_drift, section_stats, compute_ravito_stops,
        compute_aerobic_decoupling, compute_stride_metrics,
        compute_pace_variability, compute_pace_split,
        compute_circadian_profile, detect_hitting_wall,
        load_and_process_race, compute_race_kpis,
        normalize_by_distance_pct, build_races_table,
        fetch_weather_hourly, enrich_df_with_weather,
    )
    from trail_analysis_v2 import (
        DASHBOARD_VARIABLES, plot_profil_colore, plot_dashboard,
    )

References
----------
- Minetti AE et al. (2002). J Appl Physiol 93(3):1039-1046.
"""

import io, sys
import base64
import warnings

import folium
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fitparse import FitFile
from branca.element import MacroElement
from jinja2 import Template
from folium.plugins import AntPath


import plotly.io as pio
pio.renderers.default = "notebook"


warnings.filterwarnings(
    "ignore",
    message="The behavior of DataFrame concatenation with empty or all-NA entries",
    category=FutureWarning,
)

def build_section_labels(bounds, ravito_nom):
    """Build section labels from distance bounds and ravito names.

    Handles the case where ravito_nom is empty (no aid stations).

    Parameters
    ----------
    bounds     : list[float]  distance bounds including start and end
    ravito_nom : list[str]    aid station names (may be empty)

    Returns
    -------
    list[str]
    """
    labels = []
    n = len(bounds) - 1
    for i in range(n):
        a, b = bounds[i], bounds[i + 1]
        if not ravito_nom:
            if n == 1:
                labels.append(f"Course entière ({a:.1f}–{b:.1f} km)")
            else:
                labels.append(f"Section {i + 1} ({a:.1f}–{b:.1f} km)")
        elif i == 0:
            labels.append(f"Départ → {ravito_nom[0]} ({a:.1f}–{b:.1f} km)")
        elif i == n - 1:
            labels.append(f"{ravito_nom[-1]} → Arrivée ({a:.1f}–{b:.1f} km)")
        else:
            labels.append(f"{ravito_nom[i-1]} → {ravito_nom[i]} ({a:.1f}–{b:.1f} km)")
    return labels


def build_section_labels_short(bounds, ravito_nom):
    """Build short section labels (no km range).

    Parameters
    ----------
    bounds     : list[float]
    ravito_nom : list[str]

    Returns
    -------
    list[str]
    """
    labels = []
    n = len(bounds) - 1
    for i in range(n):
        if not ravito_nom:
            labels.append(f"Section {i + 1}")
        elif i == 0:
            labels.append(f"Départ → {ravito_nom[0]}")
        elif i == n - 1:
            labels.append(f"{ravito_nom[-1]} → Arrivée")
        else:
            labels.append(f"{ravito_nom[i-1]} → {ravito_nom[i]}")
    return labels


# ===========================================================================
# 1. I/O
# ===========================================================================

def load_fit(fit_path):
    """Load FIT file records into a pandas DataFrame."""
    fitfile = FitFile(fit_path)
    records = []
    for record in fitfile.get_messages("record"):
        row = {}
        for field in record:
            row[field.name] = field.value
        records.append(row)
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["time_h"] = (
        (df["timestamp"] - df["timestamp"].iloc[0])
        .dt.total_seconds() / 3600.0
    )
    return df


def clean_df(df):
    """Select best altitude/speed columns, ensure monotone distance, compute pace.

    Returns (df_clean, alt_col).
    """
    alt_col = "enhanced_altitude" if "enhanced_altitude" in df.columns else "altitude"
    spd_col = "enhanced_speed" if "enhanced_speed" in df.columns else "speed"

    df = df.dropna(subset=["distance", alt_col]).reset_index(drop=True)

    d = df["distance"].to_numpy(dtype=float)
    df["dist_m"] = np.maximum.accumulate(d)
    df["alt_m"] = df[alt_col].to_numpy(dtype=float)

    if spd_col in df.columns:
        v = df[spd_col].to_numpy(dtype=float)
        df["speed_mps"] = v
        df["speed_kmh"] = v * 3.6
        df["pace_s_per_km"] = np.where(v > 0.5, 1000.0 / v, np.nan)
    else:
        df["speed_mps"] = np.nan
        df["speed_kmh"] = np.nan
        df["pace_s_per_km"] = np.nan

    if "position_lat" in df.columns and "position_long" in df.columns:
        df["lat"] = df["position_lat"] * (180.0 / 2**31)
        df["lon"] = df["position_long"] * (180.0 / 2**31)

    return df, alt_col


# ===========================================================================
# 2. Terrain calculations
# ===========================================================================

def compute_slope(df, window_m):
    """Compute local slope (%) over a backward distance window."""
    d = df["dist_m"].to_numpy()
    z = df["alt_m"].to_numpy()
    n = len(d)
    j = np.searchsorted(d, d - window_m, side="left")
    j = np.clip(j, 0, n - 1)
    dd = d - d[j]
    dz = z - z[j]
    with np.errstate(invalid='ignore', divide='ignore'):
        slope = np.where(dd > 0, (dz / dd) * 100.0, np.nan)
    return slope


def compute_dplus_dminus(df, dz_thr=None):
    """Compute filtered elevation gain and loss.

    If dz_thr is None, use an adaptive threshold based on altitude noise.
    """
    alt_smooth = (
        df["alt_m"]
        .rolling(7, center=True, min_periods=1).median()
        .rolling(7, center=True, min_periods=1).mean()
    )
    if dz_thr is None:
        raw_std = df["alt_m"].diff().dropna().abs().std()
        dz_thr = max(0.1, raw_std / 4.0)
    dz = alt_smooth.diff()
    return float(dz[dz > dz_thr].sum()), float((-dz[dz < -dz_thr]).sum())


def segment_updown(df, up_thr, down_thr, min_seg_m):
    """Segment track into uphill (+1) / downhill (-1) / flat (0) with hysteresis."""
    s = df["slope_pct"].to_numpy()
    state = np.zeros(len(df), dtype=int)
    state[s >= up_thr] = 1
    state[s <= down_thr] = -1
    for i in range(1, len(state)):
        if state[i] == 0:
            state[i] = state[i - 1]
    df = df.copy()
    df["ud_state"] = state
    df["seg_id"] = (df["ud_state"] != df["ud_state"].shift(1)).cumsum()
    seg_len = df.groupby("seg_id")["dist_m"].agg(
        lambda x: float(x.iloc[-1] - x.iloc[0])
    )
    valid_seg = seg_len[seg_len >= min_seg_m].index
    df["ud_clean"] = np.where(df["seg_id"].isin(valid_seg), df["ud_state"], 0)
    return df


def minetti_cost_ratio(slope_pct):
    """Cost ratio relative to flat ground from Minetti et al. (2002)."""
    i = slope_pct / 100.0
    cr = (155.4 * i**5 - 30.4 * i**4 - 43.3 * i**3
          + 46.3 * i**2 + 19.5 * i + 3.6)
    return np.clip(cr / 3.6, 0.1, None)


def compute_gap(df):
    """Compute Grade Adjusted Pace (s/km) using Minetti et al. (2002)."""
    df = df.copy()
    ratio = minetti_cost_ratio(df["slope_pct"].fillna(0))
    df["gap_s_per_km"] = df["pace_s_per_km"] / ratio
    return df


# ===========================================================================
# 3. Physiological calculations
# ===========================================================================

def compute_hr_zones(df, fc_max, hr_col="heart_rate"):
    """Compute time in heart rate zones based on %FCmax."""
    zones = [
        (0.50, 0.60, "Z1 — Récupération"),
        (0.60, 0.70, "Z2 — Endurance fond."),
        (0.70, 0.80, "Z3 — Tempo"),
        (0.80, 0.90, "Z4 — Seuil"),
        (0.90, 1.00, "Z5 — VO₂max"),
    ]
    df = df.copy()
    df["hr_frac"] = df[hr_col] / fc_max
    results = []
    for low, high, label in zones:
        mask = (df["hr_frac"] >= low) & (df["hr_frac"] < high)
        t_s = mask.sum()
        results.append({
            "Zone": label,
            "FC min": int(low * fc_max),
            "FC max": int(high * fc_max),
            "Temps (min)": round(t_s / 60, 1),
            "Temps (%)": round(t_s / len(df.dropna(subset=[hr_col])) * 100, 1),
        })
    return pd.DataFrame(results)


def classify_walk_run(df, walk_thr_kmh, walk_thr_cad):
    """Classify each point as walking (1) or running (0)."""
    df = df.copy()
    df["is_walk"] = (
        (df["speed_kmh"] < walk_thr_kmh) & (df["cadence"] < walk_thr_cad)
    ).astype(int)
    return df


def compute_cardiac_drift(df, smoothing_sec=300, min_speed_kmh=3.0):
    """Compute smoothed HR/GAP ratio as cardiac drift proxy."""
    df = df.copy()
    mask = df["speed_kmh"] > min_speed_kmh
    df_moving = df.loc[mask].copy()

    if df_moving.empty:
        print("Pas assez de points après filtrage.")
        return df

    if "gap_s_per_km" in df_moving.columns:
        gap_kmh = 3600.0 / df_moving["gap_s_per_km"].replace(0, np.nan)
        df.loc[mask, "hr_speed_ratio"] = df_moving["heart_rate"] / gap_kmh
    else:
        df.loc[mask, "hr_speed_ratio"] = (
            df_moving["heart_rate"] / df_moving["speed_kmh"]
        )

    df_t = df.dropna(subset=["hr_speed_ratio"]).set_index("timestamp")
    smooth = (
        df_t["hr_speed_ratio"]
        .rolling(f"{smoothing_sec}s", min_periods=10)
        .mean()
    )
    df["hr_speed_smooth"] = smooth.reindex(
        df["timestamp"], method="nearest"
    ).values
    return df


# ===========================================================================
# 4. Section / ravito statistics
# ===========================================================================

def section_stats(df, ravito_km, ravito_nom, hr_col="heart_rate"):
    """Compute per-section statistics between aid stations."""
    bounds = np.concatenate((
        [float(df["dist_m"].min() / 1000.0)],
        np.array(ravito_km, dtype=float),
        [float(df["dist_m"].max() / 1000.0)]
    ))
    labels = build_section_labels(list(bounds), ravito_nom)
    x_km = df["dist_m"].to_numpy() / 1000.0
    df = df.copy()
    df["section_id"] = np.searchsorted(bounds[1:], x_km, side="right")
    rows = []
    for i, lbl in enumerate(labels):
        sec = df[df["section_id"] == i]
        if len(sec) < 10:
            continue
        alt = sec["alt_m"].to_numpy()
        dz = np.diff(alt)
        row = {
            "Section": lbl,
            "Dist (km)": round(
                (sec["dist_m"].iloc[-1] - sec["dist_m"].iloc[0]) / 1000, 1
            ),
            "D+ (m)": round(float(np.clip(dz, 0, None).sum()), 0),
            "D- (m)": round(float((-np.clip(dz, None, 0)).sum()), 0),
            "Allure méd. (s/km)": round(sec["pace_s_per_km"].median(), 0),
        }
        if hr_col in sec.columns:
            row["FC méd. (bpm)"] = round(sec[hr_col].median(), 0)
        if "is_walk" in sec.columns:
            row["Marche (%)"] = round(sec["is_walk"].mean() * 100, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_ravito_stops(df, ravito_km, ravito_nom,
                         speed_thr_kmh=0.5, radius_m=300, min_stop_sec=10):
    """Compute time stopped near each aid station."""
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    df["stopped"] = (df["speed_kmh"] < speed_thr_kmh).astype(int)
    df["block_id"] = (df["stopped"] != df["stopped"].shift(1)).cumsum()

    stop_blocks = (
        df[df["stopped"] == 1]
        .groupby("block_id")
        .agg(
            t_start=("timestamp", "first"),
            t_end=("timestamp", "last"),
            dist_mid_m=("dist_m", "mean"),
        )
        .reset_index(drop=True)
    )
    stop_blocks["duration_s"] = (
        (stop_blocks["t_end"] - stop_blocks["t_start"]).dt.total_seconds()
    )
    stop_blocks = stop_blocks[stop_blocks["duration_s"] >= min_stop_sec]

    rows = []
    for km, nom in zip(ravito_km, ravito_nom):
        ravito_m = km * 1000.0
        nearby = stop_blocks[
            (stop_blocks["dist_mid_m"] - ravito_m).abs() <= radius_m
        ]
        total_s = nearby["duration_s"].sum()
        rows.append({
            "Ravito": nom,
            "Position (km)": km,
            "Nb arrêts": len(nearby),
            "Temps total arrêt (s)": round(total_s, 0),
            "Temps total arrêt (min)": round(total_s / 60, 1),
        })

    result = pd.DataFrame(rows)
    total_s_all = result["Temps total arrêt (s)"].sum()
    total_row = pd.DataFrame([{
        "Ravito": "TOTAL",
        "Position (km)": np.nan,
        "Nb arrêts": int(result["Nb arrêts"].sum()),
        "Temps total arrêt (s)": total_s_all,
        "Temps total arrêt (min)": round(total_s_all / 60, 1),
    }])
    result = result.astype({
        "Position (km)": float,
        "Nb arrêts": int,
        "Temps total arrêt (s)": float,
        "Temps total arrêt (min)": float,
    })
    return pd.concat([result, total_row], ignore_index=True)


# ===========================================================================
# 5. Cartographie (folium — inchangé)
# ===========================================================================

class ElevationControl(MacroElement):
    """Folium MacroElement overlaying an elevation profile on the map."""

    def __init__(self, svg_b64):
        """Init with a base64-encoded SVG string."""
        super().__init__()
        self._name = "ElevationControl"
        self.svg_b64 = svg_b64
        self._template = Template("""
            {% macro script(this, kwargs) %}
            (function() {
                var div = document.createElement('div');
                div.style.cssText = [
                    'position:absolute',
                    'bottom:30px',
                    'left:10px',
                    'z-index:1000',
                    'background:rgba(255,255,255,0.88)',
                    'border-radius:6px',
                    'padding:6px',
                    'box-shadow:0 1px 5px rgba(0,0,0,0.3)'
                ].join(';');
                div.innerHTML = '<img src="data:image/svg+xml;base64,{{ this.svg_b64 }}"'
                    + ' width="320" style="display:block;">';
                document.querySelector('#{{ this._parent.get_name() }}').appendChild(div);
            })();
            {% endmacro %}
        """)


def make_elevation_svg(df, dist_col="dist_m", alt_col="alt_m",
                       figsize=(5, 1.6), dpi=110):
    """Generate an elevation profile as a base64-encoded SVG string."""
    dist_km = df[dist_col] / 1000.0
    alt = df[alt_col]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.fill_between(dist_km, alt, alpha=0.35, color="steelblue")
    ax.plot(dist_km, alt, color="steelblue", linewidth=1.2)
    ax.set_xlabel("Distance (km)", fontsize=7)
    ax.set_ylabel("Altitude (m)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout(pad=0.4)

    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def plot_map(df, ravito_km=None, ravito_nom=None,
             color_col=None, cmap_name="RdYlGn_r",
             tile="CartoDB positron", animated=False,
             show_elevation=False):
    """Plot the GPS trace on an interactive folium map.

    Parameters
    ----------
    df             : pd.DataFrame  must contain lat, lon, dist_m, alt_m
    ravito_km      : list[float] | None
    ravito_nom     : list[str]   | None
    color_col      : str | None   column to color the track
    cmap_name      : str          matplotlib colormap name
    tile           : str          folium tile layer
    animated       : bool         add AntPath animation
    show_elevation : bool         overlay elevation profile inset

    Returns
    -------
    folium.Map
    """
    df = df.dropna(subset=["lat", "lon"]).copy()
    if df.empty:
        print("Pas de coordonnées GPS dans ce fichier.")
        return None

    center = [df["lat"].mean(), df["lon"].mean()]
    m = folium.Map(location=center, zoom_start=12, tiles=tile)

    if color_col is not None and color_col in df.columns:
        col_vals = df[color_col].to_numpy(dtype=float)
        vmin = np.nanpercentile(col_vals, 2)
        vmax = np.nanpercentile(col_vals, 98)
        cmap = plt.get_cmap(cmap_name)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        coords = list(zip(df["lat"], df["lon"]))
        for i in range(len(coords) - 1):
            v = col_vals[i]
            if np.isnan(v):
                continue
            hex_color = mpl.colors.to_hex(cmap(norm(v)))
            folium.PolyLine(
                [coords[i], coords[i + 1]],
                color=hex_color, weight=3, opacity=0.85,
            ).add_to(m)
    else:
        coords = list(zip(df["lat"], df["lon"]))
        folium.PolyLine(coords, color="steelblue", weight=3,
                        opacity=0.85).add_to(m)

    folium.Marker(
        location=[df["lat"].iloc[0], df["lon"].iloc[0]],
        popup="Départ",
        icon=folium.Icon(color="green", icon="flag", prefix="fa"),
    ).add_to(m)
    folium.Marker(
        location=[df["lat"].iloc[-1], df["lon"].iloc[-1]],
        popup="Arrivée",
        icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa"),
    ).add_to(m)

    if ravito_km and ravito_nom:
        for km, nom in zip(ravito_km, ravito_nom):
            idx = (df["dist_m"] / 1000.0 - km).abs().idxmin()
            if idx in df.index:
                folium.Marker(
                    location=[df.loc[idx, "lat"], df.loc[idx, "lon"]],
                    popup=f"{nom} ({km} km)",
                    icon=folium.Icon(color="blue", icon="cutlery", prefix="fa"),
                ).add_to(m)

    if animated:
        AntPath(
            locations=coords,
            color="steelblue", weight=4, delay=800,
            dash_array=[10, 20], pulse_color="#ffffff",
        ).add_to(m)

    if show_elevation and "alt_m" in df.columns and "dist_m" in df.columns:
        svg_b64 = make_elevation_svg(df)
        ElevationControl(svg_b64).add_to(m)

    return m


# ===========================================================================
# 6. Plotly figures — profils colorés et dashboard
# ===========================================================================

# Reference dict for dashboard variables.
# Keys   : DataFrame column names.
# Values : (label, plotly_colorscale, vmin, vmax).
# vmin / vmax = None means auto-range.
DASHBOARD_VARIABLES = {
    "heart_rate":    ("FC (bpm)",          "RdBu_r",   100, 185),
    "pace_s_per_km": ("Allure (s/km)",     "RdYlGn_r", 180, 900),
    "gap_s_per_km":  ("GAP (s/km)",        "RdYlGn_r", 180, 900),
    "temperature":   ("Température (°C)",  "plasma",   None, None),
    "speed_kmh":     ("Vitesse (km/h)",    "RdYlGn",   None, None),
    "cadence":       ("Cadence (spm)",     "viridis",  140,  200),
    "slope_pct":     ("Pente (%)",         "RdYlGn_r", -30,   30),
    "temp_api":      ("Température ERA5 (°C)", "plasma", None, None),
}

_RAVITO_LINE = dict(color="white", width=1.5, dash="dot")
_RAVITO_FONT = dict(size=10, color="white")


def _validate_variables(variables):
    """Raise ValueError if any key is not in DASHBOARD_VARIABLES."""
    unknown = set(variables) - set(DASHBOARD_VARIABLES)
    if unknown:
        raise ValueError(
            f"Unknown variable(s): {sorted(unknown)}. "
            f"Valid keys: {list(DASHBOARD_VARIABLES.keys())}"
        )


def plot_profil_colore(df, col, label=None, colorscale="Viridis",
                       vmin=None, vmax=None,
                       ravito_km=None, ravito_nom=None,
                       height=400):
    """Elevation profile colored by a data column.

    Parameters
    ----------
    df         : pd.DataFrame  must contain dist_m, alt_m, and col
    col        : str           column name to color by
    label      : str | None    colorbar label (defaults to col)
    colorscale : str           Plotly colorscale name
    vmin, vmax : float | None  color range (None = auto)
    ravito_km  : list[float] | None
    ravito_nom : list[str]   | None
    height     : int           figure height in pixels

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")

    label = label or col
    mask = df[col].notna()
    x_km = df.loc[mask, "dist_m"] / 1000.0
    y_alt = df.loc[mask, "alt_m"]
    c_val = df.loc[mask, col]
    y_top = float(df["alt_m"].max())

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["dist_m"] / 1000.0, y=df["alt_m"],
        fill="tozeroy",
        fillcolor="rgba(180,180,180,0.25)",
        line=dict(color="rgba(180,180,180,0.4)", width=1),
        hoverinfo="skip", showlegend=False, name="Altitude",
    ))

    fig.add_trace(go.Scatter(
        x=x_km, y=y_alt, mode="markers",
        marker=dict(
            color=c_val, colorscale=colorscale,
            cmin=vmin, cmax=vmax, size=3, opacity=0.85,
            colorbar=dict(title=dict(text=label, side="right"), thickness=14),
        ),
        name=label,
        hovertemplate=(
            "Distance : %{x:.2f} km<br>Altitude : %{y:.0f} m<br>"
            f"{label} : %{{marker.color:.1f}}<extra></extra>"
        ),
    ))

    for km, nom in zip(ravito_km or [], ravito_nom or []):
        fig.add_vline(x=km, line=_RAVITO_LINE)
        fig.add_annotation(
            x=km, y=y_top, xref="x", yref="y",
            text=nom, showarrow=False, font=_RAVITO_FONT,
            textangle=-45, xanchor="left",
        )

    fig.update_layout(
        title=f"Profil coloré — {label}",
        xaxis_title="Distance (km)", yaxis_title="Altitude (m)",
        height=height, template="plotly_dark",
        margin=dict(l=60, r=80, t=50, b=50),
    )
    return fig


def plot_dashboard(df, fc_min, fc_max, variables=None,
                   ravito_km=None, ravito_nom=None,
                   height_per_panel=150):
    """Multi-panel elevation profiles, one per physiological variable.

    Parameters
    ----------
    df               : pd.DataFrame
    fc_min, fc_max   : float   heart rate range for color scaling
    variables        : dict | None
        Subset of DASHBOARD_VARIABLES. Keys = column names,
        Values = (label, colorscale, vmin, vmax).
        If None, all DASHBOARD_VARIABLES columns present in df are used.
    ravito_km        : list[float] | None
    ravito_nom       : list[str]   | None
    height_per_panel : int   height in pixels per subplot row

    Returns
    -------
    plotly.graph_objects.Figure

    Examples
    --------
    fig = plot_dashboard(df, fc_min=47, fc_max=185)

    fig = plot_dashboard(
        df, fc_min=47, fc_max=185,
        variables={
            "heart_rate":    DASHBOARD_VARIABLES["heart_rate"],
            "pace_s_per_km": DASHBOARD_VARIABLES["pace_s_per_km"],
        }
    )
    """
    if variables is None:
        variables = {k: v for k, v in DASHBOARD_VARIABLES.items()
                     if k in df.columns}
    else:
        _validate_variables(variables)
        variables = {k: v for k, v in variables.items() if k in df.columns}

    if "heart_rate" in variables:
        label, cscale, _, _ = variables["heart_rate"]
        variables["heart_rate"] = (label, cscale, fc_min, fc_max)

    if not variables:
        raise ValueError("No valid columns found for the requested variables.")

    cols_list = list(variables.keys())
    n = len(cols_list)

    fig = make_subplots(
        rows=n, cols=1, shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[variables[c][0] for c in cols_list],
    )

    x_km_full = df["dist_m"] / 1000.0
    y_alt_full = df["alt_m"]
    y_top = float(y_alt_full.max())

    for row, col in enumerate(cols_list, start=1):
        label, colorscale, vmin, vmax = variables[col]
        mask = df[col].notna()
        x_km = df.loc[mask, "dist_m"] / 1000.0
        y_alt = df.loc[mask, "alt_m"]
        c_val = df.loc[mask, col]

        fig.add_trace(go.Scatter(
            x=x_km_full, y=y_alt_full,
            fill="tozeroy", fillcolor="rgba(180,180,180,0.2)",
            line=dict(color="rgba(180,180,180,0.3)", width=1),
            hoverinfo="skip", showlegend=False, name=f"Alt_{row}",
        ), row=row, col=1)

        fig.add_trace(go.Scatter(
            x=x_km, y=y_alt, mode="markers",
            marker=dict(
                color=c_val, colorscale=colorscale,
                cmin=vmin, cmax=vmax, size=3, opacity=0.85,
                showscale=True,
                colorbar=dict(
                    title=dict(text=label, side="right"),
                    thickness=12, len=1 / n,
                    y=1 - (row - 0.5) / n, yanchor="middle",
                ),
            ),
            name=label,
            hovertemplate=(
                "Distance : %{x:.2f} km<br>Altitude : %{y:.0f} m<br>"
                f"{label} : %{{marker.color:.1f}}<extra></extra>"
            ),
        ), row=row, col=1)

        fig.update_yaxes(title_text="Alt. (m)", row=row, col=1)

        for km, nom in zip(ravito_km or [], ravito_nom or [""]*len(ravito_km or [])):
            fig.add_vline(x=km, line=_RAVITO_LINE, row=row, col=1)
            if row == 1:
                fig.add_annotation(
                    x=km, y=y_top, xref="x", yref="y1",
                    text=nom, showarrow=False, font=_RAVITO_FONT,
                    textangle=-45, xanchor="left",
                )

    fig.update_xaxes(title_text="Distance (km)", row=n, col=1)
    fig.update_layout(
        height=height_per_panel * n, template="plotly_dark",
        showlegend=False, title="Dashboard — Profils physiologiques",
        margin=dict(l=60, r=90, t=50, b=50),
    )
    return fig


# ===========================================================================
# 7. Split analysis
# ===========================================================================

def compute_pace_split(df, ravito_km, ravito_nom):
    """Compute positive / negative split based on GAP.

    Parameters
    ----------
    df         : pd.DataFrame  must contain gap_s_per_km, dist_m
    ravito_km  : list[float]
    ravito_nom : list[str]

    Returns
    -------
    dict with keys: split_ratio, split_type, gap_half1, gap_half2, section_df
    """
    if "gap_s_per_km" not in df.columns:
        return None

    dist_max = df["dist_m"].max()
    mid = dist_max / 2.0
    mask1 = (df["dist_m"] <= mid) & df["gap_s_per_km"].notna() & (df["gap_s_per_km"] > 0)
    mask2 = (df["dist_m"] > mid) & df["gap_s_per_km"].notna() & (df["gap_s_per_km"] > 0)
    gap1 = float(df.loc[mask1, "gap_s_per_km"].median())
    gap2 = float(df.loc[mask2, "gap_s_per_km"].median())
    ratio = gap2 / gap1 if gap1 > 0 else np.nan

    if ratio < 0.97:
        split_type = "negatif (acceleration)"
    elif ratio <= 1.03:
        split_type = "equilibre"
    else:
        split_type = "positif (ralentissement)"

    bounds = np.concatenate((
        [df["dist_m"].min() / 1000.0],
        np.array(ravito_km, dtype=float),
        [df["dist_m"].max() / 1000.0]
    ))
    labels = build_section_labels_short(list(bounds), ravito_nom)

    rows = []
    ref_gap = None
    for i, (nom, a, b) in enumerate(zip(labels, bounds[:-1], bounds[1:])):
        mask = (
            (df["dist_m"] / 1000.0 >= a) &
            (df["dist_m"] / 1000.0 < b) &
            df["gap_s_per_km"].notna() &
            (df["gap_s_per_km"] > 0)
        )
        sec_gap = df.loc[mask, "gap_s_per_km"].median()
        if np.isnan(sec_gap):
            continue
        if ref_gap is None:
            ref_gap = sec_gap
        delta_pct = (sec_gap / ref_gap - 1.0) * 100.0
        m, s = int(sec_gap) // 60, int(sec_gap) % 60
        rows.append({
            "Section":            nom,
            "GAP med.":           f"{m}'{s:02d}\"",
            "GAP (s/km)":         round(sec_gap, 0),
            "D vs section 1 (%)": round(delta_pct, 1),
            "Tendance":           "=" if abs(delta_pct) < 3 else (
                                  "plus lent" if delta_pct > 0 else "plus rapide"),
        })

    return {
        "split_ratio": round(ratio, 3),
        "split_type":  split_type,
        "gap_half1":   round(gap1, 0),
        "gap_half2":   round(gap2, 0),
        "section_df":  pd.DataFrame(rows),
    }


def plot_pace_split(df, ravito_km, ravito_nom, height=700):
    """Two-panel split analysis : smoothed GAP + per-section delta.

    Parameters
    ----------
    df         : pd.DataFrame  must contain gap_s_per_km, dist_m
    ravito_km  : list[float]
    ravito_nom : list[str]
    height     : int

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if "gap_s_per_km" not in df.columns:
        raise ValueError("Colonne gap_s_per_km manquante.")

    result = compute_pace_split(df, ravito_km, ravito_nom)
    if result is None:
        raise ValueError("compute_pace_split a retourne None.")

    def fmt_gap(g):
        """Format gap seconds to MM'SS string."""
        return f"{int(g)//60}'{int(g)%60:02d}\""

    x_km = df["dist_m"] / 1000.0
    win = max(10, int(3000 / df["dist_m"].diff().median()))
    gap_smooth = (
        df["gap_s_per_km"]
        .where(df["gap_s_per_km"] < 1200)
        .rolling(win, center=True, min_periods=10)
        .median()
    )
    ref_gap = float(
        df.loc[df["dist_m"] <= df["dist_m"].max() / 2, "gap_s_per_km"].median()
    )
    mid_km = df["dist_m"].max() / 2000.0

    fig = make_subplots(
        rows=2, cols=1,
        vertical_spacing=0.12,
        subplot_titles=[
            (f"GAP lisse — Split : {result['split_type']} "
             f"(ratio {result['split_ratio']:.3f})  |  "
             f"1ere moitie : {fmt_gap(result['gap_half1'])}/km  "
             f"2eme moitie : {fmt_gap(result['gap_half2'])}/km"),
            "D GAP par section vs section 1  (positif = ralentissement)",
        ],
    )

    gap_arr = gap_smooth.to_numpy()
    x_arr = x_km.to_numpy()

    fig.add_trace(
        go.Scatter(x=x_arr, y=np.where(gap_arr > ref_gap, gap_arr, ref_gap),
                   fill="tonexty", fillcolor="rgba(220,50,50,0.12)",
                   line=dict(width=0), hoverinfo="skip", showlegend=False),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=x_arr, y=np.full_like(x_arr, ref_gap),
                   fill=None, line=dict(width=0), hoverinfo="skip", showlegend=False),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=x_arr, y=np.where(gap_arr < ref_gap, gap_arr, ref_gap),
                   fill="tonexty", fillcolor="rgba(50,180,80,0.12)",
                   line=dict(width=0), hoverinfo="skip", showlegend=False),
        row=1, col=1,
    )

    hover_gap = [
        f"Distance : {x:.2f} km<br>GAP : {fmt_gap(g)}/km"
        if not np.isnan(g) else ""
        for x, g in zip(x_arr, gap_arr)
    ]
    fig.add_trace(
        go.Scatter(x=x_arr, y=gap_arr, mode="lines",
                   line=dict(color="#4a9ede", width=2), name="GAP lisse",
                   hovertext=hover_gap, hoverinfo="text"),
        row=1, col=1,
    )

    fig.add_hline(y=ref_gap, line=dict(color="grey", dash="dash", width=1),
                  annotation_text=f"Ref. {fmt_gap(ref_gap)}/km",
                  annotation_font_size=10, row=1, col=1)
    fig.add_vline(x=mid_km, line=dict(color="white", dash="dot", width=1.2),
                  annotation_text="Mi-course", annotation_font_size=9,
                  row=1, col=1)

    for km, nom in zip(ravito_km, ravito_nom):
        fig.add_vline(x=km,
                      line=dict(color="rgba(180,180,180,0.4)", dash="dot", width=1),
                      row=1, col=1)

    fig.update_yaxes(title_text="GAP (s/km)", autorange="reversed", row=1, col=1)
    fig.update_xaxes(title_text="Distance (km)", row=1, col=1)

    sec_df = result["section_df"]
    deltas = sec_df["D vs section 1 (%)"].values
    bar_colors = [
        "seagreen" if d <= 0 else ("orange" if d <= 10 else "crimson")
        for d in deltas
    ]
    fig.add_trace(
        go.Bar(x=sec_df["Section"], y=deltas,
               marker_color=bar_colors,
               marker_line_color="rgba(255,255,255,0.3)", marker_line_width=1,
               text=[f"{d:+.1f}%" for d in deltas], textposition="outside",
               hovertemplate="%{x}<br>D : %{y:+.1f}%<br>GAP : %{customdata}<extra></extra>",
               customdata=sec_df["GAP med."]),
        row=2, col=1,
    )
    for y_val, color in [(5, "orange"), (-5, "seagreen"), (0, "white")]:
        fig.add_hline(y=y_val,
                      line=dict(color=color,
                                dash="dash" if y_val != 0 else "solid",
                                width=0.8),
                      row=2, col=1)

    fig.update_yaxes(title_text="D GAP vs section 1 (%)", row=2, col=1)
    fig.update_xaxes(tickangle=-20, row=2, col=1)
    fig.update_layout(height=height, template="plotly_dark", showlegend=False,
                      margin=dict(l=60, r=40, t=60, b=60))
    return fig


# ===========================================================================
# 7. Hitting the wall detection
# ===========================================================================

def detect_hitting_wall(df, ref_start_km=5.0, ref_end_km=None,
                        threshold_pct=25.0, min_duration_km=5.0):
    """Detect sustained pace degradation episodes.

    Parameters
    ----------
    df              : pd.DataFrame  must contain gap_s_per_km, dist_m
    ref_start_km    : float
    ref_end_km      : float | None  (default: first quarter)
    threshold_pct   : float  degradation threshold (%)
    min_duration_km : float  minimum episode duration (km)

    Returns
    -------
    dict with keys: ref_gap, threshold_gap, episodes (DataFrame), flagged (bool)

    References
    ----------
    - Prigent G et al. (2024). Front Physiol. PMC12575221.
    """
    if "gap_s_per_km" not in df.columns:
        return None

    dist_max_km = df["dist_m"].max() / 1000.0
    if ref_end_km is None:
        ref_end_km = dist_max_km * 0.25

    df = df.copy().sort_values("dist_m")
    gap = df["gap_s_per_km"].copy()
    gap = gap.where((gap > 0) & (gap < 1200))

    ref_mask = (
        (df["dist_m"] / 1000.0 >= ref_start_km) &
        (df["dist_m"] / 1000.0 <= ref_end_km) &
        gap.notna()
    )
    ref_gap = float(gap[ref_mask].median())
    threshold_gap = ref_gap * (1.0 + threshold_pct / 100.0)

    # Fix to avoid division by zero
    median_diff = df["dist_m"].diff().median()
    win = max(10, int(1000 / median_diff)) if median_diff > 0 else 10


    #win = max(10, int(1000 / df["dist_m"].diff().median()))
    gap_smooth = gap.rolling(win, center=True, min_periods=5).median()
    df["_flagged"] = gap_smooth > threshold_gap
    df["_block"] = (df["_flagged"] != df["_flagged"].shift(1)).cumsum()

    episodes = []
    for _, grp in df[df["_flagged"]].groupby("_block"):
        start_km = float(grp["dist_m"].iloc[0] / 1000.0)
        end_km = float(grp["dist_m"].iloc[-1] / 1000.0)
        duration = end_km - start_km
        if duration < min_duration_km:
            continue
        gap_ep = float(grp["gap_s_per_km"].median())
        deg_pct = (gap_ep / ref_gap - 1.0) * 100.0
        episodes.append({
            "Debut (km)":      round(start_km, 1),
            "Fin (km)":        round(end_km, 1),
            "Duree (km)":      round(duration, 1),
            "GAP med. ep.":    round(gap_ep, 0),
            "Degradation (%)": round(deg_pct, 1),
        })

    return {
        "ref_gap":       round(ref_gap, 0),
        "threshold_gap": round(threshold_gap, 0),
        "episodes":      pd.DataFrame(episodes),
        "flagged":       len(episodes) > 0,
    }


def plot_hitting_wall(df, ravito_km, ravito_nom,
                      ref_start_km=5.0, ref_end_km=None, ref_pct=None,
                      threshold_pct=25.0, min_duration_km=5.0,
                      show_elevation=False, height=450):
    """GAP profile with hitting-the-wall episodes highlighted.

    La fenêtre de référence peut être définie de trois façons,
    par ordre de priorité :
      1. ref_start_km + ref_end_km  : bornes absolues en km
      2. ref_start_km + ref_pct     : de ref_start_km à X% de la distance
      3. ref_start_km seul          : de ref_start_km au premier quart

    Parameters
    ----------
    df              : pd.DataFrame  must contain gap_s_per_km, dist_m
    ravito_km       : list[float]
    ravito_nom      : list[str]
    ref_start_km    : float   début de la fenêtre de référence (km). Default 5.0.
    ref_end_km      : float | None  fin de la fenêtre en km (prioritaire)
    ref_pct         : float | None  fin de la fenêtre en % de la distance
                      (ex. 25.0 = premier quart). Ignoré si ref_end_km fourni.
    threshold_pct   : float   seuil de dégradation (%). Default 25.
    min_duration_km : float   durée minimale d'un épisode (km). Default 5.
    show_elevation  : bool    profil altimétrique sur axe secondaire
    height          : int     hauteur de la figure (pixels)

    Returns
    -------
    plotly.graph_objects.Figure

    Examples
    --------
    # Fenêtre km 5 → km 15
    fig = plot_hitting_wall(df, ravito_km, ravito_nom,
                            ref_start_km=5.0, ref_end_km=15.0)

    # Fenêtre km 3 → 20% de la distance
    fig = plot_hitting_wall(df, ravito_km, ravito_nom,
                            ref_start_km=3.0, ref_pct=20.0)

    # Fenêtre par défaut : km 5 → premier quart
    fig = plot_hitting_wall(df, ravito_km, ravito_nom)
    """
    if "gap_s_per_km" not in df.columns:
        raise ValueError("Colonne gap_s_per_km manquante — appeler compute_gap() d'abord.")

    # Résolution de la fenêtre de référence
    dist_max_km = df["dist_m"].max() / 1000.0
    if ref_end_km is not None:
        ref_end = float(ref_end_km)
    elif ref_pct is not None:
        ref_end = dist_max_km * float(ref_pct) / 100.0
    else:
        ref_end = dist_max_km * 0.25

    ref_end = min(ref_end, dist_max_km)  # garde-fou

    result = detect_hitting_wall(
        df,
        ref_start_km    = ref_start_km,
        ref_end_km      = ref_end,
        threshold_pct   = threshold_pct,
        min_duration_km = min_duration_km,
    )
    if result is None:
        raise ValueError("detect_hitting_wall a retourné None.")

    def fmt_gap(g):
        """Format gap seconds to MM'SS string."""
        return f"{int(g)//60}'{int(g)%60:02d}\""

    x_km = df["dist_m"] / 1000.0

    #win = max(10, int(2000 / df["dist_m"].diff().median()))
    median_diff = df["dist_m"].diff().median()
    win = max(10, int(1000 / median_diff)) if median_diff > 0 else 10

    
    
    gap_smooth = (
        df["gap_s_per_km"]
        .where((df["gap_s_per_km"] > 0) & (df["gap_s_per_km"] < 1200))
        .rolling(win, center=True, min_periods=5)
        .median()
    )
    gap_arr = gap_smooth.to_numpy()
    y_annot = float(np.nanmax(gap_arr)) * 1.01 if np.any(~np.isnan(gap_arr)) else 400

    fig = go.Figure()

    # ── Fenêtre de référence ──────────────────────────────────────────────────
    fig.add_vrect(
        x0=ref_start_km, x1=ref_end,
        fillcolor="rgba(50,180,80,0.08)", line_width=0,
        annotation_text=(
            f"Fenêtre réf. [{ref_start_km:.1f}–{ref_end:.1f} km]"
            f"  GAP réf. : {fmt_gap(result['ref_gap'])}/km"
        ),
        annotation_position="top left",
        annotation_font=dict(size=9, color="seagreen"),
    )

    # ── Épisodes de dégradation ───────────────────────────────────────────────
    for _, ep in result["episodes"].iterrows():
        fig.add_vrect(
            x0=ep["Début (km)"], x1=ep["Fin (km)"],
            fillcolor="rgba(220,50,50,0.15)", line_width=0,
            annotation_text=f"⚠️ +{ep['Dégradation (%)']:.0f}%",
            annotation_position="top center",
            annotation_font=dict(color="crimson", size=10),
        )

    # ── Ravitos ───────────────────────────────────────────────────────────────
    for km, nom in zip(ravito_km, ravito_nom):
        fig.add_vline(
            x=km,
            line=dict(color="rgba(180,180,180,0.35)", dash="dot", width=1),
        )
        fig.add_annotation(
            x=km, y=y_annot, xref="x", yref="y",
            text=nom, showarrow=False,
            font=dict(size=9, color="rgba(180,180,180,0.7)"),
            textangle=-45, xanchor="left",
        )

    # ── Lignes horizontales ───────────────────────────────────────────────────
    fig.add_hline(
        y=result["ref_gap"],
        line=dict(color="seagreen", dash="dash", width=1.5),
        annotation_text=f"Réf. {fmt_gap(result['ref_gap'])}/km",
        annotation_font=dict(color="seagreen", size=10),
        annotation_position="bottom right",
    )
    fig.add_hline(
        y=result["threshold_gap"],
        line=dict(color="crimson", dash="dash", width=1.5),
        annotation_text=(
            f"Seuil ×{1 + threshold_pct/100:.2f} "
            f"{fmt_gap(result['threshold_gap'])}/km"
        ),
        annotation_font=dict(color="crimson", size=10),
        annotation_position="top right",
    )

    # ── Courbe GAP lissée ─────────────────────────────────────────────────────
    hover = [
        f"Distance : {x:.2f} km<br>GAP : {fmt_gap(g)}/km"
        if not np.isnan(g) else ""
        for x, g in zip(x_km, gap_arr)
    ]
    fig.add_trace(go.Scatter(
        x=x_km, y=gap_arr,
        mode="lines",
        line=dict(color="#4a9ede", width=2),
        name="GAP lissé",
        hovertext=hover, hoverinfo="text",
    ))

    # ── Profil altimétrique optionnel (axe secondaire) ────────────────────────
    if show_elevation and "alt_m" in df.columns:
        alt_range = df["alt_m"].max() - df["alt_m"].min()
        fig.add_trace(go.Scatter(
            x=x_km, y=df["alt_m"],
            fill="tozeroy",
            fillcolor="rgba(139,90,43,0.08)",
            line=dict(color="rgba(139,90,43,0.2)", width=1),
            hoverinfo="skip", showlegend=False, name="Altitude",
            yaxis="y2",
        ))
        fig.update_layout(yaxis2=dict(
            title="Altitude (m)",
            overlaying="y", side="right",
            range=[df["alt_m"].min() - 50,
                   df["alt_m"].max() + alt_range * 2.0],
            showgrid=False,
            color="rgba(139,90,43,0.4)",
        ))

    n_ep = len(result["episodes"])
    status = (f"⚠️ {n_ep} épisode(s) détecté(s)"
              if result["flagged"] else "✅ Aucune dégradation soutenue")
    fig.update_layout(
        title=(
            f"Hitting the Wall — seuil +{threshold_pct:.0f}% "
            f"sur >{min_duration_km:.0f} km  |  {status}<br>"
            "<sup>Adapté de Prigent et al. (2024) — Front Physiol.</sup>"
        ),
        xaxis_title="Distance (km)",
        yaxis_title="GAP (s/km)",
        yaxis_autorange="reversed",
        height=height,
        template="plotly_dark",
        showlegend=False,
        margin=dict(l=60, r=80, t=80, b=50),
    )
    return fig

# ===========================================================================
# 8. Multi-race utilities (non-graphiques)
# ===========================================================================

def normalize_by_distance_pct(df, n_bins=100, cols=None):
    """Resample a race DataFrame onto a uniform 0-100% distance grid.

    Parameters
    ----------
    df     : pd.DataFrame  must contain dist_m
    n_bins : int
    cols   : list | None

    Returns
    -------
    pd.DataFrame
    """
    if cols is None:
        cols = [c for c in ["gap_s_per_km", "heart_rate", "speed_kmh",
                             "alt_m", "slope_pct", "is_walk", "cadence"]
                if c in df.columns]

    dist_pct = np.linspace(0.0, 100.0, n_bins)
    dist_m_grid = dist_pct / 100.0 * df["dist_m"].max()
    result = pd.DataFrame({"dist_pct": dist_pct})
    for col in cols:
        vals = df[col].to_numpy(dtype=float)
        dist = df["dist_m"].to_numpy(dtype=float)
        mask = ~np.isnan(vals)
        if mask.sum() > 2:
            result[col] = np.interp(dist_m_grid, dist[mask], vals[mask])
        else:
            result[col] = np.nan
    return result


def build_races_table(races_list):
    """Build a comparison DataFrame from a list of race dicts.

    Parameters
    ----------
    races_list : list[dict]

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for race in races_list:
        row = {"name": race["meta"]["name"], "date": race["meta"]["date"]}
        row.update(race["kpis"])
        rows.append(row)
    df_table = pd.DataFrame(rows)
    if "date" in df_table.columns:
        df_table = df_table.sort_values("date").reset_index(drop=True)
    return df_table


# ===========================================================================
# 9. Multi-race plots (Plotly)
# ===========================================================================

_METRIC_LABELS = {
    "gap_med_s_km":   "GAP median (s/km)",
    "cv_gap_pct":     "CV GAP (%)",
    "split_ratio":    "Ratio split (GAP2/GAP1)",
    "fc_frac":        "FC moy / FC max",
    "pct_walk":       "% marche",
    "decoupling_max": "Decouplage max (%)",
    "speed_kmh":      "Vitesse moy (km/h)",
    "dplus_m":        "D+ (m)",
}

_TAB10 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def plot_races_comparison(df_table, metrics=None, date_col="date",
                          name_col="name", height_per_row=320):
    """Temporal evolution of scalar KPIs across races.

    Parameters
    ----------
    df_table      : pd.DataFrame
    metrics       : list | None
    date_col      : str
    name_col      : str
    height_per_row: int

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if metrics is None:
        metrics = [m for m in _METRIC_LABELS if m in df_table.columns]
    if not metrics:
        raise ValueError("Aucune metrique disponible dans df_table.")

    ncols = min(len(metrics), 3)
    nrows = int(np.ceil(len(metrics) / ncols))
    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=[_METRIC_LABELS.get(m, m) for m in metrics],
        vertical_spacing=0.12, horizontal_spacing=0.08,
    )

    x_vals = (
        df_table[date_col].astype(str).tolist()
        if date_col in df_table.columns
        else list(range(len(df_table)))
    )
    names = (
        df_table[name_col].tolist()
        if name_col in df_table.columns
        else x_vals
    )

    for idx, metric in enumerate(metrics):
        row = idx // ncols + 1
        col = idx % ncols + 1
        vals = df_table[metric].values.astype(float)
        valid = ~np.isnan(vals)
        x_valid = [x_vals[i] for i in range(len(vals)) if valid[i]]
        y_valid = vals[valid]
        n_valid = [names[i] for i in range(len(vals)) if valid[i]]

        fig.add_trace(
            go.Scatter(
                x=x_valid, y=y_valid,
                mode="lines+markers+text",
                marker=dict(size=8, color="#4a9ede"),
                line=dict(color="#4a9ede", width=2),
                text=n_valid, textposition="top center",
                textfont=dict(size=8),
                hovertemplate=(
                    "%{text}<br>" + _METRIC_LABELS.get(metric, metric)
                    + " : %{y:.2f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row, col=col,
        )

        x_idx = np.where(valid)[0]
        if len(x_idx) >= 3:
            p = np.polyfit(x_idx, y_valid, 1)
            fig.add_trace(
                go.Scatter(x=x_valid, y=np.polyval(p, x_idx),
                           mode="lines",
                           line=dict(color="grey", dash="dash", width=1),
                           showlegend=False, hoverinfo="skip"),
                row=row, col=col,
            )

        fig.update_yaxes(
            title_text=_METRIC_LABELS.get(metric, metric), row=row, col=col
        )

    fig.update_layout(
        height=height_per_row * nrows, template="plotly_dark",
        title="Evolution des metriques de performance — multi-courses",
        showlegend=False, margin=dict(l=60, r=40, t=80, b=50),
    )
    return fig


def plot_normalized_profiles(races_list, col="gap_s_per_km",
                              n_bins=100, smooth_bins=5,
                              show_mean=True, show_ci=True, height=500):
    """Superposed normalized profiles (0-100% distance) for several races.

    Parameters
    ----------
    races_list  : list[dict]
    col         : str
    n_bins      : int
    smooth_bins : int
    show_mean   : bool
    show_ci     : bool
    height      : int

    Returns
    -------
    plotly.graph_objects.Figure

    References
    ----------
    - Kerheve HA et al. (2015). PLoS ONE 10(12):e0145482.
    """
    labels_map = {
        "gap_s_per_km": "GAP (s/km)", "heart_rate": "FC (bpm)",
        "speed_kmh": "Vitesse (km/h)", "is_walk": "Marche (0/1)",
        "cadence": "Cadence (spm)", "alt_m": "Altitude (m)",
    }
    ylabel = labels_map.get(col, col)
    dist_pct = np.linspace(0, 100, n_bins)
    fig = go.Figure()
    all_profiles = []

    for i, race in enumerate(races_list):
        df = race["df"]
        if col not in df.columns:
            continue
        norm_df = normalize_by_distance_pct(df, n_bins=n_bins, cols=[col])
        profile = norm_df[col].values.astype(float)
        if smooth_bins > 1:
            profile = (pd.Series(profile)
                       .rolling(smooth_bins, center=True, min_periods=1)
                       .mean().values)
        if col == "gap_s_per_km":
            profile = np.where(profile > 1200, np.nan, profile)

        label = f"{race['meta']['name']} ({race['meta']['date']})"
        color = _TAB10[i % len(_TAB10)]
        fig.add_trace(
            go.Scatter(x=dist_pct, y=profile, mode="lines",
                       line=dict(color=color, width=2), opacity=0.8,
                       name=label,
                       hovertemplate=(
                           "Distance : %{x:.0f}%<br>" + ylabel
                           + " : %{y:.1f}" + f"<extra>{label}</extra>"
                       ))
        )
        all_profiles.append(profile)

    if len(all_profiles) >= 2:
        stack = np.vstack(all_profiles)
        mean_p = np.nanmean(stack, axis=0)
        std_p = np.nanstd(stack, axis=0)
        if show_ci:
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([dist_pct, dist_pct[::-1]]),
                    y=np.concatenate([mean_p + std_p, (mean_p - std_p)[::-1]]),
                    fill="toself", fillcolor="rgba(255,255,255,0.06)",
                    line=dict(width=0), showlegend=True,
                    name="+/- 1 ecart-type", hoverinfo="skip",
                )
            )
        if show_mean:
            fig.add_trace(
                go.Scatter(x=dist_pct, y=mean_p, mode="lines",
                           line=dict(color="white", dash="dash", width=2.5),
                           name="Moyenne",
                           hovertemplate=(
                               "Distance : %{x:.0f}%<br>Moy "
                               + ylabel + " : %{y:.1f}<extra>Moyenne</extra>"
                           ))
            )

    fig.update_layout(
        title=(f"Profils normalises — {ylabel} sur 0-100% de la distance<br>"
               "<sup>Kerheve et al. (2015) — PLoS ONE</sup>"),
        xaxis_title="Distance (% de la course)", yaxis_title=ylabel,
        yaxis_autorange="reversed" if col == "gap_s_per_km" else True,
        height=height, template="plotly_dark",
        legend=dict(orientation="v", x=1.02, y=1, xanchor="left"),
        margin=dict(l=60, r=160, t=80, b=50),
    )
    return fig


def plot_decay_model(races_list, col="gap_s_per_km", n_bins=100,
                     degree=2, height=500):
    """Polynomial decay model fitted on mean normalized profile.

    Parameters
    ----------
    races_list : list[dict]
    col        : str
    n_bins     : int
    degree     : int
    height     : int

    Returns
    -------
    plotly.graph_objects.Figure

    References
    ----------
    - Matta GG et al. (2020). Eur J Sport Sci 20(3):347-356.
    - Kerheve HA et al. (2015). PLoS ONE 10(12):e0145482.
    """
    dist_pct = np.linspace(0, 100, n_bins)
    all_profiles, race_labels = [], []

    for race in races_list:
        df = race["df"]
        if col not in df.columns:
            continue
        norm_df = normalize_by_distance_pct(df, n_bins=n_bins, cols=[col])
        profile = norm_df[col].values.astype(float)
        if col == "gap_s_per_km":
            profile = np.where(profile > 1200, np.nan, profile)
        all_profiles.append(profile)
        race_labels.append(race["meta"]["name"])

    if len(all_profiles) < 2:
        raise ValueError("Au moins 2 courses necessaires.")

    stack = np.vstack(all_profiles)
    mean_p = np.nanmean(stack, axis=0)
    valid = ~np.isnan(mean_p)
    coeffs = (np.polyfit(dist_pct[valid], mean_p[valid], degree)
              if valid.sum() > degree + 1 else None)
    fitted = np.polyval(coeffs, dist_pct) if coeffs is not None else mean_p

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Profils normalises + modele de declin",
                        "Ecart au modele (positif = plus lent)"],
        horizontal_spacing=0.1,
    )

    for i, (profile, label) in enumerate(zip(all_profiles, race_labels)):
        color = _TAB10[i % len(_TAB10)]
        fig.add_trace(
            go.Scatter(x=dist_pct, y=profile, mode="lines",
                       line=dict(color=color, width=1.8), opacity=0.75,
                       name=label,
                       hovertemplate=(
                           "Distance : %{x:.0f}%<br>GAP : %{y:.0f} s/km"
                           + f"<extra>{label}</extra>"
                       )),
            row=1, col=1,
        )
        residual = profile - fitted
        fig.add_trace(
            go.Scatter(x=dist_pct, y=residual, mode="lines",
                       line=dict(color=color, width=1.8), opacity=0.75,
                       showlegend=False,
                       hovertemplate=(
                           "Distance : %{x:.0f}%<br>Residu : %{y:+.1f} s/km"
                           + f"<extra>{label}</extra>"
                       )),
            row=1, col=2,
        )

    fig.add_trace(
        go.Scatter(x=dist_pct, y=fitted, mode="lines",
                   line=dict(color="white", dash="dash", width=2.5),
                   name=f"Modele (deg. {degree})", hoverinfo="skip"),
        row=1, col=1,
    )
    fig.add_hline(y=0, line=dict(color="white", dash="dash", width=1),
                  row=1, col=2)
    fig.add_hrect(y0=-10, y1=10, fillcolor="rgba(50,180,80,0.06)",
                  line_width=0, row=1, col=2)

    fig.update_yaxes(
        autorange="reversed" if col == "gap_s_per_km" else True,
        title_text="GAP (s/km)", row=1, col=1,
    )
    fig.update_yaxes(title_text="Residu vs modele (s/km)", row=1, col=2)
    fig.update_xaxes(title_text="Distance (% de la course)")
    fig.update_layout(
        height=height, template="plotly_dark",
        title=("Modele de declin d'allure individuel<br>"
               "<sup>Matta et al. (2020) / Kerheve et al. (2015)</sup>"),
        legend=dict(orientation="v", x=1.02, y=1, xanchor="left"),
        margin=dict(l=60, r=160, t=80, b=50),
    )
    return fig


def plot_pace_vs_slope_overlay(races, bins_slope=None, show_minetti=True,
                                height=450):
    """Pace and HR vs slope for multiple races.

    Parameters
    ----------
    races        : list[dict]
    bins_slope   : list | None
    show_minetti : bool
    height       : int

    Returns
    -------
    plotly.graph_objects.Figure

    References
    ----------
    - Minetti AE et al. (2002). J Appl Physiol 93(3):1039-1046.
    """
    if bins_slope is None:
        bins_slope = [-30, -15, -10, -7, -5, -3, -1, 1, 3, 5, 7, 10, 15, 30]

    has_fc = any("heart_rate" in r["df"].columns for r in races)
    ncols = 2 if has_fc else 1
    titles = ["Allure vs pente"] + (["FC vs pente"] if has_fc else [])
    fig = make_subplots(rows=1, cols=ncols, subplot_titles=titles,
                        horizontal_spacing=0.1)

    for i, race in enumerate(races):
        color = _TAB10[i % len(_TAB10)]
        label = race["meta"].get("name", f"Course {i+1}")
        df = race["df"].copy()
        df["slope_bin"] = pd.cut(df["slope_pct"], bins=bins_slope)

        cols_needed = ["slope_pct", "pace_s_per_km"]
        if "heart_rate" in df.columns:
            cols_needed.append("heart_rate")
        tmp = df.dropna(subset=cols_needed)
        if tmp.empty:
            continue

        agg = {"n": ("slope_pct", "size"), "slope_med": ("slope_pct", "median"),
               "pace_med": ("pace_s_per_km", "median")}
        if "heart_rate" in df.columns:
            agg["hr_med"] = ("heart_rate", "median")

        summary = tmp.groupby("slope_bin", observed=True).agg(**agg).reset_index()
        summary = summary[summary["n"] >= 20]
        if summary.empty:
            continue

        fig.add_trace(
            go.Scatter(x=summary["slope_med"], y=summary["pace_med"],
                       mode="lines+markers",
                       marker=dict(size=7, color=color),
                       line=dict(color=color, width=2), name=label,
                       hovertemplate=(
                           "Pente : %{x:.1f}%<br>Allure : %{y:.0f} s/km"
                           + f"<extra>{label}</extra>"
                       )),
            row=1, col=1,
        )

        if show_minetti:
            slopes_ref = np.linspace(
                float(summary["slope_med"].min()),
                float(summary["slope_med"].max()), 200
            )
            flat_mask = summary["slope_med"].abs() < 3
            if flat_mask.sum() == 0:
                flat_mask = summary["slope_med"].abs() < 5
            pace_flat = float(summary.loc[flat_mask, "pace_med"].median())
            minetti_pace = pace_flat * minetti_cost_ratio(slopes_ref)
            fig.add_trace(
                go.Scatter(x=slopes_ref, y=minetti_pace, mode="lines",
                           line=dict(color=color, dash="dash", width=1.2),
                           opacity=0.6, name=f"{label} — Minetti",
                           hoverinfo="skip"),
                row=1, col=1,
            )

        if has_fc and "hr_med" in summary.columns:
            fig.add_trace(
                go.Scatter(x=summary["slope_med"], y=summary["hr_med"],
                           mode="lines+markers",
                           marker=dict(size=7, color=color),
                           line=dict(color=color, width=2),
                           showlegend=False,
                           hovertemplate=(
                               "Pente : %{x:.1f}%<br>FC : %{y:.0f} bpm"
                               + f"<extra>{label}</extra>"
                           )),
                row=1, col=2,
            )

    fig.update_xaxes(title_text="Pente mediane (%)")
    fig.update_yaxes(title_text="Allure (s/km)", row=1, col=1)
    if has_fc:
        fig.update_yaxes(title_text="FC mediane (bpm)", row=1, col=2)
    fig.update_layout(
        height=height, template="plotly_dark",
        title=("Allure et FC par classe de pente<br>"),
        margin=dict(l=60, r=40, t=80, b=50),
    )
    return fig


def plot_pace_vs_slope_deviation(races, bins_slope=None, height=400):
    """Relative deviation from Minetti (2002) predicted pace per slope bin.

    Parameters
    ----------
    races      : list[dict]
    bins_slope : list | None
    height     : int

    Returns
    -------
    plotly.graph_objects.Figure

    References
    ----------
    - Minetti AE et al. (2002). J Appl Physiol 93(3):1039-1046.
    """
    if bins_slope is None:
        bins_slope = [-30, -15, -10, -7, -5, -3, -1, 1, 3, 5, 7, 10, 15, 30]

    fig = go.Figure()

    for i, race in enumerate(races):
        color = _TAB10[i % len(_TAB10)]
        label = race["meta"].get("name", f"Course {i+1}")
        df = race["df"].copy()
        df["slope_bin"] = pd.cut(df["slope_pct"], bins=bins_slope)

        tmp = df.dropna(subset=["slope_pct", "pace_s_per_km"])
        if tmp.empty:
            continue

        summary = (
            tmp.groupby("slope_bin", observed=True)
            .agg(n=("slope_pct", "size"),
                 slope_med=("slope_pct", "median"),
                 pace_med=("pace_s_per_km", "median"))
            .reset_index()
        )
        summary = summary[summary["n"] >= 20]
        if summary.empty:
            continue

        flat_mask = summary["slope_med"].abs() < 3
        if flat_mask.sum() == 0:
            flat_mask = summary["slope_med"].abs() < 5
        pace_flat = float(summary.loc[flat_mask, "pace_med"].median())
        minetti_pace = pace_flat * minetti_cost_ratio(summary["slope_med"].values)
        deviation = (summary["pace_med"].values - minetti_pace) / minetti_pace

        fig.add_trace(
            go.Scatter(x=summary["slope_med"], y=deviation,
                       mode="lines+markers",
                       marker=dict(size=7, color=color),
                       line=dict(color=color, width=2), name=label,
                       hovertemplate=(
                           "Pente : %{x:.1f}%<br>Ecart Minetti : %{y:+.1%}"
                           + f"<extra>{label}</extra>"
                       ))
        )

    fig.add_hline(y=0, line=dict(color="white", dash="dash", width=1))
    fig.add_hrect(y0=-0.05, y1=0.05,
                  fillcolor="rgba(50,180,80,0.07)", line_width=0)
    fig.update_layout(
        title=("Ecart relatif a Minetti par classe de pente<br>"),
        xaxis_title="Pente mediane (%)",
        yaxis_title="Ecart relatif a Minetti",
        yaxis_tickformat="+.0%",
        height=400, template="plotly_dark",
        margin=dict(l=60, r=40, t=80, b=50),
    )
    return fig


# ===========================================================================
# 10. Aerobic decoupling
# ===========================================================================

def compute_aerobic_decoupling(df, ravito_km, ravito_nom,
                                threshold_pct=2.5):
    """Compute aerobic decoupling per section — first half vs second half.

    Follows the TrainingPeaks / Coggan method : the race is split at the
    midpoint (by distance). The HR/GAP ratio of the first half serves as
    the reference; the decoupling of each section is expressed relative
    to that reference.

    This is consistent with the canonical Pa:Hr definition used in
    TrainingPeaks and validated in Maunder et al. (2021).

    Parameters
    ----------
    df            : pd.DataFrame  must contain heart_rate, gap_s_per_km, dist_m
    ravito_km     : list[float]
    ravito_nom    : list[str]
    threshold_pct : float         alert threshold (%). Default 2.5.

    Returns
    -------
    pd.DataFrame  one row per section with decoupling metrics,
                  plus a "Première moitié" reference row.

    References
    ----------
    - TrainingPeaks / Coggan A (2003). Pa:Hr aerobic decoupling method.
    - Maunder E et al. (2021). Sports Med 51(8):1648-1651.
    - Smyth B et al. (2022). Front Physiol.
    """
    if "heart_rate" not in df.columns or "gap_s_per_km" not in df.columns:
        return pd.DataFrame()

    df = df.copy().sort_values("dist_m")
    gap_kmh = 3600.0 / df["gap_s_per_km"].replace(0, np.nan)
    df["hr_gap_ratio"] = df["heart_rate"] / gap_kmh

    # Référence : première moitié de la course (par distance)
    dist_mid = df["dist_m"].max() / 2.0
    ref_mask = (df["dist_m"] <= dist_mid) & df["hr_gap_ratio"].notna()
    ref_ratio = df.loc[ref_mask, "hr_gap_ratio"].median()
    if np.isnan(ref_ratio) or ref_ratio == 0:
        return pd.DataFrame()

    bounds = np.concatenate((
        [df["dist_m"].min() / 1000.0],
        np.array(ravito_km),
        [df["dist_m"].max() / 1000.0]
    ))
    section_labels = build_section_labels_short(list(bounds), ravito_nom)
    labels = [
        (section_labels[i], bounds[i], bounds[i + 1])
        for i in range(len(bounds) - 1)
    ]

    rows = []
    for nom, a, b in labels:
        mask = (
            (df["dist_m"] / 1000.0 >= a) &
            (df["dist_m"] / 1000.0 < b) &
            df["hr_gap_ratio"].notna()
        )
        sec_ratio = df.loc[mask, "hr_gap_ratio"].median()
        if np.isnan(sec_ratio):
            continue
        decoupling_pct = (sec_ratio / ref_ratio - 1.0) * 100.0
        rows.append({
            "Section":           nom,
            "Ratio méd. FC/GAP": round(sec_ratio, 3),
            "Découplage (%)":    round(decoupling_pct, 1),
            "Alerte":            "⚠️" if decoupling_pct > threshold_pct else "✅",
        })

    result = pd.DataFrame(rows)
    result.attrs["ref_ratio"] = ref_ratio
    result.attrs["threshold_pct"] = threshold_pct
    return result


def plot_aerobic_decoupling(df, ravito_km, ravito_nom,
                             smoothing_km=2.0, threshold_pct=2.5,
                             show_elevation=False, height=400):
    """Continuous aerobic decoupling curve along the race.

    Follows the TrainingPeaks / Coggan method : the HR/GAP ratio is
    normalized by its median over the first half of the race (by distance).
    A dashed line marks the alert threshold.

    Parameters
    ----------
    df              : pd.DataFrame  must contain heart_rate, gap_s_per_km, dist_m
    ravito_km       : list[float]
    ravito_nom      : list[str]
    smoothing_km    : float  rolling window in km
    threshold_pct   : float  alert threshold (%)
    show_elevation  : bool   overlay elevation profile on secondary y-axis
    height          : int    figure height in pixels

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if "heart_rate" not in df.columns or "gap_s_per_km" not in df.columns:
        raise ValueError("Colonnes heart_rate ou gap_s_per_km manquantes.")

    df = df.copy().sort_values("dist_m")
    gap_kmh = 3600.0 / df["gap_s_per_km"].replace(0, np.nan)
    df["hr_gap_ratio"] = df["heart_rate"] / gap_kmh

    window_pts = max(10, int(smoothing_km * 1000 / df["dist_m"].diff().median()))
    df["ratio_smooth"] = (
        df["hr_gap_ratio"]
        .rolling(window_pts, center=True, min_periods=5)
        .median()
    )

    # Référence : première moitié de la course (méthode TrainingPeaks/Coggan)
    dist_mid_m = df["dist_m"].max() / 2.0
    ref_mask = df["dist_m"] <= dist_mid_m
    ref_ratio = df.loc[ref_mask, "ratio_smooth"].median()
    df["decoupling_pct"] = (df["ratio_smooth"] / ref_ratio - 1.0) * 100.0

    x_km = df["dist_m"] / 1000.0
    dec = df["decoupling_pct"].to_numpy()
    y_top = float(np.nanmax(dec)) * 1.05

    fig = go.Figure()

    # Zone d'alerte
    fig.add_hrect(
        y0=threshold_pct, y1=max(float(np.nanmax(dec)) * 1.1, threshold_pct + 5),
        fillcolor="rgba(220,50,50,0.08)", line_width=0,
    )

    # Ravitos
    for km, nom in zip(ravito_km, ravito_nom):
        fig.add_vline(
            x=km, line=dict(color="rgba(180,180,180,0.35)", dash="dot", width=1),
        )
        fig.add_annotation(
            x=km, y=y_top, xref="x", yref="y",
            text=nom, showarrow=False,
            font=dict(size=9, color="rgba(180,180,180,0.7)"),
            textangle=-45, xanchor="left",
        )

    # Ligne zéro et seuil
    fig.add_hline(y=0, line=dict(color="grey", dash="dash", width=0.8))
    fig.add_hline(
        y=threshold_pct,
        line=dict(color="crimson", dash="dash", width=1.5),
        annotation_text=f"Seuil +{threshold_pct}%",
        annotation_font=dict(color="crimson", size=10),
        annotation_position="top right",
    )

    # Zone colorée au-dessus du seuil
    fig.add_trace(go.Scatter(
        x=x_km, y=np.where(dec > threshold_pct, dec, threshold_pct),
        fill="tonexty", fillcolor="rgba(220,50,50,0.12)",
        line=dict(width=0), hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=x_km, y=np.full(len(x_km), threshold_pct),
        fill=None, line=dict(width=0),
        hoverinfo="skip", showlegend=False,
    ))

    # Courbe principale
    fig.add_trace(go.Scatter(
        x=x_km, y=dec,
        mode="lines",
        line=dict(color="#b06ecc", width=2),
        name="Découplage FC/GAP",
        hovertemplate=(
            "Distance : %{x:.2f} km<br>"
            "Découplage : %{y:+.1f}%<extra></extra>"
        ),
    ))

    # ── Profil altimétrique optionnel (axe secondaire) ────────────────────────
    if show_elevation and "alt_m" in df.columns:
        alt_range = df["alt_m"].max() - df["alt_m"].min()
        fig.add_trace(
            go.Scatter(
                x=x_km, y=df["alt_m"],
                fill="tozeroy",
                fillcolor="rgba(139,90,43,0.08)",
                line=dict(color="rgba(139,90,43,0.2)", width=1),
                hoverinfo="skip", showlegend=False, name="Altitude",
                yaxis="y2",
            )
        )
        fig.update_layout(
            yaxis2=dict(
                title="Altitude (m)",
                overlaying="y", side="right",
                range=[df["alt_m"].min() - 50,
                       df["alt_m"].max() + alt_range * 2.0],
                showgrid=False,
                color="rgba(139,90,43,0.4)",
            )
        )

    fig.update_layout(
        title=(
            "D\xe9couplage a\xe9robie \u2014 ratio FC/GAP : 1\xe8re moiti\xe9 vs 2\xe8me moiti\xe9 (m\xe9thode TrainingPeaks/Coggan)<br>"
            "<sup>Smyth et al. (2022) / Maunder et al. (2021)</sup>"
        ),
        xaxis_title="Distance (km)",
        yaxis_title="D\xe9couplage FC/GAP (%)",
        height=height,
        template="plotly_dark",
        showlegend=False,
        margin=dict(l=60, r=60, t=80, b=50),
    )
    return fig


# ===========================================================================
# 11. Stride metrics
# ===========================================================================

def compute_stride_metrics(df, ravito_km, ravito_nom):
    """Compute stride length and cadence variability per section.

    Stride length = speed_mps / (cadence_hz) [m per stride = 2 steps].
    Cadence CV = coefficient of variation per section (fatigue proxy).

    Parameters
    ----------
    df         : pd.DataFrame  must contain speed_kmh, cadence, dist_m
    ravito_km  : list[float]
    ravito_nom : list[str]

    Returns
    -------
    pd.DataFrame  stride metrics per section
    """
    if "cadence" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    cadence_hz = df["cadence"] / 60.0
    speed_mps = df["speed_kmh"] / 3.6
    df["stride_length_m"] = np.where(
        cadence_hz > 0, speed_mps / cadence_hz, np.nan
    )

    bounds = np.concatenate((
        [df["dist_m"].min() / 1000.0],
        np.array(ravito_km),
        [df["dist_m"].max() / 1000.0]
    ))
    section_labels = build_section_labels_short(list(bounds), ravito_nom)
    labels = [
        (section_labels[i], bounds[i], bounds[i + 1])
        for i in range(len(bounds) - 1)
    ]

    rows = []
    for nom, a, b in labels:
        mask = (
            (df["dist_m"] / 1000.0 >= a) &
            (df["dist_m"] / 1000.0 < b) &
            df["cadence"].notna() &
            (df["speed_kmh"] > 4.0)
        )
        sec = df.loc[mask]
        if len(sec) < 20:
            continue
        cad_mean = sec["cadence"].mean()
        cad_std = sec["cadence"].std()
        rows.append({
            "Section":           nom,
            "Cadence méd. (spm)": round(sec["cadence"].median(), 1),
            "CV cadence (%)":    round(cad_std / cad_mean * 100, 1),
            "Foulée méd. (m)":   round(sec["stride_length_m"].median(), 2),
        })
    return pd.DataFrame(rows)


# ===========================================================================
# 12. Weather — ERA5-Land via openmeteo-requests SDK
# ===========================================================================

try:
    import openmeteo_requests
    _OPENMETEO_AVAILABLE = True
except ImportError:
    _OPENMETEO_AVAILABLE = False
    warnings.warn(
        "openmeteo-requests non installé. "
        "Installer avec : pip install openmeteo-requests\n"
        "Les fonctions météo seront indisponibles.",
        ImportWarning,
        stacklevel=2,
    )

_HOURLY_VARS = [
    "temperature_2m",        # 0  °C
    "relative_humidity_2m",  # 1  %
    "apparent_temperature",  # 2  °C
    "precipitation",         # 3  mm
    "wind_speed_10m",        # 4  km/h
    "shortwave_radiation",   # 5  W/m²
]

_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def compute_wbgt(temp_c, humidity_pct, solar_rad=None):
    """Wet Bulb Globe Temperature — Bernard & Kenney (1994).

    Identical to core/ffm/loads.py for consistency with Twinity.
    Works on scalars or numpy arrays.

    Parameters
    ----------
    temp_c       : float or array  air temperature (°C)
    humidity_pct : float or array  relative humidity (%)
    solar_rad    : float or array | None  global radiation (W/m²)

    Returns
    -------
    float or array  WBGT (°C)

    References
    ----------
    - Bernard TE & Kenney WL (1994). AIHAJ 55(1):32-36.
    - Liljegren JC et al. (2008). J Occup Environ Hyg 5(10):645-655.
    """
    scalar = np.ndim(temp_c) == 0
    t = np.asarray(temp_c, dtype=float)
    rh = np.asarray(humidity_pct, dtype=float)
    e_a = (rh / 100.0) * 6.105 * np.exp(17.27 * t / (237.7 + t))
    wbgt = 0.567 * t + 0.393 * e_a + 3.94
    if solar_rad is not None:
        sr = np.asarray(solar_rad, dtype=float)
        wbgt = np.where(sr > 0, wbgt + 0.0006 * sr, wbgt)
    if scalar:
        return round(float(wbgt), 2)
    return np.round(wbgt, 2)


def fetch_weather_hourly(lat, lon, date_str, timezone="Europe/Paris",
                         client=None, model="era5_land", date_end=None):
    """Fetch hourly reanalysis data via openmeteo-requests SDK.

    Parameters
    ----------
    lat      : float  latitude WGS84
    lon      : float  longitude WGS84
    date_str : str    start date 'YYYY-MM-DD'
    timezone : str    IANA timezone (default: 'Europe/Paris')
    client   : openmeteo_requests.Client | None
    model    : str    reanalysis model (default: 'era5_land')
    date_end : str | None  end date (for races crossing midnight)

    Returns
    -------
    pd.DataFrame  columns: time, temperature_2m, relative_humidity_2m,
                  apparent_temperature, precipitation, wind_speed_10m,
                  shortwave_radiation, wbgt — or None on error.

    References
    ----------
    - Muñoz Sabater J (2019). ERA5-Land hourly data. ECMWF.
      https://doi.org/10.24381/cds.e2161bac
    """
    if not _OPENMETEO_AVAILABLE:
        warnings.warn("openmeteo-requests indisponible — météo ignorée.")
        return None

    if client is None:
        client = openmeteo_requests.Client()

    end_str = date_end if date_end is not None else date_str
    params = {
        "latitude":        lat,
        "longitude":       lon,
        "start_date":      date_str,
        "end_date":        end_str,
        "hourly":          _HOURLY_VARS,
        "wind_speed_unit": "kmh",
        "timezone":        timezone,
        "models":          model,
    }

    try:
        responses = client.weather_api(_ARCHIVE_URL, params=params)
        r = responses[0]
        hourly = r.Hourly()

        times = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )

        def safe_var(i):
            """Extract variable i; return NaN array if unavailable."""
            try:
                arr = hourly.Variables(i).ValuesAsNumpy()
                return arr if arr is not None else np.full(len(times), np.nan)
            except Exception:
                return np.full(len(times), np.nan)

        df_w = pd.DataFrame({
            "time":                 times,
            "temperature_2m":       safe_var(0),
            "relative_humidity_2m": safe_var(1),
            "apparent_temperature": safe_var(2),
            "precipitation":        safe_var(3),
            "wind_speed_10m":       safe_var(4),
            "shortwave_radiation":  safe_var(5),
        })

        for col in df_w.columns:
            if col == "time":
                continue
            df_w[col] = pd.to_numeric(df_w[col], errors="coerce")
            df_w.loc[df_w[col] < -999, col] = np.nan

        solar = df_w["shortwave_radiation"].to_numpy()
        df_w["wbgt"] = compute_wbgt(
            df_w["temperature_2m"].to_numpy(),
            df_w["relative_humidity_2m"].to_numpy(),
            solar if not np.all(np.isnan(solar)) else None,
        )
        return df_w

    except Exception as exc:
        warnings.warn(f"Open-Meteo SDK error ({date_str}→{end_str}) : {exc}")
        return None


def enrich_df_with_weather(df, df_weather):
    """Interpolate hourly weather onto the per-second GPS DataFrame.

    Parameters
    ----------
    df         : pd.DataFrame  GPS DataFrame with timestamp column
    df_weather : pd.DataFrame  output of fetch_weather_hourly()

    Returns
    -------
    pd.DataFrame  enriched with temp_api, humidity_api, wind_kmh_api,
                  precip_api, solar_rad_api, apparent_temp_api, wbgt_api
    """
    if df_weather is None or df_weather.empty:
        warnings.warn("Données météo indisponibles — enrichissement ignoré.")
        return df

    df = df.copy()
    t_w = df_weather["time"].astype("int64").to_numpy() // 10**9

    print(df_weather["time"].min(), df_weather["time"].max())
    print(df_weather.shape)
    df_weather[["time", "temperature_2m", "wbgt"]].head(10)



    ts = df["timestamp"].copy()
    if hasattr(ts.dt, "tz") and ts.dt.tz is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    t_gps = ts.astype("int64").to_numpy() // 10**9

    col_map = {
        "temperature_2m":       "temp_api",
        "relative_humidity_2m": "humidity_api",
        "wind_speed_10m":       "wind_kmh_api",
        "precipitation":        "precip_api",
        "shortwave_radiation":  "solar_rad_api",
        "apparent_temperature": "apparent_temp_api",
        "wbgt":                 "wbgt_api",
    }
    for src, dst in col_map.items():
        if src in df_weather.columns:
            df[dst] = np.interp(t_gps, t_w,
                                df_weather[src].to_numpy(dtype=float))
        else:
            df[dst] = np.nan
    return df


def plot_weather_along_race(df, ravito_km, ravito_nom, fc_max=None,
                             show_watch_temp=False, height_per_panel=220):
    """Three-panel weather plot along the race distance axis.

    Panel 1 : ERA5-Land temperature + apparent temperature
              (+ watch temperature if show_watch_temp=True)
    Panel 2 : relative humidity (%) and wind speed (km/h)
    Panel 3 : WBGT with risk thresholds + elevation profile overlay

    Parameters
    ----------
    df               : pd.DataFrame  enriched with enrich_df_with_weather()
    ravito_km        : list[float]
    ravito_nom       : list[str]
    fc_max           : float | None  (kept for API homogeneity)
    show_watch_temp  : bool  show watch temperature curve (default False)
                       The watch sensor is biased +2–5°C by body heat;
                       use only for temporal dynamics, not absolute values.
    height_per_panel : int   height in pixels per subplot

    Returns
    -------
    plotly.graph_objects.Figure

    References
    ----------
    - Muñoz Sabater J (2019). ERA5-Land. ECMWF.
    - Bernard TE & Kenney WL (1994). AIHAJ 55(1):32-36.
    - Périard JD et al. (2021). Br J Sports Med 55(15):865-876.
    - Casa DJ et al. (2015). J Athl Train 50(9):986-1000.
    """
    x_km = df["dist_m"] / 1000.0
    has_api = "temp_api" in df.columns and df["temp_api"].notna().sum() > 10
    has_watch = (show_watch_temp
                 and "temperature" in df.columns
                 and df["temperature"].notna().sum() > 100)

    if not has_api and not has_watch:
        raise ValueError(
            "Aucune donnée de température disponible. "
            "Appeler enrich_df_with_weather() d'abord."
        )

    n_panels = 3 if has_api else 1
    fig = make_subplots(
        rows=n_panels, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.3, 0.2, 0.5],
        
        subplot_titles=[
            "Température (ERA5-Land)",
            "Humidité relative et vent (ERA5-Land)",
            "WBGT ",
        ][:n_panels],
        specs=[[{"secondary_y": True}]] * n_panels,
    )

    def add_ravitos(row):
        """Add ravito vertical lines and annotations to a subplot row."""
        y_arr = np.full(len(x_km), 0)
        y_max = float(y_arr.max()) if y_arr.size else 0
        for km, nom in zip(ravito_km, ravito_nom):
            fig.add_vline(
                x=km,
                line=dict(color="rgba(180,180,180,0.3)", dash="dot", width=1),
                row=row, col=1,
            )

    # ── Panel 1 : températures ───────────────────────────────────────────────
    if has_api:
        fig.add_trace(go.Scatter(
            x=x_km, y=df["temp_api"],
            mode="lines", line=dict(color="#4a9ede", width=2),
            name="ERA5-Land",
            hovertemplate="Distance : %{x:.2f} km<br>Temp. : %{y:.1f}°C<extra></extra>",
        ), row=1, col=1, secondary_y=False)

        if "apparent_temp_api" in df.columns:
            fig.add_trace(go.Scatter(
                x=x_km, y=df["apparent_temp_api"],
                mode="lines",
                line=dict(color="navy", width=1, dash="dot"),
                opacity=0.6, name="Ressentie",
                hovertemplate="Distance : %{x:.2f} km<br>Ressentie : %{y:.1f}°C<extra></extra>",
            ), row=1, col=1, secondary_y=False)

    if has_watch:
        win = max(10, int(3000 / df["dist_m"].diff().median()))
        t_smooth = (df["temperature"]
                    .rolling(win, center=True, min_periods=5).median())
        fig.add_trace(go.Scatter(
            x=x_km, y=t_smooth,
            mode="lines",
            line=dict(color="tomato", width=1.2, dash="dash"),
            opacity=0.6,
            name="Montre ⚠️ biais +2–5°C",
        ), row=1, col=1, secondary_y=False)

    fig.update_yaxes(title_text="Température (°C)", row=1, col=1,
                     secondary_y=False)
    add_ravitos(1)

    if not has_api:
        fig.update_xaxes(title_text="Distance (km)", row=1, col=1)
        fig.update_layout(
            height=height_per_panel, template="plotly_dark",
            margin=dict(l=60, r=60, t=60, b=50),
        )
        return fig

    # ── Panel 2 : humidité + vent ────────────────────────────────────────────
    if "humidity_api" in df.columns:
        fig.add_trace(go.Scatter(
            x=x_km, y=df["humidity_api"],
            mode="lines", line=dict(color="teal", width=1.8),
            name="Humidité (%)",
            hovertemplate="Distance : %{x:.2f} km<br>Humidité : %{y:.0f}%<extra></extra>",
        ), row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Humidité (%)", row=2, col=1,
                         secondary_y=False)

    if "wind_kmh_api" in df.columns:
        fig.add_trace(go.Scatter(
            x=x_km, y=df["wind_kmh_api"],
            mode="lines", line=dict(color="goldenrod", width=1.5),
            opacity=0.8, name="Vent (km/h)",
            hovertemplate="Distance : %{x:.2f} km<br>Vent : %{y:.1f} km/h<extra></extra>",
        ), row=2, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Vent (km/h)", row=2, col=1,
                         secondary_y=True,
                         color="goldenrod")

    add_ravitos(2)

    # ── Panel 3 : WBGT + profil altimétrique ─────────────────────────────────
    if "wbgt_api" in df.columns and df["wbgt_api"].notna().sum() > 10:
        wbgt = df["wbgt_api"].to_numpy()

        # Seuils de risque (Périard 2021 / Casa 2015)
        seuils = [
            (32.0, "darkred",  "Danger extrême > 32°C"),
            (28.0, "crimson",  "Danger > 28°C"),
            (23.0, "orange",   "Vigilance > 23°C"),
        ]
        for s_val, s_color, s_lbl in seuils:
            fig.add_hline(
                y=s_val,
                line=dict(color=s_color, dash="dash", width=1),
                annotation_text=s_lbl,
                annotation_font=dict(color=s_color, size=9),
                annotation_position="top right",
                row=3, col=1,
            )

        # Zones colorées
        for y0, y1, color in [
            (32.0, float(np.nanmax(wbgt)), "rgba(139,0,0,0.15)"),
            (28.0, 32.0,                   "rgba(220,20,60,0.10)"),
            (23.0, 28.0,                   "rgba(255,165,0,0.08)"),
        ]:
            if float(np.nanmax(wbgt)) > y0:
                fig.add_hrect(
                    y0=y0, y1=y1,
                    fillcolor=color, line_width=0,
                    row=3, col=1,
                )

        fig.add_trace(go.Scatter(
            x=x_km, y=wbgt,
            mode="lines", line=dict(color="#b06ecc", width=2),
            name="WBGT",
            hovertemplate="Distance : %{x:.2f} km<br>WBGT : %{y:.1f}°C<extra></extra>",
        ), row=3, col=1, secondary_y=False)
        fig.update_yaxes(title_text="WBGT (°C)", row=3, col=1,
                         secondary_y=False)

    # Profil altimétrique en arrière-plan (axe secondaire)
    alt_range = df["alt_m"].max() - df["alt_m"].min()
    fig.add_trace(go.Scatter(
        x=x_km, y=df["alt_m"],
        fill="tozeroy",
        fillcolor="rgba(139,90,43,0.08)",
        line=dict(color="rgba(139,90,43,0.2)", width=1),
        hoverinfo="skip", showlegend=False, name="Altitude",
    ), row=3, col=1, secondary_y=True)
    fig.update_yaxes(
        title_text="Altitude (m)",
        range=[df["alt_m"].min() - 50,
               df["alt_m"].max() + alt_range * 2.0],
        row=3, col=1, secondary_y=True,
        color="rgba(139,90,43,0.4)",
    )

    add_ravitos(3)

    fig.update_xaxes(title_text="Distance (km)", row=n_panels, col=1)
    fig.update_layout(
        height=height_per_panel * n_panels,
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", x=0, y=-0.08, font=dict(size=9)),
        margin=dict(l=60, r=80, t=60, b=60),
    )
    return fig


# ===========================================================================
# 13. Race pipeline
# ===========================================================================

def load_and_process_race(fit_path, fc_max, fc_min, poids_kg,
                           ravito_km, ravito_nom,
                           window_m=100.0, up_thr=3.0, down_thr=-3.0,
                           min_seg_m=200.0, walk_thr_kmh=6.0,
                           walk_thr_cad=140.0):
    """Load a FIT file and compute all standard derived metrics.

    Single entry point for multi-race notebooks. Returns a structured
    dict rather than printing intermediate results.

    Parameters
    ----------
    fit_path      : str    path to the .fit file
    fc_max        : int    max heart rate (bpm)
    fc_min        : int    resting heart rate (bpm)
    poids_kg      : float  athlete mass (kg)
    ravito_km     : list[float]  aid station positions (km)
    ravito_nom    : list[str]    aid station names
    window_m      : float  slope computation window (m)
    up_thr        : float  uphill threshold (%)
    down_thr      : float  downhill threshold (%)
    min_seg_m     : float  minimum segment length (m)
    walk_thr_kmh  : float  walk/run speed threshold (km/h)
    walk_thr_cad  : float  walk/run cadence threshold (spm)

    Returns
    -------
    dict with keys: df, kpis, meta, ravito_km, ravito_nom, fc_max, fc_min, poids_kg
    """
    import os

    df_raw = load_fit(fit_path)
    df, _ = clean_df(df_raw)
    df["slope_pct"] = compute_slope(df, window_m)
    df = compute_gap(df)
    df = segment_updown(df, up_thr, down_thr, min_seg_m)

    if "cadence" in df.columns:
        df = classify_walk_run(df, walk_thr_kmh, walk_thr_cad)
    if "heart_rate" in df.columns:
        df = compute_cardiac_drift(df, min_speed_kmh=3.0)

    dplus, dminus = compute_dplus_dminus(df)
    race_name = os.path.splitext(os.path.basename(fit_path))[0]

    kpis = compute_race_kpis(
        df, fc_max=fc_max, fc_min=fc_min, poids_kg=poids_kg,
        ravito_km=ravito_km, ravito_nom=ravito_nom,
        dplus=dplus, dminus=dminus,
    )

    meta = {
        "name":     race_name,
        "date":     df["timestamp"].iloc[0].date() if not df.empty else None,
        "fit_path": fit_path,
    }

    return {
        "df":         df,
        "kpis":       kpis,
        "meta":       meta,
        "ravito_km":  ravito_km,
        "ravito_nom": ravito_nom,
        "fc_max":     fc_max,
        "fc_min":     fc_min,
        "poids_kg":   poids_kg,
    }


def compute_race_kpis(df, fc_max, fc_min, poids_kg, ravito_km, ravito_nom,
                      dplus=None, dminus=None):
    """Extract scalar KPIs from a processed race DataFrame.

    All metrics are comparable across races of different distances and
    profiles, provided the same athlete parameters are used.

    Parameters
    ----------
    df         : pd.DataFrame
    fc_max     : int
    fc_min     : int
    poids_kg   : float
    ravito_km  : list[float]
    ravito_nom : list[str]
    dplus      : float | None  precomputed D+ (m)
    dminus     : float | None  precomputed D- (m)

    Returns
    -------
    dict  scalar KPIs
    """
    if dplus is None or dminus is None:
        dplus, dminus = compute_dplus_dminus(df)

    dist_km = float(df["dist_m"].max() / 1000.0)
    duration_h = float(df["time_h"].max())

    gap_vals = df["gap_s_per_km"].dropna()
    gap_vals = gap_vals[gap_vals > 0]
    gap_med = float(gap_vals.median()) if not gap_vals.empty else np.nan
    speed_kmh = dist_km / duration_h if duration_h > 0 else np.nan

    fc_mean = float(df["heart_rate"].mean()) if "heart_rate" in df.columns else np.nan
    fc_max_obs = float(df["heart_rate"].max()) if "heart_rate" in df.columns else np.nan
    fc_frac = (fc_mean / fc_max
               if fc_max > 0 and not np.isnan(fc_mean) else np.nan)

    pct_walk = (float(df["is_walk"].mean() * 100)
                if "is_walk" in df.columns else np.nan)
    cv_gap = (float(gap_vals.std() / gap_vals.mean() * 100)
              if len(gap_vals) > 20 else np.nan)

    split_result = compute_pace_split(df, ravito_km, ravito_nom)
    split_ratio = split_result["split_ratio"] if split_result else np.nan

    decoupling = np.nan
    if "heart_rate" in df.columns and "gap_s_per_km" in df.columns:
        dc_df = compute_aerobic_decoupling(df, ravito_km, ravito_nom)
        if not dc_df.empty and "Découplage (%)" in dc_df.columns:
            decoupling = float(dc_df["Découplage (%)"].max())

    start_time = df["timestamp"].iloc[0] if not df.empty else None
    start_hour = (start_time.hour + start_time.minute / 60.0
                  if start_time else np.nan)

    return {
        "distance_km":    round(dist_km, 2),
        "duration_h":     round(duration_h, 3),
        "dplus_m":        round(dplus, 0),
        "dminus_m":       round(dminus, 0),
        "speed_kmh":      round(speed_kmh, 2),
        "gap_med_s_km":   round(gap_med, 0),
        "fc_mean":        round(fc_mean, 0) if not np.isnan(fc_mean) else np.nan,
        "fc_max_obs":     round(fc_max_obs, 0) if not np.isnan(fc_max_obs) else np.nan,
        "fc_frac":        round(fc_frac, 3) if not np.isnan(fc_frac) else np.nan,
        "cv_gap_pct":     round(cv_gap, 1) if not np.isnan(cv_gap) else np.nan,
        "split_ratio":    round(split_ratio, 3) if not np.isnan(split_ratio) else np.nan,
        "pct_walk":       round(pct_walk, 1) if not np.isnan(pct_walk) else np.nan,
        "decoupling_max": round(decoupling, 1) if not np.isnan(decoupling) else np.nan,
        "start_hour":     round(start_hour, 2) if not np.isnan(start_hour) else np.nan,
    }


# ===========================================================================
# 14. Walk probability by slope and section
# ===========================================================================

def plot_walk_by_slope_sections(df, slope_bins, slope_labels,
                                 ravito_km=None, n_segments=None,
                                 walk_thr_kmh=6.0, walk_thr_cad=140.0,
                                 height=400):
    """Walk probability by slope class, split by race sections.

    Parameters
    ----------
    df           : pd.DataFrame  must contain slope_pct, speed_kmh, cadence, dist_m
    slope_bins   : list[float]   bin edges for slope classes (%)
    slope_labels : list[str]     labels for each slope class
    ravito_km    : list[float] | None   aid station positions (km)
    n_segments   : int | None           number of equal segments
    walk_thr_kmh : float   walk/run speed threshold (km/h)
    walk_thr_cad : float   walk/run cadence threshold (spm)
    height       : int     figure height in pixels

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if ravito_km is None and n_segments is None:
        raise ValueError("Provide either ravito_km or n_segments.")
    if ravito_km is not None and n_segments is not None:
        raise ValueError("Provide only one of ravito_km or n_segments.")

    df = df.copy()
    if "is_walk" not in df.columns:
        df["is_walk"] = (
            (df["speed_kmh"] < walk_thr_kmh) & (df["cadence"] < walk_thr_cad)
        ).astype(int)

    dist_min_km = df["dist_m"].min() / 1000.0
    dist_max_km = df["dist_m"].max() / 1000.0
    bounds = (
        [dist_min_km] + list(ravito_km) + [dist_max_km]
        if ravito_km is not None
        else list(np.linspace(dist_min_km, dist_max_km, n_segments + 1))
    )
    section_labels = [
        f"{bounds[i]:.1f}–{bounds[i+1]:.1f} km"
        for i in range(len(bounds) - 1)
    ]

    df["section"] = pd.cut(
        df["dist_m"] / 1000.0, bins=bounds,
        labels=section_labels, include_lowest=True,
    )
    df["slope_bin"] = pd.cut(
        df["slope_pct"], bins=slope_bins, labels=slope_labels
    )
    tmp = df.dropna(subset=["section", "slope_bin", "cadence"])
    walk_matrix = (
        tmp.groupby(["slope_bin", "section"], observed=True)["is_walk"]
        .mean() * 100
    ).unstack("section")

    colors = [
        "#4a9ede", "#e07b54", "#5bc45b", "#c45bc4",
        "#e0c454", "#54c4c4", "#c45454", "#8884d8",
    ]
    fig = go.Figure()

    for i, section in enumerate(walk_matrix.columns):
        fig.add_trace(go.Bar(
            name=section,
            x=walk_matrix.index.astype(str),
            y=walk_matrix[section].values,
            marker_color=colors[i % len(colors)],
            marker_line_color="rgba(255,255,255,0.3)",
            marker_line_width=1,
            hovertemplate=(
                f"Section : {section}<br>"
                "Pente : %{x}<br>"
                "% marche : %{y:.1f}%<extra></extra>"
            ),
        ))

    title = (
        "Probabilité de marcher par classe de pente — sections ravitos"
        if ravito_km is not None
        else f"Probabilité de marcher par classe de pente — {n_segments} segments"
    )
    fig.update_layout(
        barmode="group",
        title=title,
        xaxis_title="Classe de pente",
        yaxis_title="% de marche",
        height=height,
        template="plotly_dark",
        legend=dict(title="Section", orientation="v", x=1.01, y=1,
                    font=dict(size=9)),
        margin=dict(l=60, r=160, t=60, b=80),
    )
    return fig


# ===========================================================================
# 15. HR heatmap (speed × slope) by section
# ===========================================================================

def plot_heatmap_sections(df, fc_max, walk_thr_kmh=6.0,
                           ravito_km=None, n_segments=None,
                           slope_min=-25, slope_max=25, slope_step=1.0,
                           speed_min=0, speed_max=18, speed_step=0.5,
                           min_count=5, height_per_panel=250):
    """HR heatmaps (speed × slope) split by race sections.

    Parameters
    ----------
    df              : pd.DataFrame  must contain speed_kmh, slope_pct, heart_rate
    fc_max          : float
    walk_thr_kmh    : float  walk/run threshold (km/h)
    ravito_km       : list[float] | None
    n_segments      : int | None
    slope_min/max   : float  slope axis range (%)
    slope_step      : float  slope bin size (%)
    speed_min/max   : float  speed axis range (km/h)
    speed_step      : float  speed bin size (km/h)
    min_count       : int    minimum points per cell
    height_per_panel: int    height in pixels per subplot row

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if ravito_km is None and n_segments is None:
        raise ValueError("Provide either ravito_km or n_segments.")
    if ravito_km is not None and n_segments is not None:
        raise ValueError("Provide only one of ravito_km or n_segments.")
    if "heart_rate" not in df.columns:
        raise ValueError("Colonne heart_rate manquante.")

    req = ["speed_kmh", "slope_pct", "heart_rate", "dist_m"]
    dh = df.dropna(subset=req).copy()

    slope_bins = np.arange(slope_min, slope_max + slope_step, slope_step)
    speed_bins = np.arange(speed_min, speed_max + speed_step, speed_step)
    vmin, vmax = fc_max * 0.6, fc_max * 0.95

    dist_min_km = dh["dist_m"].min() / 1000.0
    dist_max_km = dh["dist_m"].max() / 1000.0
    bounds = (
        [dist_min_km] + list(ravito_km) + [dist_max_km]
        if ravito_km is not None
        else list(np.linspace(dist_min_km, dist_max_km, n_segments + 1))
    )
    section_labels = [
        f"{bounds[i]:.1f}–{bounds[i+1]:.1f} km"
        for i in range(len(bounds) - 1)
    ]
    all_labels = ["Course entière"] + section_labels
    all_masks = [np.ones(len(dh), dtype=bool)] + [
        ((dh["dist_m"] / 1000.0 >= bounds[i]) &
         (dh["dist_m"] / 1000.0 < bounds[i + 1])).values
        for i in range(len(bounds) - 1)
    ]

    def compute_heat(mask):
        """Compute median HR grid for a section mask."""
        sub = dh.loc[mask]
        sidx = np.digitize(sub["speed_kmh"].values, speed_bins) - 1
        pidx = np.digitize(sub["slope_pct"].values, slope_bins) - 1
        sumhr = np.zeros((len(slope_bins), len(speed_bins)))
        count = np.zeros_like(sumhr)
        for si, pi, hr in zip(sidx, pidx, sub["heart_rate"].values):
            if 0 <= pi < len(slope_bins) and 0 <= si < len(speed_bins):
                sumhr[pi, si] += hr
                count[pi, si] += 1
        with np.errstate(invalid="ignore"):
            return np.where(count >= min_count, sumhr / count, np.nan)

    n_panels = len(all_labels)
    ncols = min(n_panels, 4)
    nrows = int(np.ceil(n_panels / ncols))

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=all_labels,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    for idx, (label, mask) in enumerate(zip(all_labels, all_masks)):
        row = idx // ncols + 1
        col = idx % ncols + 1
        heat = compute_heat(mask)

        showscale = (idx == 0)
        fig.add_trace(
            go.Heatmap(
                z=heat,
                x=speed_bins,
                y=slope_bins,
                colorscale="RdYlGn_r",
                zmin=vmin, zmax=vmax,
                showscale=showscale,
                colorbar=dict(title="FC méd. (bpm)", thickness=14) if showscale else None,
                hovertemplate=(
                    "Vitesse : %{x:.1f} km/h<br>"
                    "Pente : %{y:.0f}%<br>"
                    "FC méd. : %{z:.0f} bpm<extra></extra>"
                ),
                name=label,
            ),
            row=row, col=col,
        )
        # Lignes de référence : pente 0 et vitesse seuil marche
        fig.add_hline(y=0, line=dict(color="black", width=1, dash="dash"),
                      row=row, col=col)
        fig.add_vline(x=walk_thr_kmh, line=dict(color="black", width=1, dash="dash"),
                      row=row, col=col)

        fig.update_xaxes(title_text="Vitesse (km/h)", row=row, col=col)
        fig.update_yaxes(title_text="Pente (%)", row=row, col=col)

    title = (
        "Heatmap FC médiane — sections ravitos"
        if ravito_km is not None
        else f"Heatmap FC médiane — {n_segments} segments"
    )
    fig.update_layout(
        title=title,
        height=height_per_panel * nrows,
        template="plotly_dark",
        margin=dict(l=60, r=60, t=80, b=60),
    )
    return fig


# ===========================================================================
# 16. Pace variability
# ===========================================================================

def compute_pace_variability(df, ravito_km, ravito_nom):
    """Compute GAP coefficient of variation per section (pace regularity).

    A lower CV indicates a more even pacing strategy.
    CV of GAP correlates negatively with performance.

    Parameters
    ----------
    df         : pd.DataFrame  must contain gap_s_per_km, dist_m
    ravito_km  : list[float]
    ravito_nom : list[str]

    Returns
    -------
    pd.DataFrame  one row per section with CV and related metrics

    References
    ----------
    - Haney TA & Mercer JA (2011). Int J Exerc Sci 4(2):133-140.
    - Cuk I et al. (2024). Medicina 60(2):218.
      https://doi.org/10.3390/medicina60020218
    """
    if "gap_s_per_km" not in df.columns:
        return pd.DataFrame()

    bounds = np.concatenate((
        [df["dist_m"].min() / 1000.0],
        np.array(ravito_km, dtype=float),
        [df["dist_m"].max() / 1000.0]
    ))
    section_labels = build_section_labels_short(list(bounds), ravito_nom)
    labels = [
        (section_labels[i], bounds[i], bounds[i + 1])
        for i in range(len(bounds) - 1)
    ]

    rows = []
    for nom, a, b in labels:
        mask = (
            (df["dist_m"] / 1000.0 >= a) &
            (df["dist_m"] / 1000.0 < b) &
            df["gap_s_per_km"].notna() &
            (df["gap_s_per_km"] > 0)
        )
        sec = df.loc[mask, "gap_s_per_km"]
        if len(sec) < 20:
            continue
        mean_gap = sec.mean()
        std_gap = sec.std()
        cv = std_gap / mean_gap * 100.0
        rows.append({
            "Section":         nom,
            "GAP méd. (s/km)": round(sec.median(), 0),
            "GAP moy. (s/km)": round(mean_gap, 0),
            "Écart-type":      round(std_gap, 1),
            "CV GAP (%)":      round(cv, 1),
            "Régularité":      "✅ régulier" if cv < 15 else (
                               "⚠️ variable" if cv < 25 else "❌ très variable"),
        })
    return pd.DataFrame(rows)


def plot_pace_variability(df, ravito_km, ravito_nom, height=700):
    """GAP distribution per section as box plots + CV bar chart.

    Panel 1 : box plot of GAP per section with median annotation.
    Panel 2 : CV of GAP per section with threshold lines.

    Parameters
    ----------
    df         : pd.DataFrame  must contain gap_s_per_km, dist_m
    ravito_km  : list[float]
    ravito_nom : list[str]
    height     : int  total figure height in pixels

    Returns
    -------
    plotly.graph_objects.Figure

    References
    ----------
    - Haney TA & Mercer JA (2011). Int J Exerc Sci 4(2):133-140.
    - Cuk I et al. (2024). Medicina 60(2):218.
    """
    if "gap_s_per_km" not in df.columns:
        raise ValueError("Colonne gap_s_per_km manquante.")

    df_var = compute_pace_variability(df, ravito_km, ravito_nom)
    if df_var.empty:
        raise ValueError("Pas assez de données pour l'analyse de variabilité.")

    bounds = np.concatenate((
        [df["dist_m"].min() / 1000.0],
        np.array(ravito_km, dtype=float),
        [df["dist_m"].max() / 1000.0]
    ))

    short_labels = []
    data_boxes = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        mask = (
            (df["dist_m"] / 1000.0 >= a) &
            (df["dist_m"] / 1000.0 < b) &
            df["gap_s_per_km"].notna() &
            (df["gap_s_per_km"] > 0) &
            (df["gap_s_per_km"] < 1200)
        )
        vals = df.loc[mask, "gap_s_per_km"].dropna().values
        if len(vals) < 20:
            continue
        if not ravito_nom:
            short_labels.append(f"Section {i + 1}")
        elif i == 0:
            short_labels.append(f"→ {ravito_nom[0]}")
        elif i == len(bounds) - 2:
            short_labels.append(f"{ravito_nom[-1]} →")
        else:
            short_labels.append(ravito_nom[i - 1])
        data_boxes.append(vals)

    colors = [
        "#4a9ede", "#e07b54", "#5bc45b", "#c45bc4",
        "#e0c454", "#54c4c4", "#c45454", "#8884d8",
    ]

    fig = make_subplots(
        rows=2, cols=1,
        vertical_spacing=0.12,
        subplot_titles=[
            "Distribution du GAP par section — Haney & Mercer (2011) / Cuk et al. (2024)",
            "Régularité d'allure par section (CV du GAP)",
        ],
    )

    # ── Panel 1 : box plots ───────────────────────────────────────────────────
    for i, (label, vals) in enumerate(zip(short_labels, data_boxes)):
        med = float(np.median(vals))
        m, s = int(med) // 60, int(med) % 60
        fig.add_trace(
            go.Box(
                y=vals,
                name=label,
                marker_color=colors[i % len(colors)],
                boxmean=False,
                hovertemplate=(
                    f"{label}<br>"
                    "GAP : %{y:.0f} s/km<extra></extra>"
                ),
            ),
            row=1, col=1,
        )
        # Annotation médiane au format MM'SS" au-dessus de chaque boîte
        fig.add_annotation(
            x=label, y=float(np.percentile(vals, 75)),
            xref="x1", yref="y1",
            text=f"{m}'{s:02d}\"",
            showarrow=False,
            font=dict(size=8, color="white"),
            yshift=-18,
        )

    fig.update_yaxes(title_text="GAP (s/km)", autorange="reversed",
                     row=1, col=1)

    # ── Panel 2 : barres CV ───────────────────────────────────────────────────
    cv_vals = df_var["CV GAP (%)"].values
    bar_colors = [
        "seagreen" if v < 15 else ("orange" if v < 25 else "crimson")
        for v in cv_vals
    ]

    fig.add_trace(
        go.Bar(
            x=df_var["Section"],
            y=cv_vals,
            marker_color=bar_colors,
            marker_line_color="rgba(255,255,255,0.3)",
            marker_line_width=1,
            text=[f"{v:.1f}%" for v in cv_vals],
            textposition="outside",
            name="CV GAP",
            hovertemplate=(
                "%{x}<br>CV : %{y:.1f}%<extra></extra>"
            ),
        ),
        row=2, col=1,
    )

    for y_val, color, label in [
        (15, "orange", "Variable (15%)"),
        (25, "crimson", "Très variable (25%)"),
    ]:
        fig.add_hline(
            y=y_val,
            line=dict(color=color, dash="dash", width=1),
            annotation_text=label,
            annotation_font=dict(color=color, size=9),
            annotation_position="top right",
            row=2, col=1,
        )

    fig.update_yaxes(title_text="CV du GAP (%)", row=2, col=1)
    fig.update_xaxes(tickangle=-20, row=2, col=1)
    fig.update_layout(
        height=height,
        template="plotly_dark",
        showlegend=False,
        margin=dict(l=60, r=40, t=60, b=60),
    )
    return fig


# ===========================================================================
# 17. Circadian profile
# ===========================================================================

def compute_circadian_profile(df, bin_hours=2):
    """Compute performance metrics per time-of-day bin.

    Particularly relevant for nocturnal races (SaintéLyon, UTMB) where
    the circadian nadir (02h–06h) causes a performance drop distinct from
    topographic or homeostatic fatigue.

    Parameters
    ----------
    df         : pd.DataFrame  must contain timestamp, gap_s_per_km
    bin_hours  : int  width of time bins in hours (default 2)

    Returns
    -------
    pd.DataFrame  one row per time bin with performance metrics

    References
    ----------
    - Czeisler CA et al. (1999). Science 284(5423):2177-2181.
    - Bearden SE & van Woerden I (2025). PLoS ONE.
      https://doi.org/10.1371/journal.pone.0322883
    """
    if "gap_s_per_km" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["hour_bin"] = (df["hour"] // bin_hours) * bin_hours

    all_bins = sorted(df["hour_bin"].unique())
    start_h = int(df["timestamp"].iloc[0].hour)
    start_bin = (start_h // bin_hours) * bin_hours
    if start_bin in all_bins:
        idx = all_bins.index(start_bin)
        all_bins = all_bins[idx:] + all_bins[:idx]

    rows = []
    for h_start in all_bins:
        h_end = h_start + bin_hours
        mask = (df["hour_bin"] == h_start) & df["gap_s_per_km"].notna()
        sec = df.loc[mask]
        if len(sec) < 30:
            continue
        row = {
            "Tranche horaire": f"{h_start:02d}h–{h_end:02d}h",
            "N points":        len(sec),
            "GAP méd. (s/km)": round(sec["gap_s_per_km"].median(), 0),
        }
        if "heart_rate" in sec.columns:
            row["FC méd. (bpm)"] = round(sec["heart_rate"].median(), 0)
        if "is_walk" in sec.columns:
            row["Marche (%)"] = round(sec["is_walk"].mean() * 100, 1)
        if "cadence" in sec.columns:
            row["Cadence méd."] = round(sec["cadence"].median(), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_circadian_profile(df, bin_hours=2, height=600):
    """Performance metrics per time-of-day bin.

    Highlights the circadian nadir window (22h–06h) in purple.
    Up to three panels: GAP, HR, walk rate (depending on available columns).

    Parameters
    ----------
    df         : pd.DataFrame  must contain timestamp, gap_s_per_km
    bin_hours  : int  time bin width in hours
    height     : int  total figure height in pixels

    Returns
    -------
    plotly.graph_objects.Figure

    References
    ----------
    - Czeisler CA et al. (1999). Science 284(5423):2177-2181.
    - Bearden SE & van Woerden I (2025). PLoS ONE.
      https://doi.org/10.1371/journal.pone.0322883
    """
    circ_df = compute_circadian_profile(df, bin_hours=bin_hours)
    if circ_df.empty:
        raise ValueError("Pas assez de données pour l'analyse circadienne.")

    labels = circ_df["Tranche horaire"].tolist()
    has_fc = "FC méd. (bpm)" in circ_df.columns
    has_walk = "Marche (%)" in circ_df.columns
    n_panels = 1 + int(has_fc) + int(has_walk)

    def is_nocturnal(label):
        """Return True if the time bin falls in the circadian nadir."""
        h = int(label[:2])
        return h >= 22 or h < 6

    noc_mask = [is_nocturnal(l) for l in labels]
    bar_colors = ["#b06ecc" if noc else "#4a9ede" for noc in noc_mask]

    start_ts = df["timestamp"].iloc[0]
    start_bin_lbl = f"{((start_ts.hour // bin_hours) * bin_hours):02d}h"

    subplot_titles = ["GAP médian par tranche horaire"]
    if has_fc:
        subplot_titles.append("FC médiane par tranche horaire")
    if has_walk:
        subplot_titles.append("% marche par tranche horaire")

    fig = make_subplots(
        rows=n_panels, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=subplot_titles,
    )

    gap_vals = circ_df["GAP méd. (s/km)"].values
    gap_med_global = float(np.median(gap_vals))

    def fmt_gap(g):
        """Format gap seconds to MM'SS string."""
        return f"{int(g)//60}'{int(g)%60:02d}\""

    # ── Panel 1 : GAP ────────────────────────────────────────────────────────
    fig.add_trace(go.Bar(
        x=labels, y=gap_vals,
        marker_color=bar_colors,
        marker_line_color="rgba(255,255,255,0.2)",
        marker_line_width=1,
        text=[fmt_gap(v) for v in gap_vals],
        textposition="outside",
        name="GAP",
        hovertemplate=(
            "%{x}<br>GAP : %{text}/km<extra></extra>"
        ),
    ), row=1, col=1)

    fig.add_hline(
        y=gap_med_global,
        line=dict(color="white", dash="dash", width=1),
        annotation_text=f"Médiane {fmt_gap(gap_med_global)}/km",
        annotation_font=dict(size=9),
        annotation_position="top right",
        row=1, col=1,
    )
    fig.update_yaxes(title_text="GAP (s/km)", autorange="reversed",
                     row=1, col=1)

    panel_idx = 2

    # ── Panel 2 : FC ─────────────────────────────────────────────────────────
    if has_fc:
        fc_vals = circ_df["FC méd. (bpm)"].values
        fig.add_trace(go.Bar(
            x=labels, y=fc_vals,
            marker_color=bar_colors,
            marker_line_color="rgba(255,255,255,0.2)",
            marker_line_width=1,
            name="FC",
            hovertemplate="%{x}<br>FC : %{y:.0f} bpm<extra></extra>",
        ), row=panel_idx, col=1)
        fig.add_hline(
            y=float(np.median(fc_vals)),
            line=dict(color="white", dash="dash", width=1),
            row=panel_idx, col=1,
        )
        fig.update_yaxes(title_text="FC méd. (bpm)", row=panel_idx, col=1)
        panel_idx += 1

    # ── Panel 3 : % marche ───────────────────────────────────────────────────
    if has_walk:
        walk_vals = circ_df["Marche (%)"].values
        fig.add_trace(go.Bar(
            x=labels, y=walk_vals,
            marker_color=bar_colors,
            marker_line_color="rgba(255,255,255,0.2)",
            marker_line_width=1,
            name="Marche",
            hovertemplate="%{x}<br>Marche : %{y:.1f}%<extra></extra>",
        ), row=panel_idx, col=1)
        fig.update_yaxes(title_text="% marche", row=panel_idx, col=1)

    fig.update_xaxes(tickangle=-20, row=n_panels, col=1)
    fig.update_layout(
        title=(
            f"Profil circadien — tranches de {bin_hours}h | "
            f"Départ : {start_ts.strftime('%Hh%M')} | "
            "Violet = nadir circadien (22h–06h)<br>"
            "<sup>Czeisler (1999) / Bearden & van Woerden (2025)</sup>"
        ),
        height=height,
        template="plotly_dark",
        showlegend=False,
        margin=dict(l=60, r=40, t=80, b=60),
    )
    return fig


# ===========================================================================
# 18. Radar chart by section
# ===========================================================================

def plot_radar_sections(df, ravito_km, ravito_nom, fc_max, height=550):
    """Radar chart with 5 performance axes per race section.

    Axes (normalized 0-1, higher = better):
      1. GAP speed (inverted — faster = higher)
      2. Cardiac economy (lower HR%FCmax = higher)
      3. Run ratio (1 - walk rate)
      4. Cadence regularity (inverted CV)
      5. Durability (inverted HR/GAP ratio drift)

    Parameters
    ----------
    df         : pd.DataFrame  fully processed
    ravito_km  : list[float]
    ravito_nom : list[str]
    fc_max     : float
    height     : int  figure height in pixels

    Returns
    -------
    plotly.graph_objects.Figure
    """
    bounds = np.concatenate((
        [df["dist_m"].min() / 1000.0],
        np.array(ravito_km),
        [df["dist_m"].max() / 1000.0]
    ))
    section_names = []
    section_names = build_section_labels_short(list(bounds), ravito_nom)

    gap_kmh = 3600.0 / df["gap_s_per_km"].replace(0, np.nan)
    df = df.copy()
    df["hr_gap_ratio"] = (
        df["heart_rate"] / gap_kmh if "heart_rate" in df.columns else np.nan
    )

    raw = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        mask = (df["dist_m"] / 1000.0 >= a) & (df["dist_m"] / 1000.0 < b)
        sec = df.loc[mask]
        if len(sec) < 20:
            raw.append(None)
            continue
        gap_med = sec["gap_s_per_km"].median() if "gap_s_per_km" in sec else np.nan
        fc_med = (sec["heart_rate"].median() / fc_max
                  if "heart_rate" in sec.columns else 0.5)
        pct_run = (1.0 - sec["is_walk"].mean()
                   if "is_walk" in sec.columns else 0.5)
        cad = sec["cadence"].dropna() if "cadence" in sec.columns else pd.Series([])
        cv_cad = (cad.std() / cad.mean()
                  if len(cad) > 5 and cad.mean() > 0 else 0.1)
        hr_gap = (sec["hr_gap_ratio"].median()
                  if "hr_gap_ratio" in sec.columns else np.nan)
        raw.append({"gap_med": gap_med, "fc_frac": fc_med,
                    "pct_run": pct_run, "cv_cad": cv_cad, "hr_gap": hr_gap})

    valid = [r for r in raw if r is not None]
    if not valid:
        raise ValueError("Pas assez de données pour le radar.")

    gap_all = [r["gap_med"] for r in valid if not np.isnan(r["gap_med"])]
    hr_gap_all = [r["hr_gap"] for r in valid if not np.isnan(r["hr_gap"])]
    cv_all = [r["cv_cad"] for r in valid]

    gap_min, gap_max = min(gap_all), max(gap_all)
    hr_gap_min = min(hr_gap_all) if hr_gap_all else 0
    hr_gap_max = max(hr_gap_all) if hr_gap_all else 1
    cv_min, cv_max = min(cv_all), max(cv_all)

    def norm(val, vmin, vmax, invert=False):
        """Normalize value to [0, 1]."""
        if vmax == vmin:
            return 0.5
        n = (val - vmin) / (vmax - vmin)
        return 1.0 - n if invert else n

    axes_labels = ["Vitesse GAP", "Économie cardio",
                   "% Course", "Régularité cadence", "Durabilité"]
    colors = [
        "#4a9ede", "#e07b54", "#5bc45b", "#c45bc4",
        "#e0c454", "#54c4c4", "#c45454", "#8884d8",
    ]

    fig = go.Figure()

    for i, (r, name) in enumerate(zip(raw, section_names)):
        if r is None:
            continue
        values = [
            norm(r["gap_med"], gap_min, gap_max, invert=True),
            1.0 - r["fc_frac"],
            r["pct_run"],
            norm(r["cv_cad"], cv_min, cv_max, invert=True),
            norm(r["hr_gap"], hr_gap_min, hr_gap_max, invert=True)
            if not np.isnan(r["hr_gap"]) else 0.5,
        ]
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=axes_labels + [axes_labels[0]],
            fill="toself",
            fillcolor=colors[i % len(colors)],
            opacity=0.15,
            line=dict(color=colors[i % len(colors)], width=2),
            name=name,
            hovertemplate=(
                f"{name}<br>%{{theta}} : %{{r:.2f}}<extra></extra>"
            ),
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1],
                            tickvals=[0.25, 0.5, 0.75, 1.0],
                            ticktext=["25%", "50%", "75%", "100%"],
                            tickfont=dict(size=8)),
        ),
        title="Profil de performance par section<br>"
              "<sup>Normalisé — vers l'extérieur = mieux</sup>",
        height=height,
        template="plotly_dark",
        legend=dict(orientation="v", x=1.1, y=1, font=dict(size=9)),
        margin=dict(l=60, r=160, t=80, b=60),
    )
    return fig


# ===========================================================================
# 19. Speed by slope class (mean ± std)
# ===========================================================================

def plot_speed_by_slope(df, slope_bins=None, slope_labels=None,
                        ravito_km=None, height=450):
    """Mean speed and standard deviation per slope class.

    Useful to identify where the runner is fast/slow relative to gradient,
    and to compare technique between uphill and downhill sections.
    Can be split by race section when ravito_km is provided.

    Parameters
    ----------
    df           : pd.DataFrame  must contain speed_kmh, slope_pct, dist_m
    slope_bins   : list[float] | None  bin edges (%). Default: standard bins.
    slope_labels : list[str]   | None  labels for each class.
    ravito_km    : list[float] | None  split by section when provided.
    height       : int  figure height in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if slope_bins is None:
        slope_bins   = [-30, -15, -10, -7, -5, -3, -1, 1, 3, 5, 7, 10, 15, 30]
        slope_labels = ["<-15", "-15/-10", "-10/-7", "-7/-5", "-5/-3",
                        "-3/-1", "-1/+1", "+1/+3", "+3/+5", "+5/+7",
                        "+7/+10", "+10/+15", ">+15"]

    df = df.copy()
    df["slope_bin"] = pd.cut(df["slope_pct"], bins=slope_bins,
                              labels=slope_labels)

    colors = [
        "#4a9ede", "#e07b54", "#5bc45b", "#c45bc4",
        "#e0c454", "#54c4c4", "#c45454", "#8884d8",
    ]

    if ravito_km is None:
        # Vue globale — une trace avec barres d'erreur
        agg = (df.dropna(subset=["slope_bin", "speed_kmh"])
               .groupby("slope_bin", observed=True)["speed_kmh"]
               .agg(["mean", "std", "count"])
               .reset_index())
        agg.columns = ["slope_bin", "mean", "std", "count"]
        agg = agg[agg["count"] >= 10]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=agg["slope_bin"].astype(str),
            y=agg["mean"],
            error_y=dict(type="data", array=agg["std"].values,
                         visible=True, color="rgba(255,255,255,0.4)"),
            marker_color="#4a9ede",
            marker_line_color="rgba(255,255,255,0.2)",
            marker_line_width=1,
            text=[f"{v:.1f}" for v in agg["mean"]],
            textposition="outside",
            hovertemplate=(
                "Pente : %{x}<br>"
                "Vitesse moy. : %{y:.2f} km/h<br>"
                "Écart-type : %{error_y.array:.2f} km/h<extra></extra>"
            ),
            name="Vitesse moy.",
        ))
        subtitle = "Course entière"
    else:
        # Vue par section
        dist_min_km = df["dist_m"].min() / 1000.0
        dist_max_km = df["dist_m"].max() / 1000.0
        bounds = [dist_min_km] + list(ravito_km) + [dist_max_km]
        section_labels = [
            f"{bounds[i]:.1f}–{bounds[i+1]:.1f} km"
            for i in range(len(bounds) - 1)
        ]
        df["section"] = pd.cut(
            df["dist_m"] / 1000.0, bins=bounds,
            labels=section_labels, include_lowest=True,
        )

        fig = go.Figure()
        for i, sec in enumerate(section_labels):
            sub = df[df["section"] == sec].dropna(subset=["slope_bin", "speed_kmh"])
            agg = (sub.groupby("slope_bin", observed=True)["speed_kmh"]
                   .agg(["mean", "std", "count"]).reset_index())
            agg.columns = ["slope_bin", "mean", "std", "count"]
            agg = agg[agg["count"] >= 10]
            if agg.empty:
                continue
            fig.add_trace(go.Bar(
                x=agg["slope_bin"].astype(str),
                y=agg["mean"],
                error_y=dict(type="data", array=agg["std"].values,
                             visible=True, color="rgba(255,255,255,0.3)"),
                marker_color=colors[i % len(colors)],
                marker_line_color="rgba(255,255,255,0.2)",
                marker_line_width=1,
                name=sec,
                hovertemplate=(
                    f"{sec}<br>Pente : %{{x}}<br>"
                    "Vitesse : %{y:.2f} ± %{error_y.array:.2f} km/h<extra></extra>"
                ),
            ))
        subtitle = "par section"

    fig.update_layout(
        barmode="group",
        title=f"Vitesse moyenne ± écart-type par classe de pente ({subtitle})",
        xaxis_title="Classe de pente (%)",
        yaxis_title="Vitesse (km/h)",
        height=height,
        template="plotly_dark",
        legend=dict(orientation="v", x=1.01, y=1, font=dict(size=9)),
        margin=dict(l=60, r=160, t=60, b=80),
    )
    return fig


# ===========================================================================
# 20. Two-runner comparison on same race
# ===========================================================================

def plot_comparison_two_runners(race_a, race_b, col="gap_s_per_km",
                                 n_bins=100, smooth_bins=5, height=500):
    """Compare two runners on the same race course.

    Three panels:
      1. Normalized profiles (0-100% distance) for both runners.
      2. Delta (runner B - runner A) per distance bin — positive = B slower.
      3. KPI radar comparing the two runners.

    Parameters
    ----------
    race_a   : dict  output of load_and_process_race() for runner A
    race_b   : dict  output of load_and_process_race() for runner B
    col      : str   column to compare (default: gap_s_per_km)
    n_bins   : int   number of distance bins
    smooth_bins : int  rolling smoothing on normalized axis
    height   : int   total figure height in pixels

    Returns
    -------
    plotly.graph_objects.Figure
    """
    labels_map = {
        "gap_s_per_km": "GAP (s/km)",
        "heart_rate":   "FC (bpm)",
        "speed_kmh":    "Vitesse (km/h)",
    }
    ylabel = labels_map.get(col, col)
    dist_pct = np.linspace(0, 100, n_bins)

    def get_profile(race):
        """Extract normalized profile for one runner."""
        df = race["df"]
        if col not in df.columns:
            return np.full(n_bins, np.nan)
        norm = normalize_by_distance_pct(df, n_bins=n_bins, cols=[col])
        p = norm[col].values.astype(float)
        if smooth_bins > 1:
            p = pd.Series(p).rolling(smooth_bins, center=True,
                                      min_periods=1).mean().values
        if col == "gap_s_per_km":
            p = np.where(p > 1200, np.nan, p)
        return p

    name_a = race_a["meta"]["name"]
    name_b = race_b["meta"]["name"]
    prof_a = get_profile(race_a)
    prof_b = get_profile(race_b)
    delta  = prof_b - prof_a  # positif = B plus lent

    fig = make_subplots(
        rows=2, cols=1,
        vertical_spacing=0.08,
        subplot_titles=[
            f"Profils normalisés — {ylabel}",
            f"Écart {name_b} − {name_a}  (positif = {name_b} plus lent)",
        ],
        specs=[
            [{"type": "xy"}],
            [{"type": "xy"}],
        ],
    )

    # ── Panel 1 : profils ─────────────────────────────────────────────────────
    for prof, name, color in [
        (prof_a, name_a, "#4a9ede"),
        (prof_b, name_b, "#e07b54"),
    ]:
        fig.add_trace(go.Scatter(
            x=dist_pct, y=prof,
            mode="lines", line=dict(color=color, width=2),
            name=name,
            hovertemplate=f"{name}<br>%{{x:.0f}}% : %{{y:.1f}}<extra></extra>",
        ), row=1, col=1)

    fig.update_yaxes(
        title_text=ylabel,
        autorange="reversed" if col == "gap_s_per_km" else True,
        row=1, col=1,
    )

    # ── Panel 2 : delta ───────────────────────────────────────────────────────
    fig.add_hline(y=0, line=dict(color="white", dash="dash", width=1),
                  row=2, col=1)
    fig.add_hrect(y0=-10, y1=10,
                  fillcolor="rgba(50,180,80,0.06)", line_width=0,
                  row=2, col=1)

    delta_colors = np.where(delta > 0, "rgba(220,50,50,0.7)",
                            "rgba(50,180,80,0.7)")
    fig.add_trace(go.Bar(
        x=dist_pct, y=delta,
        marker_color=delta_colors,
        name="Écart",
        hovertemplate=(
            "Distance : %{x:.0f}%<br>"
            f"Écart : %{{y:+.1f}} {ylabel.split('(')[-1].rstrip(')')}<extra></extra>"
        ),
        showlegend=False,
    ), row=2, col=1)

    fig.update_yaxes(title_text=f"Δ {ylabel}", row=2, col=1)
    fig.update_xaxes(title_text="Distance (% de la course)", row=2, col=1)

    # # ── Panel 3 : radar KPI ───────────────────────────────────────────────────
    # kpi_labels = ["GAP médian", "Split ratio", "% marche",
    #               "Découplage", "Économie cardiaque"]
    # kpi_keys   = ["gap_med_s_km", "split_ratio", "pct_walk",
    #               "decoupling_max", "fc_frac"]

    # def kpi_scores(race):
    #     """Normalize KPIs to [0,1] — higher = better."""
    #     k = race["kpis"]
    #     scores = []
    #     for key in kpi_keys:
    #         v = k.get(key, np.nan)
    #         scores.append(float(v) if not np.isnan(float(v) if v is not None else np.nan) else 0.5)
    #     # Invert metrics where lower = better
    #     invert = {"gap_med_s_km", "split_ratio", "pct_walk", "decoupling_max", "fc_frac"}
    #     return scores

    # scores_a = kpi_scores(race_a)
    # scores_b = kpi_scores(race_b)

    # # Normalize together
    # for i in range(len(kpi_labels)):
    #     vmin = min(scores_a[i], scores_b[i])
    #     vmax = max(scores_a[i], scores_b[i])
    #     if vmax > vmin:
    #         scores_a[i] = (scores_a[i] - vmin) / (vmax - vmin)
    #         scores_b[i] = (scores_b[i] - vmin) / (vmax - vmin)
    #     else:
    #         scores_a[i] = scores_b[i] = 0.5

    # for scores, name, color in [
    #     (scores_a, name_a, "#4a9ede"),
    #     (scores_b, name_b, "#e07b54"),
    # ]:
    #     fig.add_trace(go.Scatterpolar(
    #         r=scores + [scores[0]],
    #         theta=kpi_labels + [kpi_labels[0]],
    #         fill="toself",
    #         fillcolor=color,
    #         opacity=0.15,
    #         line=dict(color=color, width=2),
    #         name=name,
    #     ), row=3, col=1)

    # fig.update_layout(
    #     height=height,
    #     template="plotly_dark",
    #     title=f"Comparaison {name_a} vs {name_b} — même course",
    #     legend=dict(orientation="h", x=0.3, y=1.02, font=dict(size=10)),
    #     margin=dict(l=60, r=60, t=80, b=60),
    #     polar3=dict(
    #         radialaxis=dict(
    #             visible=True, range=[0, 1],
    #             tickvals=[0.25, 0.5, 0.75, 1.0],
    #             ticktext=["25%", "50%", "75%", "100%"],
    #             tickfont=dict(size=8),
    #         ),
    #     ),
    #)

    fig.update_layout(
    height=height,
    template="plotly_dark",
    title=f"Comparaison {name_a} vs {name_b} — même course",
    legend=dict(orientation="h", x=0.3, y=1.02, font=dict(size=10)),
    margin=dict(l=60, r=60, t=80, b=60),
    )
    return fig



def build_comparison_table(race_a, race_b):
    """Build a side-by-side KPI comparison DataFrame for two runners.

    Parameters
    ----------
    race_a : dict  output of load_and_process_race()
    race_b : dict  output of load_and_process_race()

    Returns
    -------
    pd.DataFrame  KPI, value A, value B, delta (B - A)
    """
    kpi_labels = {
        "distance_km":    "Distance (km)",
        "dplus_m":        "D+ (m)",
        "duration_h":     "Durée (h)",
        "gap_med_s_km":   "GAP médian (s/km)",
        "cv_gap_pct":     "CV GAP (%)",
        "split_ratio":    "Ratio split",
        "fc_frac":        "FC moy / FC max",
        "pct_walk":       "% marche",
        "decoupling_max": "Découplage max (%)",
    }
    rows = []
    for key, label in kpi_labels.items():
        va = race_a["kpis"].get(key, np.nan)
        vb = race_b["kpis"].get(key, np.nan)
        try:
            delta = round(float(vb) - float(va), 3)
        except (TypeError, ValueError):
            delta = np.nan
        rows.append({
            "KPI":   label,
            race_a["meta"]["name"]: va,
            race_b["meta"]["name"]: vb,
            "Δ (B − A)": delta,
        })
    return pd.DataFrame(rows)


# ===========================================================================
# 21. GPS utilities
# ===========================================================================

def extract_gps_centroid(df):
    """Extract the GPS centroid from a processed race DataFrame.

    Used as a fallback when lat/lon are not provided manually for
    weather queries. Returns the median lat/lon over the race to
    avoid influence of outlier GPS points.

    Parameters
    ----------
    df : pd.DataFrame  must contain lat, lon columns

    Returns
    -------
    tuple (lat, lon) or (None, None) if GPS data is unavailable
    """
    if "lat" not in df.columns or "lon" not in df.columns:
        return None, None
    lat = df["lat"].dropna()
    lon = df["lon"].dropna()
    if lat.empty or lon.empty:
        return None, None
    return float(lat.median()), float(lon.median())


# ===========================================================================
# 22. Altitude vs time — two-runner comparison
# ===========================================================================

def plot_altitude_vs_time(race_a, race_b, height=500):
    """Altitude profiles of two runners + stretch subplot.

    Panel 1 : altitude vs normalized time for both runners.
              Time axis normalized by runner A's total duration —
              runner B extends beyond x=1 if slower.
    Panel 2 : stretch in km (dist_A - dist_B at same normalized time).
              Positive = runner A is ahead.

    Parameters
    ----------
    race_a : dict  reference runner (load_and_process_race output)
    race_b : dict  second runner
    height : int   total figure height in pixels

    Returns
    -------
    plotly.graph_objects.Figure
    """
    df_a = race_a["df"].copy().sort_values("timestamp").reset_index(drop=True)
    df_b = race_b["df"].copy().sort_values("timestamp").reset_index(drop=True)
    name_a = race_a["meta"]["name"]
    name_b = race_b["meta"]["name"]

    dur_a = float(df_a["time_h"].max())
    dur_b = float(df_b["time_h"].max())

    # Temps normalisé : axe référence = durée de A
    # Chaque coureur est rééchantillonné sur cet axe
    t_norm_a = df_a["time_h"].values / dur_a
    t_norm_b = df_b["time_h"].values / dur_a   # divisé par dur_a, pas dur_b

    alt_a = df_a["alt_m"].values
    alt_b = df_b["alt_m"].values
    dist_a = df_a["dist_m"].values / 1000.0
    dist_b = df_b["dist_m"].values / 1000.0

    # Grille fine pour le stretch
    t_max = max(t_norm_a[-1], t_norm_b[-1])
    t_grid = np.linspace(0.0, t_max, 1000)

    dist_a_g = np.interp(t_grid, t_norm_a, dist_a)
    dist_b_g = np.interp(t_grid, t_norm_b, dist_b,
                          left=np.nan, right=np.nan)
    stretch = dist_a_g - dist_b_g  # positif = A devant

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=[
            (f"Profil altimétrique — {name_a} vs {name_b}  "
             f"(axe temps normalisé par la durée de {name_a} : "
             f"{int(dur_a*60)//60}h{int(dur_a*60)%60:02d})"),
            (f"Stretch {name_a} − {name_b} (km)  "
             f"— positif = {name_a} en avance"),
        ],
    )

    # ── Panel 1 : profils altitude ────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=t_norm_a, y=alt_a,
        mode="lines",
        line=dict(color="#4a9ede", width=2),
        name=name_a,
        hovertemplate=(
            f"{name_a}<br>"
            "t/T_ref : %{x:.3f}<br>"
            "Altitude : %{y:.0f} m<extra></extra>"
        ),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t_norm_b, y=alt_b,
        mode="lines",
        line=dict(color="#e07b54", width=2),
        name=name_b,
        hovertemplate=(
            f"{name_b}<br>"
            "t/T_ref : %{x:.3f}<br>"
            "Altitude : %{y:.0f} m<extra></extra>"
        ),
    ), row=1, col=1)

    # Ligne verticale à x=1 (arrivée de A)
    fig.add_vline(
        x=1.0,
        line=dict(color="#4a9ede", dash="dot", width=1.5),
        annotation_text=f"Arrivée {name_a}",
        annotation_font=dict(color="#4a9ede", size=9),
        annotation_position="top right",
        row=1, col=1,
    )
    if t_norm_b[-1] > 1.0:
        fig.add_vline(
            x=t_norm_b[-1],
            line=dict(color="#e07b54", dash="dot", width=1.5),
            annotation_text=f"Arrivée {name_b}",
            annotation_font=dict(color="#e07b54", size=9),
            annotation_position="top left",
            row=1, col=1,
        )

    fig.update_yaxes(title_text="Altitude (m)", row=1, col=1)

    # ── Panel 2 : stretch ─────────────────────────────────────────────────────
    # Zone colorée : vert quand A devant, rouge quand B devant
    fig.add_trace(go.Scatter(
        x=np.concatenate([t_grid, t_grid[::-1]]),
        y=np.concatenate([np.where(stretch > 0, stretch, 0),
                          np.zeros(len(t_grid))]),
        fill="toself",
        fillcolor="rgba(50,180,80,0.15)",
        line=dict(width=0),
        hoverinfo="skip",
        showlegend=False,
        name=f"{name_a} devant",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=np.concatenate([t_grid, t_grid[::-1]]),
        y=np.concatenate([np.where(stretch < 0, stretch, 0),
                          np.zeros(len(t_grid))]),
        fill="toself",
        fillcolor="rgba(220,50,50,0.15)",
        line=dict(width=0),
        hoverinfo="skip",
        showlegend=False,
        name=f"{name_b} devant",
    ), row=2, col=1)

    # Courbe du stretch
    fig.add_trace(go.Scatter(
        x=t_grid, y=stretch,
        mode="lines",
        line=dict(color="white", width=1.5),
        name="Stretch (km)",
        hovertemplate=(
            "t/T_ref : %{x:.3f}<br>"
            "Stretch : %{y:+.2f} km<extra></extra>"
        ),
        showlegend=False,
    ), row=2, col=1)

    fig.add_hline(
        y=0,
        line=dict(color="rgba(180,180,180,0.5)", width=1, dash="dash"),
        row=2, col=1,
    )

    fig.update_yaxes(title_text="Stretch (km)", row=2, col=1)
    fig.update_xaxes(
        title_text=f"Temps normalisé (1.0 = durée de {name_a})",
        row=2, col=1,
    )

    fig.update_layout(
        height=height,
        template="plotly_dark",
        legend=dict(orientation="h", x=0.3, y=1.04, font=dict(size=10)),
        margin=dict(l=60, r=40, t=80, b=60),
    )
    return fig


# ===========================================================================
# 23. Raw physiological profiles
# ===========================================================================

def plot_raw_profiles(df, variables=None, smoothing_s=30,
                      x_axis="distance", ravito_km=None, ravito_nom=None,
                      height_per_panel=150):
    """Multi-panel raw physiological profiles vs distance or time.

    Parameters
    ----------
    df              : pd.DataFrame  processed race DataFrame
    variables       : list[str] | None  columns to plot. If None, all
                      available columns among the defaults are used.
    smoothing_s     : int   smoothing window in seconds (default 30s)
    x_axis          : str   'distance' or 'time'
    ravito_km       : list[float] | None
    ravito_nom      : list[str]   | None
    height_per_panel: int   height in pixels per subplot row

    Returns
    -------
    plotly.graph_objects.Figure
    """
    # Variables disponibles par défaut avec leurs labels et couleurs
    VAR_META = {
        "heart_rate":    ("FC (bpm)",          "#e05c5c"),
        "pace_s_per_km": ("Allure (s/km)",      "#4a9ede"),
        "gap_s_per_km":  ("GAP (s/km)",         "#5bc45b"),
        "speed_kmh":     ("Vitesse (km/h)",     "#e0c454"),
        "cadence":       ("Cadence (spm)",      "#b06ecc"),
        "altitude":      ("Altitude (m)",       "#a0785a"),
        "alt_m":         ("Altitude (m)",       "#a0785a"),
        "slope_pct":     ("Pente (%)",          "#54c4c4"),
        "temperature":   ("Température montre (°C)", "#e07b54"),
        "temp_api":      ("Température ERA5 (°C)",   "#e07b54"),
        "power":         ("Puissance (W)",      "#f0a500"),
    }

    if variables is None:
        variables = [v for v in VAR_META if v in df.columns]
    else:
        variables = [v for v in variables if v in df.columns]

    if not variables:
        raise ValueError("Aucune variable disponible dans le DataFrame.")

    # Calcul de l'axe x
    df = df.copy().sort_values("timestamp" if "timestamp" in df.columns
                                else "dist_m")

    if x_axis == "distance":
        x = df["dist_m"] / 1000.0
        x_label = "Distance (km)"
        ravito_x = ravito_km or []
    else:
        if "time_h" not in df.columns:
            raise ValueError("Colonne time_h manquante pour l'axe temps.")
        x = df["time_h"] * 60.0  # en minutes
        x_label = "Temps (min)"
        # Convertir les positions ravito de km en minutes
        if ravito_km and "dist_m" in df.columns:
            ravito_x = []
            for km in ravito_km:
                idx = (df["dist_m"] / 1000.0 - km).abs().idxmin()
                ravito_x.append(float(df.loc[idx, "time_h"] * 60.0))
        else:
            ravito_x = []

    # Calcul de la fenêtre de lissage en points
    if "timestamp" in df.columns:
        dt = df["timestamp"].diff().dt.total_seconds().median()
        dt = dt if dt and dt > 0 else 1.0
    else:
        dt = 1.0
    win_pts = max(3, int(smoothing_s / dt))

    n = len(variables)
    fig = make_subplots(
        rows=n, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=[VAR_META.get(v, (v, ""))[0] for v in variables],
    )

    for row, col in enumerate(variables, start=1):
        label, color = VAR_META.get(col, (col, "#aaaaaa"))
        raw = df[col].to_numpy(dtype=float)

        # Lissage
        smooth = (pd.Series(raw)
                  .rolling(win_pts, center=True, min_periods=3)
                  .mean()
                  .to_numpy())

        # Axe inversé pour l'allure
        inverted = col in ("pace_s_per_km", "gap_s_per_km")

        # Courbe brute (transparente)
        fig.add_trace(go.Scatter(
            x=x, y=raw,
            mode="lines",
            line=dict(color=color, width=0.8),
            opacity=0.25,
            hoverinfo="skip",
            showlegend=False,
            name=f"{label} brut",
        ), row=row, col=1)

        # Courbe lissée
        fig.add_trace(go.Scatter(
            x=x, y=smooth,
            mode="lines",
            line=dict(color=color, width=2),
            name=label,
            hovertemplate=(
                f"{x_label.split(' ')[0]} : %{{x:.2f}}<br>"
                f"{label} : %{{y:.1f}}<extra></extra>"
            ),
        ), row=row, col=1)

        fig.update_yaxes(
            title_text=label,
            autorange="reversed" if inverted else True,
            row=row, col=1,
        )

        # Ravitos
        for rx, rnom in zip(ravito_x, ravito_nom or [""]*len(ravito_x)):
            fig.add_vline(
                x=rx,
                line=dict(color="rgba(180,180,180,0.35)", dash="dot", width=1),
                row=row, col=1,
            )
            if row == 1 and rnom:
                fig.add_annotation(
                    x=rx,
                    y=float(np.nanmax(smooth[~np.isnan(smooth)])),
                    xref="x", yref=f"y{row}",
                    text=rnom, showarrow=False,
                    font=dict(size=9, color="rgba(180,180,180,0.7)"),
                    textangle=-45, xanchor="left",
                )

    fig.update_xaxes(title_text=x_label, row=n, col=1)
    fig.update_layout(
        height=height_per_panel * n,
        template="plotly_dark",
        showlegend=False,
        title=(f"Profils physiologiques bruts — "
               f"lissage {smoothing_s}s — axe {x_label.split(' ')[0].lower()}"),
        margin=dict(l=70, r=40, t=60, b=50),
    )
    return fig


# ===========================================================================
# 25. EPOC, PTE, CTL/ATL/TSB, Calories
# ===========================================================================

def compute_epoc(df, fc_max, vo2max=None, dt_s=1.0):
    """Estimate EPOC accumulation during exercise.

    Approximation of the Firstbeat recursive model (Saalasti 2003 /
    Rusko et al. 2003) based on publicly available white paper structure.
    Not identical to the proprietary Firstbeat/Suunto implementation.

    EPOC accumulates when intensity > 50% VO2max and decreases during
    low-intensity periods. The model uses %VO2max derived from %HRmax
    via the linear approximation: %VO2max ≈ 1.2 * %HRmax - 0.2.

    Parameters
    ----------
    df      : pd.DataFrame  must contain heart_rate, dist_m, timestamp
    fc_max  : float         athlete max HR (bpm)
    vo2max  : float | None  athlete VO2max (ml/kg/min). If None, 50 is used.
    dt_s    : float         sampling interval in seconds (default 1.0)

    Returns
    -------
    pd.Series  EPOC at each timepoint (ml/kg), same index as df

    References
    ----------
    - Saalasti S (2003). Neural networks for heart rate time series analysis.
      PhD thesis, University of Jyväskylä.
    - Rusko H et al. (2003). Heart beat based automatic and individual
      assessment of anaerobic threshold. Med Sci Sports Exerc 35:S191.
    - Firstbeat Technologies (2012). Indirect EPOC Prediction Method Based
      on Heart Rate Measurement. White Paper.
    """
    if "heart_rate" not in df.columns:
        raise ValueError("Colonne heart_rate manquante.")

    vo2max = vo2max or 50.0

    # %HRmax → %VO2max (approximation linéaire, valide > 50% HRmax)
    hr_frac = (df["heart_rate"] / fc_max).clip(0, 1).to_numpy()
    vo2_frac = np.clip(1.2 * hr_frac - 0.2, 0, 1)
    pct_vo2max = vo2_frac * 100.0

    # Calcul du dt réel depuis les timestamps si disponible
    if "timestamp" in df.columns:
        dt_arr = df["timestamp"].diff().dt.total_seconds().fillna(1.0).to_numpy()
        dt_arr = np.clip(dt_arr, 0.5, 10.0)
    else:
        dt_arr = np.full(len(df), dt_s)

    # Modèle récursif EPOC
    # Composante montante : accumulation quand intensité élevée
    # Composante descendante : dissipation pendant récupération
    # Coefficients calés sur les données publiées (Rusko 2003, MAE ~14 ml/kg)
    EPOC_THRESHOLD = 50.0   # %VO2max seuil d'accumulation
    K_UP   = 0.0065         # taux d'accumulation (ml/kg/s par %VO2max)
    K_DOWN = 0.020          # taux de dissipation (1/s)

    epoc = np.zeros(len(df))
    for i in range(1, len(df)):
        dt = dt_arr[i]
        pct = pct_vo2max[i]
        prev = epoc[i - 1]

        if pct > EPOC_THRESHOLD:
            # Accumulation — non linéaire : s'accélère avec l'intensité
            rate = K_UP * (pct - EPOC_THRESHOLD) ** 1.5
            epoc[i] = prev + rate * dt
        else:
            # Dissipation exponentielle pendant récupération
            epoc[i] = prev * np.exp(-K_DOWN * dt)

        epoc[i] = max(0.0, epoc[i])

    return pd.Series(epoc, index=df.index, name="epoc_ml_kg")


def compute_pte(epoc_peak, vo2max=50.0):
    """Compute Peak Training Effect (PTE) from peak EPOC.

    PTE is EPOC scaled by the athlete's fitness level (VO2max).
    A fitter athlete needs a higher EPOC to reach the same PTE level.
    Scale 0–5, consistent with Suunto/Firstbeat PTE definition.

    Parameters
    ----------
    epoc_peak : float   peak EPOC value from the session (ml/kg)
    vo2max    : float   athlete VO2max (ml/kg/min). Default 50.

    Returns
    -------
    float  PTE on a 0.0–5.0 scale

    References
    ----------
    - Firstbeat Technologies (2014). Training Effect White Paper.
    - Suunto (2022). Glossary — Peak Training Effect.
    """
    # Normalisation par le fitness level
    # Un athlète avec VO2max = 50 atteint PTE=5 à ~200 ml/kg d'EPOC
    # La relation est logarithmique selon le white paper Firstbeat
    fitness_factor = vo2max / 50.0
    epoc_scaled = epoc_peak / fitness_factor

    # Mapping logarithmique EPOC → PTE (0–5)
    # Calé sur les valeurs Suunto documentées dans les forums
    if epoc_scaled <= 0:
        return 0.0
    pte = 5.0 * np.log(1.0 + epoc_scaled / 30.0) / np.log(1.0 + 200.0 / 30.0)
    return float(np.clip(round(pte, 1), 0.0, 5.0))


def compute_calories(df, fc_max, fc_rest, poids_kg,
                     age_years=35, gender="male", dplus_m=0.0):
    """Estimate energy expenditure during exercise.

    Two methods combined :
      1. Keytel et al. (2005) — HR-based, validated on treadmill.
         Used when HR data is available.
      2. Mechanical work heuristic — distance + D+ correction.
         Used as fallback or cross-check.

    Parameters
    ----------
    df        : pd.DataFrame  must contain heart_rate, dist_m
    fc_max    : float
    fc_rest   : float
    poids_kg  : float
    age_years : float   Default 35.
    gender    : str     'male' | 'female'
    dplus_m   : float   cumulative elevation gain (m)

    Returns
    -------
    dict with keys:
        calories_keytel    : float  kcal (HR-based)
        calories_mechanical: float  kcal (distance + D+ heuristic)
        calories_combined  : float  kcal (weighted mean)

    References
    ----------
    - Keytel LR et al. (2005). Prediction of energy expenditure from heart
      rate monitoring during submaximal exercise. J Sports Sci 23(3):289-297.
    - di Prampero PE (1986). The energy cost of human locomotion on land
      and in water. Int J Sports Med 7(2):55-72.
    """
    duration_min = 0.0
    hr_mean = None

    if "timestamp" in df.columns and "heart_rate" in df.columns:
        dt_s = df["timestamp"].diff().dt.total_seconds().fillna(1.0)
        duration_min = float(dt_s.sum() / 60.0)
        hr_mean = float(df["heart_rate"].dropna().mean())

    dist_km = float(df["dist_m"].max() / 1000.0) if "dist_m" in df.columns else 0.0

    # ── Méthode Keytel (2005) ────────────────────────────────────────────────
    cal_keytel = None
    if hr_mean and duration_min > 0:
        if gender == "male":
            kcal_min = (-55.0969 + 0.6309 * hr_mean
                        + 0.1988 * poids_kg
                        + 0.2017 * age_years) / 4.184
        else:
            kcal_min = (-20.4022 + 0.4472 * hr_mean
                        - 0.1263 * poids_kg
                        + 0.0740 * age_years) / 4.184
        cal_keytel = float(max(0.0, kcal_min) * duration_min)

    # ── Méthode mécanique (di Prampero 1986 + correction D+) ─────────────────
    # Coût horizontal : ~1 kcal/kg/km
    # Coût vertical   : ~0.00235 * poids * D+ (équivalence course/montée)
    cal_meca = float(poids_kg * dist_km + poids_kg * 0.00235 * dplus_m)

    # ── Combinaison pondérée ──────────────────────────────────────────────────
    if cal_keytel is not None:
        # Keytel pondéré 70%, mécanique 30%
        cal_combined = 0.7 * cal_keytel + 0.3 * cal_meca
    else:
        cal_combined = cal_meca

    return {
        "calories_keytel":     round(cal_keytel, 0) if cal_keytel else None,
        "calories_mechanical": round(cal_meca, 0),
        "calories_combined":   round(cal_combined, 0),
    }


def compute_session_load(df, fc_max, fc_rest, poids_kg,
                          vo2max=None, age_years=35, gender="male",
                          dplus_m=None):
    """Compute all load metrics for a single session.

    Combines TRIMP, EPOC, PTE and calories in a single call.

    Parameters
    ----------
    df        : pd.DataFrame  processed race DataFrame
    fc_max    : float
    fc_rest   : float
    poids_kg  : float
    vo2max    : float | None   athlete VO2max (ml/kg/min)
    age_years : float
    gender    : str
    dplus_m   : float | None   if None, computed from df

    Returns
    -------
    dict  all load metrics
    """
    if dplus_m is None:
        dplus_m, _ = compute_dplus_dminus(df)

    # TRIMP (Banister 1975)
    hr_frac = (df["heart_rate"] - fc_rest) / (fc_max - fc_rest)
    hr_frac = hr_frac.clip(0, 1)
    b = 1.92 if gender == "male" else 1.67
    if "timestamp" in df.columns:
        dt_min = df["timestamp"].diff().dt.total_seconds().fillna(1.0) / 60.0
    else:
        dt_min = pd.Series(np.ones(len(df)) / 60.0)
    trimp = float((dt_min * hr_frac * 0.64 * np.exp(b * hr_frac)).sum())

    # EPOC
    epoc_series = compute_epoc(df, fc_max, vo2max=vo2max)
    epoc_peak = float(epoc_series.max())
    epoc_final = float(epoc_series.iloc[-1])

    # PTE
    pte = compute_pte(epoc_peak, vo2max=vo2max or 50.0)

    # Calories
    cal = compute_calories(df, fc_max, fc_rest, poids_kg,
                           age_years=age_years, gender=gender,
                           dplus_m=dplus_m)

    duration_h = float(df["time_h"].max()) if "time_h" in df.columns else 0.0
    dist_km = float(df["dist_m"].max() / 1000.0) if "dist_m" in df.columns else 0.0

    return {
        "trimp":               round(trimp, 1),
        "epoc_peak_ml_kg":     round(epoc_peak, 1),
        "epoc_final_ml_kg":    round(epoc_final, 1),
        "pte":                 pte,
        "calories_keytel":     cal["calories_keytel"],
        "calories_mechanical": cal["calories_mechanical"],
        "calories_combined":   cal["calories_combined"],
        "duration_h":          round(duration_h, 3),
        "distance_km":         round(dist_km, 2),
        "dplus_m":             round(dplus_m, 0),
    }

def plot_epoc(df, fc_max, vo2max=None, ravito_km=None, ravito_nom=None,
              height=350):
    """EPOC accumulation curve along the race distance.

    Parameters
    ----------
    df         : pd.DataFrame  processed race DataFrame
    fc_max     : float         athlete max HR (bpm)
    vo2max     : float | None  athlete VO2max (ml/kg/min). Default 50.
    ravito_km  : list[float] | None
    ravito_nom : list[str]   | None
    height     : int           figure height in pixels

    Returns
    -------
    plotly.graph_objects.Figure
    """
    epoc_series = compute_epoc(df, fc_max, vo2max=vo2max)
    epoc_peak   = float(epoc_series.max())
    pte         = compute_pte(epoc_peak, vo2max=vo2max or 50.0)

    x_km = df["dist_m"] / 1000.0

    fig = go.Figure()

    # ── Courbe EPOC ───────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_km,
        y=epoc_series,
        mode="lines",
        line=dict(color="#b06ecc", width=2),
        fill="tozeroy",
        fillcolor="rgba(176,110,204,0.10)",
        name="EPOC",
        hovertemplate=(
            "Distance : %{x:.2f} km<br>"
            "EPOC : %{y:.1f} ml/kg<extra></extra>"
        ),
    ))

    # ── Marqueur du pic ───────────────────────────────────────────────────────
    idx_peak = epoc_series.idxmax()
    x_peak   = float(df.loc[idx_peak, "dist_m"] / 1000.0)
    fig.add_trace(go.Scatter(
        x=[x_peak],
        y=[epoc_peak],
        mode="markers+text",
        marker=dict(color="#b06ecc", size=10, symbol="diamond"),
        text=[f"Peak {epoc_peak:.1f} ml/kg"],
        textposition="top center",
        textfont=dict(size=9, color="#b06ecc"),
        showlegend=False,
        hoverinfo="skip",
    ))

    # ── Zones de référence EPOC (valeurs Firstbeat documentées) ──────────────
    zones = [
        (0,   40,  "rgba(80,180,80,0.04)",   "Maintien"),
        (40,  100, "rgba(240,180,0,0.04)",   "Amélioration"),
        (100, 200, "rgba(220,100,0,0.04)",   "Amélioration importante"),
        (200, 400, "rgba(220,50,50,0.04)",   "Effort maximal"),
    ]
    y_max = max(epoc_peak * 1.2, 50)
    for y0, y1, color, label in zones:
        if y0 >= y_max:
            break
        fig.add_hrect(
            y0=y0, y1=min(y1, y_max),
            fillcolor=color, line_width=0,
            annotation_text=label,
            annotation_position="right",
            annotation_font=dict(size=8, color="rgba(200,200,200,0.6)"),
        )

    # ── Ravitos ───────────────────────────────────────────────────────────────
    for km, nom in zip(ravito_km or [], ravito_nom or []):
        fig.add_vline(
            x=km,
            line=dict(color="rgba(180,180,180,0.35)", dash="dot", width=1),
        )
        fig.add_annotation(
            x=km, y=epoc_peak * 1.05,
            xref="x", yref="y",
            text=nom, showarrow=False,
            font=dict(size=9, color="rgba(180,180,180,0.6)"),
            textangle=-45, xanchor="left",
        )

    # ── PTE zones en annotation ───────────────────────────────────────────────
    pte_color = (
        "seagreen"  if pte < 3.0 else
        "goldenrod" if pte < 4.0 else
        "crimson"
    )
    fig.update_layout(
        title=(
            f"EPOC au fil de la course — "
            f"Peak : {epoc_peak:.1f} ml/kg  |  "
            f"PTE : <span style='color:{pte_color}'>{pte:.1f} / 5.0</span><br>"
            "<sup>Modèle Saalasti (2003) / Firstbeat White Paper — "
            "approximation, non identique aux valeurs Suunto</sup>"
        ),
        xaxis_title="Distance (km)",
        yaxis_title="EPOC (ml/kg)",
        yaxis=dict(range=[0, y_max]),
        template="plotly_dark",
        height=height,
        showlegend=False,
        margin=dict(l=60, r=100, t=80, b=50),
    )
    return fig