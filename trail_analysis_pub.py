"""trail_analysis.py — Utility functions for trail running FIT file analysis.

Usage in a notebook:
    from trail_analysis_pub import *


    
"""

import io
import base64
import warnings

import folium
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fitparse import FitFile
from branca.element import MacroElement
from jinja2 import Template

import requests

# ---------------------------------------------------------------------------
# Silence FutureWarnings from pandas concat with all-NA columns
# ---------------------------------------------------------------------------
warnings.filterwarnings(
    "ignore",
    message="The behavior of DataFrame concatenation with empty or all-NA entries",
    category=FutureWarning,
)


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

    # GPS coordinates (semicircles → degrees)
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
    return np.where(dd > 0, (dz / dd) * 100.0, np.nan)


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



def classify_walk_run(df, walk_thr_kmh, walk_thr_cad):
    """Classify each point as walking (1) or running (0)."""
    df = df.copy()
    df["is_walk"] = (
        (df["speed_kmh"] < walk_thr_kmh) & (df["cadence"] < walk_thr_cad)
    ).astype(int)
    return df




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
    # Cast explicite pour éviter le FutureWarning pandas concat
    result = result.astype({
        "Position (km)": float,
        "Nb arrêts": int,
        "Temps total arrêt (s)": float,
        "Temps total arrêt (min)": float,
    })
    return pd.concat([result, total_row], ignore_index=True)


# ===========================================================================
# 5. Cartographie (folium)
# ===========================================================================

class ElevationControl(MacroElement):
    """Folium MacroElement that overlays an elevation profile on the map.

    Injects a floating <div> in the bottom-left corner of the Leaflet map
    containing the profile as a base64-encoded SVG image. The HTML output
    is self-contained (no external image file required).
    """

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
    """Generate an elevation profile as a base64-encoded SVG string.

    Parameters
    ----------
    df : DataFrame with distance and altitude columns.
    dist_col : str, distance column name (metres).
    alt_col : str, altitude column name (metres).
    figsize : tuple, matplotlib figure size in inches.
    dpi : int, figure resolution.

    Returns
    -------
    str : base64-encoded SVG.
    """
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

    Colors the track by a data column (heart_rate, speed_kmh, slope_pct…)
    if color_col is provided. Aid stations, start/finish markers, an
    animated AntPath trace, and an elevation profile inset can be added
    via optional parameters.

    Parameters
    ----------
    df : DataFrame with lat, lon, dist_m, alt_m columns.
    ravito_km : list of float. Optional.
    ravito_nom : list of str. Optional.
    color_col : str, column to color the track. None = single color.
    cmap_name : str, matplotlib colormap name.
    tile : str, folium tile layer.
    animated : bool, add AntPath animation over the full trace.
    show_elevation : bool, overlay elevation profile inset (bottom-left).

    Returns
    -------
    folium.Map
    """
    import matplotlib as mpl
    from folium.plugins import AntPath

    df = df.dropna(subset=["lat", "lon"]).copy()
    if df.empty:
        print("Pas de coordonnées GPS dans ce fichier.")
        return None

    center = [df["lat"].mean(), df["lon"].mean()]
    m = folium.Map(location=center, zoom_start=12, tiles=tile)

    # --- Trace colorée ou monochrome ---
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
                color=hex_color,
                weight=3,
                opacity=0.85,
            ).add_to(m)
    else:
        coords = list(zip(df["lat"], df["lon"]))
        folium.PolyLine(coords, color="steelblue", weight=3,
                        opacity=0.85).add_to(m)

    # --- Marqueurs départ / arrivée ---
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

    # --- Ravitaillements ---
    if ravito_km and ravito_nom:
        for km, nom in zip(ravito_km, ravito_nom):
            idx = (df["dist_m"] / 1000.0 - km).abs().idxmin()
            if idx in df.index:
                folium.Marker(
                    location=[df.loc[idx, "lat"], df.loc[idx, "lon"]],
                    popup=f"{nom} ({km} km)",
                    icon=folium.Icon(color="blue", icon="cutlery",
                                     prefix="fa"),
                ).add_to(m)

    # --- Trace animée AntPath ---
    if animated:
        AntPath(
            locations=coords,
            color="steelblue",
            weight=4,
            delay=800,
            dash_array=[10, 20],
            pulse_color="#ffffff",
        ).add_to(m)

    # --- Profil altimétrique en incrustation ---
    if show_elevation and "alt_m" in df.columns and "dist_m" in df.columns:
        svg_b64 = make_elevation_svg(df)
        ElevationControl(svg_b64).add_to(m)

    return m


# ===========================================================================
# 6. Plots — profils et distributions
# ===========================================================================

def plot_profil_colore(df, col, label, cmap="viridis", vmin=None, vmax=None):
    """Plot elevation profile with scatter colored by a data column."""
    mask = df[col].notna()
    x = df.loc[mask, "dist_m"] / 1000.0
    y = df.loc[mask, "alt_m"]
    c = df.loc[mask, col]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(
        df["dist_m"] / 1000.0, df["alt_m"], df["alt_m"].min(),
        color="lightgrey", alpha=0.5, zorder=1
    )
    sc = ax.scatter(x, y, c=c, cmap=cmap, s=4, alpha=0.8,
                    vmin=vmin, vmax=vmax, zorder=2)
    plt.colorbar(sc, ax=ax, label=label)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Altitude (m)")
    ax.set_title(f"Profil coloré — {label}")
    fig.tight_layout()
    plt.show()


def plot_dashboard(df, fc_min, fc_max):
    """Multi-panel elevation profiles colored by HR, pace, and temperature."""
    variables = [
        ("heart_rate",    "FC (bpm)",         "coolwarm",  fc_min, fc_max),
        ("pace_s_per_km", "Allure (s/km)",     "RdYlGn_r",  180,   900),
        ("temperature",   "Température (°C)",  "plasma",    None,  None),
    ]
    variables = [(c, l, cm, vn, vx) for c, l, cm, vn, vx in variables
                 if c in df.columns]
    n = len(variables)
    if n == 0:
        print("Aucune variable disponible pour le dashboard.")
        return

    fig, axes = plt.subplots(n, 1, figsize=(13, 2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (col, label, cmap, vmin, vmax) in zip(axes, variables):
        mask = df[col].notna()
        x = df.loc[mask, "dist_m"] / 1000.0
        y = df.loc[mask, "alt_m"]
        c = df.loc[mask, col]
        ax.fill_between(
            df["dist_m"] / 1000.0, df["alt_m"], df["alt_m"].min(),
            color="lightgrey", alpha=0.5, zorder=1
        )
        sc = ax.scatter(x, y, c=c, cmap=cmap, s=4, alpha=0.8,
                        vmin=vmin, vmax=vmax, zorder=2)
        plt.colorbar(sc, ax=ax, label=label)
        ax.set_ylabel("Altitude (m)")
        ax.set_title(f"Profil — {label}")

    axes[-1].set_xlabel("Distance (km)")
    fig.tight_layout()
    plt.show()


def plot_walk_by_slope_sections(df, slope_bins, slope_labels,
                                ravito_km=None, n_segments=None,
                                walk_thr_kmh=6.0, walk_thr_cad=140.0):
    """Plot walk probability by slope class, split by race sections."""
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
        labels=section_labels, include_lowest=True
    )
    df["slope_bin"] = pd.cut(
        df["slope_pct"], bins=slope_bins, labels=slope_labels
    )
    tmp = df.dropna(subset=["section", "slope_bin", "cadence"])
    walk_matrix = (
        tmp.groupby(["slope_bin", "section"], observed=True)["is_walk"]
        .mean() * 100
    ).unstack("section")

    n_sections = len(section_labels)
    fig, ax = plt.subplots(figsize=(max(8, 2 * n_sections), 3))
    walk_matrix.plot(kind="bar", ax=ax, edgecolor="white", width=0.8)
    ax.set_ylabel("% de marche")
    ax.set_xlabel("Classe de pente")
    title = (
        "Probabilité de marcher — sections ravitos"
        if ravito_km is not None
        else f"Probabilité de marcher — {n_segments} segments"
    )
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Section", bbox_to_anchor=(1.01, 1),
              loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.show()

