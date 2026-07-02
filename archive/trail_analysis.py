"""trail_analysis.py — Utility functions for trail running FIT file analysis.

Usage in a notebook:
    from trail_analysis import *
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
    """Compute smoothed HR/GAP ratio as cardiac drift proxy.

    Filters stops and uses GAP instead of raw speed.
    """
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
    labels = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if i == 0:
            labels.append(f"Départ → {ravito_nom[0]} ({a:.1f}–{b:.1f} km)")
        elif i == len(bounds) - 2:
            labels.append(f"{ravito_nom[-1]} → Arrivée ({a:.1f}–{b:.1f} km)")
        else:
            labels.append(
                f"{ravito_nom[i-1]} → {ravito_nom[i]} ({a:.1f}–{b:.1f} km)"
            )
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


def plot_heatmap_sections(df, fc_max, walk_thr_kmh=6.0,
                          ravito_km=None, n_segments=None,
                          slope_min=-25, slope_max=25, slope_step=1.0,
                          speed_min=0, speed_max=18, speed_step=0.5,
                          min_count=5):
    """Plot HR heatmaps (speed × slope) split by race sections."""
    if ravito_km is None and n_segments is None:
        raise ValueError("Provide either ravito_km or n_segments.")
    if ravito_km is not None and n_segments is not None:
        raise ValueError("Provide only one of ravito_km or n_segments.")

    if "heart_rate" not in df.columns:
        print("Pas de données FC.")
        return

    req = ["speed_kmh", "slope_pct", "heart_rate", "dist_m"]
    dh = df.dropna(subset=req).copy()

    slope_bins = np.arange(slope_min, slope_max + slope_step, slope_step)
    speed_bins = np.arange(speed_min, speed_max + speed_step, speed_step)
    extent = [speed_min, speed_max, slope_min, slope_max]
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

    n_panels = len(all_labels)
    ncols = min(n_panels, 3)
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7 * ncols, 6 * nrows), squeeze=False
    )
    axes_flat = axes.flatten()

    def _compute_heat(mask):
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

    im_ref = None
    for idx, (label, mask) in enumerate(zip(all_labels, all_masks)):
        ax = axes_flat[idx]
        heat = _compute_heat(mask)
        im = ax.imshow(heat, aspect="auto", origin="lower", extent=extent,
                       cmap="RdYlGn_r", vmin=vmin, vmax=vmax)
        if im_ref is None:
            im_ref = im
        ax.axhline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.8)
        ax.axvline(walk_thr_kmh, color="black", linewidth=1.2,
                   linestyle="--", alpha=0.8)
        ax.set_xlabel("Vitesse (km/h)")
        ax.set_ylabel("Pente (%)")
        ax.set_title(f"{label}  (n={int(mask.sum())})", fontsize=10)

    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im_ref, cax=cbar_ax, label="FC médiane (bpm)")
    title = (
        "Heatmap FC médiane — sections ravitos"
        if ravito_km is not None
        else f"Heatmap FC médiane — {n_segments} segments"
    )
    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout(rect=[0, 0, 0.89, 1])
    plt.show()

# ------------------------------------------------------------
# A. Découplage aérobie (aerobic decoupling)
# ------------------------------------------------------------

def compute_aerobic_decoupling(df, ravito_km, ravito_nom,
                               ref_start_km=None, ref_end_km=None,
                               threshold_pct=2.5):
    """Compute aerobic decoupling per section relative to a reference window.

    Aerobic decoupling = increase in HR/GAP ratio relative to baseline.
    A value > threshold_pct signals cardiovascular drift (Smyth et al., 2022).

    The reference window defaults to the first section (km 0 to ravito_km[0]).
    Override with ref_start_km / ref_end_km for a custom baseline.

    Parameters
    ----------
    df : DataFrame with heart_rate, gap_s_per_km, dist_m columns.
    ravito_km : list of float.
    ravito_nom : list of str.
    ref_start_km : float. Start of reference window (km). Default: 0.
    ref_end_km : float. End of reference window (km). Default: ravito_km[0].
    threshold_pct : float. Decoupling alert threshold (%). Default: 2.5.

    Returns
    -------
    DataFrame with one row per section and decoupling metrics.
    """
    if "heart_rate" not in df.columns or "gap_s_per_km" not in df.columns:
        print("Colonnes heart_rate ou gap_s_per_km manquantes.")
        return pd.DataFrame()

    df = df.copy()
    # ratio interne/externe : FC normalisée / vitesse GAP
    gap_kmh = 3600.0 / df["gap_s_per_km"].replace(0, np.nan)
    df["hr_gap_ratio"] = df["heart_rate"] / gap_kmh

    # Fenêtre de référence
    if ref_start_km is None:
        ref_start_km = df["dist_m"].min() / 1000.0
    if ref_end_km is None:
        ref_end_km = ravito_km[0] if ravito_km else df["dist_m"].max() / 1000.0

    ref_mask = (
        (df["dist_m"] / 1000.0 >= ref_start_km) &
        (df["dist_m"] / 1000.0 <= ref_end_km) &
        df["hr_gap_ratio"].notna()
    )
    ref_ratio = df.loc[ref_mask, "hr_gap_ratio"].median()
    if np.isnan(ref_ratio) or ref_ratio == 0:
        print("Ratio de référence invalide.")
        return pd.DataFrame()

    # Sections
    bounds = np.concatenate((
        [df["dist_m"].min() / 1000.0],
        np.array(ravito_km),
        [df["dist_m"].max() / 1000.0]
    ))
    labels = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if i == 0:
            nom = f"Départ → {ravito_nom[0]}"
        elif i == len(bounds) - 2:
            nom = f"{ravito_nom[-1]} → Arrivée"
        else:
            nom = f"{ravito_nom[i-1]} → {ravito_nom[i]}"
        labels.append((nom, a, b))

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
            "Section": nom,
            "Ratio méd. FC/GAP": round(sec_ratio, 3),
            "Découplage (%)": round(decoupling_pct, 1),
            "Alerte": "⚠️" if decoupling_pct > threshold_pct else "✅",
        })

    result = pd.DataFrame(rows)
    result.attrs["ref_ratio"] = ref_ratio
    result.attrs["threshold_pct"] = threshold_pct
    return result


def plot_aerobic_decoupling(df, ravito_km, ravito_nom,
                            smoothing_km=2.0, threshold_pct=2.5):
    """Plot continuous aerobic decoupling curve along the race.

    Shows the rolling HR/GAP ratio normalized to the first section.
    A horizontal dashed line marks the +2.5 % alert threshold.

    Parameters
    ----------
    df : DataFrame with heart_rate, gap_s_per_km, dist_m columns.
    smoothing_km : float, rolling window in km.
    threshold_pct : float, alert threshold (%).
    """
    if "heart_rate" not in df.columns or "gap_s_per_km" not in df.columns:
        print("Colonnes heart_rate ou gap_s_per_km manquantes.")
        return

    df = df.copy().sort_values("dist_m")
    gap_kmh = 3600.0 / df["gap_s_per_km"].replace(0, np.nan)
    df["hr_gap_ratio"] = df["heart_rate"] / gap_kmh

    # Lissage spatial (fenêtre glissante en mètres)
    window_pts = max(
        10,
        int(smoothing_km * 1000 / df["dist_m"].diff().median())
    )
    df["ratio_smooth"] = (
        df["hr_gap_ratio"]
        .rolling(window_pts, center=True, min_periods=5)
        .median()
    )

    # Référence : médiane des 20 % premiers km
    ref_km = df["dist_m"].max() / 1000.0 * 0.20
    ref_mask = df["dist_m"] / 1000.0 <= ref_km
    ref_ratio = df.loc[ref_mask, "ratio_smooth"].median()
    df["decoupling_pct"] = (df["ratio_smooth"] / ref_ratio - 1.0) * 100.0

    x = df["dist_m"] / 1000.0

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.axhline(threshold_pct, color="crimson", linewidth=1.2,
               linestyle="--", label=f"Seuil +{threshold_pct}%")
    ax.plot(x, df["decoupling_pct"], color="purple", linewidth=1.5,
            label="Découplage FC/GAP")
    ax.fill_between(x, 0, df["decoupling_pct"],
                    where=df["decoupling_pct"] > threshold_pct,
                    color="crimson", alpha=0.15, label="Zone alerte")

    for km, nom in zip(ravito_km, ravito_nom):
        ax.axvline(km, color="black", linestyle=":", alpha=0.5, linewidth=1)
        ax.text(km, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else -1,
                nom, fontsize=7, rotation=90, va="top",
                ha="right", color="grey")

    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Découplage FC/GAP (%)")
    ax.set_title(
        "Découplage aérobie — dérive FC/GAP normalisée aux 20 premiers km\n"
        "(Smyth et al., 2022 ; Maunder et al., 2021)"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()


# ------------------------------------------------------------
# B. Longueur de foulée et variabilité
# ------------------------------------------------------------

def compute_stride_metrics(df, ravito_km, ravito_nom):
    """Compute stride length and cadence variability per section.

    Stride length = speed_kmh / (cadence / 60) * 1000 / 2  [m per step]
    Cadence CV = coefficient of variation per section (fatigue proxy).

    Parameters
    ----------
    df : DataFrame with speed_kmh, cadence, dist_m columns.

    Returns
    -------
    DataFrame with stride metrics per section.
    """
    if "cadence" not in df.columns:
        print("Colonne cadence manquante.")
        return pd.DataFrame()

    df = df.copy()
    # Longueur de foulée en mètres (stride = 2 steps)
    cadence_hz = df["cadence"] / 60.0
    speed_mps = df["speed_kmh"] / 3.6
    df["stride_length_m"] = np.where(
        cadence_hz > 0,
        speed_mps / cadence_hz,  # longueur d'une enjambée (2 pas)
        np.nan
    )

    bounds = np.concatenate((
        [df["dist_m"].min() / 1000.0],
        np.array(ravito_km),
        [df["dist_m"].max() / 1000.0]
    ))
    labels = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if i == 0:
            nom = f"Départ → {ravito_nom[0]}"
        elif i == len(bounds) - 2:
            nom = f"{ravito_nom[-1]} → Arrivée"
        else:
            nom = f"{ravito_nom[i-1]} → {ravito_nom[i]}"
        labels.append((nom, a, b))

    rows = []
    for nom, a, b in labels:
        mask = (
            (df["dist_m"] / 1000.0 >= a) &
            (df["dist_m"] / 1000.0 < b) &
            df["cadence"].notna() &
            (df["speed_kmh"] > 4.0)  # exclure marche lente
        )
        sec = df.loc[mask]
        if len(sec) < 20:
            continue
        cad_mean = sec["cadence"].mean()
        cad_std = sec["cadence"].std()
        rows.append({
            "Section": nom,
            "Cadence méd. (spm)": round(sec["cadence"].median(), 1),
            "CV cadence (%)": round(cad_std / cad_mean * 100, 1),
            "Foulée méd. (m)": round(sec["stride_length_m"].median(), 2),
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# C. Radar / Spider par section
# ------------------------------------------------------------

def plot_radar_sections(df, ravito_km, ravito_nom, fc_max):
    """Plot a radar chart with 5 performance axes per section.

    Axes (all normalized 0–1, higher = better):
      1. Allure GAP relative (vitesse — inversée)
      2. Économie cardiaque (FC basse = bon = 1 - FC%FCmax)
      3. % course (1 - %marche)
      4. Stabilité cadence (1 - CV_cadence normalisé)
      5. Découplage faible (1 - découplage normalisé)

    Parameters
    ----------
    df : DataFrame fully processed (gap_s_per_km, is_walk, cadence, heart_rate).
    ravito_km : list of float.
    ravito_nom : list of str.
    fc_max : float.
    """
    import matplotlib.patches as mpatches

    bounds = np.concatenate((
        [df["dist_m"].min() / 1000.0],
        np.array(ravito_km),
        [df["dist_m"].max() / 1000.0]
    ))
    section_names = []
    for i in range(len(bounds) - 1):
        if i == 0:
            section_names.append(f"→ {ravito_nom[0]}")
        elif i == len(bounds) - 2:
            section_names.append(f"{ravito_nom[-1]} →")
        else:
            section_names.append(ravito_nom[i])

    # Calcul des métriques brutes par section
    gap_kmh = 3600.0 / df["gap_s_per_km"].replace(0, np.nan)
    df = df.copy()
    df["hr_gap_ratio"] = df["heart_rate"] / gap_kmh if "heart_rate" in df.columns else np.nan

    raw = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        mask = (df["dist_m"] / 1000.0 >= a) & (df["dist_m"] / 1000.0 < b)
        sec = df.loc[mask]
        if len(sec) < 20:
            raw.append(None)
            continue

        gap_med = sec["gap_s_per_km"].median() if "gap_s_per_km" in sec else np.nan
        fc_med = sec["heart_rate"].median() / fc_max if "heart_rate" in sec.columns else 0.5
        pct_run = 1.0 - sec["is_walk"].mean() if "is_walk" in sec.columns else 0.5
        cad = sec["cadence"].dropna()
        cv_cad = cad.std() / cad.mean() if len(cad) > 5 and cad.mean() > 0 else 0.1
        hr_gap = sec["hr_gap_ratio"].median() if "hr_gap_ratio" in sec.columns else np.nan

        raw.append({
            "gap_med": gap_med,
            "fc_frac": fc_med,
            "pct_run": pct_run,
            "cv_cad": cv_cad,
            "hr_gap": hr_gap,
        })

    # Normalisation globale sur toutes les sections
    valid = [r for r in raw if r is not None]
    if not valid:
        print("Pas assez de données pour le radar.")
        return

    gap_all = [r["gap_med"] for r in valid if not np.isnan(r["gap_med"])]
    hr_gap_all = [r["hr_gap"] for r in valid if not np.isnan(r["hr_gap"])]
    cv_all = [r["cv_cad"] for r in valid]

    gap_min, gap_max = min(gap_all), max(gap_all)
    hr_gap_min, hr_gap_max = min(hr_gap_all), max(hr_gap_all)
    cv_min, cv_max = min(cv_all), max(cv_all)

    def norm(val, vmin, vmax, invert=False):
        if vmax == vmin:
            return 0.5
        n = (val - vmin) / (vmax - vmin)
        return 1.0 - n if invert else n

    labels = ["Vitesse\nGAP", "Économie\ncardio", "% Course", "Régularité\ncadence", "Durabilité"]
    n_axes = len(labels)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8),
                           subplot_kw=dict(polar=True))
    cmap = plt.get_cmap("tab10")
    handles = []

    for i, (r, name) in enumerate(zip(raw, section_names)):
        if r is None:
            continue
        gap_score = norm(r["gap_med"], gap_min, gap_max, invert=True)
        fc_score = 1.0 - r["fc_frac"]
        run_score = r["pct_run"]
        cad_score = norm(r["cv_cad"], cv_min, cv_max, invert=True)
        dur_score = (
            norm(r["hr_gap"], hr_gap_min, hr_gap_max, invert=True)
            if not np.isnan(r["hr_gap"]) else 0.5
        )

        values = [gap_score, fc_score, run_score, cad_score, dur_score]
        values += values[:1]

        color = cmap(i % 10)
        ax.plot(angles, values, color=color, linewidth=1.8, linestyle="solid")
        ax.fill(angles, values, color=color, alpha=0.08)
        handles.append(mpatches.Patch(color=color, label=name))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], size=7, color="grey")
    ax.set_title("Profil de performance par section\n(normalisé — vers l'extérieur = mieux)",
                 size=12, pad=20)
    ax.legend(handles=handles, loc="upper right",
              bbox_to_anchor=(1.35, 1.15), fontsize=8)
    fig.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# ERA5-Land via openmeteo-requests SDK (FlatBuffers)
# ─────────────────────────────────────────────────────────────────────────────

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

# Variables horaires demandées à ERA5-Land (ordre fixe pour le SDK FlatBuffers)
_HOURLY_VARS = [
    "temperature_2m",           # 0  °C
    "relative_humidity_2m",     # 1  %
    "apparent_temperature",     # 2  °C
    "precipitation",            # 3  mm
    "wind_speed_10m",           # 4  km/h
    "shortwave_radiation",      # 5  W/m²  (flux instantané horaire)
]

# URL archive Open-Meteo (même endpoint que dans fetch_weather_open_meteo Twinity)
_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


# ─────────────────────────────────────────────────────────────────────────────
# WBGT — même formule que loads.py (Bernard & Kenney 1994)
# ─────────────────────────────────────────────────────────────────────────────

def compute_wbgt(temp_c, humidity_pct, solar_rad=None):
    """Wet Bulb Globe Temperature — Bernard & Kenney (1994).

    Identique à core/ffm/loads.py pour cohérence avec Twinity.
    Fonctionne sur scalaires ou arrays numpy.

    Parameters
    ----------
    temp_c       : float or array  température de l'air (°C)
    humidity_pct : float or array  humidité relative (%)
    solar_rad    : float or array | None  rayonnement global (W/m²)

    Returns
    -------
    float or array  WBGT (°C), None si entrées invalides (scalaire)
    """
    scalar = np.ndim(temp_c) == 0

    t  = np.asarray(temp_c,       dtype=float)
    rh = np.asarray(humidity_pct, dtype=float)

    # Pression de vapeur d'eau partielle (hPa)
    e_a = (rh / 100.0) * 6.105 * np.exp(17.27 * t / (237.7 + t))

    wbgt = 0.567 * t + 0.393 * e_a + 3.94

    # Correction rayonnement solaire (Liljegren et al. 2008)
    if solar_rad is not None:
        sr = np.asarray(solar_rad, dtype=float)
        wbgt = np.where(sr > 0, wbgt + 0.0006 * sr, wbgt)

    if scalar:
        return round(float(wbgt), 2)
    return np.round(wbgt, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Récupération ERA5-Land via SDK openmeteo-requests (FlatBuffers)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_weather_hourly(lat, lon, date_str, timezone="Europe/Paris",
                         client=None, model="era5_land", date_end=None):
    """Fetch hourly reanalysis data via openmeteo-requests SDK.

    Utilise le même endpoint archive-api.open-meteo.com que Twinity.
    Retourne 24 h (ou 48 h si date_end differ) de données horaires :
    température, humidité, vent, précipitations, rayonnement et WBGT.

    Parameters
    ----------
    lat      : float  latitude WGS84 (degrés décimaux)
    lon      : float  longitude WGS84 (degrés décimaux)
    date_str : str    date de début 'YYYY-MM-DD'
    timezone : str    fuseau IANA (défaut : 'Europe/Paris')
    client   : openmeteo_requests.Client | None
               Si None, un client temporaire est créé (sans cache).
               Passer le client global Twinity pour mutualiser.
    model    : str    modèle de réanalyse (défaut : 'era5_land')
               'era5_land' : résolution 0.1° (~9 km) — précis en altitude,
                             peut sous-estimer en zone urbaine dense.
               'era5'      : résolution 0.25° (~25 km) — parfois plus juste
                             en plaine et zone péri-urbaine.
    date_end : str | None  date de fin 'YYYY-MM-DD' (défaut : date_str).
               Passer la date d'arrivée pour les courses nocturnes
               franchissant minuit (ex. SaintéLyon : start=2025-11-29,
               end=2025-11-30). Retourne alors 48 heures de données.

    Returns
    -------
    pd.DataFrame  colonnes : time (datetime, UTC), temperature_2m,
                  relative_humidity_2m, apparent_temperature,
                  precipitation, wind_speed_10m, shortwave_radiation,
                  wbgt — ou None en cas d'erreur.
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

        # Reconstruction du DataFrame depuis le buffer FlatBuffers.
        # Le SDK retourne toujours des timestamps UTC (secondes epoch).
        times = pd.date_range(
            start=pd.to_datetime(hourly.Time(),    unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(),   unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )

        def _safe_var(i):
            """Extract variable i; return NaN array if unavailable."""
            try:
                arr = hourly.Variables(i).ValuesAsNumpy()
                return arr if arr is not None else np.full(len(times), np.nan)
            except Exception:
                return np.full(len(times), np.nan)

        df_w = pd.DataFrame({
            "time":                   times,
            "temperature_2m":         _safe_var(0),
            "relative_humidity_2m":   _safe_var(1),
            "apparent_temperature":   _safe_var(2),
            "precipitation":          _safe_var(3),
            "wind_speed_10m":         _safe_var(4),
            "shortwave_radiation":    _safe_var(5),
        })

        # Remplacer les sentinelles -9999 du SDK par NaN
        for col in df_w.columns:
            if col == "time":
                continue
            df_w[col] = pd.to_numeric(df_w[col], errors="coerce")
            df_w.loc[df_w[col] < -999, col] = np.nan

        # WBGT horaire (Bernard & Kenney 1994 + correction Liljegren 2008)
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


# ─────────────────────────────────────────────────────────────────────────────
# Interpolation sur le DataFrame GPS
# ─────────────────────────────────────────────────────────────────────────────

def enrich_df_with_weather(df, df_weather):
    """Interpolate hourly weather onto the per-second GPS DataFrame.

    Aligne sur les timestamps GPS via interpolation linéaire.
    Ajoute les colonnes : temp_api, humidity_api, wind_kmh_api,
    precip_api, solar_rad_api, apparent_temp_api, wbgt_api.

    Parameters
    ----------
    df         : GPS DataFrame avec colonne 'timestamp'.
    df_weather : DataFrame issu de fetch_weather_hourly().

    Returns
    -------
    df  enrichi, inchangé si df_weather est None.
    """
    if df_weather is None or df_weather.empty:
        warnings.warn("Données météo indisponibles — enrichissement ignoré.")
        return df

    df = df.copy()

    # Timestamps en secondes epoch pour np.interp
    t_w = df_weather["time"].astype("int64").to_numpy() // 10**9

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
            df[dst] = np.interp(
                t_gps,
                t_w,
                df_weather[src].to_numpy(dtype=float),
            )
        else:
            df[dst] = np.nan

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_weather_along_race(df, ravito_km, ravito_nom, fc_max=None,
                            show_watch_temp=False):
    """Plot weather variables along the race (distance axis).

    Trois panneaux :
      1. Température ERA5-Land + ressentie (montre optionnelle)
      2. Humidité (%) et vent (km/h) — twin axis
      3. WBGT horaire avec seuils de risque + profil altimétrique

    La température montre est désactivée par défaut (show_watch_temp=False)
    car biaisée +2 à +5°C par la chaleur corporelle. Activer uniquement
    pour comparer la dynamique temporelle, pas les valeurs absolues.

    Parameters
    ----------
    df              : DataFrame enriched with enrich_df_with_weather().
    ravito_km       : list of float.
    ravito_nom      : list of str.
    fc_max          : float | None  (conservé pour homogénéité API)
    show_watch_temp : bool  afficher la courbe montre (défaut False).
    """
    x_km = df["dist_m"] / 1000.0
    has_watch = (show_watch_temp
                 and "temperature" in df.columns
                 and df["temperature"].notna().sum() > 100)
    has_api   = ("temp_api" in df.columns
                 and df["temp_api"].notna().sum() > 10)

    if not has_watch and not has_api:
        print("Aucune donnée de température disponible.")
        return

    def _add_ravitos(ax):
        for km, nom in zip(ravito_km, ravito_nom):
            ax.axvline(km, color="grey", linestyle=":", alpha=0.5, linewidth=1)
            ylim = ax.get_ylim()
            ax.text(km, ylim[1], nom, fontsize=7, rotation=90,
                    va="top", ha="right", color="grey")

    n_panels = 3 if has_api else 1
    fig, axes = plt.subplots(n_panels, 1,
                             figsize=(13, 4 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    # ── Panel 1 : températures ───────────────────────────────────────────────
    ax0 = axes[0]

    if has_api:
        ax0.plot(x_km, df["temp_api"], color="steelblue", linewidth=1.8,
                 label="ERA5-Land (réanalyse, ~9 km)")
        if "apparent_temp_api" in df.columns:
            ax0.plot(x_km, df["apparent_temp_api"],
                     color="navy", linewidth=0.9, alpha=0.5,
                     linestyle=":", label="Température ressentie")

    if has_watch:
        win = max(10, int(3000 / df["dist_m"].diff().median()))
        t_smooth = (df["temperature"]
                    .rolling(win, center=True, min_periods=5)
                    .median())
        ax0.plot(x_km, t_smooth, color="tomato", linewidth=1.2,
                 alpha=0.6, linestyle="--",
                 label="Montre lissée ⚠️ biais +2–5°C (non fiable en absolu)")

    title = "Température au fil de la course — ERA5-Land (Muñoz Sabater 2019)"
    if has_watch:
        title += " + montre (indicatif)"
    ax0.set_ylabel("Température (°C)")
    ax0.set_title(title)
    ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.4)
    _add_ravitos(ax0)

    if not has_api:
        axes[-1].set_xlabel("Distance (km)")
        fig.tight_layout()
        plt.show()
        return

    # ── Panel 2 : humidité + vent ────────────────────────────────────────────
    ax1 = axes[1]
    if "humidity_api" in df.columns:
        ax1.plot(x_km, df["humidity_api"], color="teal", linewidth=1.4,
                 label="Humidité relative (%)")
        ax1.set_ylabel("Humidité (%)")

    ax1b = ax1.twinx()
    if "wind_kmh_api" in df.columns:
        ax1b.plot(x_km, df["wind_kmh_api"], color="goldenrod",
                  linewidth=1.2, alpha=0.7, label="Vent (km/h)")
        ax1b.set_ylabel("Vent (km/h)", color="goldenrod")
        ax1b.tick_params(axis="y", labelcolor="goldenrod")

    ax1.set_title("Humidité relative et vent (ERA5-Land)")
    ax1.legend(fontsize=8, loc="upper left")
    ax1b.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.4)
    _add_ravitos(ax1)

    # ── Panel 3 : WBGT + profil ──────────────────────────────────────────────
    ax2 = axes[2]
    if "wbgt_api" in df.columns and df["wbgt_api"].notna().sum() > 10:
        wbgt = df["wbgt_api"].to_numpy()

        # Seuils de risque (Périard 2021 / Casa 2015)
        seuils = [
            (32.0, "darkred",  "Danger extrême > 32°C"),
            (28.0, "crimson",  "Danger > 28°C"),
            (23.0, "orange",   "Vigilance > 23°C"),
        ]
        for s_val, s_col, s_lbl in seuils:
            ax2.axhline(s_val, color=s_col, linewidth=0.9,
                        linestyle="--", alpha=0.6, label=s_lbl)

        ax2.plot(x_km, wbgt, color="purple", linewidth=1.8,
                 label="WBGT (Bernard & Kenney 1994)")
        ax2.fill_between(x_km, wbgt, 32.0,
                         where=wbgt >= 32.0,
                         color="darkred", alpha=0.20)
        ax2.fill_between(x_km, wbgt, 28.0,
                         where=(wbgt >= 28.0) & (wbgt < 32.0),
                         color="crimson", alpha=0.15)
        ax2.fill_between(x_km, wbgt, 23.0,
                         where=(wbgt >= 23.0) & (wbgt < 28.0),
                         color="orange", alpha=0.10)

        ax2.set_ylabel("WBGT (°C)")
        ax2.set_title(
            "WBGT au fil du parcours — "
            "Bernard & Kenney (1994) + correction Liljegren (2008)\n"
            "Seuils : Périard et al. (2021) / Casa et al. (2015)"
        )
        ax2.legend(fontsize=8, loc="upper left")
        ax2.grid(True, alpha=0.4)

    # Profil altimétrique en arrière-plan
    ax2b = ax2.twinx()
    ax2b.fill_between(x_km, df["alt_m"], df["alt_m"].min(),
                      color="saddlebrown", alpha=0.08, zorder=0)
    ax2b.plot(x_km, df["alt_m"], color="saddlebrown",
              linewidth=0.8, alpha=0.20, zorder=0)
    ax2b.set_ylabel("Altitude (m)", color="saddlebrown", alpha=0.5, fontsize=9)
    ax2b.tick_params(axis="y", labelcolor="saddlebrown", labelsize=8)
    alt_range = df["alt_m"].max() - df["alt_m"].min()
    ax2b.set_ylim(df["alt_m"].min() - 50,
                  df["alt_m"].max() + alt_range * 2.0)
    ax2b.set_zorder(0)
    ax2.set_zorder(1)
    ax2.patch.set_visible(False)
    _add_ravitos(ax2)

    axes[-1].set_xlabel("Distance (km)")
    fig.tight_layout()
    plt.show()

    # ── Tableau résumé ───────────────────────────────────────────────────────
    print("\n── Météo ERA5-Land pendant la course ───────────────────────────")
    print(f"Température     : min {df['temp_api'].min():.1f}°C "
          f"/ moy {df['temp_api'].mean():.1f}°C "
          f"/ max {df['temp_api'].max():.1f}°C")
    if "humidity_api" in df.columns:
        print(f"Humidité        : moy {df['humidity_api'].mean():.0f}%")
    if "wind_kmh_api" in df.columns:
        print(f"Vent            : moy {df['wind_kmh_api'].mean():.1f} km/h "
              f"/ max {df['wind_kmh_api'].max():.1f} km/h")
    if "solar_rad_api" in df.columns:
        print(f"Rayonnement     : moy {df['solar_rad_api'].mean():.0f} W/m²")
    if "wbgt_api" in df.columns:
        wbgt_max = float(df["wbgt_api"].max())
        wbgt_moy = float(df["wbgt_api"].mean())
        print(f"WBGT            : moy {wbgt_moy:.1f}°C / max {wbgt_max:.1f}°C")
        if wbgt_max >= 32:
            print("  ⚠️  WBGT > 32°C — danger extrême, performance très dégradée")
        elif wbgt_max >= 28:
            print("  ⚠️  WBGT > 28°C — risque de coup de chaleur (seuil World Athletics)")
        elif wbgt_max >= 23:
            print("  ⚠️  WBGT > 23°C — vigilance digestive, hydratation renforcée")
        else:
            print("  ✅  WBGT < 23°C — conditions thermiques favorables")

# ===========================================================================
# Analyses de performance avancées — single race
# ===========================================================================

# ---------------------------------------------------------------------------
# A. Variabilité d'allure (Pace Variability)
# ---------------------------------------------------------------------------

def compute_pace_variability(df, ravito_km, ravito_nom):
    """Compute GAP coefficient of variation per section (pace regularity).

    A lower CV indicates a more even pacing strategy.
    CV of GAP correlates negatively with performance (r = -0.71,
    Haney & Mercer 2011 ; Cuk et al. 2024).

    Uses GAP (Grade Adjusted Pace) instead of raw pace to remove
    the topographic bias — sections with more elevation gain are
    otherwise artificially slower.

    Parameters
    ----------
    df         : DataFrame with gap_s_per_km, dist_m columns.
    ravito_km  : list of float.
    ravito_nom : list of str.

    Returns
    -------
    pd.DataFrame  one row per section with CV and related metrics.

    References
    ----------
    - Haney TA & Mercer JA (2011). A description of variability of pacing
      in marathon distance running. Int J Exerc Sci 4(2):133-140.
    - Cuk I et al. (2024). Running variability in marathon — evaluation of
      pacing variables. Medicina 60(2):218.
      https://doi.org/10.3390/medicina60020218
    """
    if "gap_s_per_km" not in df.columns:
        print("Colonne gap_s_per_km manquante.")
        return pd.DataFrame()

    bounds = np.concatenate((
        [df["dist_m"].min() / 1000.0],
        np.array(ravito_km, dtype=float),
        [df["dist_m"].max() / 1000.0]
    ))
    labels = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if i == 0:
            nom = f"Départ → {ravito_nom[0]}"
        elif i == len(bounds) - 2:
            nom = f"{ravito_nom[-1]} → Arrivée"
        else:
            nom = f"{ravito_nom[i-1]} → {ravito_nom[i]}"
        labels.append((nom, a, b))

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
            "Section":          nom,
            "GAP méd. (s/km)":  round(sec.median(), 0),
            "GAP moy. (s/km)":  round(mean_gap, 0),
            "Écart-type":       round(std_gap, 1),
            "CV GAP (%)":       round(cv, 1),
            "Régularité":       "✅ régulier" if cv < 15 else (
                                "⚠️ variable" if cv < 25 else "❌ très variable"),
        })

    return pd.DataFrame(rows)


def plot_pace_variability(df, ravito_km, ravito_nom):
    """Plot GAP distribution per section as boxplots + CV evolution.

    Two panels:
      1. Boxplot du GAP par section (distribution des allures)
      2. CV du GAP par section (évolution de la régularité)

    Parameters
    ----------
    df        : DataFrame with gap_s_per_km, dist_m columns.
    ravito_km : list of float.
    ravito_nom: list of str.
    """
    if "gap_s_per_km" not in df.columns:
        print("Colonne gap_s_per_km manquante.")
        return

    df_var = compute_pace_variability(df, ravito_km, ravito_nom)
    if df_var.empty:
        return

    bounds = np.concatenate((
        [df["dist_m"].min() / 1000.0],
        np.array(ravito_km, dtype=float),
        [df["dist_m"].max() / 1000.0]
    ))
    labels = []
    data_boxes = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        mask = (
            (df["dist_m"] / 1000.0 >= a) &
            (df["dist_m"] / 1000.0 < b) &
            df["gap_s_per_km"].notna() &
            (df["gap_s_per_km"] > 0) &
            (df["gap_s_per_km"] < 1200)  # exclure arrêts
        )
        vals = df.loc[mask, "gap_s_per_km"].dropna().values
        if len(vals) < 20:
            continue
        if i == 0:
            labels.append(f"→ {ravito_nom[0]}")
        elif i == len(bounds) - 2:
            labels.append(f"{ravito_nom[-1]} →")
        else:
            labels.append(ravito_nom[i - 1])
        data_boxes.append(vals)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8))

    # Panel 1 : boxplots GAP
    ax0 = axes[0]
    bp = ax0.boxplot(data_boxes, labels=labels, patch_artist=True,
                     medianprops=dict(color="black", linewidth=2))
    colors = plt.get_cmap("RdYlGn_r")(
        np.linspace(0.2, 0.8, len(data_boxes))
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax0.set_ylabel("GAP (s/km)")
    ax0.set_title(
        "Distribution du GAP par section — "
        "Haney & Mercer (2011) / Cuk et al. (2024)"
    )
    ax0.tick_params(axis="x", rotation=20)
    ax0.grid(True, axis="y", alpha=0.4)

    # Annotation : médiane en allure min'sec"
    for i, vals in enumerate(data_boxes):
        med = np.median(vals)
        m, s = int(med) // 60, int(med) % 60
        ax0.text(i + 1, ax0.get_ylim()[1] * 0.97,
                 f"{m}'{s:02d}\"", ha="center", fontsize=7, color="navy")

    # Panel 2 : CV par section
    ax1 = axes[1]
    cv_vals = df_var["CV GAP (%)"].values
    x_pos = np.arange(len(cv_vals))
    colors_cv = ["seagreen" if v < 15 else ("orange" if v < 25 else "crimson")
                 for v in cv_vals]
    bars = ax1.bar(x_pos, cv_vals, color=colors_cv, alpha=0.8, edgecolor="white")
    ax1.axhline(15, color="orange", linestyle="--", linewidth=1,
                label="Seuil variable (15%)")
    ax1.axhline(25, color="crimson", linestyle="--", linewidth=1,
                label="Seuil très variable (25%)")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("CV du GAP (%)")
    ax1.set_title("Régularité d'allure par section (CV du GAP)")
    ax1.legend(fontsize=8)
    ax1.grid(True, axis="y", alpha=0.4)

    for bar, val in zip(bars, cv_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 f"{val:.1f}%", ha="center", fontsize=8)

    fig.tight_layout()
    plt.show()

    print("\n── Variabilité d'allure par section ────────────────────────────")
    print(df_var[["Section", "GAP méd. (s/km)", "CV GAP (%)", "Régularité"]]
          .to_string(index=False))


# ---------------------------------------------------------------------------
# B. Positive / negative split sur le GAP
# ---------------------------------------------------------------------------

def compute_pace_split(df, ravito_km, ravito_nom):
    """Compute positive / negative split based on GAP.

    Compares first half vs second half of the race (by distance)
    and also section-by-section vs the reference section (first).

    Uses GAP to remove elevation bias from the split analysis.

    Parameters
    ----------
    df        : DataFrame with gap_s_per_km, dist_m columns.
    ravito_km : list of float.
    ravito_nom: list of str.

    Returns
    -------
    dict with keys:
        'split_ratio'   : float  GAP_half2 / GAP_half1 (>1 = positive split)
        'split_type'    : str    'négatif', 'équilibré', 'positif'
        'gap_half1'     : float  median GAP first half (s/km)
        'gap_half2'     : float  median GAP second half (s/km)
        'section_df'    : pd.DataFrame  GAP and delta per section
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
        split_type = "négatif ✅ (accélération)"
    elif ratio <= 1.03:
        split_type = "équilibré ✅"
    else:
        split_type = "positif ⚠️ (ralentissement)"

    # Section-par-section vs référence (première section)
    bounds = np.concatenate((
        [df["dist_m"].min() / 1000.0],
        np.array(ravito_km, dtype=float),
        [df["dist_m"].max() / 1000.0]
    ))
    labels = []
    for i in range(len(bounds) - 1):
        if i == 0:
            labels.append(f"Départ → {ravito_nom[0]}")
        elif i == len(bounds) - 2:
            labels.append(f"{ravito_nom[-1]} → Arrivée")
        else:
            labels.append(f"{ravito_nom[i-1]} → {ravito_nom[i]}")

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
            "Section":          nom,
            "GAP méd.":         f"{m}'{s:02d}\"",
            "GAP (s/km)":       round(sec_gap, 0),
            "Δ vs section 1 (%)": round(delta_pct, 1),
            "Tendance":         "=" if abs(delta_pct) < 3 else (
                                "↑ plus lent" if delta_pct > 0 else "↓ plus rapide"),
        })

    return {
        "split_ratio": round(ratio, 3),
        "split_type":  split_type,
        "gap_half1":   round(gap1, 0),
        "gap_half2":   round(gap2, 0),
        "section_df":  pd.DataFrame(rows),
    }


def plot_pace_split(df, ravito_km, ravito_nom):
    """Plot split analysis : GAP evolution + half split summary.

    Two panels:
      1. GAP lissé au fil de la course avec ligne médiane de référence
      2. Δ GAP section par section vs la première section

    Parameters
    ----------
    df        : DataFrame with gap_s_per_km, dist_m columns.
    ravito_km : list of float.
    ravito_nom: list of str.
    """
    if "gap_s_per_km" not in df.columns:
        print("Colonne gap_s_per_km manquante.")
        return

    result = compute_pace_split(df, ravito_km, ravito_nom)
    if result is None:
        return

    x_km = df["dist_m"] / 1000.0
    win = max(10, int(3000 / df["dist_m"].diff().median()))
    gap_smooth = (
        df["gap_s_per_km"]
        .where(df["gap_s_per_km"] < 1200)  # exclure arrêts
        .rolling(win, center=True, min_periods=10)
        .median()
    )
    ref_gap = float(df.loc[df["dist_m"] <= df["dist_m"].max() / 2,
                           "gap_s_per_km"].median())

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=False)

    # Panel 1 : GAP lissé
    ax0 = axes[0]
    ax0.plot(x_km, gap_smooth, color="steelblue", linewidth=1.6,
             label="GAP lissé (3 km)")
    ax0.axhline(ref_gap, color="grey", linestyle="--", linewidth=1.0,
                label=f"Référence 1ère moitié ({ref_gap/60:.0f}'{ref_gap%60:02.0f}\")")

    # Demi-course
    mid_km = df["dist_m"].max() / 2000.0
    ax0.axvline(mid_km, color="black", linestyle=":", linewidth=1.2,
                label="Mi-course")

    # Coloriage positif/négatif
    ax0.fill_between(x_km, gap_smooth, ref_gap,
                     where=gap_smooth > ref_gap,
                     color="crimson", alpha=0.12, label="Plus lent que référence")
    ax0.fill_between(x_km, gap_smooth, ref_gap,
                     where=gap_smooth < ref_gap,
                     color="seagreen", alpha=0.12, label="Plus rapide que référence")

    for km, nom in zip(ravito_km, ravito_nom):
        ax0.axvline(km, color="grey", linestyle=":", alpha=0.5, linewidth=1)
        ax0.text(km, ax0.get_ylim()[1] if ax0.get_ylim()[1] > 0 else ref_gap * 1.3,
                 nom, fontsize=7, rotation=90, va="top", ha="right", color="grey")

    ax0.set_ylabel("GAP (s/km)")
    ax0.set_xlabel("Distance (km)")
    ax0.invert_yaxis()  # allure inversée : bas = plus rapide
    ax0.set_title(
        f"GAP lissé au fil de la course — Split : {result['split_type']} "
        f"(ratio {result['split_ratio']:.3f})\n"
        "GAP 1ère moitié : "
        f"{result['gap_half1']//60:.0f}'{result['gap_half1']%60:02.0f}\""
        " | 2ème moitié : "
        f"{result['gap_half2']//60:.0f}'{result['gap_half2']%60:02.0f}\""
    )
    ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.4)

    # Panel 2 : Δ par section
    ax1 = axes[1]
    sec_df = result["section_df"]
    x_pos = np.arange(len(sec_df))
    deltas = sec_df["Δ vs section 1 (%)"].values
    colors = ["seagreen" if d <= 0 else ("orange" if d <= 10 else "crimson")
              for d in deltas]
    bars = ax1.bar(x_pos, deltas, color=colors, alpha=0.8, edgecolor="white")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.axhline(5, color="orange", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.axhline(-5, color="seagreen", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(sec_df["Section"], rotation=20, ha="right", fontsize=8)
    ax1.set_ylabel("Δ GAP vs section 1 (%)")
    ax1.set_title("Évolution du GAP par section — positif = ralentissement")
    ax1.grid(True, axis="y", alpha=0.4)

    for bar, val in zip(bars, deltas):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (0.3 if val >= 0 else -1.0),
                 f"{val:+.1f}%", ha="center", fontsize=8)

    fig.tight_layout()
    plt.show()

    print("\n── Split analyse ────────────────────────────────────────────────")
    print(f"Type de split   : {result['split_type']}")
    print(f"Ratio 2ème/1ère : {result['split_ratio']:.3f}")
    print(f"GAP 1ère moitié : {result['gap_half1']//60:.0f}'{result['gap_half1']%60:02.0f}\"  /km")
    print(f"GAP 2ème moitié : {result['gap_half2']//60:.0f}'{result['gap_half2']%60:02.0f}\"  /km")
    print()
    print(sec_df[["Section", "GAP méd.", "Δ vs section 1 (%)", "Tendance"]]
          .to_string(index=False))


# ---------------------------------------------------------------------------
# C. Analyse circadienne (effet heure de la journée)
# ---------------------------------------------------------------------------

def compute_circadian_profile(df, bin_hours=2):
    """Compute performance metrics per time-of-day bin.

    Segments the race by wall-clock hour bins and computes median
    GAP, HR fraction, walk rate and cadence per bin.

    Particularly relevant for nocturnal races (SaintéLyon, UTMB) where
    the circadian nadir (02h–06h) causes a performance drop that is
    distinct from topographic or homeostatic fatigue.

    Parameters
    ----------
    df        : DataFrame with timestamp, gap_s_per_km columns.
    bin_hours : int  width of time bins in hours (default 2).

    Returns
    -------
    pd.DataFrame  one row per time bin with performance metrics.

    References
    ----------
    - Czeisler CA et al. (1999). Stability of the human circadian pacemaker.
      Science 284(5423):2177-2181.
    - Youngstedt SD & O'Connor PJ (1999). The influence of air travel on
      athletic performance. Sports Med 28(3):197-207.
    - Bearden SE & van Woerden I (2025). Pacing and placing in 161-km
      ultramarathons. PLoS ONE. https://doi.org/10.1371/journal.pone.0322883
    """
    if "gap_s_per_km" not in df.columns:
        print("Colonne gap_s_per_km manquante.")
        return pd.DataFrame()

    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["hour_bin"] = (df["hour"] // bin_hours) * bin_hours

    # Tri chronologique depuis l'heure de départ de la course (pas numérique).
    # Ex : départ à 23h31 → ordre [22, 0, 2, 4, 6, 8, 10] pas [0, 2, 4, 6, 8, 10, 22].
    all_bins = sorted(df["hour_bin"].unique())
    start_h   = int(df["timestamp"].iloc[0].hour)
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


def plot_circadian_profile(df, bin_hours=2):
    """Plot performance metrics per time-of-day bin.

    Highlights the circadian nadir window (02h–06h) where performance
    is typically degraded in nocturnal ultramarathons, independent of
    topographic difficulty.

    Parameters
    ----------
    df        : DataFrame with timestamp, gap_s_per_km, heart_rate columns.
    bin_hours : int  time bin width in hours.
    """
    circ_df = compute_circadian_profile(df, bin_hours=bin_hours)
    if circ_df.empty:
        print("Pas assez de données pour l'analyse circadienne.")
        return

    labels = circ_df["Tranche horaire"].tolist()
    n = len(labels)
    gap_vals = circ_df["GAP méd. (s/km)"].values
    has_fc   = "FC méd. (bpm)" in circ_df.columns
    has_walk = "Marche (%)" in circ_df.columns

    n_panels = 1 + int(has_fc) + int(has_walk)
    fig, axes = plt.subplots(n_panels, 1, figsize=(max(10, n * 1.5), 4 * n_panels))
    if n_panels == 1:
        axes = [axes]

    # Identifier les tranches nocturnes (22h–06h)
    def is_nocturnal(label):
        h = int(label[:2])
        return h >= 22 or h < 6

    noc_mask = [is_nocturnal(l) for l in labels]

    def _add_nadir_shading(ax):
        for i, (lbl, is_noc) in enumerate(zip(labels, noc_mask)):
            if is_noc:
                ax.axvspan(i - 0.5, i + 0.5, color="midnightblue",
                           alpha=0.08, zorder=0)

    x = np.arange(n)

    # Panel 1 : GAP
    ax0 = axes[0]
    colors_gap = ["steelblue" if not noc else "mediumpurple" for noc in noc_mask]
    bars = ax0.bar(x, gap_vals, color=colors_gap, alpha=0.8, edgecolor="white")
    ax0.axhline(np.median(gap_vals), color="black", linestyle="--",
                linewidth=1.0, label=f"Médiane globale ({np.median(gap_vals)/60:.0f}'{np.median(gap_vals)%60:02.0f}\")")
    _add_nadir_shading(ax0)
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=20)
    ax0.set_ylabel("GAP médian (s/km)")
    start_ts  = df["timestamp"].iloc[0]
    start_lbl = start_ts.strftime("%Hh%M")
    ax0.set_title(
        f"Profil circadien de la performance — tranches de {bin_hours}h\n"
        f"Départ : {start_lbl} | Axe trié depuis le départ | "
        "Nadir typique : 02h–06h\n"
        "(Czeisler 1999 ; Bearden & van Woerden 2025)"
    )
    ax0.legend(fontsize=8)
    ax0.grid(True, axis="y", alpha=0.4)
    # Annotations allure + marquage de l'heure de départ
    start_bin_lbl = f"{((start_ts.hour // bin_hours) * bin_hours):02d}h"
    for i, (bar, val, lbl) in enumerate(zip(bars, gap_vals, labels)):
        m, s = int(val) // 60, int(val) % 60
        ax0.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 2,
                 f"{m}'{s:02d}\"", ha="center", fontsize=7)
        if lbl.startswith(start_bin_lbl):
            ax0.get_xticklabels()  # force render before style
    # Mettre en évidence le bin de départ après le rendu
    for i, (lbl, tick) in enumerate(zip(labels, ax0.get_xticklabels())):
        if lbl.startswith(start_bin_lbl):
            tick.set_fontweight("bold")
            tick.set_color("darkred")

    panel_idx = 1

    # Panel 2 : FC
    if has_fc:
        ax1 = axes[panel_idx]
        fc_vals = circ_df["FC méd. (bpm)"].values
        ax1.bar(x, fc_vals, color=["steelblue" if not noc else "mediumpurple"
                                    for noc in noc_mask], alpha=0.8, edgecolor="white")
        ax1.axhline(np.median(fc_vals), color="black", linestyle="--",
                    linewidth=1.0, label=f"Médiane ({np.median(fc_vals):.0f} bpm)")
        _add_nadir_shading(ax1)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=20)
        ax1.set_ylabel("FC médiane (bpm)")
        ax1.set_title("FC médiane par tranche horaire")
        ax1.legend(fontsize=8)
        ax1.grid(True, axis="y", alpha=0.4)
        panel_idx += 1

    # Panel 3 : % marche
    if has_walk:
        ax2 = axes[panel_idx]
        walk_vals = circ_df["Marche (%)"].values
        ax2.bar(x, walk_vals, color=["steelblue" if not noc else "mediumpurple"
                                      for noc in noc_mask], alpha=0.8, edgecolor="white")
        _add_nadir_shading(ax2)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=20)
        ax2.set_ylabel("% marche")
        ax2.set_title("Proportion de marche par tranche horaire")
        ax2.grid(True, axis="y", alpha=0.4)

    fig.tight_layout()
    plt.show()

    print(f"\n── Profil circadien — tranches de {bin_hours}h ─────────────────────")
    print(circ_df.to_string(index=False))


# ---------------------------------------------------------------------------
# D. Hitting the wall detector
# ---------------------------------------------------------------------------

def detect_hitting_wall(df, ref_start_km=5.0, ref_end_km=None,
                        threshold_pct=25.0, min_duration_km=5.0):
    """Detect sustained pace degradation episodes ('hitting the wall').

    Adapted from marathon literature for trail ultra context.
    Flags segments where GAP exceeds threshold_pct above the reference
    pace continuously for more than min_duration_km.

    Definition adapted from : Prigent et al. (PMC12575221, 2024) who
    defined HtW as pace > 125% of mean pace (5–20 km) over >5 km after km 25.
    Threshold and reference window are adjustable for ultra-trail context.

    ⚠️  Limitations :
    - Single race only : no between-subject comparison.
    - Trail context : a steep climb can trigger a false positive.
      Use GAP (not raw pace) to remove topographic bias.
    - Recommended threshold for ultra-trail : 25–35% vs 25% for marathon.

    Parameters
    ----------
    df             : DataFrame with gap_s_per_km, dist_m columns.
    ref_start_km   : float  start of reference window (km).
    ref_end_km     : float | None  end of reference window (km).
                     Default : first quarter of the race.
    threshold_pct  : float  GAP degradation threshold (%). Default 25.
    min_duration_km: float  minimum episode duration (km). Default 5.

    Returns
    -------
    dict with keys:
        'ref_gap'     : float   reference GAP (s/km)
        'threshold_gap': float  GAP above which = flagged (s/km)
        'episodes'    : pd.DataFrame  flagged episodes with start/end km
        'flagged'     : bool    True if at least one episode detected

    References
    ----------
    - Prigent G et al. (2024). Early marathon metrics from IMU predict
      significant pace reduction. Front Physiol. PMC12575221.
      https://doi.org/10.3389/fphys.2024.1612880
    """
    if "gap_s_per_km" not in df.columns:
        return None

    dist_max_km = df["dist_m"].max() / 1000.0
    if ref_end_km is None:
        ref_end_km = dist_max_km * 0.25

    df = df.copy().sort_values("dist_m")
    gap = df["gap_s_per_km"].copy()
    gap = gap.where((gap > 0) & (gap < 1200))  # exclure arrêts

    ref_mask = (
        (df["dist_m"] / 1000.0 >= ref_start_km) &
        (df["dist_m"] / 1000.0 <= ref_end_km) &
        gap.notna()
    )
    ref_gap = float(gap[ref_mask].median())
    threshold_gap = ref_gap * (1.0 + threshold_pct / 100.0)

    # Lissage pour éviter les faux positifs sur les montées brèves
    win = max(10, int(1000 / df["dist_m"].diff().median()))
    gap_smooth = gap.rolling(win, center=True, min_periods=5).median()

    df["_flagged"] = gap_smooth > threshold_gap

    # Identifier les épisodes continus
    df["_block"] = (df["_flagged"] != df["_flagged"].shift(1)).cumsum()
    episodes = []
    for block_id, grp in df[df["_flagged"]].groupby("_block"):
        start_km = float(grp["dist_m"].iloc[0] / 1000.0)
        end_km   = float(grp["dist_m"].iloc[-1] / 1000.0)
        duration = end_km - start_km
        if duration < min_duration_km:
            continue
        gap_ep = float(grp["gap_s_per_km"].median())
        deg_pct = (gap_ep / ref_gap - 1.0) * 100.0
        episodes.append({
            "Début (km)":      round(start_km, 1),
            "Fin (km)":        round(end_km, 1),
            "Durée (km)":      round(duration, 1),
            "GAP méd. ép.":    round(gap_ep, 0),
            "Dégradation (%)": round(deg_pct, 1),
        })

    ep_df = pd.DataFrame(episodes)
    return {
        "ref_gap":       round(ref_gap, 0),
        "threshold_gap": round(threshold_gap, 0),
        "episodes":      ep_df,
        "flagged":       len(ep_df) > 0,
    }


def plot_hitting_wall(df, ravito_km, ravito_nom, ref_start_km=5.0,
                      ref_end_km=None, threshold_pct=25.0,
                      min_duration_km=5.0):
    """Plot GAP along the race with hitting-the-wall episodes highlighted.

    Parameters
    ----------
    df              : DataFrame with gap_s_per_km, dist_m columns.
    ravito_km       : list of float.
    ravito_nom      : list of str.
    ref_start_km    : float  start of reference window.
    ref_end_km      : float | None  end of reference window.
    threshold_pct   : float  degradation threshold (%).
    min_duration_km : float  minimum episode duration (km).
    """
    if "gap_s_per_km" not in df.columns:
        print("Colonne gap_s_per_km manquante.")
        return

    result = detect_hitting_wall(
        df, ref_start_km=ref_start_km, ref_end_km=ref_end_km,
        threshold_pct=threshold_pct, min_duration_km=min_duration_km
    )
    if result is None:
        return

    x_km = df["dist_m"] / 1000.0
    win = max(10, int(2000 / df["dist_m"].diff().median()))
    gap_smooth = (
        df["gap_s_per_km"]
        .where((df["gap_s_per_km"] > 0) & (df["gap_s_per_km"] < 1200))
        .rolling(win, center=True, min_periods=5)
        .median()
    )

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(x_km, gap_smooth, color="steelblue", linewidth=1.6,
            label="GAP lissé (2 km)", zorder=3)
    ax.axhline(result["ref_gap"], color="seagreen", linestyle="--",
               linewidth=1.2,
               label=f"GAP référence ({result['ref_gap']/60:.0f}'{result['ref_gap']%60:02.0f}\")")
    ax.axhline(result["threshold_gap"], color="crimson", linestyle="--",
               linewidth=1.2,
               label=f"Seuil ×{1+threshold_pct/100:.2f} "
                     f"({result['threshold_gap']/60:.0f}'{result['threshold_gap']%60:02.0f}\")")

    # Épisodes flaggés
    for _, ep in result["episodes"].iterrows():
        ax.axvspan(ep["Début (km)"], ep["Fin (km)"],
                   color="crimson", alpha=0.15, zorder=1)
        ax.text((ep["Début (km)"] + ep["Fin (km)"]) / 2,
                result["threshold_gap"] * 1.02,
                f"⚠️ +{ep['Dégradation (%)']:.0f}%",
                ha="center", fontsize=8, color="crimson", zorder=4)

    # Fenêtre de référence
    ref_end = ref_end_km if ref_end_km else df["dist_m"].max() / 4000.0
    ax.axvspan(ref_start_km, ref_end, color="seagreen", alpha=0.07,
               label="Fenêtre de référence")

    for km, nom in zip(ravito_km, ravito_nom):
        ax.axvline(km, color="grey", linestyle=":", alpha=0.5, linewidth=1)
        ax.text(km, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else result["threshold_gap"] * 1.2,
                nom, fontsize=7, rotation=90, va="top", ha="right", color="grey")

    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("GAP (s/km)")
    ax.invert_yaxis()
    ax.set_title(
        f"Détection de dégradation d'allure ('hitting the wall') — "
        f"seuil +{threshold_pct:.0f}% sur >{min_duration_km:.0f} km\n"
        "Adapté de Prigent et al. (2024) pour ultra-trail"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    plt.show()

    print("\n── Hitting the wall ─────────────────────────────────────────────")
    print(f"GAP de référence  : {result['ref_gap']/60:.0f}'{result['ref_gap']%60:02.0f}\" /km")
    print(f"Seuil de détection: {result['threshold_gap']/60:.0f}'{result['threshold_gap']%60:02.0f}\" /km (+{threshold_pct:.0f}%)")
    if result["flagged"]:
        print(f"\n⚠️  {len(result['episodes'])} épisode(s) détecté(s) :")
        print(result["episodes"].to_string(index=False))
    else:
        print("\n✅  Aucun épisode de dégradation soutenue détecté.")
        print(f"   (critère : GAP > seuil pendant >{min_duration_km:.0f} km consécutifs)")


# ===========================================================================
# Multi-race utilities
# ===========================================================================
#
# Ces fonctions constituent la couche d'abstraction entre les notebooks
# single-race et multi-race. Elles ne dépendent d'aucune base de données
# externe — tout est calculé depuis les fichiers FIT à la demande.
#
# Architecture :
#   load_and_process_race()     → charge + calcule tout depuis un FIT
#   compute_race_kpis()         → extrait les métriques scalaires comparables
#   normalize_by_distance_pct() → rééchantillonne sur axe 0–100 %
#   build_races_table()         → DataFrame de KPI pour N courses
#   plot_races_comparison()     → graphiques comparatifs
#   plot_normalized_profiles()  → profils superposés sur 0–100 %
#   plot_decay_model()          → modèle de déclin individuel + courbe théorique
# ===========================================================================


def load_and_process_race(fit_path, fc_max, fc_min, poids_kg,
                           ravito_km, ravito_nom,
                           window_m=100.0, up_thr=3.0, down_thr=-3.0,
                           min_seg_m=200.0, walk_thr_kmh=6.0,
                           walk_thr_cad=140.0):
    """Load a FIT file and compute all standard derived metrics.

    Single entry point for multi-race notebooks. Runs the same pipeline
    as race_single.ipynb but returns a structured dict rather than
    printing intermediate results.

    Parameters
    ----------
    fit_path    : str   path to the .fit file
    fc_max      : int   max heart rate (bpm)
    fc_min      : int   resting heart rate (bpm)
    poids_kg    : float athlete mass (kg)
    ravito_km   : list of float  aid station positions (km)
    ravito_nom  : list of str    aid station names
    window_m    : float slope computation window (m)
    up_thr      : float uphill threshold (%)
    down_thr    : float downhill threshold (%)
    min_seg_m   : float minimum segment length (m)
    walk_thr_kmh: float walk/run speed threshold (km/h)
    walk_thr_cad: float walk/run cadence threshold (spm)

    Returns
    -------
    dict with keys:
        'df'       : pd.DataFrame  full per-second GPS DataFrame
        'kpis'     : dict          scalar metrics (see compute_race_kpis)
        'meta'     : dict          race metadata (name, date, path)
        'ravito_km': list
        'ravito_nom': list
        'fc_max'   : int
        'fc_min'   : int
        'poids_kg' : float
    """
    import os

    df_raw = load_fit(fit_path)
    df, alt_col = clean_df(df_raw)
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
        "df":        df,
        "kpis":      kpis,
        "meta":      meta,
        "ravito_km": ravito_km,
        "ravito_nom": ravito_nom,
        "fc_max":    fc_max,
        "fc_min":    fc_min,
        "poids_kg":  poids_kg,
    }


def compute_race_kpis(df, fc_max, fc_min, poids_kg, ravito_km, ravito_nom,
                      dplus=None, dminus=None):
    """Extract scalar KPIs from a processed race DataFrame.

    All metrics are comparable across races of different distances and
    profiles, provided the same athlete parameters are used.

    Parameters
    ----------
    df          : pd.DataFrame  processed race DataFrame
    fc_max      : int
    fc_min      : int
    poids_kg    : float
    ravito_km   : list of float
    ravito_nom  : list of str
    dplus       : float | None  precomputed D+ (m). Recomputed if None.
    dminus      : float | None  precomputed D- (m). Recomputed if None.

    Returns
    -------
    dict  scalar KPIs — one value per metric per race.
    """
    if dplus is None or dminus is None:
        dplus, dminus = compute_dplus_dminus(df)

    dist_km   = float(df["dist_m"].max() / 1000.0)
    duration_h = float(df["time_h"].max())

    # Allure et vitesse globales
    gap_vals = df["gap_s_per_km"].dropna()
    gap_vals = gap_vals[gap_vals > 0]
    gap_med  = float(gap_vals.median()) if not gap_vals.empty else np.nan

    speed_kmh = dist_km / duration_h if duration_h > 0 else np.nan

    # FC
    fc_mean = float(df["heart_rate"].mean()) if "heart_rate" in df.columns else np.nan
    fc_max_obs = float(df["heart_rate"].max()) if "heart_rate" in df.columns else np.nan
    fc_frac = fc_mean / fc_max if fc_max > 0 and not np.isnan(fc_mean) else np.nan

    # Marche
    pct_walk = float(df["is_walk"].mean() * 100) if "is_walk" in df.columns else np.nan

    # Variabilité d'allure (CV global)
    cv_gap = float(gap_vals.std() / gap_vals.mean() * 100) if len(gap_vals) > 20 else np.nan

    # Split ratio
    split_result = compute_pace_split(df, ravito_km, ravito_nom)
    split_ratio  = split_result["split_ratio"] if split_result else np.nan

    # Découplage aérobie max
    decoupling = np.nan
    if "heart_rate" in df.columns and "gap_s_per_km" in df.columns:
        dc_df = compute_aerobic_decoupling(df, ravito_km, ravito_nom)
        if not dc_df.empty and "Découplage (%)" in dc_df.columns:
            decoupling = float(dc_df["Découplage (%)"].max())

    # Heure de départ
    start_time = df["timestamp"].iloc[0] if not df.empty else None
    start_hour = start_time.hour + start_time.minute / 60.0 if start_time else np.nan

    return {
        # Métriques de course
        "distance_km":    round(dist_km, 2),
        "duration_h":     round(duration_h, 3),
        "dplus_m":        round(dplus, 0),
        "dminus_m":       round(dminus, 0),
        "speed_kmh":      round(speed_kmh, 2),
        "gap_med_s_km":   round(gap_med, 0),
        # FC
        "fc_mean":        round(fc_mean, 0) if not np.isnan(fc_mean) else np.nan,
        "fc_max_obs":     round(fc_max_obs, 0) if not np.isnan(fc_max_obs) else np.nan,
        "fc_frac":        round(fc_frac, 3) if not np.isnan(fc_frac) else np.nan,
        # Gestion d'allure
        "cv_gap_pct":     round(cv_gap, 1) if not np.isnan(cv_gap) else np.nan,
        "split_ratio":    round(split_ratio, 3) if not np.isnan(split_ratio) else np.nan,
        "pct_walk":       round(pct_walk, 1) if not np.isnan(pct_walk) else np.nan,
        # Fatigue
        "decoupling_max": round(decoupling, 1) if not np.isnan(decoupling) else np.nan,
        # Contexte
        "start_hour":     round(start_hour, 2) if not np.isnan(start_hour) else np.nan,
    }


def normalize_by_distance_pct(df, n_bins=100, cols=None):
    """Resample a race DataFrame onto a uniform 0–100 % distance grid.

    Enables direct visual comparison between races of different lengths.
    Uses linear interpolation between GPS points.

    Parameters
    ----------
    df     : pd.DataFrame  processed race DataFrame with dist_m column.
    n_bins : int           number of bins (default 100 → 1 % steps).
    cols   : list | None   columns to resample. Default: gap_s_per_km,
             heart_rate, speed_kmh, alt_m, slope_pct, is_walk, cadence.

    Returns
    -------
    pd.DataFrame  n_bins rows, columns: dist_pct + requested cols.
    """
    if cols is None:
        cols = [c for c in ["gap_s_per_km", "heart_rate", "speed_kmh",
                             "alt_m", "slope_pct", "is_walk", "cadence",
                             "wbgt_api"]
                if c in df.columns]

    dist_pct = np.linspace(0.0, 100.0, n_bins)
    dist_m_grid = dist_pct / 100.0 * df["dist_m"].max()

    result = pd.DataFrame({"dist_pct": dist_pct})
    for col in cols:
        vals = df[col].to_numpy(dtype=float)
        dist = df["dist_m"].to_numpy(dtype=float)
        # Filtrer NaN pour l'interpolation
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
    races_list : list of dict  output of load_and_process_race()

    Returns
    -------
    pd.DataFrame  one row per race, columns = KPI names + meta.
    """
    rows = []
    for race in races_list:
        row = {
            "name": race["meta"]["name"],
            "date": race["meta"]["date"],
        }
        row.update(race["kpis"])
        rows.append(row)

    df_table = pd.DataFrame(rows)
    if "date" in df_table.columns:
        df_table = df_table.sort_values("date").reset_index(drop=True)
    return df_table


def plot_races_comparison(df_table, metrics=None, date_col="date",
                          name_col="name"):
    """Plot temporal evolution of scalar KPIs across races.

    One sub-panel per metric, x-axis = race date (or index if no date).

    Parameters
    ----------
    df_table : pd.DataFrame  output of build_races_table()
    metrics  : list | None   KPI columns to plot. Default: main metrics.
    date_col : str
    name_col : str
    """
    if metrics is None:
        metrics = [m for m in [
            "gap_med_s_km", "cv_gap_pct", "split_ratio",
            "fc_frac", "pct_walk", "decoupling_max",
        ] if m in df_table.columns]

    labels_map = {
        "gap_med_s_km":   "GAP médian (s/km)",
        "cv_gap_pct":     "CV GAP (%)",
        "split_ratio":    "Ratio split (GAP₂/GAP₁)",
        "fc_frac":        "FC moy / FC max",
        "pct_walk":       "% marche",
        "decoupling_max": "Découplage max (%)",
        "speed_kmh":      "Vitesse moy (km/h)",
        "dplus_m":        "D+ (m)",
    }

    n = len(metrics)
    if n == 0:
        print("Aucune métrique disponible.")
        return

    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    x = (df_table[date_col].astype(str).tolist()
         if date_col in df_table.columns
         else list(range(len(df_table))))
    names = df_table[name_col].tolist() if name_col in df_table.columns else x

    for idx, metric in enumerate(metrics):
        ax = axes_flat[idx]
        vals = df_table[metric].values.astype(float)
        valid = ~np.isnan(vals)

        ax.plot(np.array(x)[valid], vals[valid],
                marker="o", linewidth=1.8, color="steelblue",
                markersize=7, zorder=3)

        # Annotations des noms de course
        for xi, vi, ni in zip(np.array(x)[valid], vals[valid],
                               np.array(names)[valid]):
            ax.annotate(ni, (xi, vi), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7,
                        color="navy", rotation=30)

        # Ligne de tendance si ≥ 3 points
        x_idx = np.where(valid)[0]
        if len(x_idx) >= 3:
            p = np.polyfit(x_idx, vals[valid], 1)
            ax.plot(np.array(x)[valid],
                    np.polyval(p, x_idx),
                    linestyle="--", color="grey", linewidth=1.0,
                    alpha=0.6, label="Tendance linéaire")

        ax.set_title(labels_map.get(metric, metric), fontsize=10)
        ax.set_ylabel(labels_map.get(metric, metric), fontsize=8)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.grid(True, alpha=0.35)

    for idx in range(len(metrics), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Évolution des métriques de performance — multi-courses",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    plt.show()


def plot_normalized_profiles(races_list, col="gap_s_per_km",
                              n_bins=100, smooth_bins=5,
                              show_mean=True, show_ci=True):
    """Superpose normalized profiles (0–100 % distance) for several races.

    Allows direct visual comparison of effort distribution patterns
    across races of different lengths and profiles.

    Parameters
    ----------
    races_list : list of dict  output of load_and_process_race()
    col        : str           column to plot (default: gap_s_per_km)
    n_bins     : int           number of distance bins (default 100)
    smooth_bins: int           rolling average window on normalized axis
    show_mean  : bool          overlay mean profile across all races
    show_ci    : bool          overlay ±1 std confidence band

    References
    ----------
    - Kerhervé HA, Millet GY, Solomon C (2015). PLoS ONE 10(12):e0145482.
      → normalization by distance percentage for multi-race comparison
    """
    labels_map = {
        "gap_s_per_km": "GAP (s/km)",
        "heart_rate":   "FC (bpm)",
        "speed_kmh":    "Vitesse (km/h)",
        "is_walk":      "% marche (0/1)",
        "cadence":      "Cadence (spm)",
        "alt_m":        "Altitude (m)",
    }

    fig, ax = plt.subplots(figsize=(13, 5))
    cmap = plt.get_cmap("tab10")
    all_profiles = []

    for i, race in enumerate(races_list):
        df = race["df"]
        if col not in df.columns:
            continue

        norm_df = normalize_by_distance_pct(df, n_bins=n_bins, cols=[col])
        profile  = norm_df[col].values.astype(float)

        # Lissage léger sur l'axe normalisé
        if smooth_bins > 1:
            profile = (pd.Series(profile)
                       .rolling(smooth_bins, center=True, min_periods=1)
                       .mean()
                       .values)

        # Masquer les valeurs aberrantes (arrêts)
        if col == "gap_s_per_km":
            profile = np.where(profile > 1200, np.nan, profile)

        label = f"{race['meta']['name']} ({race['meta']['date']})"
        ax.plot(norm_df["dist_pct"], profile,
                color=cmap(i % 10), linewidth=1.5,
                alpha=0.75, label=label)
        all_profiles.append(profile)

    # Moyenne et bande de confiance
    if len(all_profiles) >= 2:
        stack = np.vstack(all_profiles)
        mean_profile = np.nanmean(stack, axis=0)
        std_profile  = np.nanstd(stack, axis=0)
        dist_pct = np.linspace(0, 100, n_bins)

        if show_mean:
            ax.plot(dist_pct, mean_profile,
                    color="black", linewidth=2.2,
                    linestyle="--", label="Moyenne")
        if show_ci:
            ax.fill_between(dist_pct,
                            mean_profile - std_profile,
                            mean_profile + std_profile,
                            color="black", alpha=0.07,
                            label="± 1 écart-type")

    ylabel = labels_map.get(col, col)
    ax.set_xlabel("Distance (% de la course)")
    ax.set_ylabel(ylabel)

    # Inverser l'axe Y pour le GAP (bas = rapide)
    if col == "gap_s_per_km":
        ax.invert_yaxis()

    ax.set_title(
        f"Profils normalisés — {ylabel} sur 0–100 % de la distance\n"
        "Normalisé par distance % pour comparaison inter-courses "
        "(Kerhervé et al. 2015)"
    )
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    plt.show()


def plot_decay_model(races_list, col="gap_s_per_km", n_bins=100,
                     degree=2):
    """Fit and plot an individual pace decay model across races.

    Fits a polynomial decay curve on the mean normalized profile,
    then overlays individual races and the fitted model.
    Useful to identify which races degraded more or less than expected.

    Parameters
    ----------
    races_list : list of dict
    col        : str    column to model (default: gap_s_per_km)
    n_bins     : int    distance bins
    degree     : int    polynomial degree for decay fit (default 2)

    References
    ----------
    - Matta GG et al. (2020). Influence of a slow-start on overall
      performance and running kinematics during 6-h ultramarathon races.
      Eur J Sport Sci 20(3):347-356.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.get_cmap("tab10")
    all_profiles = []
    dist_pct = np.linspace(0, 100, n_bins)

    for i, race in enumerate(races_list):
        df = race["df"]
        if col not in df.columns:
            continue
        norm_df = normalize_by_distance_pct(df, n_bins=n_bins, cols=[col])
        profile = norm_df[col].values.astype(float)
        if col == "gap_s_per_km":
            profile = np.where(profile > 1200, np.nan, profile)
        all_profiles.append(profile)

        label = race["meta"]["name"]
        axes[0].plot(dist_pct, profile, color=cmap(i % 10),
                     linewidth=1.4, alpha=0.7, label=label)

    if len(all_profiles) < 2:
        print("Pas assez de courses pour fitter un modèle.")
        return

    stack = np.vstack(all_profiles)
    mean_profile = np.nanmean(stack, axis=0)

    # Fit polynomial sur la moyenne (sur les points non-NaN)
    valid = ~np.isnan(mean_profile)
    if valid.sum() > degree + 1:
        coeffs = np.polyfit(dist_pct[valid], mean_profile[valid], degree)
        fitted  = np.polyval(coeffs, dist_pct)
    else:
        fitted = mean_profile.copy()

    axes[0].plot(dist_pct, fitted, color="black", linewidth=2.5,
                 linestyle="--", label=f"Modèle déclin (deg. {degree})")
    axes[0].set_xlabel("Distance (% de la course)")
    axes[0].set_ylabel("GAP (s/km)")
    if col == "gap_s_per_km":
        axes[0].invert_yaxis()
    axes[0].set_title("Profils normalisés + modèle de déclin individuel")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.35)

    # Panel 2 : résidu de chaque course vs le modèle
    for i, (race, profile) in enumerate(zip(races_list, all_profiles)):
        residual = profile - fitted
        label = race["meta"]["name"]
        axes[1].plot(dist_pct, residual, color=cmap(i % 10),
                     linewidth=1.4, alpha=0.75, label=label)

    axes[1].axhline(0, color="black", linewidth=1.0, linestyle="--")
    axes[1].fill_between(dist_pct, -10, 10,
                         color="seagreen", alpha=0.06,
                         label="±10 s/km (tolérance)")
    axes[1].set_xlabel("Distance (% de la course)")
    axes[1].set_ylabel("Résidu vs modèle (s/km)")
    axes[1].set_title(
        "Écart au modèle de déclin individuel\n"
        "Positif = plus lent que le modèle attendu"
    )
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.35)

    fig.suptitle(
        "Modèle de déclin d'allure individuel — "
        "Matta et al. (2020) / Kerhervé et al. (2015)",
        fontsize=11, y=1.01
    )
    fig.tight_layout()
    plt.show()


def plot_pace_vs_slope_overlay(races, bins_slope=None, show_minetti=True):
    """Plot pace and heart rate vs slope for multiple races on shared axes.

    One figure with two panels (pace left, heart rate right).
    Each race gets a distinct color. Minetti (2002) reference is plotted
    as a dashed line of the same color when show_minetti=True.

    Parameters
    ----------
    races       : list of dict  each dict has keys 'df' and 'meta' (with 'name').
    bins_slope  : list | None   slope bin edges (%). Default: standard bins.
    show_minetti: bool          overlay Minetti (2002) predicted pace per race.

    References
    ----------
    - Minetti AE et al. (2002). J. Appl. Physiol. 93(3):1039-1046.
    """
    if bins_slope is None:
        bins_slope = [-30, -15, -10, -7, -5, -3, -1, 1, 3, 5, 7, 10, 15, 30]

    colors = plt.cm.tab10.colors

    has_fc = any("heart_rate" in r["df"].columns for r in races)
    ncols = 2 if has_fc else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    for i, race in enumerate(races):
        color = colors[i % len(colors)]
        label = race["meta"].get("name", f"Course {i+1}")
        df = race["df"].copy()
        df["slope_bin"] = pd.cut(df["slope_pct"], bins=bins_slope)

        cols_needed = ["slope_pct", "pace_s_per_km"]
        if "heart_rate" in df.columns:
            cols_needed.append("heart_rate")

        tmp = df.dropna(subset=cols_needed)
        if tmp.empty:
            print(f"{label} : données insuffisantes.")
            continue

        agg = {
            "n":         ("slope_pct",     "size"),
            "slope_med": ("slope_pct",     "median"),
            "pace_med":  ("pace_s_per_km", "median"),
        }
        if "heart_rate" in df.columns:
            agg["hr_med"] = ("heart_rate", "median")

        summary = tmp.groupby("slope_bin", observed=True).agg(**agg).reset_index()
        summary = summary[summary["n"] >= 20]

        if summary.empty:
            print(f"{label} : pas assez de points par classe de pente.")
            continue

        axes[0].plot(summary["slope_med"], summary["pace_med"],
                     marker="o", color=color, linewidth=1.6, label=label)

        if show_minetti:
            slopes_ref = np.linspace(
                summary["slope_med"].min(), summary["slope_med"].max(), 200
            )
            flat_mask = summary["slope_med"].abs() < 3
            if flat_mask.sum() == 0:
                flat_mask = summary["slope_med"].abs() < 5
            pace_flat = float(summary.loc[flat_mask, "pace_med"].median())
            minetti_pace = pace_flat * minetti_cost_ratio(slopes_ref)
            axes[0].plot(slopes_ref, minetti_pace, "--", color=color,
                         linewidth=1.2, alpha=0.7, label=f"{label} — Minetti")

        if has_fc and "hr_med" in summary.columns:
            axes[1].plot(summary["slope_med"], summary["hr_med"],
                         marker="o", color=color, linewidth=1.6, label=label)

    axes[0].set_xlabel("Pente médiane (%)")
    axes[0].set_ylabel("Allure (s/km)")
    axes[0].set_title("Allure vs pente")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.35)

    if has_fc:
        axes[1].set_xlabel("Pente médiane (%)")
        axes[1].set_ylabel("FC médiane (bpm)")
        axes[1].set_title("FC vs pente")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.35)

    fig.tight_layout()
    plt.show()



def plot_pace_vs_slope_deviation(races, bins_slope=None, title="Écart relatif à Minetti par classe de pente"):
    """Plot relative deviation from Minetti (2002) predicted pace per slope bin.

    For each race, computes (pace_real - pace_minetti) / pace_minetti per slope
    bin. Positive values mean slower than Minetti; negative values mean faster.

    Parameters
    ----------
    races      : list of dict  each dict has keys 'df' and 'meta' (with 'name').
    bins_slope : list | None   slope bin edges (%). Default: standard bins.

    References
    ----------
    - Minetti AE et al. (2002). J. Appl. Physiol. 93(3):1039-1046.
    """
    if bins_slope is None:
        bins_slope = [-30, -15, -10, -7, -5, -3, -1, 1, 3, 5, 7, 10, 15, 30]

    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(8, 4))

    for i, race in enumerate(races):
        color = colors[i % len(colors)]
        label = race["meta"].get("name", f"Course {i+1}")
        df = race["df"].copy()
        df["slope_bin"] = pd.cut(df["slope_pct"], bins=bins_slope)

        tmp = df.dropna(subset=["slope_pct", "pace_s_per_km"])
        if tmp.empty:
            print(f"{label} : données insuffisantes.")
            continue

        summary = (
            tmp.groupby("slope_bin", observed=True)
            .agg(
                n=("slope_pct", "size"),
                slope_med=("slope_pct", "median"),
                pace_med=("pace_s_per_km", "median"),
            )
            .reset_index()
        )
        summary = summary[summary["n"] >= 20]

        if summary.empty:
            print(f"{label} : pas assez de points par classe de pente.")
            continue

        flat_mask = summary["slope_med"].abs() < 3
        if flat_mask.sum() == 0:
            flat_mask = summary["slope_med"].abs() < 5
        pace_flat = float(summary.loc[flat_mask, "pace_med"].median())
        minetti_pace = pace_flat * minetti_cost_ratio(summary["slope_med"].values)

        deviation = (summary["pace_med"].values - minetti_pace) / minetti_pace

        ax.plot(summary["slope_med"], deviation,
                marker="o", color=color, linewidth=1.6, label=label)

    ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.5)
    ax.set_xlabel("Pente médiane (%)")
    ax.set_ylabel("Écart relatif à Minetti")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _:  f"{y:+.0%}"))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    plt.show()


















def plot_pace_vs_slope(df, label="", bins_slope=None):
    """Plot median pace and heart rate vs slope class, with Minetti reference.

    Two-panel figure (single race):
      Left  : median pace per slope bin (real) + Minetti (2002) predicted pace
               on a twin Y-axis (same shape, different scale).
      Right : median heart rate per slope bin (if available).

    Designed to be called in a loop for multi-race comparison
    (one figure per race, stacked vertically in race_compare.ipynb).

    Parameters
    ----------
    df         : pd.DataFrame  processed race DataFrame.
    label      : str           race name shown in the title.
    bins_slope : list | None   slope bin edges (%). Default: standard bins.

    References
    ----------
    - Minetti AE et al. (2002). J. Appl. Physiol. 93(3):1039-1046.
    """
    if bins_slope is None:
        bins_slope = [-30, -15, -10, -7, -5, -3, -1, 1, 3, 5, 7, 10, 15, 30]

    df = df.copy()
    df["slope_bin"] = pd.cut(df["slope_pct"], bins=bins_slope)

    cols_needed = ["slope_pct", "pace_s_per_km"]
    if "heart_rate" in df.columns:
        cols_needed.append("heart_rate")

    tmp = df.dropna(subset=cols_needed)
    if tmp.empty:
        print(f"{label} : données insuffisantes pour le plot allure vs pente.")
        return

    agg = {
        "n":         ("slope_pct",    "size"),
        "slope_med": ("slope_pct",    "median"),
        "pace_med":  ("pace_s_per_km","median"),
    }
    if "heart_rate" in df.columns:
        agg["hr_med"] = ("heart_rate", "median")

    summary = tmp.groupby("slope_bin", observed=True).agg(**agg).reset_index()
    summary = summary[summary["n"] >= 20]  # exclure les bins trop peu fournis

    if summary.empty:
        print(f"{label} : pas assez de points par classe de pente.")
        return

    # Courbe Minetti de référence
    slopes_ref = np.linspace(
        summary["slope_med"].min(), summary["slope_med"].max(), 200
    )
    flat_mask = summary["slope_med"].abs() < 3
    if flat_mask.sum() == 0:
        flat_mask = summary["slope_med"].abs() < 5
    pace_flat = float(summary.loc[flat_mask, "pace_med"].median())
    minetti_pace = pace_flat * minetti_cost_ratio(slopes_ref)

    has_fc = "hr_med" in summary.columns
    ncols  = 2 if has_fc else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    # Panel gauche : allure réelle + Minetti (twin axis)
    ax0  = axes[0]
    ax0t = ax0.twinx()

    ax0.plot(summary["slope_med"], summary["pace_med"],
             marker="o", color="steelblue", linewidth=1.6,
             label="Allure réelle")
    ax0t.plot(slopes_ref, minetti_pace, "--", color="darkorange",
              linewidth=1.5, label="Minetti (2002)")

    ax0.set_xlabel("Pente médiane (%)")
    ax0.set_ylabel("Allure réelle (s/km)", color="steelblue")
    ax0t.set_ylabel("Allure Minetti (s/km)", color="darkorange")
    ax0.tick_params(axis="y", labelcolor="steelblue")
    ax0t.tick_params(axis="y", labelcolor="darkorange")
    ax0.set_title(f"Allure vs pente + Minetti (2002){' — ' + label if label else ''}")

    lines = ax0.get_lines() + ax0t.get_lines()
    ax0.legend(lines, [l.get_label() for l in lines],
               loc="upper left", fontsize=8)
    ax0.grid(True, alpha=0.35)

    # Panel droit : FC vs pente
    if has_fc:
        axes[1].plot(summary["slope_med"], summary["hr_med"],
                     marker="o", color="crimson", linewidth=1.6)
        axes[1].set_xlabel("Pente médiane (%)")
        axes[1].set_ylabel("FC médiane (bpm)")
        axes[1].set_title(f"FC vs pente{' — ' + label if label else ''}")
        axes[1].grid(True, alpha=0.35)

    fig.tight_layout()
    plt.show()
