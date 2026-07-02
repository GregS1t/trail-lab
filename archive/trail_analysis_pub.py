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
# Visualisation méteo le long de la course
# ─────────────────────────────────────────────────────────────────────────────

def plot_weather_along_race(df, ravito_km, ravito_nom, fc_max=None,
                            show_watch_temp=False):
    """Plot weather variables along the race (distance axis).

    Deux panneaux :
      1. Température ERA5-Land + ressentie (montre optionnelle)
      2. Humidité (%) et vent (km/h) 

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
            ax.axvline(km, color="grey", linestyle=":", alpha=0.9, linewidth=1)
            ylim = ax.get_ylim()
            ax.text(km, ylim[1], nom, fontsize=7, rotation=90,
                    va="top", ha="right", color="grey")

    n_panels = 2 if has_api else 1
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
                 label="Montre lissée - biais +2–5°C (non fiable en absolu)")

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
    ax1.grid(True, alpha=0.7)
    _add_ravitos(ax1)

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
        sr_moy = df["solar_rad_api"].dropna()
        if not sr_moy.empty:
            print(f"Rayonnement     : moy {sr_moy.mean():.0f} W/m²")

