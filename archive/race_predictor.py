# core/race_predictor.py
# -----------------------
# Estimation de la vitesse sur chaque segment d'une trace de course,
# calibrée sur l'historique de l'athlète.
#
# Modèle de base : Minetti AE et al. (2002)
# Coût métabolique C(i) en fonction de la pente i (sans unité) :
#
#   C(i) = 155.4·i⁵ − 30.4·i⁴ − 43.3·i³ + 46.3·i² + 19.5·i + 3.6
#
# Unité : J/(kg·m)  — énergie par unité de masse et de distance horizontale.
#
# La vitesse estimée est :
#   V_est(i) = V_ref / (C(i) / C(0))
#
# où V_ref est la vitesse de référence de l'athlète sur terrain plat,
# extraite de son historique de sessions ou renseignée manuellement.
#
# Références
# ----------
# - Minetti AE, Moia C, Roi GS, Susta D, Ferretti G (2002).
#   Energy cost of walking and running at extreme uphill and downhill slopes.
#   J Appl Physiol 93(3):1039-1046.
#   https://doi.org/10.1152/japplphysiol.01177.2001
# - Scarf P (2007). Route choice in mountain navigation.
#   J Sports Sci 25(4):371-376.

import math
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Modèle Minetti (2002)
# ─────────────────────────────────────────────────────────────────────────────

def minetti_cost(slope_pct):
    """
    Coût métabolique de la course en fonction de la pente (Minetti 2002).

    Parameters
    ----------
    slope_pct : float  pente en % (positif = montée, négatif = descente)
                       clippée à [-45, 45] %

    Returns
    -------
    float  coût en J/(kg·m)  — toujours > 0
    """
    i = np.clip(float(slope_pct) / 100.0, -0.45, 0.45)
    c = (155.4 * i**5
         - 30.4  * i**4
         - 43.3  * i**3
         + 46.3  * i**2
         + 19.5  * i
         + 3.6)
    # Le coût ne peut pas être négatif (limite physique)
    return max(float(c), 0.5)


def minetti_cost_flat():
    """Coût de référence sur terrain plat (pente = 0 %)."""
    return minetti_cost(0.0)   # ≈ 3.6 J/(kg·m)


def minetti_speed_ratio(slope_pct):
    """
    Rapport V(pente) / V(plat) selon le modèle Minetti.

    Un rapport < 1 signifie qu'on ralentit, > 1 qu'on accélère
    (descente douce optimale ≈ −10 %).

    Parameters
    ----------
    slope_pct : float  pente en %

    Returns
    -------
    float  ratio vitesse (sans unité, > 0)
    """
    return minetti_cost_flat() / minetti_cost(slope_pct)


# ─────────────────────────────────────────────────────────────────────────────
# Calibration sur l'historique athlète
# ─────────────────────────────────────────────────────────────────────────────

def estimate_flat_speed_from_history(df_kpis, sport_filter=None,
                                     percentile=50):
    """
    Estime la vitesse de référence sur terrain plat à partir des KPI
    historiques de l'athlète.

    Utilise les sessions dont l'efficiency_index est disponible et dont
    le dénivelé relatif est faible (< 15 m/km → terrain quasiment plat).

    Parameters
    ----------
    df_kpis       : pd.DataFrame  issu de load_kpis_from_db()
    sport_filter  : list[str] | None  filtrer par sport ('trail', 'running'…)
    percentile    : int           percentile de vitesse à utiliser (50 = médiane)

    Returns
    -------
    float | None  vitesse en km/h, ou None si pas assez de données
    """
    df = df_kpis.copy()

    if sport_filter:
        df = df[df['sport'].isin(sport_filter)]

    needed = {'speed_mean_kmh', 'distance_km', 'dplus_m'}
    if not needed.issubset(df.columns):
        return None

    df = df.dropna(subset=list(needed))
    df = df[df['distance_km'] > 0]

    # Terrain plat : dénivelé < 15 m/km
    df['dplus_per_km'] = df['dplus_m'] / df['distance_km']
    flat = df[df['dplus_per_km'] < 15]

    if len(flat) < 3:
        # Fallback : toutes les sessions disponibles
        flat = df

    if flat.empty:
        return None

    return float(np.percentile(flat['speed_mean_kmh'].dropna(), percentile))


def calibrate_athlete_factor(df_kpis, sport_filter=None):
    """
    Calcule un facteur de correction global entre la vitesse réelle
    de l'athlète et la prédiction Minetti pure.

    Le facteur est calculé session par session (sessions avec dplus > 0)
    puis moyenné. Un facteur < 1 indique que l'athlète est plus lent
    que la prédiction (fatigue, terrain technique, etc.).

    Parameters
    ----------
    df_kpis      : pd.DataFrame
    sport_filter : list[str] | None

    Returns
    -------
    float  facteur de correction (1.0 si pas assez de données)
    """
    df = df_kpis.copy()
    if sport_filter:
        df = df[df['sport'].isin(sport_filter)]

    needed = {'speed_mean_kmh', 'distance_km', 'dplus_m', 'dmoins_m'}
    if not needed.issubset(df.columns):
        return 1.0

    df = df.dropna(subset=list(needed))
    df = df[(df['distance_km'] > 2) & (df['dplus_m'] > 50)]

    if len(df) < 3:
        return 1.0

    factors = []
    for _, row in df.iterrows():
        dist_m       = row['distance_km'] * 1000
        slope_equiv  = (row['dplus_m'] - row['dmoins_m']) / dist_m * 100
        v_minetti    = (row.get('speed_mean_flat', row['speed_mean_kmh'])
                        * minetti_speed_ratio(slope_equiv))
        v_real       = row['speed_mean_kmh']
        if v_minetti > 0:
            factors.append(v_real / v_minetti)

    return float(np.median(factors)) if factors else 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Prédiction par segment
# ─────────────────────────────────────────────────────────────────────────────

def predict_segments(df_segments, v_flat_kmh, athlete_factor=1.0):
    """
    Estime la vitesse et le temps sur chaque segment de la trace.

    Parameters
    ----------
    df_segments    : pd.DataFrame  issu de segment_trace()
    v_flat_kmh     : float         vitesse de référence sur terrain plat (km/h)
    athlete_factor : float         correctif athlète (1.0 = modèle pur)

    Returns
    -------
    pd.DataFrame  avec les colonnes supplémentaires :
        speed_kmh, pace_min_per_km, time_min, time_cum_min,
        effort_index   (coût relatif vs plat)
    """
    rows = []
    t_cum = 0.0

    for _, seg in df_segments.iterrows():
        ratio  = minetti_speed_ratio(seg['slope_mean_pct'])
        v_pred = v_flat_kmh * ratio * athlete_factor
        v_pred = max(v_pred, 1.0)  # plancher physique 1 km/h

        pace   = 60.0 / v_pred
        t_min  = seg['length_km'] / v_pred * 60.0
        t_cum += t_min

        effort = minetti_cost(seg['slope_mean_pct']) / minetti_cost_flat()

        rows.append({
            **seg.to_dict(),
            'speed_kmh':       round(v_pred, 2),
            'pace_min_per_km': round(pace, 2),
            'time_min':        round(t_min, 1),
            'time_cum_min':    round(t_cum, 1),
            'effort_index':    round(effort, 2),
        })

    return pd.DataFrame(rows)


def format_time(minutes):
    """
    Formate un nombre de minutes en chaîne 'Xh YYmin'.

    Parameters
    ----------
    minutes : float

    Returns
    -------
    str
    """
    if minutes is None or math.isnan(minutes):
        return '–'
    h   = int(minutes) // 60
    m   = int(minutes) % 60
    if h > 0:
        return f'{h}h {m:02d}min'
    return f'{m}min'


def format_pace(min_per_km):
    """
    Formate une allure (min/km) en chaîne "X'YY\"".

    Parameters
    ----------
    min_per_km : float

    Returns
    -------
    str
    """
    if min_per_km is None or math.isnan(min_per_km):
        return '–'
    m = int(min_per_km)
    s = int((min_per_km - m) * 60)
    return f"{m}'{s:02d}\""


# ─────────────────────────────────────────────────────────────────────────────
# Résumé de la course
# ─────────────────────────────────────────────────────────────────────────────

def race_summary(df_pred, race_name=''):
    """
    Calcule les statistiques globales d'une prédiction de course.

    Parameters
    ----------
    df_pred   : pd.DataFrame  issu de predict_segments()
    race_name : str

    Returns
    -------
    dict
    """
    total_km    = float(df_pred['length_km'].sum())
    total_dplus = float(df_pred['dplus_m'].sum())
    total_dmoins = float(df_pred['dmoins_m'].sum())
    total_min   = float(df_pred['time_min'].sum())
    avg_speed   = total_km / (total_min / 60.0) if total_min > 0 else 0.0
    avg_pace    = 60.0 / avg_speed if avg_speed > 0 else None

    return {
        'race_name':      race_name,
        'distance_km':    round(total_km, 2),
        'dplus_m':        round(total_dplus),
        'dmoins_m':       round(total_dmoins),
        'total_min':      round(total_min, 1),
        'total_time_str': format_time(total_min),
        'avg_speed_kmh':  round(avg_speed, 2),
        'avg_pace_str':   format_pace(avg_pace),
        'n_segments':     len(df_pred),
    }
