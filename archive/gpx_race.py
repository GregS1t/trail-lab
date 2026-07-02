# core/gpx_race.py
# -----------------
# Parsing de fichiers GPX et segmentation d'une trace de course en vue
# de l'estimation des temps de passage.
#
# Dépendances : numpy, scipy (standard dans l'environnement Twinity)
# Pas de gpxpy requis — parsing XML natif.
#
# Références scientifiques
# -------------------------
# - Minetti AE et al. (2002). Energy cost of walking and running at
#   extreme uphill and downhill slopes.
#   J Appl Physiol 93(3):1039-1046.
#   https://doi.org/10.1152/japplphysiol.01177.2001
# - Ramer DH (1972). An iterative procedure for the polygonal
#   approximation of plane curves. Computer Graphics and Image Processing.
# - Douglas DH & Peucker TK (1973). Algorithms for the reduction of
#   the number of points required to represent a digitized line.
#   Cartographica 10(2):112-122.

import math
import xml.etree.ElementTree as ET

import numpy as np
from scipy.signal import savgol_filter

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

# Namespaces GPX courants
_NS = {
    'gpx10': 'http://www.topografix.com/GPX/1/0',
    'gpx11': 'http://www.topografix.com/GPX/1/1',
}

# Rayon moyen de la Terre (m)
_EARTH_R = 6_371_000.0


# ─────────────────────────────────────────────────────────────────────────────
# Géodésie
# ─────────────────────────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    """
    Distance haversine entre deux points GPS (degrés décimaux).

    Parameters
    ----------
    lat1, lon1, lat2, lon2 : float  coordonnées en degrés

    Returns
    -------
    float  distance en mètres
    """
    r = _EARTH_R
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# ─────────────────────────────────────────────────────────────────────────────
# Parsing GPX → DataFrame de points
# ─────────────────────────────────────────────────────────────────────────────

def parse_gpx(gpx_path):
    """
    Parse un fichier GPX (v1.0 ou v1.1) et retourne un DataFrame de points.

    Colonnes retournées :
        lat, lon, ele (m), dist_cum (m), dist_step (m),
        slope_pct, dplus (m cumulé), dmoins (m cumulé)

    Parameters
    ----------
    gpx_path : str  chemin vers le fichier .gpx

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError  si aucun trackpoint n'est trouvé dans le fichier
    """
    import pandas as pd

    tree = ET.parse(gpx_path)
    root = tree.getroot()

    # Détection automatique du namespace
    tag = root.tag   # ex. '{http://www.topografix.com/GPX/1/1}gpx'
    ns_uri = tag.split('}')[0].lstrip('{') if '}' in tag else ''
    ns = {'g': ns_uri} if ns_uri else {}

    def find_all(node, xpath):
        if ns:
            return node.findall(xpath.replace('/', '/g:').replace('g:g:', 'g:'),
                                ns)
        return node.findall(xpath)

    # Collecte des trackpoints (trkpt) — et waypoints (wpt) en fallback
    trkpts = find_all(root, 'g:trk/g:trkseg/g:trkpt') if ns else (
        root.findall('.//{%s}trkpt' % ns_uri) if ns_uri else root.findall('.//trkpt')
    )
    if not trkpts:
        # Fallback : waypoints
        trkpts = (
            root.findall('.//{%s}wpt' % ns_uri) if ns_uri
            else root.findall('.//wpt')
        )
    if not trkpts:
        raise ValueError("Aucun trackpoint trouvé dans le fichier GPX.")

    lats, lons, eles = [], [], []
    for pt in trkpts:
        lats.append(float(pt.attrib['lat']))
        lons.append(float(pt.attrib['lon']))
        ele_node = (
            pt.find('{%s}ele' % ns_uri) if ns_uri else pt.find('ele')
        )
        eles.append(float(ele_node.text) if ele_node is not None else 0.0)

    n = len(lats)
    dist_step = np.zeros(n)
    for i in range(1, n):
        dist_step[i] = haversine(lats[i - 1], lons[i - 1], lats[i], lons[i])

    dist_cum = np.cumsum(dist_step)

    # Lissage de l'altitude (fenêtre adaptée à la densité de points)
    eles_arr = np.array(eles, dtype=float)
    win = min(21, n if n % 2 == 1 else n - 1)
    win = max(win, 5)
    if n >= win:
        eles_smooth = savgol_filter(eles_arr, window_length=win, polyorder=3)
    else:
        eles_smooth = eles_arr.copy()

    # Pente (%) et dénivelé cumulé
    d_ele   = np.diff(eles_smooth, prepend=eles_smooth[0])
    d_dist  = np.where(dist_step > 0, dist_step, np.nan)
    slope   = np.where(np.isfinite(d_dist), d_ele / d_dist * 100, 0.0)
    slope   = np.clip(slope, -60, 60)

    dplus   = np.cumsum(np.where(d_ele > 0, d_ele, 0.0))
    dmoins  = np.cumsum(np.where(d_ele < 0, -d_ele, 0.0))

    return pd.DataFrame({
        'lat':       lats,
        'lon':       lons,
        'ele':       eles_smooth,
        'ele_raw':   eles_arr,
        'dist_step': dist_step,
        'dist_cum':  dist_cum,
        'slope_pct': slope,
        'dplus':     dplus,
        'dmoins':    dmoins,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation par pente (Ramer-Douglas-Peucker sur le profil)
# ─────────────────────────────────────────────────────────────────────────────

def rdp_indices(points, epsilon):
    """
    Algorithme de Ramer-Douglas-Peucker.
    Retourne les indices des points à conserver parmi `points`.

    Parameters
    ----------
    points  : np.ndarray  shape (N, 2)  — (x, y) du profil
    epsilon : float       tolérance en unité de y (mètres d'altitude ici)

    Returns
    -------
    list[int]  indices triés dans l'ordre croissant
    """
    if len(points) < 3:
        return list(range(len(points)))

    stack = [(0, len(points) - 1)]
    keep  = {0, len(points) - 1}

    while stack:
        start, end = stack.pop()
        if end - start < 2:
            continue

        # Vecteur de la droite start→end
        p1, p2 = points[start], points[end]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        norm = math.hypot(dx, dy) or 1e-10

        # Distance perpendiculaire de chaque point intermédiaire
        segment = points[start + 1: end]
        dists = np.abs(
            dy * segment[:, 0] - dx * segment[:, 1]
            + p2[0] * p1[1] - p2[1] * p1[0]
        ) / norm

        i_max = int(np.argmax(dists))
        if dists[i_max] > epsilon:
            pivot = start + 1 + i_max
            keep.add(pivot)
            stack.append((start, pivot))
            stack.append((pivot, end))

    return sorted(keep)


def segment_trace(df_gpx, epsilon_m=8.0, min_seg_km=0.5):
    """
    Découpe la trace GPX en segments homogènes par profil altimétrique.

    Utilise RDP sur le profil altitude/distance pour identifier les
    ruptures de pente significatives, puis fusionne les micro-segments
    en dessous de `min_seg_km`.

    Parameters
    ----------
    df_gpx     : pd.DataFrame   issu de parse_gpx()
    epsilon_m  : float          tolérance RDP en mètres d'altitude (8 m par défaut)
    min_seg_km : float          longueur minimale d'un segment en km (0.5 km)

    Returns
    -------
    pd.DataFrame  une ligne par segment, colonnes :
        seg_id, dist_start_km, dist_end_km, length_km,
        ele_start, ele_end, dplus_m, dmoins_m,
        slope_mean_pct, slope_category,
        n_points
    """
    import pandas as pd

    dist = df_gpx['dist_cum'].values
    ele  = df_gpx['ele'].values

    # Points pour RDP : (distance_km, altitude)
    pts = np.column_stack([dist / 1000.0, ele])
    kept_idx = rdp_indices(pts, epsilon=epsilon_m)

    # Construction des segments bruts
    raw_segs = []
    for k in range(len(kept_idx) - 1):
        i0, i1 = kept_idx[k], kept_idx[k + 1]
        sub = df_gpx.iloc[i0: i1 + 1]

        d_ele   = np.diff(ele[i0: i1 + 1])
        dp      = float(d_ele[d_ele > 0].sum())
        dm      = float((-d_ele[d_ele < 0]).sum())
        length  = float(dist[i1] - dist[i0]) / 1000.0
        slope   = (ele[i1] - ele[i0]) / max(dist[i1] - dist[i0], 1) * 100

        raw_segs.append({
            'i0': i0, 'i1': i1,
            'dist_start_km': dist[i0] / 1000.0,
            'dist_end_km':   dist[i1] / 1000.0,
            'length_km':     length,
            'ele_start':     float(ele[i0]),
            'ele_end':       float(ele[i1]),
            'dplus_m':       dp,
            'dmoins_m':      dm,
            'slope_mean_pct': float(slope),
            'n_points':      i1 - i0 + 1,
        })

    # Fusion des segments trop courts avec leur voisin le plus proche
    merged = _merge_short_segments(raw_segs, min_seg_km)

    # Catégorisation de la pente
    def categorize(slope):
        if slope > 8:
            return 'montée raide'
        if slope > 3:
            return 'montée'
        if slope > -3:
            return 'plat'
        if slope > -8:
            return 'descente'
        return 'descente raide'

    result = []
    for seg_id, seg in enumerate(merged):
        seg['seg_id']         = seg_id
        seg['slope_category'] = categorize(seg['slope_mean_pct'])
        result.append(seg)

    cols = ['seg_id', 'dist_start_km', 'dist_end_km', 'length_km',
            'ele_start', 'ele_end', 'dplus_m', 'dmoins_m',
            'slope_mean_pct', 'slope_category', 'n_points']
    return pd.DataFrame(result)[cols]


def _merge_short_segments(segs, min_km):
    """Fusionne les segments inférieurs à min_km avec leur voisin."""
    if not segs:
        return segs

    changed = True
    while changed:
        changed = False
        merged = []
        i = 0
        while i < len(segs):
            if segs[i]['length_km'] < min_km and len(segs) > 1:
                # Fusionner avec le suivant (ou le précédent si dernier)
                if i + 1 < len(segs):
                    segs[i + 1] = _fuse(segs[i], segs[i + 1])
                    i += 1
                    changed = True
                elif merged:
                    merged[-1] = _fuse(merged[-1], segs[i])
                    i += 1
                    changed = True
                else:
                    merged.append(segs[i])
                    i += 1
            else:
                merged.append(segs[i])
                i += 1
        segs = merged

    return segs


def _fuse(a, b):
    """Fusionne deux segments adjacents en un seul."""
    d_ele = b['ele_end'] - a['ele_start']
    dist  = (b['dist_end_km'] - a['dist_start_km']) * 1000
    return {
        'i0':             a['i0'],
        'i1':             b['i1'],
        'dist_start_km':  a['dist_start_km'],
        'dist_end_km':    b['dist_end_km'],
        'length_km':      a['length_km'] + b['length_km'],
        'ele_start':      a['ele_start'],
        'ele_end':        b['ele_end'],
        'dplus_m':        a['dplus_m']  + b['dplus_m'],
        'dmoins_m':       a['dmoins_m'] + b['dmoins_m'],
        'slope_mean_pct': d_ele / dist * 100 if dist > 0 else 0.0,
        'n_points':       a['n_points'] + b['n_points'] - 1,
    }
