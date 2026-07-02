"""trail_report.py — Generate a coach-ready PDF report from trail race data.

Covers all analyses from analyse_trail_dashboard_v4.ipynb (except the map):
  - Race summary card (fiche résumé)
  - HR zones table + pie
  - Section stats table
  - Aid station stops table
  - Aerobic decoupling table + figure
  - Stride / cadence table + figure
  - Pace variability table + figure
  - Pace split table + figure
  - Hitting the wall figure + text
  - Pace vs slope figure (Minetti)
  - Circadian profile table + figure (night races)
  - Running power table + figure (if available)
  - Weather summary table + figure (if ERA5 data)
  - Dashboard coloured profiles
  - Radar chart per section
  - Multi-race comparison table + progression figures

Usage:
    from trail_analysis import load_and_process_race
    from trail_report import generate_report

    race = load_and_process_race(fit_path="...", fc_max=185, ...)
    race["meta"]["name"] = "Ecotrail80 2026"
    generate_report(races=[race], output_path="report.pdf", athlete_name="Greg")
"""

import io
import datetime
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
C_DARK       = colors.HexColor("#1a2332")
C_ACCENT     = colors.HexColor("#2563eb")
C_GREEN      = colors.HexColor("#16a34a")
C_ORANGE     = colors.HexColor("#ea580c")
C_RED        = colors.HexColor("#dc2626")
C_GREY_LIGHT = colors.HexColor("#f1f5f9")
C_GREY_MID   = colors.HexColor("#94a3b8")
C_GREY_DARK  = colors.HexColor("#334155")

THRESHOLDS = {
    "split_ratio":    {"green": (0.0,  1.10), "orange": (1.10, 1.20), "red": (1.20, 99.0)},
    "cv_gap_pct":     {"green": (0.0,  25.0), "orange": (25.0, 35.0), "red": (35.0, 999.0)},
    "decoupling_max": {"green": (0.0,   5.0), "orange": (5.0,  10.0), "red": (10.0, 999.0)},
    "pct_walk":       {"green": (0.0,  30.0), "orange": (30.0, 50.0), "red": (50.0, 100.0)},
    "fc_frac":        {"green": (0.75, 0.85), "orange": (0.85, 0.92), "red": (0.92, 1.0)},
}

PAGE_W = 17.0 * cm


# ===========================================================================
# Formatters
# ===========================================================================

def fmt_pace(s_per_km):
    """Format seconds/km as m'ss\"."""
    if s_per_km is None or (isinstance(s_per_km, float) and np.isnan(s_per_km)):
        return "—"
    m, s = int(s_per_km) // 60, int(s_per_km) % 60
    return f"{m}'{s:02d}\""


def fmt_duration(hours):
    """Format decimal hours as Xh YYmin."""
    if hours is None or (isinstance(hours, float) and np.isnan(hours)):
        return "—"
    h = int(hours)
    return f"{h}h {int(round((hours-h)*60)):02d}min"


def fmt_float(val, decimals=1, suffix=""):
    """Safe float to string."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{val:.{decimals}f}{suffix}"


def traffic_light(metric, value):
    """Return (hex_color, label) traffic-light for a KPI."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "#94a3b8", "—"
    thrs = THRESHOLDS.get(metric)
    if thrs is None:
        return "#94a3b8", "—"
    for lbl, (lo, hi) in [("green", thrs["green"]),
                            ("orange", thrs["orange"]),
                            ("red", thrs["red"])]:
        if lo <= value < hi:
            hx = {"green": "#16a34a", "orange": "#ea580c", "red": "#dc2626"}[lbl]
            tx = {"green": "Bon", "orange": "Moyen", "red": "Attention"}[lbl]
            return hx, tx
    return "#94a3b8", "—"


def fig_to_image(fig, width_cm=17.0, dpi=130):
    """Convert matplotlib figure to a reportlab Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    w = width_cm * cm
    fw, fh = fig.get_size_inches()
    return Image(buf, width=w, height=w * fh / fw)


# ===========================================================================
# ReportLab helpers
# ===========================================================================

def sec_title(text, styles):
    """Blue-accent section title."""
    st = ParagraphStyle(
        "ST", parent=styles["Normal"], fontSize=11,
        textColor=C_DARK, fontName="Helvetica-Bold",
        spaceBefore=12, spaceAfter=5,
    )
    return Paragraph(f'<font color="#2563eb">■</font>  {text}', st)


def body_p(text, styles):
    """Normal body paragraph."""
    st = ParagraphStyle(
        "BP", parent=styles["Normal"], fontSize=9,
        textColor=C_GREY_DARK, spaceAfter=4, leading=13,
    )
    return Paragraph(text, st)


def bullet_p(text, styles):
    """Bullet paragraph."""
    st = ParagraphStyle(
        "BUL", parent=styles["Normal"], fontSize=9,
        textColor=C_GREY_DARK, spaceAfter=4, leading=13, leftIndent=10,
    )
    return Paragraph(f"• {text}", st)


def make_table(data, col_widths=None, header_bg=None):
    """2-tone table from list-of-lists (row 0 = header)."""
    if header_bg is None:
        header_bg = C_DARK
    if col_widths is None:
        n = len(data[0])
        col_widths = [PAGE_W / n] * n
    tbl = Table(data, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1,  0), header_bg),
        ("TEXTCOLOR",     (0, 0), (-1,  0), colors.white),
        ("FONTNAME",      (0, 0), (-1,  0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_GREY_LIGHT, colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.25, C_GREY_MID),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
    ]))
    return tbl


def df_to_tbl(df, col_widths=None):
    """pandas DataFrame → reportlab table."""
    data = [list(df.columns)]
    for _, row in df.iterrows():
        data.append([str(v) for v in row.values])
    return make_table(data, col_widths=col_widths)


def make_kpi_table(kpi_rows, styles):
    """KPI table with traffic-light column.

    kpi_rows: list of (label, value_str, metric_key, raw_value)
    """
    cw = [7.5*cm, 4.5*cm, 5.0*cm]
    data = [["Indicateur", "Valeur", "Évaluation"]]
    cmds = [
        ("BACKGROUND",    (0, 0), (-1,  0), C_DARK),
        ("TEXTCOLOR",     (0, 0), (-1,  0), colors.white),
        ("FONTNAME",      (0, 0), (-1,  0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_GREY_LIGHT, colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.25, C_GREY_MID),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ]
    for i, (label, val_str, metric_key, raw) in enumerate(kpi_rows):
        hx, ev = traffic_light(metric_key, raw)
        data.append([label, val_str, ev])
        if ev != "—":
            ri = i + 1
            cmds += [("TEXTCOLOR", (2, ri), (2, ri), colors.HexColor(hx)),
                     ("FONTNAME",  (2, ri), (2, ri), "Helvetica-Bold")]
    tbl = Table(data, colWidths=cw)
    tbl.setStyle(TableStyle(cmds))
    return tbl


# ===========================================================================
# Race summary card
# ===========================================================================

def build_summary_card(story, styles, race):
    """2-column info card at the top of each race section."""
    kpis = race["kpis"]
    meta = race["meta"]
    fc_mean    = kpis.get("fc_mean",    float("nan"))
    fc_max_obs = kpis.get("fc_max_obs", float("nan"))
    fc_frac    = kpis.get("fc_frac",    float("nan"))

    fields = [
        ("Course",        meta.get("name", "—")),
        ("Date",          str(meta.get("date", "—"))),
        ("Distance",      f"{kpis['distance_km']:.1f} km"),
        ("Durée",         fmt_duration(kpis["duration_h"])),
        ("D+",            fmt_float(kpis["dplus_m"], 0, " m")),
        ("D-",            fmt_float(kpis.get("dminus_m", float("nan")), 0, " m")),
        ("Vitesse moy.",  fmt_float(kpis.get("speed_kmh", float("nan")), 1, " km/h")),
        ("GAP médian",    fmt_pace(kpis.get("gap_med_s_km"))),
        ("FC moy / max",  f"{fc_mean:.0f} / {fc_max_obs:.0f} bpm"
                          if not (np.isnan(fc_mean) or np.isnan(fc_max_obs)) else "—"),
        ("FC / FCmax",    fmt_float(fc_frac * 100 if not np.isnan(fc_frac)
                                   else float("nan"), 1, "%")),
    ]
    half = len(fields) // 2 + len(fields) % 2
    left, right = fields[:half], fields[half:]
    data = []
    for i in range(half):
        ll, vl = left[i]
        lr, vr = right[i] if i < len(right) else ("", "")
        data.append([
            Paragraph(f"<b>{ll}</b>", styles["Normal"]),
            Paragraph(vl, styles["Normal"]),
            Paragraph(f"<b>{lr}</b>", styles["Normal"]),
            Paragraph(vr, styles["Normal"]),
        ])
    cw = [3.2*cm, 5.3*cm, 3.2*cm, 5.3*cm]
    tbl = Table(data, colWidths=cw)
    tbl.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [C_GREY_LIGHT, colors.white]),
        ("GRID",           (0, 0), (-1, -1), 0.25, C_GREY_MID),
        ("FONTSIZE",       (0, 0), (-1, -1), 9),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
        ("LEFTPADDING",    (0, 0), (-1, -1), 8),
    ]))
    story.append(sec_title(f"Fiche course — {meta.get('name','')}", styles))
    story.append(tbl)
    story.append(Spacer(1, 0.3*cm))


# ===========================================================================
# Figure generators  (return fig, never call plt.show())
# ===========================================================================

def _ravitos(ax, rav_km, rav_nom):
    """Add aid-station verticals to an axis."""
    for km, nom in zip(rav_km, rav_nom):
        ax.axvline(km, color="#94a3b8", linestyle=":", linewidth=0.8, alpha=0.7)
        yt = ax.get_ylim()[1] if ax.get_ylim()[1] else 1
        ax.text(km, yt, nom, fontsize=6, rotation=90,
                va="top", ha="right", color="#64748b")


def fig_elevation(df, rav_km, rav_nom, title=""):
    """Elevation profile."""
    fig, ax = plt.subplots(figsize=(14, 2.6))
    x = df["dist_m"] / 1000.0
    z = df["alt_m"]
    ax.fill_between(x, z, alpha=0.20, color="#2563eb")
    ax.plot(x, z, color="#2563eb", linewidth=1.2)
    _ravitos(ax, rav_km, rav_nom)
    ax.set_xlabel("Distance (km)", fontsize=8)
    ax.set_ylabel("Altitude (m)", fontsize=8)
    ax.set_title(f"Profil altimétrique — {title}", fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def fig_dashboard(df, fc_min, fc_max):
    """Elevation scatter coloured by HR, pace, temperature, WBGT."""
    variables = [
        ("heart_rate",    "FC (bpm)",        "coolwarm", fc_min, fc_max),
        ("pace_s_per_km", "Allure (s/km)",   "RdYlGn_r", 180,   900),
        ("temperature",   "Température (°C)","plasma",   None,  None),
        ("wbgt_api",      "WBGT (°C)",       "YlOrRd",   15,    35),
    ]
    variables = [(c, l, cm_, vn, vx) for c, l, cm_, vn, vx in variables
                 if c in df.columns and df[c].notna().sum() > 50]
    if not variables:
        return None
    n = len(variables)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.0*n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (col, label, cmap, vmin, vmax) in zip(axes, variables):
        mask = df[col].notna()
        sc = ax.scatter(df.loc[mask, "dist_m"]/1000.0, df.loc[mask, "alt_m"],
                        c=df.loc[mask, col], cmap=cmap, s=3, alpha=0.7,
                        vmin=vmin, vmax=vmax, zorder=2)
        ax.fill_between(df["dist_m"]/1000.0, df["alt_m"], df["alt_m"].min(),
                        color="#cbd5e1", alpha=0.35, zorder=1)
        plt.colorbar(sc, ax=ax, label=label, fraction=0.02, pad=0.01)
        ax.set_ylabel("Alt (m)", fontsize=7)
        ax.set_title(label, fontsize=8)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("Distance (km)", fontsize=8)
    fig.suptitle("Dashboard — profil coloré par variable", fontsize=9, y=1.01)
    fig.tight_layout()
    return fig


def fig_hr_pie(df, fc_max):
    """HR zone pie."""
    from trail_analysis import compute_hr_zones
    hz = compute_hr_zones(df, fc_max)
    if hz.empty:
        return None
    sizes  = hz["Temps (%)"].tolist()
    labels = hz["Zone"].tolist()
    pal    = ["#93c5fd","#34d399","#fbbf24","#f97316","#ef4444"]
    nz = [(l, s, c) for l, s, c in zip(labels, sizes, pal) if s > 0]
    if not nz:
        return None
    lz, sz, cz = zip(*nz)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    wedges, _, autotexts = ax.pie(sz, labels=None, colors=cz, autopct="%1.0f%%",
                                   startangle=90, pctdistance=0.72,
                                   wedgeprops={"edgecolor":"white","linewidth":1.0})
    for at in autotexts:
        at.set_fontsize(8)
    ax.legend(wedges, lz, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    ax.set_title("Temps en zones FC", fontsize=9)
    fig.tight_layout()
    return fig


def fig_gap_profile(df, rav_km, rav_nom):
    """Smoothed GAP with split colouring."""
    x_km = df["dist_m"] / 1000.0
    win  = max(10, int(3000 / df["dist_m"].diff().median()))
    gs   = (df["gap_s_per_km"].where(df["gap_s_per_km"] < 1200)
            .rolling(win, center=True, min_periods=10).median())
    ref  = float(df.loc[df["dist_m"] <= df["dist_m"].max()/2,
                         "gap_s_per_km"].median())
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.plot(x_km, gs, color="#2563eb", linewidth=1.4, label="GAP lissé (3 km)")
    ax.axhline(ref, color="#64748b", linestyle="--", linewidth=0.9,
               label=f"Réf. 1re moitié ({fmt_pace(ref)})")
    ax.axvline(df["dist_m"].max()/2000.0, color="#1a2332", linestyle=":",
               linewidth=0.9, label="Mi-course")
    ax.fill_between(x_km, gs, ref, where=gs > ref,
                    color="#dc2626", alpha=0.09, label="Plus lent")
    ax.fill_between(x_km, gs, ref, where=gs < ref,
                    color="#16a34a", alpha=0.09, label="Plus rapide")
    _ravitos(ax, rav_km, rav_nom)
    ax.set_xlabel("Distance (km)", fontsize=8)
    ax.set_ylabel("GAP (s/km)", fontsize=8)
    ax.invert_yaxis()
    ax.set_title("GAP lissé — évolution au fil de la course", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def fig_pace_slope(df, label=""):
    """Pace and HR vs slope bins + Minetti reference."""
    from trail_analysis import minetti_cost_ratio
    bins = [-30,-15,-10,-7,-5,-3,-1,1,3,5,7,10,15,30]
    df   = df.copy()
    df["slope_bin"] = pd.cut(df["slope_pct"], bins=bins)
    cols = ["slope_pct","pace_s_per_km"] + (["heart_rate"] if "heart_rate" in df.columns else [])
    tmp  = df.dropna(subset=cols)
    if tmp.empty:
        return None
    agg = {"n":("slope_pct","size"), "slope_med":("slope_pct","median"),
           "pace_med":("pace_s_per_km","median")}
    if "heart_rate" in df.columns:
        agg["hr_med"] = ("heart_rate","median")
    s = tmp.groupby("slope_bin", observed=True).agg(**agg).reset_index()
    s = s[s["n"] >= 20]
    if s.empty:
        return None
    slopes_ref = np.linspace(s["slope_med"].min(), s["slope_med"].max(), 200)
    fm = s["slope_med"].abs() < 3
    if fm.sum() == 0:
        fm = s["slope_med"].abs() < 5
    pf = float(s.loc[fm, "pace_med"].median())
    minetti_pace = pf * minetti_cost_ratio(slopes_ref)
    has_fc = "hr_med" in s.columns
    fig, axes = plt.subplots(1, 2 if has_fc else 1,
                              figsize=(14 if has_fc else 7, 3.8))
    if not has_fc:
        axes = [axes]
    ax0, ax0t = axes[0], axes[0].twinx()
    ax0.plot(s["slope_med"], s["pace_med"], marker="o", markersize=4,
             color="#2563eb", linewidth=1.3, label="Allure réelle")
    ax0t.plot(slopes_ref, minetti_pace, "--", color="#ea580c",
              linewidth=1.2, label="Minetti (2002)")
    ax0.set_xlabel("Pente médiane (%)", fontsize=8)
    ax0.set_ylabel("Allure réelle (s/km)", color="#2563eb", fontsize=8)
    ax0t.set_ylabel("Allure Minetti (s/km)", color="#ea580c", fontsize=8)
    ax0.tick_params(axis="y", labelcolor="#2563eb", labelsize=7)
    ax0t.tick_params(axis="y", labelcolor="#ea580c", labelsize=7)
    ax0.set_title(f"Allure vs pente + Minetti (2002){' — '+label if label else ''}",
                  fontsize=9)
    lines = ax0.get_lines() + ax0t.get_lines()
    ax0.legend(lines, [l.get_label() for l in lines], fontsize=7)
    ax0.grid(True, alpha=0.2)
    if has_fc:
        axes[1].plot(s["slope_med"], s["hr_med"], marker="o", markersize=4,
                     color="#dc2626", linewidth=1.3)
        axes[1].set_xlabel("Pente médiane (%)", fontsize=8)
        axes[1].set_ylabel("FC médiane (bpm)", fontsize=8)
        axes[1].set_title(f"FC vs pente{' — '+label if label else ''}", fontsize=9)
        axes[1].tick_params(labelsize=7)
        axes[1].grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def fig_decoupling_curve(df, rav_km, rav_nom, threshold_pct=2.5):
    """Continuous aerobic decoupling curve."""
    if "heart_rate" not in df.columns or "gap_s_per_km" not in df.columns:
        return None
    df  = df.copy().sort_values("dist_m")
    gk  = 3600.0 / df["gap_s_per_km"].replace(0, np.nan)
    df["hgr"] = df["heart_rate"] / gk
    win = max(10, int(2000 / df["dist_m"].diff().median()))
    df["hgr_s"] = df["hgr"].rolling(win, center=True, min_periods=5).median()
    ref_km   = df["dist_m"].max() / 1000.0 * 0.20
    ref_mask = df["dist_m"] / 1000.0 <= ref_km
    ref_r    = df.loc[ref_mask, "hgr_s"].median()
    if np.isnan(ref_r) or ref_r == 0:
        return None
    df["dc"] = (df["hgr_s"] / ref_r - 1.0) * 100.0
    x = df["dist_m"] / 1000.0
    fig, ax = plt.subplots(figsize=(14, 3.2))
    ax.axhline(0, color="#94a3b8", linewidth=0.6, linestyle="--")
    ax.axhline(threshold_pct, color="#dc2626", linewidth=0.9, linestyle="--",
               label=f"Seuil +{threshold_pct}%")
    ax.plot(x, df["dc"], color="#7c3aed", linewidth=1.3, label="Découplage FC/GAP")
    ax.fill_between(x, 0, df["dc"], where=df["dc"] > threshold_pct,
                    color="#dc2626", alpha=0.11, label="Zone alerte")
    _ravitos(ax, rav_km, rav_nom)
    ax.set_xlabel("Distance (km)", fontsize=8)
    ax.set_ylabel("Découplage (%)", fontsize=8)
    ax.set_title("Découplage aérobie — dérive FC/GAP aux 20 premiers km\n"
                 "(Smyth et al. 2022 ; Maunder et al. 2021)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def fig_stride(df, rav_km, rav_nom):
    """Stride length and cadence along the race."""
    if "cadence" not in df.columns:
        return None
    df   = df.copy()
    chz  = df["cadence"] / 60.0
    smps = df["speed_kmh"] / 3.6
    df["stride_m"] = np.where((chz > 0) & (df["speed_kmh"] > 4.0),
                               smps / chz, np.nan)
    win = max(10, int(3000 / df["dist_m"].diff().median()))
    df["stride_s"] = df["stride_m"].rolling(win, center=True, min_periods=5).median()
    df["cad_s"]    = df["cadence"].rolling(win, center=True, min_periods=5).median()
    x = df["dist_m"] / 1000.0
    fig, axes = plt.subplots(2, 1, figsize=(14, 4.8), sharex=True)
    axes[0].plot(x, df["stride_s"], color="#2563eb", linewidth=1.2)
    _ravitos(axes[0], rav_km, rav_nom)
    axes[0].set_ylabel("Foulée (m)", fontsize=8)
    axes[0].set_title("Longueur de foulée lissée (3 km)", fontsize=9)
    axes[0].grid(True, alpha=0.2)
    axes[1].plot(x, df["cad_s"], color="#7c3aed", linewidth=1.2)
    _ravitos(axes[1], rav_km, rav_nom)
    axes[1].set_xlabel("Distance (km)", fontsize=8)
    axes[1].set_ylabel("Cadence (spm)", fontsize=8)
    axes[1].set_title("Cadence lissée (3 km)", fontsize=9)
    axes[1].grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def fig_pace_variability(df, rav_km, rav_nom):
    """GAP boxplots + CV bars per section."""
    if "gap_s_per_km" not in df.columns:
        return None
    bounds = np.concatenate(([df["dist_m"].min()/1000.0],
                              np.array(rav_km, dtype=float),
                              [df["dist_m"].max()/1000.0]))
    labels = []
    for i in range(len(bounds)-1):
        if i == 0:
            labels.append(f"→ {rav_nom[0]}")
        elif i == len(bounds)-2:
            labels.append(f"{rav_nom[-1]} →")
        else:
            labels.append(rav_nom[i])
    secs, cvs = [], []
    for i in range(len(bounds)-1):
        mask = ((df["dist_m"]/1000.0 >= bounds[i]) &
                (df["dist_m"]/1000.0 <  bounds[i+1]) &
                df["gap_s_per_km"].notna() &
                (df["gap_s_per_km"] > 0) &
                (df["gap_s_per_km"] < 1200))
        sec = df.loc[mask, "gap_s_per_km"].values
        secs.append(sec)
        cvs.append(sec.std()/sec.mean()*100 if len(sec) > 10 else np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    valid = [(l, s) for l, s in zip(labels, secs) if len(s) > 0]
    if valid:
        lv, sv = zip(*valid)
        axes[0].boxplot(sv, labels=lv, patch_artist=True,
                        boxprops=dict(facecolor="#bfdbfe", color="#2563eb"),
                        medianprops=dict(color="#dc2626", linewidth=1.4),
                        whiskerprops=dict(color="#64748b"),
                        capprops=dict(color="#64748b"),
                        flierprops=dict(marker=".", markersize=2, color="#94a3b8"))
    axes[0].set_ylabel("GAP (s/km)", fontsize=8)
    axes[0].set_title("Distribution du GAP par section", fontsize=9)
    axes[0].tick_params(axis="x", rotation=25, labelsize=7)
    axes[0].invert_yaxis()
    axes[0].grid(True, axis="y", alpha=0.2)
    x_cv  = np.arange(len(labels))
    c_cv  = ["#16a34a" if (v and v < 25) else
              "#ea580c" if (v and v < 35) else "#dc2626"
              for v in cvs]
    axes[1].bar(x_cv, cvs, color=c_cv, alpha=0.82, edgecolor="white")
    axes[1].axhline(25, color="#ea580c", linestyle="--", linewidth=0.8, alpha=0.7)
    axes[1].axhline(35, color="#dc2626", linestyle="--", linewidth=0.8, alpha=0.7)
    axes[1].set_xticks(x_cv)
    axes[1].set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
    axes[1].set_ylabel("CV GAP (%)", fontsize=8)
    axes[1].set_title("CV du GAP par section (Haney & Mercer 2011)", fontsize=9)
    axes[1].grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def fig_pace_split(df, rav_km, rav_nom):
    """GAP curve + per-section delta bars."""
    from trail_analysis import compute_pace_split
    res = compute_pace_split(df, rav_km, rav_nom)
    if res is None:
        return None
    x_km = df["dist_m"] / 1000.0
    win  = max(10, int(3000 / df["dist_m"].diff().median()))
    gs   = (df["gap_s_per_km"].where(df["gap_s_per_km"] < 1200)
            .rolling(win, center=True, min_periods=10).median())
    ref  = float(df.loc[df["dist_m"] <= df["dist_m"].max()/2,
                         "gap_s_per_km"].median())
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    axes[0].plot(x_km, gs, color="#2563eb", linewidth=1.3)
    axes[0].axhline(ref, color="#64748b", linestyle="--", linewidth=0.9,
                    label=f"Réf. ({fmt_pace(ref)})")
    axes[0].axvline(df["dist_m"].max()/2000.0, color="#1a2332",
                    linestyle=":", linewidth=0.9, label="Mi-course")
    axes[0].fill_between(x_km, gs, ref, where=gs > ref,
                         color="#dc2626", alpha=0.09)
    axes[0].fill_between(x_km, gs, ref, where=gs < ref,
                         color="#16a34a", alpha=0.09)
    _ravitos(axes[0], rav_km, rav_nom)
    axes[0].set_ylabel("GAP (s/km)", fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_title(f"GAP lissé — split {res['split_type']} "
                      f"(ratio {res['split_ratio']:.3f})", fontsize=9)
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.2)
    axes[0].set_xlabel("Distance (km)", fontsize=8)
    sdf   = res["section_df"]
    x_pos = np.arange(len(sdf))
    delts = sdf["Δ vs section 1 (%)"].values
    c_b   = ["#16a34a" if d <= 0 else ("#ea580c" if d <= 10 else "#dc2626")
              for d in delts]
    bars = axes[1].bar(x_pos, delts, color=c_b, alpha=0.82, edgecolor="white")
    axes[1].axhline(0, color="#1a2332", linewidth=0.6)
    axes[1].axhline(5, color="#ea580c", linestyle="--", linewidth=0.6, alpha=0.6)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(sdf["Section"], rotation=22, ha="right", fontsize=7)
    axes[1].set_ylabel("Δ GAP vs section 1 (%)", fontsize=8)
    axes[1].set_title("Évolution du GAP par section vs section 1", fontsize=9)
    axes[1].grid(True, axis="y", alpha=0.2)
    for bar, val in zip(bars, delts):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + (0.2 if val >= 0 else -0.8),
                     f"{val:+.1f}%", ha="center", fontsize=7)
    fig.tight_layout()
    return fig


def fig_hitting_wall(df, rav_km, rav_nom, threshold_pct=30.0):
    """Hitting-the-wall detection."""
    from trail_analysis import detect_hitting_wall
    res = detect_hitting_wall(df, threshold_pct=threshold_pct)
    if res is None:
        return None
    x_km = df["dist_m"] / 1000.0
    win  = max(10, int(2000 / df["dist_m"].diff().median()))
    gs   = (df["gap_s_per_km"]
            .where((df["gap_s_per_km"] > 0) & (df["gap_s_per_km"] < 1200))
            .rolling(win, center=True, min_periods=5).median())
    fig, ax = plt.subplots(figsize=(14, 3.4))
    ax.plot(x_km, gs, color="#2563eb", linewidth=1.3, label="GAP lissé (2 km)")
    ax.axhline(res["ref_gap"], color="#16a34a", linestyle="--", linewidth=0.9,
               label=f"Réf. ({fmt_pace(res['ref_gap'])})")
    ax.axhline(res["threshold_gap"], color="#dc2626", linestyle="--", linewidth=0.9,
               label=f"Seuil ({fmt_pace(res['threshold_gap'])})")
    for _, ep in res["episodes"].iterrows():
        ax.axvspan(ep["Début (km)"], ep["Fin (km)"],
                   color="#dc2626", alpha=0.11, zorder=1)
        ax.text((ep["Début (km)"]+ep["Fin (km)"])/2,
                res["threshold_gap"]*1.01,
                f"⚠ +{ep['Dégradation (%)']:.0f}%",
                ha="center", fontsize=7, color="#dc2626")
    _ravitos(ax, rav_km, rav_nom)
    ax.set_xlabel("Distance (km)", fontsize=8)
    ax.set_ylabel("GAP (s/km)", fontsize=8)
    ax.invert_yaxis()
    ax.set_title(f"Dégradation d'allure — seuil +{threshold_pct:.0f}% / >5 km "
                 "(Prigent et al. 2024)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def fig_radar(df, rav_km, rav_nom, fc_max):
    """Radar chart (5 axes) per section."""
    bounds = np.concatenate(([df["dist_m"].min()/1000.0],
                              np.array(rav_km),
                              [df["dist_m"].max()/1000.0]))
    sec_names = []
    for i in range(len(bounds)-1):
        if i == 0:
            sec_names.append(f"→ {rav_nom[0]}")
        elif i == len(bounds)-2:
            sec_names.append(f"{rav_nom[-1]} →")
        else:
            sec_names.append(rav_nom[i])
    gk = 3600.0 / df["gap_s_per_km"].replace(0, np.nan)
    df = df.copy()
    if "heart_rate" in df.columns:
        df["hgr"] = df["heart_rate"] / gk
    raw = []
    for i in range(len(bounds)-1):
        a, b = bounds[i], bounds[i+1]
        sec  = df.loc[(df["dist_m"]/1000.0 >= a) & (df["dist_m"]/1000.0 < b)]
        if len(sec) < 20:
            raw.append(None)
            continue
        gm  = sec["gap_s_per_km"].median() if "gap_s_per_km" in sec else np.nan
        fcf = sec["heart_rate"].median()/fc_max if "heart_rate" in sec.columns else 0.5
        pr  = 1.0 - sec["is_walk"].mean() if "is_walk" in sec.columns else 0.5
        cad = sec["cadence"].dropna() if "cadence" in sec.columns else pd.Series()
        cvc = cad.std()/cad.mean() if len(cad) > 5 and cad.mean() > 0 else 0.1
        hgr = sec["hgr"].median() if "hgr" in sec.columns else np.nan
        raw.append({"gm": gm, "fcf": fcf, "pr": pr, "cvc": cvc, "hgr": hgr})
    valid = [r for r in raw if r is not None]
    if not valid:
        return None
    gall  = [r["gm"]  for r in valid if not np.isnan(r["gm"])]
    hall  = [r["hgr"] for r in valid if not np.isnan(r["hgr"])]
    call  = [r["cvc"] for r in valid]
    if not gall:
        return None
    gmin, gmax = min(gall), max(gall)
    hmin, hmax = (min(hall), max(hall)) if hall else (0, 1)
    cmin, cmax = min(call), max(call)

    def _n(val, lo, hi, inv=False):
        if hi == lo:
            return 0.5
        n = (val - lo) / (hi - lo)
        return 1.0-n if inv else n

    ax_lbl = ["Vitesse\nGAP","Éco.\ncardio","% Course","Régularité\ncad.","Durabilité"]
    n_ax   = len(ax_lbl)
    angles = np.linspace(0, 2*np.pi, n_ax, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    cmap    = plt.get_cmap("tab10")
    handles = []
    for i, (r, name) in enumerate(zip(raw, sec_names)):
        if r is None:
            continue
        vals = [_n(r["gm"], gmin, gmax, inv=True),
                1.0 - r["fcf"],
                r["pr"],
                _n(r["cvc"], cmin, cmax, inv=True),
                _n(r["hgr"], hmin, hmax, inv=True) if not np.isnan(r["hgr"]) else 0.5]
        vals += vals[:1]
        color = cmap(i % 10)
        ax.plot(angles, vals, color=color, linewidth=1.5)
        ax.fill(angles, vals, color=color, alpha=0.07)
        handles.append(mpatches.Patch(color=color, label=name))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(ax_lbl, size=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%","50%","75%","100%"], size=7, color="#94a3b8")
    ax.set_title("Profil de performance par section\n(normalisé — vers l'extérieur = mieux)",
                 size=10, pad=18)
    ax.legend(handles=handles, loc="upper right",
              bbox_to_anchor=(1.35, 1.15), fontsize=8)
    fig.tight_layout()
    return fig


def fig_circadian(df):
    """Circadian profile: GAP, HR, walk% by time-of-day bin."""
    from trail_analysis import compute_circadian_profile
    circ = compute_circadian_profile(df, bin_hours=2)
    if circ.empty:
        return None
    labels   = circ["Tranche horaire"].tolist()
    gap_vals = circ["GAP méd. (s/km)"].values
    has_fc   = "FC méd. (bpm)"  in circ.columns
    has_walk = "Marche (%)"     in circ.columns
    n        = 1 + int(has_fc) + int(has_walk)
    fig, axes = plt.subplots(n, 1, figsize=(max(10, len(labels)*1.3), 3.3*n))
    if n == 1:
        axes = [axes]

    def is_noc(lbl):
        h = int(lbl[:2])
        return h >= 22 or h < 6

    noc  = [is_noc(l) for l in labels]
    cd, cn = "#2563eb", "#7c3aed"
    x    = np.arange(len(labels))

    bars = axes[0].bar(x, gap_vals/60.0,
                       color=[cn if n_ else cd for n_ in noc],
                       alpha=0.82, edgecolor="white")
    axes[0].axhline(np.nanmedian(gap_vals)/60.0, color="#64748b",
                    linestyle="--", linewidth=0.8)
    for bar, val in zip(bars, gap_vals):
        m, s = int(val)//60, int(val)%60
        axes[0].text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+0.02,
                     f"{m}'{s:02d}\"", ha="center", fontsize=6.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
    axes[0].set_ylabel("GAP (min/km)", fontsize=8)
    axes[0].set_title("Profil circadien — GAP par tranche de 2h\n"
                      "Violet = plage nocturne (Czeisler 1999 ; Bearden & van Woerden 2025)",
                      fontsize=9)
    axes[0].grid(True, axis="y", alpha=0.2)
    axes[0].invert_yaxis()

    idx = 1
    if has_fc:
        fv = circ["FC méd. (bpm)"].values
        axes[idx].bar(x, fv, color=[cn if n_ else cd for n_ in noc],
                      alpha=0.82, edgecolor="white")
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
        axes[idx].set_ylabel("FC médiane (bpm)", fontsize=8)
        axes[idx].set_title("FC médiane par tranche horaire", fontsize=9)
        axes[idx].grid(True, axis="y", alpha=0.2)
        idx += 1
    if has_walk:
        wv = circ["Marche (%)"].values
        axes[idx].bar(x, wv, color=[cn if n_ else cd for n_ in noc],
                      alpha=0.82, edgecolor="white")
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
        axes[idx].set_ylabel("% marche", fontsize=8)
        axes[idx].set_title("% marche par tranche horaire", fontsize=9)
        axes[idx].grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def fig_weather(df):
    """Temperature, humidity, wind, WBGT along the race."""
    has_api   = "temp_api"    in df.columns and df["temp_api"].notna().sum() > 10
    has_watch = "temperature" in df.columns and df["temperature"].notna().sum() > 10
    if not has_api and not has_watch:
        return None
    x_km   = df["dist_m"] / 1000.0
    panels = (["temp"]
              + (["hum_wind"] if has_api and "humidity_api" in df.columns else [])
              + (["wbgt"]     if has_api and "wbgt_api"    in df.columns else []))
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.8*n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, panel in zip(axes, panels):
        if panel == "temp":
            if has_api:
                ax.plot(x_km, df["temp_api"], color="#2563eb", linewidth=1.3,
                        label="ERA5-Land")
            if has_watch:
                win = max(10, int(3000/df["dist_m"].diff().median()))
                ax.plot(x_km,
                        df["temperature"].rolling(win,center=True,min_periods=5).median(),
                        color="#ea580c", linewidth=1.0, linestyle="--",
                        alpha=0.6, label="Montre (indicatif)")
            ax.set_ylabel("°C", fontsize=8)
            ax.set_title("Température", fontsize=9)
            ax.legend(fontsize=7)
        elif panel == "hum_wind":
            ax.plot(x_km, df["humidity_api"], color="#0d9488", linewidth=1.2,
                    label="Humidité (%)")
            ax.set_ylabel("Humidité (%)", fontsize=8)
            if "wind_kmh_api" in df.columns:
                ax2 = ax.twinx()
                ax2.plot(x_km, df["wind_kmh_api"], color="#ca8a04", linewidth=1.0,
                         alpha=0.7, label="Vent (km/h)")
                ax2.set_ylabel("Vent (km/h)", color="#ca8a04", fontsize=8)
                ax2.tick_params(axis="y", labelcolor="#ca8a04", labelsize=7)
            ax.set_title("Humidité & vent (ERA5-Land)", fontsize=9)
            ax.legend(fontsize=7, loc="upper left")
        elif panel == "wbgt":
            wbgt = df["wbgt_api"].values
            for sv, sc, sl in [(32,"#7f1d1d","Danger extrême > 32°C"),
                                (28,"#dc2626","Danger > 28°C"),
                                (23,"#ea580c","Vigilance > 23°C")]:
                ax.axhline(sv, color=sc, linewidth=0.7, linestyle="--",
                           alpha=0.6, label=sl)
            ax.plot(x_km, wbgt, color="#7c3aed", linewidth=1.3,
                    label="WBGT (Bernard & Kenney 1994)")
            ax.fill_between(x_km, wbgt, 28,
                            where=np.array(wbgt)>=28,
                            color="#dc2626", alpha=0.10)
            ax.fill_between(x_km, wbgt, 23,
                            where=(np.array(wbgt)>=23)&(np.array(wbgt)<28),
                            color="#ea580c", alpha=0.07)
            ax.set_ylabel("WBGT (°C)", fontsize=8)
            ax.set_title("WBGT — Périard et al. (2021)", fontsize=9)
            ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("Distance (km)", fontsize=8)
    fig.tight_layout()
    return fig


def fig_power(df, rav_km, rav_nom):
    """Running power and power/HR along the race."""
    if "power" not in df.columns or df["power"].notna().sum() < 100:
        return None
    win = max(10, int(2000/df["dist_m"].diff().median()))
    df  = df.copy()
    df["pow_s"] = df["power"].rolling(win, center=True, min_periods=5).median()
    x   = df["dist_m"] / 1000.0
    has_hr = "heart_rate" in df.columns
    fig, axes = plt.subplots(2 if has_hr else 1, 1,
                              figsize=(14, 4.8 if has_hr else 2.8),
                              sharex=True)
    if not has_hr:
        axes = [axes]
    axes[0].plot(x, df["pow_s"], color="#0d9488", linewidth=1.2)
    _ravitos(axes[0], rav_km, rav_nom)
    axes[0].set_ylabel("Puissance (W)", fontsize=8)
    axes[0].set_title("Puissance lissée — estimation algorithmique (valeur relative)",
                      fontsize=9)
    axes[0].grid(True, alpha=0.2)
    if has_hr:
        df["phr"] = np.where(df["heart_rate"] > 0,
                              df["pow_s"] / df["heart_rate"], np.nan)
        df["phr_s"] = df["phr"].rolling(win, center=True, min_periods=5).median()
        axes[1].plot(x, df["phr_s"], color="#7c3aed", linewidth=1.2)
        _ravitos(axes[1], rav_km, rav_nom)
        axes[1].set_ylabel("Puissance / FC (W/bpm)", fontsize=8)
        axes[1].set_title("Efficacité mécanique (Puissance / FC)", fontsize=9)
        axes[1].grid(True, alpha=0.2)
    axes[-1].set_xlabel("Distance (km)", fontsize=8)
    fig.tight_layout()
    return fig


# Multi-race figures
def fig_kpi_evolution(df_table):
    """Temporal evolution of KPIs across races."""
    metrics = {
        "gap_med_s_km": "GAP médian",
        "fc_frac":       "FC/FCmax (%)",
        "split_ratio":   "Split ratio",
        "decoupling_max":"Découplage max (%)",
        "pct_walk":      "% marche",
        "cv_gap_pct":    "CV GAP (%)",
    }
    avail = [k for k in metrics if k in df_table.columns]
    if not avail:
        return None
    n    = len(avail)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5*cols, 3.5*rows))
    axes = np.array(axes).flatten()
    for i, key in enumerate(avail):
        ax    = axes[i]
        label = metrics[key]
        vals  = df_table[key].to_numpy(dtype=float)
        y     = vals.copy()
        if key == "gap_med_s_km":
            y, label = vals/60.0, "GAP médian (min/km)"
        elif key == "fc_frac":
            y = vals*100.0
        x     = np.arange(len(y))
        c_pts = []
        for v in vals:
            _, ev = traffic_light(key, v)
            c_pts.append({"Bon":"#16a34a","Moyen":"#ea580c",
                          "Attention":"#dc2626","—":"#94a3b8"}.get(ev,"#94a3b8"))
        ax.plot(x, y, color="#2563eb", linewidth=1.2, zorder=2)
        ax.scatter(x, y, c=c_pts, s=65, zorder=3,
                   edgecolors="white", linewidths=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(df_table["name"].tolist(),
                           rotation=20, ha="right", fontsize=7)
        ax.set_ylabel(label, fontsize=7)
        ax.set_title(label, fontsize=8, fontweight="bold")
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)
    for j in range(len(avail), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Évolution des indicateurs clés entre les courses",
                 fontsize=10, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def fig_normalized(races_list, col, ylabel, invert=False):
    """Normalized 0-100% profiles for a given column."""
    from trail_analysis import normalize_by_distance_pct
    n_bins   = 100
    dist_pct = np.linspace(0, 100, n_bins)
    cmap     = plt.get_cmap("tab10")
    all_p    = []
    fig, ax  = plt.subplots(figsize=(14, 4))
    for i, race in enumerate(races_list):
        df = race["df"]
        if col not in df.columns:
            continue
        norm_df = normalize_by_distance_pct(df, n_bins=n_bins, cols=[col])
        p = norm_df[col].values.astype(float)
        if col == "gap_s_per_km":
            p = np.where(p > 1200, np.nan, p) / 60.0
        all_p.append(p)
        ax.plot(dist_pct, p, color=cmap(i%10), linewidth=1.2,
                alpha=0.8, label=race["meta"]["name"])
    if len(all_p) >= 2:
        stack = np.vstack(all_p)
        mp    = np.nanmean(stack, axis=0)
        sp    = np.nanstd(stack, axis=0)
        ax.plot(dist_pct, mp, color="black", linewidth=1.7,
                linestyle="--", label="Moyenne")
        ax.fill_between(dist_pct, mp-sp, mp+sp,
                        color="black", alpha=0.06, label="± 1 σ")
    if invert:
        ax.invert_yaxis()
    ax.set_xlabel("Distance (% de la course)", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(f"Profils normalisés — {ylabel} (0–100% distance)\n"
                 "Kerhervé et al. (2015)", fontsize=9)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


# ===========================================================================
# Text analysis blocks
# ===========================================================================

def analyse_single(kpis):
    """Interpretation bullets for a single race."""
    bullets = []
    sr = kpis.get("split_ratio")
    if sr and not np.isnan(sr):
        if sr < 1.0:
            bullets.append(f"<b>Gestion d'allure :</b> Split négatif (ratio {sr:.3f}) — "
                           "départ conservateur, accélération 2e moitié. Excellent.")
        elif sr <= 1.10:
            bullets.append(f"<b>Gestion d'allure :</b> Split quasi-équilibré ({sr:.3f}) — "
                           "bonne maîtrise de l'effort.")
        elif sr <= 1.20:
            bullets.append(f"<b>Gestion d'allure :</b> Split positif modéré ({sr:.3f}) — "
                           "légère perte d'allure en 2e moitié. À surveiller.")
        else:
            bullets.append(f"<b>Gestion d'allure :</b> Split positif marqué ({sr:.3f}) — "
                           "ralentissement significatif. Piste : départ trop rapide ou "
                           "accumulation de fatigue non compensée.")
    dc = kpis.get("decoupling_max")
    if dc and not np.isnan(dc):
        if dc < 5.0:
            bullets.append(f"<b>Découplage aérobie :</b> Faible ({dc:.1f}%) — "
                           "coût cardiaque stable, efficacité aérobie conservée.")
        elif dc < 10.0:
            bullets.append(f"<b>Découplage aérobie :</b> Modéré ({dc:.1f}%) — "
                           "légère hausse du coût cardiaque. Possible sous-nutrition "
                           "ou fatigue accumulée.")
        else:
            bullets.append(f"<b>Découplage aérobie :</b> Élevé ({dc:.1f}%) — "
                           "coût cardiaque en forte hausse. Déshydratation, fatigue "
                           "ou manque de volume fondamental à investiguer.")
    cv = kpis.get("cv_gap_pct")
    if cv and not np.isnan(cv):
        if cv < 25.0:
            bullets.append(f"<b>Régularité :</b> Bonne (CV = {cv:.1f}%) — allure homogène.")
        elif cv < 35.0:
            bullets.append(f"<b>Régularité :</b> Moyenne (CV = {cv:.1f}%) — variabilité "
                           "notable, probablement liée au profil ou aux ravitaillements.")
        else:
            bullets.append(f"<b>Régularité :</b> Élevée (CV = {cv:.1f}%) — forte "
                           "dispersion du GAP. À corriger si profil non exceptionnel.")
    pw = kpis.get("pct_walk")
    if pw and not np.isnan(pw):
        ctx = ("La marche sur montées est stratégique."
               if kpis.get("distance_km", 0) >= 50
               else "Un % élevé peut signaler un manque de condition ou un D+ important.")
        bullets.append(f"<b>Marche :</b> {pw:.1f}% du temps de mouvement. {ctx}")
    return bullets


def analyse_progression(df_table):
    """Progression bullets across races."""
    if len(df_table) < 2:
        return ["Au moins 2 courses nécessaires pour l'analyse de progression."]
    bullets = []
    f, l = df_table.iloc[0], df_table.iloc[-1]
    if "gap_med_s_km" in df_table.columns:
        g0, g1 = f["gap_med_s_km"], l["gap_med_s_km"]
        if not (np.isnan(g0) or np.isnan(g1) or g0 == 0):
            dp   = (g0 - g1) / g0 * 100
            sens = "progression" if dp > 0 else "régression"
            bullets.append(f"<b>Allure :</b> {'↓' if dp>0 else '↑'} {abs(dp):.1f}% "
                           f"de {sens} ({fmt_pace(g0)} → {fmt_pace(g1)}).")
    if "fc_frac" in df_table.columns:
        f0, f1 = f["fc_frac"], l["fc_frac"]
        if not (np.isnan(f0) or np.isnan(f1)) and abs(f0-f1)*100 > 2:
            d = (f0-f1)*100
            s = "meilleure efficacité cardiaque" if d > 0 else "sollicitation accrue"
            bullets.append(f"<b>Cardio :</b> {'baisse' if d>0 else 'hausse'} "
                           f"de {abs(d):.1f} pts %FCmax → {s}.")
    for key, dir_txt_up, dir_txt_down in [
        ("decoupling_max",
         "Découplage en baisse — meilleure résistance à la fatigue aérobie.",
         "Découplage en hausse — surveiller la durabilité de l'effort."),
        ("split_ratio",
         "Split ratio en amélioration — meilleure gestion entre les deux moitiés.",
         "Split ratio en dégradation — tendance à partir trop vite."),
    ]:
        if key in df_table.columns:
            v = df_table[key].dropna()
            if len(v) >= 2:
                trend = np.polyfit(np.arange(len(v)), v.values, 1)[0]
                thresh = 0.5 if key == "decoupling_max" else 0.02
                lbl   = key.replace("_", " ").capitalize()
                if trend < -thresh:
                    bullets.append(f"<b>{lbl} :</b> {dir_txt_up}")
                elif trend > thresh:
                    bullets.append(f"<b>{lbl} :</b> {dir_txt_down}")
    return bullets


# ===========================================================================
# Weather table helper
# ===========================================================================

def add_weather_section(story, styles, df):
    """Weather summary table + figure (if ERA5 data present)."""
    if "temp_api" not in df.columns or df["temp_api"].notna().sum() < 10:
        return
    story.append(sec_title("Météo pendant la course (ERA5-Land)", styles))
    rows = [["Variable", "Min", "Moy.", "Max"]]
    rows.append(["Température (°C)",
                 fmt_float(df["temp_api"].min(), 1),
                 fmt_float(df["temp_api"].mean(), 1),
                 fmt_float(df["temp_api"].max(), 1)])
    if "humidity_api" in df.columns:
        rows.append(["Humidité (%)", "—",
                     fmt_float(df["humidity_api"].mean(), 0), "—"])
    if "wind_kmh_api" in df.columns:
        rows.append(["Vent (km/h)",
                     fmt_float(df["wind_kmh_api"].min(), 1),
                     fmt_float(df["wind_kmh_api"].mean(), 1),
                     fmt_float(df["wind_kmh_api"].max(), 1)])
    if "wbgt_api" in df.columns:
        wm = float(df["wbgt_api"].max())
        rows.append(["WBGT (°C)",
                     fmt_float(df["wbgt_api"].min(), 1),
                     fmt_float(df["wbgt_api"].mean(), 1),
                     fmt_float(wm, 1)])
        note = ("⚠ WBGT > 32°C — danger extrême, performance très dégradée." if wm >= 32
                else "⚠ WBGT > 28°C — risque thermique (World Athletics)." if wm >= 28
                else "⚠ WBGT > 23°C — vigilance hydratation." if wm >= 23
                else "✓ WBGT < 23°C — conditions thermiques favorables.")
        story.append(bullet_p(note, styles))
    story.append(make_table(rows, col_widths=[6*cm, 3.5*cm, 3.5*cm, 4*cm]))
    story.append(Spacer(1, 0.2*cm))
    fig = fig_weather(df)
    if fig:
        story.append(fig_to_image(fig))
    story.append(Spacer(1, 0.3*cm))


# ===========================================================================
# Single-race section
# ===========================================================================

def build_single_race(story, styles, race):
    """Full single-race PDF section."""
    from trail_analysis import (
        compute_hr_zones, compute_aerobic_decoupling,
        compute_stride_metrics, compute_pace_variability,
        compute_pace_split, detect_hitting_wall,
        compute_ravito_stops, section_stats,
        compute_circadian_profile,
    )
    df      = race["df"]
    kpis    = race["kpis"]
    rk      = race["ravito_km"]
    rn      = race["ravito_nom"]
    fc_max  = race.get("fc_max", 185)
    fc_min  = race.get("fc_min", 47)
    name    = race["meta"]["name"]

    # 0. Fiche résumé ─────────────────────────────────────────────────────────
    build_summary_card(story, styles, race)

    # 1. KPI table ─────────────────────────────────────────────────────────────
    story.append(sec_title("Indicateurs clés de performance", styles))
    fc_frac = kpis.get("fc_frac", float("nan"))
    kpi_rows = [
        ("Split ratio (2e / 1re moitié)",
         fmt_float(kpis.get("split_ratio"), 3),
         "split_ratio", kpis.get("split_ratio")),
        ("Découplage aérobie max (%)",
         fmt_float(kpis.get("decoupling_max"), 1, "%"),
         "decoupling_max", kpis.get("decoupling_max")),
        ("CV du GAP (%)",
         fmt_float(kpis.get("cv_gap_pct"), 1, "%"),
         "cv_gap_pct", kpis.get("cv_gap_pct")),
        ("% de marche",
         fmt_float(kpis.get("pct_walk"), 1, "%"),
         "pct_walk", kpis.get("pct_walk")),
        ("FC / FCmax",
         fmt_float(fc_frac*100 if not np.isnan(fc_frac) else float("nan"), 1, "%"),
         "fc_frac", fc_frac),
    ]
    story.append(make_kpi_table(kpi_rows, styles))
    story.append(Spacer(1, 0.3*cm))

    # 2. Interprétation ────────────────────────────────────────────────────────
    bullets = analyse_single(kpis)
    if bullets:
        story.append(sec_title("Interprétation automatique", styles))
        for b in bullets:
            story.append(bullet_p(b, styles))
    story.append(Spacer(1, 0.3*cm))

    # 3. Météo (si disponible) ─────────────────────────────────────────────────
    add_weather_section(story, styles, df)

    # 4. Profil altimétrique ───────────────────────────────────────────────────
    story.append(sec_title("Profil altimétrique", styles))
    story.append(fig_to_image(fig_elevation(df, rk, rn, title=name)))
    story.append(Spacer(1, 0.3*cm))

    # 5. Dashboard ─────────────────────────────────────────────────────────────
    fig = fig_dashboard(df, fc_min, fc_max)
    if fig:
        story.append(sec_title("Dashboard — profil coloré par variable", styles))
        story.append(fig_to_image(fig))
        story.append(Spacer(1, 0.3*cm))

    # 6. Zones FC — tableau + camembert ───────────────────────────────────────
    if "heart_rate" in df.columns:
        story.append(sec_title("Zones de fréquence cardiaque", styles))
        hz = compute_hr_zones(df, fc_max)
        if not hz.empty:
            story.append(df_to_tbl(hz, col_widths=[5.5*cm,2.5*cm,2.5*cm,3.0*cm,3.5*cm]))
            story.append(Spacer(1, 0.15*cm))
        fig = fig_hr_pie(df, fc_max)
        if fig:
            story.append(fig_to_image(fig, width_cm=10.0))
        story.append(Spacer(1, 0.3*cm))

    # 7. Statistiques par section ──────────────────────────────────────────────
    if rk:
        story.append(sec_title("Statistiques par section", styles))
        sec_df = section_stats(df, rk, rn)
        if not sec_df.empty:
            story.append(df_to_tbl(sec_df))
        story.append(Spacer(1, 0.3*cm))

    # 8. Arrêts aux ravitaillements ────────────────────────────────────────────
    if rk and "timestamp" in df.columns:
        story.append(sec_title("Arrêts aux ravitaillements", styles))
        stops = compute_ravito_stops(df, rk, rn)
        if not stops.empty:
            keep = [c for c in ["Ravito","Position (km)","Nb arrêts",
                                "Temps total arrêt (min)"] if c in stops.columns]
            story.append(make_table([keep] + [[str(row[c]) for c in keep]
                                              for _, row in stops.iterrows()]))
        story.append(Spacer(1, 0.3*cm))

    # 9. GAP au fil de la course ───────────────────────────────────────────────
    story.append(sec_title("Évolution du GAP au fil de la course", styles))
    story.append(fig_to_image(fig_gap_profile(df, rk, rn)))
    story.append(Spacer(1, 0.3*cm))

    # 10. Analyse du split — tableau + figure ──────────────────────────────────
    if "gap_s_per_km" in df.columns:
        story.append(sec_title("Analyse du split d'allure", styles))
        sr = compute_pace_split(df, rk, rn)
        if sr:
            sdf  = sr["section_df"]
            keep = [c for c in ["Section","GAP méd.","Δ vs section 1 (%)","Tendance"]
                    if c in sdf.columns]
            story.append(make_table([keep]+[[str(row[c]) for c in keep]
                                            for _, row in sdf.iterrows()]))
            story.append(Spacer(1, 0.1*cm))
            story.append(body_p(
                f"Type de split : <b>{sr['split_type']}</b> — "
                f"ratio {sr['split_ratio']:.3f} — "
                f"1re moitié {fmt_pace(sr['gap_half1'])} — "
                f"2e moitié {fmt_pace(sr['gap_half2'])}", styles))
        fig = fig_pace_split(df, rk, rn)
        if fig:
            story.append(fig_to_image(fig))
        story.append(Spacer(1, 0.3*cm))

    # 11. Variabilité d'allure — tableau + figure ──────────────────────────────
    if "gap_s_per_km" in df.columns:
        story.append(sec_title("Variabilité d'allure par section", styles))
        from trail_analysis import compute_pace_variability
        pv = compute_pace_variability(df, rk, rn)
        if not pv.empty:
            story.append(df_to_tbl(pv))
            story.append(Spacer(1, 0.1*cm))
        fig = fig_pace_variability(df, rk, rn)
        if fig:
            story.append(fig_to_image(fig))
        story.append(Spacer(1, 0.3*cm))

    # 12. Découplage aérobie — tableau + figure ────────────────────────────────
    if "heart_rate" in df.columns and "gap_s_per_km" in df.columns:
        story.append(sec_title("Découplage aérobie par section", styles))
        dc_df = compute_aerobic_decoupling(df, rk, rn)
        if not dc_df.empty:
            story.append(df_to_tbl(dc_df))
            story.append(Spacer(1, 0.1*cm))
        fig = fig_decoupling_curve(df, rk, rn)
        if fig:
            story.append(fig_to_image(fig))
        story.append(Spacer(1, 0.3*cm))

    # 13. Foulée et cadence — tableau + figure ─────────────────────────────────
    if "cadence" in df.columns:
        story.append(sec_title("Foulée et cadence par section", styles))
        st_df = compute_stride_metrics(df, rk, rn)
        if not st_df.empty:
            story.append(df_to_tbl(st_df))
            story.append(Spacer(1, 0.1*cm))
        fig = fig_stride(df, rk, rn)
        if fig:
            story.append(fig_to_image(fig))
        story.append(Spacer(1, 0.3*cm))

    # 14. Allure vs pente (Minetti) ────────────────────────────────────────────
    story.append(sec_title("Allure vs pente — référence Minetti (2002)", styles))
    fig = fig_pace_slope(df, label=name)
    if fig:
        story.append(fig_to_image(fig))
    story.append(Spacer(1, 0.3*cm))

    # 15. Détection dégradation d'allure ──────────────────────────────────────
    if "gap_s_per_km" in df.columns:
        story.append(sec_title("Détection de dégradation d'allure", styles))
        htw = detect_hitting_wall(df, threshold_pct=30.0)
        if htw:
            if htw["flagged"]:
                story.append(bullet_p(
                    f"⚠ {len(htw['episodes'])} épisode(s) détecté(s) — "
                    f"référence {fmt_pace(htw['ref_gap'])} / "
                    f"seuil {fmt_pace(htw['threshold_gap'])} (+30%).", styles))
                story.append(df_to_tbl(htw["episodes"]))
            else:
                story.append(bullet_p(
                    f"✓ Aucune dégradation soutenue détectée — "
                    f"référence {fmt_pace(htw['ref_gap'])}, "
                    f"seuil {fmt_pace(htw['threshold_gap'])} (+30%) non franchi.",
                    styles))
        fig = fig_hitting_wall(df, rk, rn)
        if fig:
            story.append(fig_to_image(fig))
        story.append(Spacer(1, 0.3*cm))

    # 16. Radar par section ────────────────────────────────────────────────────
    if rk:
        story.append(sec_title("Radar de performance par section", styles))
        fig = fig_radar(df, rk, rn, fc_max)
        if fig:
            story.append(fig_to_image(fig, width_cm=10.0))
        story.append(Spacer(1, 0.3*cm))

    # 17. Profil circadien (courses nocturnes ou > 6h) ─────────────────────────
    start_h = kpis.get("start_hour", 12)
    dur_h   = kpis.get("duration_h", 0)
    is_night = (start_h >= 20 or start_h < 6 or dur_h > 12)
    if "timestamp" in df.columns and dur_h > 6 and is_night:
        story.append(sec_title("Profil circadien (course nocturne / longue)", styles))
        circ = compute_circadian_profile(df, bin_hours=2)
        if not circ.empty:
            story.append(df_to_tbl(circ))
            story.append(Spacer(1, 0.1*cm))
        fig = fig_circadian(df)
        if fig:
            story.append(fig_to_image(fig, width_cm=13.0))
        story.append(Spacer(1, 0.3*cm))

    # 18. Puissance (si disponible) ────────────────────────────────────────────
    if "power" in df.columns and df["power"].notna().sum() > 100:
        story.append(sec_title("Puissance de course (estimation algorithmique)", styles))
        bounds_p = np.concatenate(([df["dist_m"].min()/1000.0],
                                    np.array(rk, dtype=float),
                                    [df["dist_m"].max()/1000.0]))
        rows_p = [["Section", "Power méd. (W)", "Power / FC (W/bpm)"]]
        for i in range(len(bounds_p)-1):
            a, b = bounds_p[i], bounds_p[i+1]
            lbl  = (f"→ {rn[0]}" if i == 0
                    else f"{rn[-1]} →" if i == len(bounds_p)-2
                    else rn[i])
            mask = ((df["dist_m"]/1000.0 >= a) &
                    (df["dist_m"]/1000.0 <  b) &
                    df["power"].notna())
            sec  = df.loc[mask]
            if len(sec) < 10:
                continue
            pm  = sec["power"].median()
            phr = (pm / sec["heart_rate"].median()
                   if "heart_rate" in sec.columns else float("nan"))
            rows_p.append([lbl, fmt_float(pm, 0), fmt_float(phr, 2)])
        if len(rows_p) > 1:
            story.append(make_table(rows_p))
            story.append(Spacer(1, 0.1*cm))
        fig = fig_power(df, rk, rn)
        if fig:
            story.append(fig_to_image(fig))
        story.append(Spacer(1, 0.3*cm))

    story.append(PageBreak())


# ===========================================================================
# Multi-race section
# ===========================================================================

def build_multi_race(story, styles, races, df_table):
    """Multi-race comparison + progression section."""
    story.append(sec_title(f"Analyse de progression — {len(races)} courses", styles))

    # Tableau comparatif ───────────────────────────────────────────────────────
    disp = {"name":"Course","date":"Date","distance_km":"Dist.(km)",
            "dplus_m":"D+(m)","duration_h":"Durée","gap_med_s_km":"GAP méd.",
            "fc_frac":"FC/FCmax","split_ratio":"Split",
            "decoupling_max":"Découpl.","cv_gap_pct":"CV GAP","pct_walk":"% marche"}
    cols   = [c for c in disp if c in df_table.columns]
    header = [disp[c] for c in cols]
    data   = [header]
    for _, row in df_table.iterrows():
        r = []
        for c in cols:
            v = row[c]
            if c == "gap_med_s_km":
                r.append(fmt_pace(v) if not pd.isna(v) else "—")
            elif c == "duration_h":
                r.append(fmt_duration(v) if not pd.isna(v) else "—")
            elif c == "fc_frac":
                r.append(fmt_float(v*100, 0, "%") if not pd.isna(v) else "—")
            elif c in ("distance_km","dplus_m"):
                r.append(fmt_float(v, 0) if not pd.isna(v) else "—")
            elif isinstance(v, float):
                r.append(fmt_float(v, 2) if not pd.isna(v) else "—")
            else:
                r.append(str(v))
        data.append(r)
    story.append(make_table(data, col_widths=[PAGE_W/len(cols)]*len(cols)))
    story.append(Spacer(1, 0.4*cm))

    # Bullets de progression ───────────────────────────────────────────────────
    bullets = analyse_progression(df_table)
    if bullets:
        story.append(sec_title("Analyse de la progression", styles))
        for b in bullets:
            story.append(bullet_p(b, styles))
    story.append(Spacer(1, 0.4*cm))

    # Évolution des KPIs ───────────────────────────────────────────────────────
    story.append(sec_title("Évolution temporelle des indicateurs", styles))
    fig = fig_kpi_evolution(df_table)
    if fig:
        story.append(fig_to_image(fig))
    story.append(Spacer(1, 0.3*cm))

    # Profils GAP normalisés ───────────────────────────────────────────────────
    story.append(sec_title("Profils GAP normalisés (0–100% distance)", styles))
    story.append(fig_to_image(fig_normalized(races, "gap_s_per_km",
                                              "GAP (min/km)", invert=True)))
    story.append(Spacer(1, 0.3*cm))

    # Profils FC normalisés ────────────────────────────────────────────────────
    if any("heart_rate" in r["df"].columns for r in races):
        story.append(sec_title("Profils FC normalisés (0–100% distance)", styles))
        enriched = []
        for r in races:
            df = r["df"].copy()
            if "heart_rate" in df.columns:
                df["hr_pct"] = df["heart_rate"] / r.get("fc_max", 185) * 100.0
            enriched.append({**r, "df": df})
        story.append(fig_to_image(fig_normalized(enriched, "hr_pct", "FC (% FCmax)")))

    story.append(PageBreak())


# ===========================================================================
# Cover page
# ===========================================================================

def build_cover(story, styles, athlete_name, races, report_date):
    """Cover page."""
    st_t = ParagraphStyle("CT", parent=styles["Normal"], fontSize=26,
                           textColor=C_DARK, fontName="Helvetica-Bold",
                           alignment=TA_CENTER, spaceAfter=6)
    st_s = ParagraphStyle("CS", parent=styles["Normal"], fontSize=12,
                           textColor=C_GREY_DARK, alignment=TA_CENTER, spaceAfter=4)
    st_i = ParagraphStyle("CI", parent=styles["Normal"], fontSize=9,
                           textColor=C_GREY_MID, alignment=TA_CENTER)
    story.append(Spacer(1, 3*cm))
    story.append(Paragraph("Rapport d'analyse trail", st_t))
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="55%", thickness=2, color=C_ACCENT, spaceAfter=10))
    if athlete_name:
        story.append(Paragraph(f"Athlète : {athlete_name}", st_s))
    for r in races:
        story.append(Paragraph(r["meta"]["name"], st_s))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        f"Généré le {report_date.strftime('%d %B %Y')} | {len(races)} course(s)",
        st_i))
    story.append(PageBreak())


# ===========================================================================
# Public entry point
# ===========================================================================

def generate_report(races, output_path="GS_Trail_Haze_report.pdf",
                    athlete_name="", include_single=True, include_multi=True):
    """Generate a complete PDF coaching report.

    Parameters
    ----------
    races         : list of dict  output of load_and_process_race()
    output_path   : str           destination PDF file
    athlete_name  : str           shown on cover page
    include_single: bool          include per-race pages (default True)
    include_multi : bool          include multi-race comparison (default True)
    """
    from trail_analysis import build_races_table

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=2.0*cm, rightMargin=2.0*cm,
        topMargin=2.0*cm,  bottomMargin=2.0*cm,
        title=f"Rapport trail — {athlete_name}",
        author="trail_report.py",
    )
    styles = getSampleStyleSheet()
    story  = []

    build_cover(story, styles, athlete_name, races, datetime.date.today())

    if include_single:
        for race in races:
            build_single_race(story, styles, race)

    if include_multi and len(races) >= 2:
        df_table = build_races_table(races)
        build_multi_race(story, styles, races, df_table)

    doc.build(story)
    print(f"✓ Rapport généré : {output_path}")
    return output_path
