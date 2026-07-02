---
layout: post
title: "[data · 101] Grade Adjusted Pace: Flattening the Mountain"
date: 2025-12-18
description: >
  Your raw pace is not comparable across gradients. GAP corrects for slope
  using the Minetti model, and VAM tells you how fast you actually climb.
tags: [trail, data, python, fit-file, gap, vam, pace, minetti, slope, explained]
categories: [explained, trail, data, running]
thumbnail: assets/img/blog/2025-12_grade_adjusted_pace/gap_vs_raw.png
related_posts: false
toc:
  sidebar: left
math: true
---

*"Running 8:00/km on a +15% climb and 8:00/km on flat ground are not the same effort. Not even close. GAP tells you the difference."*

This article picks up where [Why GPS Lies About Elevation]({% post_url 2025-09-25-gps-elevation-slope %}) left off. The DataFrame `df` already contains `slope_pct`, `ud_clean`, `is_walk`, and the columns from notebook 01.

For the science behind the formula we use here, see [The Minetti Model]({% post_url 2025-11-27-minetti-model %}). This article applies it; that one explains it.

---

## 1. From slope to energy cost ratio

The Minetti polynomial gives the energy cost of running at a given slope. To convert raw pace into grade adjusted pace, we only need the **ratio** between cost at current slope and cost on flat ground.

```python
def minetti_cost_ratio(slope_pct):
    """Cost ratio relative to flat ground, from Minetti et al. (2002)."""
    i = slope_pct / 100.0
    cr = (155.4 * i**5 - 30.4 * i**4 - 43.3 * i**3
          + 46.3 * i**2 + 19.5 * i + 3.6)
    cr_flat = 3.6
    return np.clip(cr / cr_flat, 0.1, None)
```

If the ratio is 2.0, you spend twice the energy per meter compared to flat ground. Your GAP is then half your raw pace: you were working as hard as someone running twice as fast on the flat.

---

## 2. Computing GAP

The conversion is one line once you have the ratio:

```python
def compute_gap(df):
    """Compute grade adjusted pace (s/km) using Minetti cost ratio."""
    df = df.copy()
    ratio = minetti_cost_ratio(df["slope_pct"].to_numpy())
    df["gap_s_per_km"] = df["pace_s_per_km"] / ratio
    return df

df = compute_gap(df)
```

In words: GAP = raw pace / cost ratio. If the ratio is greater than 1 (uphill), GAP is faster than raw pace, because you are working harder than the number on your watch suggests. If the ratio is below 1 (moderate downhill), GAP is slower: gravity is doing part of the work.

---

## 3. Visualizing the correction

The most instructive view is a side-by-side comparison. Same elevation profile, colored by raw pace on top and by GAP below. Wherever the colors differ between the two panels, that is the profile doing the talking rather than your fitness.

```python
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=("Raw pace (s/km)", "GAP (s/km)")
)

for row, col, label in [(1, "pace_s_per_km", "Raw"), (2, "gap_s_per_km", "GAP")]:
    mask = df[col].between(180, 900)
    fig.add_trace(go.Scattergl(
        x=df.loc[mask, "dist_m"] / 1000.0,
        y=df.loc[mask, "alt_m"],
        mode="markers",
        marker=dict(
            size=3, color=df.loc[mask, col],
            colorscale="RdYlGn_r", cmin=180, cmax=900,
            colorbar=dict(title="s/km", len=0.4, y=0.8 if row == 1 else 0.2),
        ),
        hovertemplate="Dist: %{x:.1f} km<br>Alt: %{y:.0f} m<extra></extra>",
        showlegend=False,
    ), row=row, col=1)

fig.update_yaxes(title_text="Altitude (m)", row=1, col=1)
fig.update_yaxes(title_text="Altitude (m)", row=2, col=1)
fig.update_xaxes(title_text="Distance (km)", row=2, col=1)
fig.update_layout(
    height=550, template="plotly_dark",
    margin=dict(l=60, r=40, t=50, b=50),
)
fig.show()
```

<div class="plotly-container">
  <iframe src="{{ site.baseurl }}/assets/img/blog/2025-12_grade_adjusted_pace/gap_vs_raw.html"
          width="100%" height="600" frameborder="0" scrolling="no">
  </iframe>
</div>

The difference between the two panels reveals the effect of the profile on pace. Zones where GAP is much faster than raw pace are steep climbs: you advance slowly but spend a lot of energy. The reverse happens on fast descents where gravity assists you.

---

## 4. VAM by section

VAM (Vertical Ascent Meters per hour, *Velocità Ascensionale Media* in Italian) is the standard climbing performance metric in mountain sports. It normalizes effort independently of distance and gradient: 800 m/h of VAM is universally understood, whereas "8:30/km at +15%" requires mental arithmetic.

We compute it section by section, between aid stations.

```python
def section_summary(df, ravito_km, ravito_nom):
    """Compute per-section stats: distance, D+, duration, VAM, GAP, HR, walk%."""
    bounds = np.concatenate((
        [float(df["dist_m"].min() / 1000.0)],
        np.array(ravito_km, dtype=float),
        [float(df["dist_m"].max() / 1000.0)]
    ))
    labels = []
    for i in range(len(bounds) - 1):
        if i == 0:
            labels.append(f"Start -> {ravito_nom[0]}")
        elif i == len(bounds) - 2:
            labels.append(f"{ravito_nom[-1]} -> Finish")
        else:
            labels.append(f"{ravito_nom[i-1]} -> {ravito_nom[i]}")

    df = df.copy()
    df["section_id"] = np.searchsorted(
        bounds[1:], df["dist_m"].to_numpy() / 1000.0, side="right"
    )

    rows = []
    for i, lbl in enumerate(labels):
        sec = df[df["section_id"] == i]
        if len(sec) < 10:
            continue
        alt = sec["alt_m"].to_numpy()
        dz = np.diff(alt)
        dplus = float(np.clip(dz, 0, None).sum())
        dur_h = float(sec["time_h"].iloc[-1] - sec["time_h"].iloc[0])

        row = {
            "Section": lbl,
            "Dist (km)": round((sec["dist_m"].iloc[-1] - sec["dist_m"].iloc[0]) / 1000, 1),
            "D+ (m)": round(dplus, 0),
            "Duration (h)": round(dur_h, 2),
            "VAM (m/h)": round(dplus / dur_h, 0) if dur_h > 0 else np.nan,
        }
        if "gap_s_per_km" in sec.columns:
            row["GAP med (s/km)"] = round(sec["gap_s_per_km"].median(), 0)
        if "heart_rate" in sec.columns:
            row["HR med (bpm)"] = round(sec["heart_rate"].median(), 0)
        if "is_walk" in sec.columns and sec["is_walk"].notna().any():
            row["Walk (%)"] = round(sec["is_walk"].mean() * 100, 1)
        rows.append(row)

    return pd.DataFrame(rows)


if len(RAVITO_KM) > 0:
    stats = section_summary(df, RAVITO_KM, RAVITO_NOM)
    print(stats.to_string(index=False))
```

VAM per section is more informative than pace when comparing stretches with very different profiles. A section at 700 m/h VAM with 50% walking tells a different story from 700 m/h in continuous running, and the table shows you both.

---

## What we have now

At this point, the DataFrame contains everything needed to analyze a single race in depth: raw data (speed, HR, altitude, GPS), terrain features (slope, segments, walk/run), and effort-corrected metrics (GAP, VAM).

The next articles will build on this foundation. First, a scientific look at how to quantify the total training load of a session using heart rate: [TRIMP & the Fitness-Fatigue Model]({% post_url 2026-01-22-trimp-fitness-fatigue %}). Then, a deep dive into the physiological signatures hidden in your race data.

---

The notebook for this article is available on [GitHub](https://github.com/GregS1t/trail-lab/). The only cell you need to change is the one at the top.

> **Disclaimer:** I am a research engineer, not a sports physiologist. What you read here is the notebook of a curious trail runner who likes understanding his data, not medical or training advice. Sources are provided so you can verify for yourself.
