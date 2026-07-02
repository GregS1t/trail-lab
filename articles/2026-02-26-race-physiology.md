---
layout: post
title: "[data · 101] Reading Your Race Physiology: Cardiac Drift, Fatigue Signatures, and Effort Heatmaps"
date: 2026-02-26
description: >
  Cardiac drift, TRIMP, cross-analysis of slope and HR, GAP degradation,
  and the speed-slope-HR heatmap. What Garmin Connect does not compute.
tags: [trail, data, python, fit-file, heart-rate, cardiac-drift, TRIMP, physiology, heatmap, fatigue, explained]
categories: [explained, trail, data, running]
thumbnail: assets/img/blog/2026-02_race_physiology/heatmap.png
related_posts: false
toc:
  sidebar: left
math: true
---

*"You can see your heart rate on Garmin Connect. What you cannot do there is cross your cardiac drift with the profile, compute your TRIMP by section, or compare your personal slope-pace curve with the Minetti model. That is what we do here."*

This article concludes the hands-on data series (for now). It assumes you have run notebooks 01 through 05: the DataFrame `df` already contains `slope_pct`, `gap_s_per_km`, `is_walk`, and the standard columns.

For the theory behind the models used here, see [The Minetti Model]({% post_url 2025-11-27-minetti-model %}) and [TRIMP & the Fitness-Fatigue Model]({% post_url 2026-01-22-trimp-fitness-fatigue %}).

---

## Parameters

```python
FC_MIN = 45     # resting heart rate (bpm)
FC_MAX = 190    # max heart rate (bpm)
```

---

## 1. Cardiac drift

Over a long effort, heart rate tends to creep upward even at constant pace. This is **cardiac drift**, a sign of cardiovascular fatigue, dehydration, or rising core temperature.

The simplest proxy from FIT data: the HR/speed ratio. If effort is steady but HR climbs, this ratio increases. We smooth over a 5-minute window to filter out slope-induced spikes.

```python
def compute_cardiac_drift(df, smoothing_sec=300):
    """Compute smoothed HR/speed ratio as a cardiac drift proxy."""
    df = df.copy()
    spd = df["speed_kmh"].replace(0, np.nan)
    df["hr_speed_ratio"] = df["heart_rate"] / spd
    df_t = df.set_index("timestamp")
    df["hr_speed_smooth"] = (
        df_t["hr_speed_ratio"]
        .rolling(f"{smoothing_sec}s", min_periods=30)
        .mean()
        .values
    )
    return df

df = compute_cardiac_drift(df)
```

```python
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["time_h"], y=df["hr_speed_smooth"],
    mode="lines", line=dict(color="mediumpurple", width=1.5),
    hovertemplate="Time: %{x:.2f} h<br>HR/speed: %{y:.1f}<extra></extra>",
))
# ... (aid station markers added in the notebook)
fig.update_layout(
    title="Cardiac drift (HR / speed, smoothed 5 min)",
    xaxis_title="Time (h)", yaxis_title="HR / speed",
    template="plotly_dark", height=300,
    margin=dict(l=60, r=40, t=50, b=50),
)
fig.show()
```

<div class="plotly-container">
  <iframe src="{{ site.baseurl }}/assets/img/blog/2026-02_race_physiology/cardiac_drift.html"
          width="100%" height="350" frameborder="0" scrolling="no">
  </iframe>
</div>

A curve that rises steadily, even across sections with a similar gradient, means the heart is working harder to maintain the same output. If it jumps after a specific aid station, that is often the point where something went wrong: digestion, hydration, heat.

---

## 2. GAP degradation

Raw pace degrades naturally in the second half of an ultra, but part of that degradation is simply the profile getting harder. To isolate **real fatigue** from profile effects, we track median GAP section by section.

```python
def gap_drift(df, n_bins=20):
    """Compute rolling median GAP by distance bin."""
    df = df.copy()
    df["dist_km"] = df["dist_m"] / 1000.0
    df["dist_bin"] = pd.cut(
        df["dist_km"],
        bins=np.linspace(0, df["dist_km"].max(), n_bins + 1)
    )
    return (
        df.dropna(subset=["gap_s_per_km"])
        .groupby("dist_bin", observed=True)["gap_s_per_km"]
        .median()
    )
```

<div class="plotly-container">
  <iframe src="{{ site.baseurl }}/assets/img/blog/2026-02_race_physiology/gap_degradation.html"
          width="100%" height="350" frameborder="0" scrolling="no">
  </iframe>
</div>

The y-axis is inverted by convention: faster paces (lower s/km) are at the top. A rising curve over the kilometers, independent of the profile's climbs, is the signature of cumulative fatigue.

---

## 3. Slope vs. HR vs. pace: your personal cost curve

Group by slope bin, compute medians (robust to outliers). This is the analysis that builds your **individual cost curve**, comparable to the Minetti polynomial.

```python
SLOPE_BINS = [-30, -15, -10, -7, -5, -3, -1, 1, 3, 5, 7, 10, 15, 30]

def slope_crossanalysis(df, bins=SLOPE_BINS):
    """Aggregate pace and HR by slope bin (median)."""
    df = df.copy()
    df["slope_bin"] = pd.cut(df["slope_pct"], bins=bins)
    cols = ["slope_pct", "gap_s_per_km"]
    if "heart_rate" in df.columns:
        cols.append("heart_rate")
    tmp = df.dropna(subset=cols)
    agg = {
        "n": ("slope_pct", "size"),
        "slope_med": ("slope_pct", "median"),
        "gap_med": ("gap_s_per_km", "median"),
    }
    if "heart_rate" in df.columns:
        agg["hr_med"] = ("heart_rate", "median")
    return tmp.groupby("slope_bin", observed=True).agg(**agg).reset_index()
```

<div class="plotly-container">
  <iframe src="{{ site.baseurl }}/assets/img/blog/2026-02_race_physiology/slope_crossanalysis.html"
          width="100%" height="400" frameborder="0" scrolling="no">
  </iframe>
</div>

The comparison between your actual curve (orange) and the Minetti prediction (blue dashed) is directly interpretable. If you are systematically above Minetti on descents, you brake more than the model's sample. If you are below on climbs, either your uphill running economy is better than average, or you did not hold back.

---

## 4. TRIMP by section

We compute the TRIMP between each pair of aid stations. For the full explanation of the formula and its limits, see [TRIMP & the Fitness-Fatigue Model]({% post_url 2026-01-22-trimp-fitness-fatigue %}).

```python
def compute_trimp(df, fc_min, fc_max, hr_col="heart_rate", sex="M"):
    """Compute TRIMP (Banister 1991) from HR time series."""
    b = 1.92 if sex == "M" else 1.67
    tmp = df.dropna(subset=[hr_col, "timestamp"]).copy()
    tmp = tmp.sort_values("timestamp")
    dt_min = tmp["timestamp"].diff().dt.total_seconds().fillna(1.0) / 60.0
    hrr = (tmp[hr_col] - fc_min) / (fc_max - fc_min)
    hrr = hrr.clip(0, 1)
    return float((dt_min * hrr * np.exp(b * hrr)).sum())
```

The overall TRIMP of the session, plus a breakdown by section, gives you a physiological fingerprint of the race effort.

**Reference values** (male, highly variable between individuals):

| Session type | Typical TRIMP |
|---|---|
| Easy 1h Z2 | ~50-80 |
| Long 3h endurance | ~150-250 |
| 80 km race | ~400-700 |
| 160 km ultra | ~700-1200+ |

---

## 5. Speed x slope -> HR heatmap

The densest visualization in this article. It crosses two continuous variables (speed and slope) and encodes a third (median HR) as color. You get a map of your physiological effort across the entire course.

```python
def plot_heatmap_speed_slope_hr(df, fc_max):
    """2D heatmap: speed (x) x slope (y) -> median HR (color)."""
    req = ["speed_kmh", "slope_pct", "heart_rate"]
    dh = df.dropna(subset=req).copy()

    speed_bins = np.arange(0, 18.5, 0.5)
    slope_bins = np.arange(-25, 26, 1.0)

    sidx = np.digitize(dh["speed_kmh"].to_numpy(), speed_bins) - 1
    pidx = np.digitize(dh["slope_pct"].to_numpy(), slope_bins) - 1

    sumhr = np.zeros((len(slope_bins), len(speed_bins)))
    count = np.zeros_like(sumhr)

    for si, pi, hr in zip(sidx, pidx, dh["heart_rate"].to_numpy()):
        if 0 <= pi < len(slope_bins) and 0 <= si < len(speed_bins):
            sumhr[pi, si] += hr
            count[pi, si] += 1

    with np.errstate(invalid="ignore"):
        heat = np.where(count > 5, sumhr / count, np.nan)
    return heat, speed_bins, slope_bins
```

<div class="plotly-container">
  <iframe src="{{ site.baseurl }}/assets/img/blog/2026-02_race_physiology/heatmap.html"
          width="100%" height="450" frameborder="0" scrolling="no">
  </iframe>
</div>

How to read this map: cool-colored cells are speed/slope combinations where your HR stays low (easy effort or no engagement). Hot-colored cells are high cardiac engagement zones. Dense zones (many data points) correspond to your habitual cruising rhythm on that terrain.

Two interesting readings: the **ascending diagonal** (the faster you go on flat, the higher your HR, which is coherent) and the **climb/descent asymmetry**: at the same speed, climbing engages the cardiovascular system more, while descent is primarily a muscular constraint.

---

## What we did not do

This series covers the analysis of a single race. The real value of these tools appears over **time**: comparing TRIMP week after week, checking whether the walk threshold shifted between two competitions, or whether median GAP at the same effort improved after a specific training block.

That is the next step: aggregating multiple FIT files, building a history, and starting to see trends. But that is for another series.

---

The notebook for this article is available on [GitHub](https://github.com/GregS1t/trail-lab/). The only cell you need to change is the one at the top.

> **Disclaimer:** I am a research engineer, not a sports physiologist. What you read here is the notebook of a curious trail runner who likes understanding his data, not medical or training advice. Sources are provided so you can verify for yourself.
