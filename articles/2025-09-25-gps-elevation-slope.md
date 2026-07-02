---
layout: post
title: "[data · 101] Why GPS Lies About Elevation: Slope, Segments, and Walk Detection"
date: 2025-09-25
description: >
  Robust slope estimation, uphill/downhill segmentation, and walk-vs-run classification.
  What the elevation profile alone cannot tell you.
tags: [trail, data, python, fit-file, gps, slope, elevation, walk, run, signal-processing, explained]
categories: [explained, trail, data, running]
related_posts: false
toc:
  sidebar: left
math: true
---

*"You know your elevation profile. We built it in the [previous article]({% post_url 2025-08-28-anatomy-trail-race-fit-file %}). But do you know at which slope you start walking, and whether that threshold shifted in the second half of your race?"*

This is the second article in the series. We assume you have loaded and cleaned your FIT file using the pipeline from [Anatomy of a Trail Race FIT File]({% post_url 2025-08-28-anatomy-trail-race-fit-file %}): the DataFrame `df` already contains `dist_m`, `alt_m`, `speed_kmh`, `pace_s_per_km`, and the usual columns.

---

## Parameters

A few constants to set before running anything. Adjust them to your device and your race.

```python
# Slope computation
WINDOW_M = 100.0     # backward window in meters (50-200 depending on device)

# Uphill / downhill segmentation
UP_THR    = +3.0     # slope > +3% = uphill
DOWN_THR  = -3.0     # slope < -3% = downhill
MIN_SEG_M = 200.0    # discard segments shorter than 200 m

# Walk / run classification
WALK_THR_KMH = 6.0   # speed threshold (km/h)
WALK_THR_CAD = 140.0  # cadence threshold (steps/min)
```

---

## 1. Robust local slope

The naive approach to slope is straightforward: take the altitude difference between two consecutive points and divide by the distance difference. At 1 Hz recording, with an altimeter resolution around 0.5 m, this produces garbage. A single GPS glitch creates spikes of several hundred percent.

The fix is to compute slope over a **backward distance window**. For each point, we find the point roughly `WINDOW_M` meters behind on the track and compute the slope between the two. The result is a real local gradient, stable and usable.

```python
def compute_slope(df, window_m):
    """Compute local slope (%) over a backward distance window."""
    d = df["dist_m"].to_numpy()
    z = df["alt_m"].to_numpy()

    j = np.searchsorted(d, d - window_m, side="left")
    j = np.clip(j, 0, len(d) - 1)

    dd = d - d[j]
    dz = z - z[j]

    return np.where(dd > 0, (dz / dd) * 100.0, np.nan)


df["slope_pct"] = compute_slope(df, WINDOW_M)
print(df["slope_pct"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))
```

`np.searchsorted` does the heavy lifting: for each point $i$, it finds the index $j$ such that $d[j] \approx d[i] - W$. The slope is then simply $(z[i] - z[j]) / (d[i] - d[j]) \times 100$.

The choice of `WINDOW_M` is a tradeoff. Too small (< 30 m) and the slope stays noisy on basic GPS recordings. Too large (> 200 m) and you smooth out real transitions. 100 m is a reasonable starting point. If your watch has a barometric altimeter, 50 m is often enough.

---

## 2. Visual check: slope on the elevation profile

Before going further, always plot to verify. Two panels, same x-axis: elevation on top, local slope below.

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

x_km = df["dist_m"] / 1000.0

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=("Elevation profile", f"Local slope (window {WINDOW_M:.0f} m)")
)

# Elevation
fig.add_trace(go.Scatter(
    x=x_km, y=df["alt_m"],
    mode="lines", fill="tozeroy",
    fillcolor="rgba(139, 90, 43, 0.2)",
    line=dict(color="saddlebrown", width=2),
    name="Altitude",
    hovertemplate="Dist: %{x:.1f} km<br>Alt: %{y:.0f} m<extra></extra>",
), row=1, col=1)

# Slope scatter
fig.add_trace(go.Scattergl(
    x=x_km, y=df["slope_pct"],
    mode="markers",
    marker=dict(size=2, color="steelblue", opacity=0.4),
    name="Slope",
    hovertemplate="Dist: %{x:.1f} km<br>Slope: %{y:.1f} %<extra></extra>",
), row=2, col=1)

# Reference lines for thresholds
fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=0.8, row=2, col=1)
fig.add_hline(y=UP_THR, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
fig.add_hline(y=DOWN_THR, line_dash="dash", line_color="blue", opacity=0.5, row=2, col=1)

fig.update_yaxes(title_text="Altitude (m)", row=1, col=1)
fig.update_yaxes(title_text="Slope (%)", row=2, col=1)
fig.update_xaxes(title_text="Distance (km)", row=2, col=1)
fig.update_layout(height=550, template="plotly_white", showlegend=False)
fig.show()
```

If the slope scatter shows values beyond ±50%, something is off with the altitude data or the window is too short. If the extremes stay within ±30% for a typical trail race, you are in good shape.

---

## 3. Uphill / downhill segmentation

Next, we classify each point as uphill (+1), downhill (-1), or flat (0). Two mechanisms make this robust.

**Hysteresis.** In the transition zone between +3% and -3%, instead of flipping to 0 immediately, we keep the previous state. This prevents rapid oscillations around the boundary on false flats: flat, uphill, flat, uphill, flat... you get the idea.

**Minimum segment length.** A 50 m stretch classified as "uphill" because two consecutive points happen to sit at +5% carries no useful information. We discard all segments shorter than `MIN_SEG_M`.

```python
def segment_updown(df, up_thr, down_thr, min_seg_m):
    """Segment track into uphill (+1), downhill (-1), flat (0) with hysteresis."""
    s = df["slope_pct"].to_numpy()
    state = np.zeros(len(df), dtype=int)
    state[s >= up_thr] = 1
    state[s <= down_thr] = -1

    # Hysteresis: in the dead zone, propagate previous state
    for i in range(1, len(state)):
        if state[i] == 0:
            state[i] = state[i - 1]

    df = df.copy()
    df["ud_state"] = state
    df["seg_id"] = (df["ud_state"] != df["ud_state"].shift(1)).cumsum()

    seg_len = df.groupby("seg_id")["dist_m"].agg(
        lambda x: float(x.iloc[-1] - x.iloc[0])
    )
    valid = seg_len[seg_len >= min_seg_m].index
    df["ud_clean"] = np.where(df["seg_id"].isin(valid), df["ud_state"], 0)

    return df


df = segment_updown(df, UP_THR, DOWN_THR, MIN_SEG_M)

counts = df["ud_clean"].value_counts().rename({1: "uphill", -1: "downhill", 0: "flat"})
print(counts)
```

And the visualization, with the elevation profile colored by state:

```python
color_map = {1: "crimson", -1: "steelblue", 0: "mediumseagreen"}
label_map = {1: "Uphill", -1: "Downhill", 0: "Flat"}

fig = go.Figure()
for state in [1, -1, 0]:
    mask = df["ud_clean"] == state
    fig.add_trace(go.Scattergl(
        x=df.loc[mask, "dist_m"] / 1000.0,
        y=df.loc[mask, "alt_m"],
        mode="markers",
        marker=dict(size=3, color=color_map[state], opacity=0.6),
        name=label_map[state],
        hovertemplate="Dist: %{x:.1f} km<br>Alt: %{y:.0f} m<extra></extra>",
    ))

fig.update_layout(
    title="Elevation profile: uphill / downhill / flat segmentation",
    xaxis_title="Distance (km)",
    yaxis_title="Altitude (m)",
    template="plotly_white",
    height=400,
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
fig.show()
```

---

## 4. Walk vs. run classification

Speed alone is not enough to distinguish walking from running on trail. You can walk at 6 km/h on a steep climb and run at 5 km/h on a muddy false flat. **Cadence** is a much better discriminant: it typically drops below 130-140 steps/min during active walking, regardless of terrain.

We combine both criteria. A point is classified as "walking" only if speed is low **and** cadence is low.

```python
def classify_walk_run(df, walk_thr_kmh, walk_thr_cad):
    """Classify each point as walking (1) or running (0)."""
    df = df.copy()
    if "cadence" not in df.columns:
        print("No cadence data, classification not possible.")
        df["is_walk"] = np.nan
        return df
    df["is_walk"] = (
        (df["speed_kmh"] < walk_thr_kmh) & (df["cadence"] < walk_thr_cad)
    ).astype(int)
    return df


df = classify_walk_run(df, WALK_THR_KMH, WALK_THR_CAD)

if df["is_walk"].notna().any():
    pct_walk = df["is_walk"].mean() * 100
    print(f"Time spent walking: {pct_walk:.1f}%")
```

This is useful, but the really interesting view is the next one.

---

## 5. The question that matters: when do you start walking?

Group by slope bin, compute the fraction of walking in each bin, and split between the first and second half of the race. This single chart tells you your personal walk threshold, and whether fatigue shifted it.

```python
if "is_walk" in df.columns and df["is_walk"].notna().any():
    df["slope_bin"] = pd.cut(
        df["slope_pct"],
        bins=[-30, -10, -3, 3, 10, 20, 40],
        labels=["< -10%", "-10/-3%", "flat", "+3/+10%", "+10/+20%", "> +20%"]
    )

    mid_km = df["dist_m"].max() / 2000.0
    df["half"] = np.where(
        df["dist_m"] / 1000.0 < mid_km, "1st half", "2nd half"
    )

    walk_by_slope = (
        df.dropna(subset=["slope_bin", "cadence"])
        .groupby(["slope_bin", "half"], observed=True)["is_walk"]
        .mean() * 100
    ).unstack()

    fig = go.Figure()
    colors = {"1st half": "steelblue", "2nd half": "darkorange"}
    for half_label in ["1st half", "2nd half"]:
        if half_label in walk_by_slope.columns:
            fig.add_trace(go.Bar(
                x=walk_by_slope.index.astype(str),
                y=walk_by_slope[half_label],
                name=half_label,
                marker_color=colors[half_label],
            ))

    fig.update_layout(
        title="Walk probability by slope bin: 1st vs 2nd half",
        xaxis_title="Slope bin",
        yaxis_title="% walking",
        barmode="group",
        template="plotly_white",
        height=400,
    )
    fig.show()
```

If you see the orange bars (second half) creeping higher than the blue bars (first half) across all slope bins, that is fatigue showing up. Your body switched to walking earlier and more often as the race went on. If the shift only appears in the steepest bins (+20% and above), the degradation is mostly on the hard stuff, which is normal. If it shows up on the flat too, you had a rough day.

---

## What's next

We now have a DataFrame enriched with slope, segment type, and walk/run classification. The missing piece is a way to compare effort across different gradients: running at 8:00/km on a +15% climb is a very different story from 8:00/km on flat ground.

That correction is called **Grade Adjusted Pace** (GAP), and it relies on a biomechanical model of the energy cost of running on slopes. Before we implement it, we will take a proper look at the science behind it: the Minetti cost-of-transport model.

That is the subject of the [next article]({% post_url 2025-10-23-central-governor %}).

---

The notebook for this article is available on [GitHub](https://github.com/GregS1t/trail-lab/). The only cell you need to change is the one at the top.

> **Disclaimer:** I am a research engineer, not a sports physiologist. What you read here is the notebook of a curious trail runner who likes understanding his data, not medical or training advice. Sources are provided so you can verify for yourself.
