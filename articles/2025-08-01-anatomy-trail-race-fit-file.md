---
layout: post
title: "[data] Anatomy of a trail race: reading your FIT file"
date: 2025-08-01
description: "From binary file to your first plots: loading, cleaning, and visualizing trail running data with Python."
tags: [trail, data, python, fit-file, gps, signal-processing, explained]
categories: [explained, trail, data, running]
thumbnail: assets/img/blog/2025-08_fit_file/thumbnail.png
related_posts: false
toc:
  sidebar: left
math: true
---

*Your watch knows everything about your run. Here is how to read it without going through Garmin or Strava.*

---

Your watch knows everything about your run. The exact moment you started climbing, the second your heart rate spiked, the GPS coordinates of every step you took. It recorded all of it, faithfully, for hours.

The problem? It stored everything in a `.fit` file — a compact binary format that no spreadsheet can open, no text editor can read, and no human can make sense of without the right tools.

This is the first article in a series on trail running data analysis. No black box, no app doing the thinking for you. Just Python, your own data, and a bit of signal processing. By the end of this series, you will be able to detect fatigue, model training load, and extract physiological insights from a simple GPS file.

Let's start at the beginning: opening the file.

---

## What's in a FIT file?

FIT stands for Flexible and Interoperable Data Transfer. Developed by Garmin, it has quietly become the standard format for almost every GPS sport device on the market. Whatever brand is strapped to your wrist, chances are it speaks FIT.

The format is binary, meaning it is optimized for storage and transmission rather than human reading. A one-hour run typically produces a file under 1 MB, yet it contains thousands of data points recorded roughly every second. Compact, dense, and completely opaque to a text editor.

Inside, the data is organized in messages of different types. The ones we care about are called `record` messages, one per second, each carrying the raw sensor readings of that moment: position, altitude, speed, heart rate, cadence. Think of it as a very long table where each row is a heartbeat of your race.

What is actually available depends on your device. A basic GPS watch gives you coordinates, distance, and speed. Add a heart rate strap and you get cardiac data on top. A footpod brings cadence and stride length into the picture. Some fields will be missing depending on your setup, and that is fine. We will deal with it as it comes.

To decode all of this in Python, we use `fitparse`, a lightweight open-source library that handles the binary format so we do not have to.

```bash
pip install fitparse pandas numpy plotly
```

---

## Loading the data

Let's open the file. The `load_fit` function below reads every `record` message from the FIT file and assembles them into a pandas DataFrame.

```python
from fitparse import FitFile
import pandas as pd
import numpy as np


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
```

Call it with the path to your file and you get a DataFrame, one row per second.

```python
df_raw = load_fit("my_run.fit")
print(df_raw.shape)
print(df_raw.columns.tolist())
```

A typical output looks something like this:

```
(4823, 14)
['timestamp', 'time_h', 'distance', 'enhanced_altitude', 'altitude',
 'enhanced_speed', 'speed', 'heart_rate', 'cadence',
 'position_lat', 'position_long', 'temperature', ...]
```

A few things worth noticing. First, altitude often appears twice: `altitude` and `enhanced_altitude`. The enhanced version uses barometric correction when available and is generally more accurate on climbs. Same story for speed. We will always prefer the enhanced version when it exists.

Second, GPS coordinates are not stored in degrees. They come in a unit called semicircles, an integer encoding that covers the full range of latitudes and longitudes. The conversion formula is:

$$\text{degrees} = \text{semicircles} \times \frac{180}{2^{31}}$$

The factor $2^{31}$ maps the full range of a signed 32-bit integer onto $[-180°, 180°]$. In practice, one semicircle is roughly $8.4 \times 10^{-8}$ degrees, or about 0.01 mm on the ground. More precision than any GPS antenna can actually deliver.

```python
df_raw["lat"] = df_raw["position_lat"] * (180.0 / 2**31)
df_raw["lon"] = df_raw["position_long"] * (180.0 / 2**31)
```

Third, some fields will simply be absent depending on your device. `heart_rate` requires a sensor. `cadence` requires either a footpod or an optical cadence sensor. The code throughout this series checks for column existence before using it, so you will not hit errors if your setup is minimal.

---

## Cleaning and converting

The raw DataFrame is a good start, but it needs a bit of work before we can do anything useful with it. Three things to sort out, each worth understanding before we wrap them into a single function.

### Monotone distance

GPS devices record cumulative distance as they go. In theory, this number should only ever increase. In practice, small signal glitches occasionally produce tiny dips, a few centimeters here and there, where the recorded distance briefly goes backwards.

It sounds harmless. But when you later compute slope as the ratio of altitude change over distance change, a negative distance increment produces a nonsensical spike. Better to fix it early.

```python
d = df["distance"].to_numpy(dtype=float)
df["dist_m"] = np.maximum.accumulate(d)
```

`np.maximum.accumulate` replaces each value with the maximum seen so far in the array. One line, and the problem is gone.

### Selecting the best columns

Most modern GPS watches record altitude and speed twice: a raw version from the GPS antenna, and an enhanced version that uses barometric pressure to correct the signal. The enhanced altitude in particular is significantly smoother on climbs, where GPS alone tends to produce erratic jumps.

We always prefer the enhanced version when it exists, and fall back to the raw version otherwise.

```python
alt_col = "enhanced_altitude" if "enhanced_altitude" in df.columns else "altitude"
spd_col = "enhanced_speed" if "enhanced_speed" in df.columns else "speed"

df["alt_m"] = df[alt_col].to_numpy(dtype=float)
```

One thing to keep in mind: even the enhanced altitude is far from perfect. Barometric sensors drift with weather changes, and a long race run through a storm can introduce meaningful errors. We will come back to this in the next article, where we look at elevation filtering in detail.

### Computing pace

Pace is simply the inverse of speed, expressed in seconds per kilometer. If you are running at 10 km/h, your pace is $3600 / 10 = 360$ seconds per kilometer, or 6'00"/km.

$$\text{pace (s/km)} = \frac{1000}{v \text{ (m/s)}}$$

The conversion is trivial, but there is one edge case to handle: stops. At an aid station, your speed drops to near zero, and the inverse explodes. A pace of 50,000 s/km is not useful to anyone.

The fix is a simple speed threshold. Below 0.5 m/s (roughly 1.8 km/h), we assign `NaN` instead of computing the inverse. Those points will be ignored automatically by any subsequent aggregation or plot.

```python
v = df[spd_col].to_numpy(dtype=float)
df["speed_mps"] = v
df["speed_kmh"] = v * 3.6
df["pace_s_per_km"] = np.where(v > 0.5, 1000.0 / v, np.nan)
```

### Converting GPS coordinates

As we saw earlier, GPS coordinates are stored in semicircles. The conversion needs to happen before any mapping or distance calculation.

```python
if "position_lat" in df.columns and "position_long" in df.columns:
    df["lat"] = df["position_lat"] * (180.0 / 2**31)
    df["lon"] = df["position_long"] * (180.0 / 2**31)
```

The `if` guard is deliberate. Some devices or activity types do not record GPS coordinates at all, treadmill runs being the obvious example. Rather than crashing, we simply skip the conversion and move on.

### Putting it all together

Now that we have walked through each step individually, we can assemble them into a single function.

```python
def clean_df(df):
    """Select best altitude/speed columns, ensure monotone distance, compute pace."""
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
```

Call it right after `load_fit` and you are done with the plumbing.

```python
df_raw = load_fit("my_run.fit")
df, alt_col = clean_df(df_raw)

print(f"Points    : {len(df)}")
print(f"Distance  : {df['dist_m'].max() / 1000:.1f} km")
print(f"Duration  : {df['time_h'].max():.2f} h")
```

From here, `df` is clean, consistently typed, and ready for analysis. The columns we will use throughout this series are summarized below.

| Column | Unit | Description |
|---|---|---|
| `dist_m` | m | Cumulative distance, monotone |
| `alt_m` | m | Altitude, enhanced when available |
| `speed_kmh` | km/h | Speed |
| `pace_s_per_km` | s/km | Pace, NaN during stops |
| `lat`, `lon` | degrees | GPS coordinates |
| `heart_rate` | bpm | Heart rate, if available |
| `cadence` | spm | Cadence, if available |

---

## A first look at the data

The data is clean. Let us see what it looks like.

Three figures, three signals. Nothing sophisticated yet, just the raw material laid out on the table. Think of it as meeting your data for the first time before asking it hard questions.

### Elevation profile

The most natural starting point. Distance on the x-axis, altitude on the y-axis. Simple, but already telling.

```python
import plotly.graph_objects as go

x_km = df["dist_m"] / 1000.0

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=x_km,
    y=df["alt_m"],
    mode="lines",
    fill="tozeroy",
    fillcolor="rgba(139, 90, 43, 0.2)",
    line=dict(color="saddlebrown", width=2),
    name="Altitude",
    hovertemplate="Distance: %{x:.2f} km<br>Altitude: %{y:.0f} m<extra></extra>",
))
fig.update_layout(
    title="Elevation profile",
    xaxis_title="Distance (km)",
    yaxis_title="Altitude (m)",
    template="plotly_dark",
    height=300,
)
fig.show()
```

<div class="plotly-container">
  <iframe src="{{ site.baseurl }}/assets/plotly/2025-08_fit_file/elevation.html"
          width="100%" height="350" frameborder="0" scrolling="no">
  </iframe>
</div>

Even at this stage, the profile tells a story. Where are the climbs? Where does the trail flatten out? If you have run the route, you will recognize every section immediately.

### Heart rate

If your device records heart rate, this is the second thing worth plotting. Not because it is the most sophisticated metric, but because it is the most honest one. Pace can be gamed by terrain and wind. Heart rate reflects what is actually happening inside.

```python
if "heart_rate" in df.columns:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_km,
        y=df["heart_rate"],
        mode="lines",
        line=dict(color="#e05c5c", width=1.5),
        name="Heart rate",
        hovertemplate="Distance: %{x:.2f} km<br>HR: %{y:.0f} bpm<extra></extra>",
    ))
    fig.update_layout(
        title="Heart rate along the race",
        xaxis_title="Distance (km)",
        yaxis_title="Heart rate (bpm)",
        template="plotly_dark",
        height=300,
    )
    fig.show()
```

<div class="plotly-container">
  <iframe src="{{ site.baseurl }}/assets/plotly/2025-08_fit_file/heart_rate.html"
          width="100%" height="350" frameborder="0" scrolling="no">
  </iframe>
</div>

### Raw pace

Last one, and the noisiest. Raw pace fluctuates wildly from second to second, GPS signal being what it is. Do not try to read too much into the individual spikes.

What you can read, though, is the general trend. Is pace relatively stable throughout, or does it drift upward as the run progresses? That drift, when it exists, is one of the first visible signs of fatigue. We will quantify it properly in a later article.

```python
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=x_km,
    y=df["pace_s_per_km"],
    mode="lines",
    line=dict(color="#4a9ede", width=1, opacity=0.7),
    name="Pace",
    hovertemplate="Distance: %{x:.2f} km<br>Pace: %{y:.0f} s/km<extra></extra>",
))
fig.update_layout(
    title="Raw pace along the race",
    xaxis_title="Distance (km)",
    yaxis_title="Pace (s/km)",
    yaxis=dict(autorange="reversed"),
    template="plotly_dark",
    height=300,
)
fig.show()
```

<div class="plotly-container">
  <iframe src="{{ site.baseurl }}/assets/plotly/2025-08_fit_file/pace.html"
          width="100%" height="350" frameborder="0" scrolling="no">
  </iframe>
</div>

The y-axis is intentionally reversed: lower values mean faster pace, which is the convention in running. A pace of 300 s/km (5'00"/km) sits above 400 s/km (6'40"/km) on the plot, matching the intuition that faster is better.

---

## What's next

Three functions, three figures, and you now have a working pipeline to turn a binary `.fit` file into something you can actually reason about.

But look at that pace signal. Noisy, spiky, almost unreadable at the second level. And the elevation profile, while telling, hides a subtler problem: GPS altitude is not as reliable as it looks. Barometric correction helps, but it does not solve everything. A long climb recorded through changing weather can carry meaningful errors, and those errors propagate directly into any metric that depends on elevation.

The next two articles address exactly this.

In **Why GPS lies about elevation**, we will look at how altitude noise affects cumulative elevation gain, why your watch and Strava almost never agree on D+, and how a simple filtering strategy dramatically improves the signal.

In **Grade Adjusted Pace: flattening the mountain**, we will introduce the Minetti model, a biomechanical framework that converts any pace on any slope into its flat-ground equivalent. It is the foundation of almost everything that comes later in this series, from fatigue detection to training load modeling.

The full notebook for this article is available on [GitHub](https://github.com/GregS1t/trail-lab/), with everything needed to reproduce the three figures on your own data. The only cell you need to modify is the one at the top.

---

> **Disclaimer:** I am a research engineer, not a sports physiologist. What you read here is the notebook of a curious trail runner who likes understanding his data, not medical or training advice. The analyses are provided for informational purposes only. Sources are there so you can check for yourself.
