# Trail Data Blog — English Publication Plan

## Concept

A 12-article series published on the **4th Thursday of each month**, mixing three tracks:

- **[data · hands-on]** — Practical Python tutorials. The reader opens a notebook and follows along.
- **[science · deep dive]** — Literature reviews with critical analysis. Sourced, formalized, honest about limits.
- **[mental · deep dive]** — Sport psychology, same scientific rigor, more accessible tone.

Each article carries one of two labels visible to the reader:

| Label | Meaning | Audience |
|---|---|---|
| **101** | Hands-on tutorial, code included, no prerequisites beyond the previous 101 articles | Trail runners curious about data, coaches, data scientists looking for a use case |
| **Research / Exploration** | Science-backed review, references, equations, critical discussion of limits | Researchers, advanced practitioners, anyone who wants to understand the models behind the tools |

---

## Publication calendar

All dates are the 4th Thursday of the month, except December 2025 (shifted to avoid Christmas).

| # | Date | Title | Track | Label | Source FR | Key dependency |
|---|---|---|---|---|---|---|
| 1 | Aug 28, 2025 | *Anatomy of a Trail Race FIT File* | data · hands-on | 101 | Anatomy EN (redate from Aug 1) | — |
| 2 | Sep 25, 2025 | *Why GPS Lies About Elevation: Slope, Segments, and Walk Detection* | data · hands-on | 101 | Terrain FR §1–4 | Requires #1 |
| 3 | Oct 23, 2025 | *The Central Governor: Does Your Brain Referee Fatigue?* | mental · deep dive | Exploration | Gouverneur central FR | Standalone |
| 4 | Nov 27, 2025 | *The Minetti Model: Energy Cost of Running on Slopes* | science · deep dive | Research | Minetti FR | Standalone |
| 5 | Dec 18, 2025 | *Grade Adjusted Pace: Flattening the Mountain* | data · hands-on | 101 | Terrain FR §5–6 | Requires #1, #2. References #4 for theory |
| 6 | Jan 22, 2026 | *TRIMP & the Fitness-Fatigue Model: Quantifying Training Load* | science · deep dive | Research | TRIMP/Banister FR | Standalone |
| 7 | Feb 26, 2026 | *Reading Your Race Physiology: Cardiac Drift, Fatigue Signatures, and Effort Heatmaps* | data · hands-on | 101 | Physiologie FR (unpublished) | Requires #1, #2, #5. References #4 and #6 for theory |
| 8 | Mar 26, 2026 | *Association vs. Dissociation: Where Should Your Attention Go During Effort?* | mental · deep dive | Exploration | Association/Dissociation FR | References #3 for context |
| 9 | Apr 23, 2026 | *Weather as Training Data: Retrieving Real Conditions with ERA5-Land* | data · hands-on | 101 | Météo FR | Requires #1 (for the DataFrame). Otherwise standalone |
| 10 | May 28, 2026 | *Self-Talk in Endurance: Does Talking to Yourself Actually Work?* | mental · deep dive | Exploration | Self-talk FR | References #3 and #8 for context |
| 11 | Jun 25, 2026 | *Mapping Your Race: Interactive Visualization from FIT Data* | data · hands-on | 101 | Carte interactive FR (finished, translation only) | Requires #1. Keeps Folium (exception to Plotly rule) |
| 12 | Jul 23, 2026 | *Predicting Trail Performance: A (Really) Hard Problem* | science · deep dive | Research | Prédicteurs FR | References #4, #6, #9. Capstone article |

---

## Retro-publication strategy

Today is June 2, 2026. Articles #1 through #10 (Aug 2025 — May 2026) will be retro-published.
Articles #11 (Jun 25) and #12 (Jul 23) will be published in real time.

Suggested workflow:
- Batch 1 (retro-publish together): #1 through #5 — the complete "data foundations" + first deep dives
- Batch 2 (retro-publish together): #6 through #10 — physiology, mental trilogy, weather
- Real-time: #11 and #12

This gives roughly 3 weeks to finalize #11 (interactive map) before its live date.

---

## Overlap resolution map

| Overlap | Resolution |
|---|---|
| **Anatomy EN ↔ lire-donnees FR** | `lire-donnees` is fully absorbed into Anatomy. Remove FR article. |
| **Terrain FR → split into #2 + #5** | #2 covers slope, segmentation, walk/run detection. #5 covers GAP and VAM. The split matches what Anatomy already promises in its "What's next" section. |
| **#5 (GAP) ↔ #4 (Minetti)** | #5 *uses* the Minetti polynomial as a tool (one paragraph + function call). #4 is the full scientific review. #5 links to #4 with: *"For the full derivation and limits of this model, see [The Minetti Model]."* |
| **#7 (Physiology) ↔ #6 (TRIMP)** | Same pattern. #7 *uses* `compute_trimp()` as a tool. #6 is the full review. #7 links to #6. |
| **#12 (Predictors) ↔ #4 (Minetti)** | #12 discusses Minetti as one predictor among many. It summarizes the model in 2–3 sentences and links to #4 for depth. No re-derivation of the polynomial. |
| **French articles** | All French versions are depublished. English becomes the single canonical version. |

---

## Content flow for each article

### #1 — Anatomy of a Trail Race FIT File
**What stays:** FIT format, `load_fit()`, `clean_df()`, first plots (elevation, HR, pace).
**What changes:** Redate to Aug 28. Rewrite "What's next" section to match the actual plan:
> *"Next up: why your watch and Strava never agree on elevation gain, how to compute slope robustly, and how to detect when you switch from running to walking."*

Plotly: already uses Plotly. No change needed.

---

### #2 — Why GPS Lies About Elevation
**Source:** Terrain FR §1–4.
**Content:** Slope calculation (windowed, `compute_slope()`), visual verification, segmentation up/down/flat (hysteresis + min length), walk/run classification (speed + cadence), walk probability by slope bin (1st vs 2nd half).
**Does NOT include:** GAP, VAM, Minetti polynomial.
**Plotly migration:** Replace all matplotlib with Plotly (scatter, filled area, bar chart).
**Closes with:** teaser for #5 (GAP).

---

### #3 — The Central Governor
**Source:** Gouverneur central FR.
**Content:** Historical context (peripheral fatigue model), Noakes 1997–2012, anticipatory regulation, RPE, Gandevia 2001, end-spurt phenomenon, limits and critiques.
**Label note:** First "Exploration" article. Sets the stage for the mental track.
**Plotly:** No code/notebook. Pure text + figures.

---

### #4 — The Minetti Model
**Source:** Minetti FR.
**Content:** Full review of Minetti et al. (2002). Protocol (10 elite runners, treadmill, calorimetry), 5th-order polynomial, key results (flat cost, climb scaling, descent minimum), limits (sample size, elite-only, treadmill vs. trail, no fatigue), extensions and alternatives.
**Label note:** First "Research" article. The polynomial is derived here and only here.
**Plotly migration:** Minetti curve figure in Plotly.

---

### #5 — Grade Adjusted Pace: Flattening the Mountain
**Source:** Terrain FR §5–6.
**Content:** GAP concept, `minetti_cost_ratio()` (code only, no re-derivation — link to #4), `compute_gap()`, visual comparison raw pace vs. GAP on elevation profile, VAM by section between aid stations.
**Key sentence:** *"The polynomial behind this conversion comes from Minetti et al. (2002). For the full story — the experiment, the assumptions, and where it breaks — see [The Minetti Model](#4)."*
**Plotly migration:** Dual-panel scatter (raw pace vs. GAP on profile) in Plotly.

---

### #6 — TRIMP & the Fitness-Fatigue Model
**Source:** TRIMP/Banister FR.
**Content:** Banister 1975/1991 (formula, exponential weighting, sex-specific constants), Morton FFM (fitness/fatigue, tau1/tau2, peak form), Coggan simplification (CTL/ATL/TSB, PMC chart, practical thresholds), Edwards 1993, Lucia 2003, limits (HR averaging, fixed coefficients, exertion type blindness), data-driven perspectives.
**Plotly:** Banister weighting curve in Plotly. Possibly a CTL/ATL/TSB example chart.

---

### #7 — Reading Your Race Physiology
**Source:** Physiologie FR (currently unpublished).
**Content:** Cardiac drift (`compute_cardiac_drift()`), GAP degradation by distance bin, slope × HR × pace cross-analysis (individual curve vs. Minetti), TRIMP by section, speed × slope → HR heatmap.
**Overlap handling:** `compute_trimp()` is called but not explained. Link to #6. `minetti_gap_curve()` is called but not derived. Link to #4.
**Plotly migration:** All 5 figures migrated to Plotly (line, bar, scatter, heatmap).

---

### #8 — Association vs. Dissociation
**Source:** Association/Dissociation FR.
**Content:** Morgan & Pollock 1977, associative vs. dissociative strategies, Stevinson & Biddle 1998 (4-quadrant model), intensity-dependent switching, meta-analyses, practical implications for ultra-distance.
**Link to #3:** *"If you've read [The Central Governor], you already know the brain doesn't passively receive fatigue signals — it actively regulates effort. Attentional focus is one of the levers."*
**Plotly:** No code. Pure text + figures.

---

### #9 — Weather as Training Data
**Source:** Météo FR.
**Content:** Why wrist temperature is biased (+2–5°C), ERA5-Land via Open-Meteo (API, resolution, coverage), `fetch_weather_hourly()`, interpolation onto GPS DataFrame, dual-panel visualization (temperature + humidity/wind), critical discussion (9 km resolution, linear interpolation, ERA5 vs. local stations).
**Plotly migration:** Weather panels in Plotly.

---

### #10 — Self-Talk in Endurance
**Source:** Self-talk FR.
**Content:** Definition and taxonomy (motivational, instructional, negative — Weinberg & Gould 2019), Blanchfield et al. 2014 (reference RCT), mechanism via RPE modulation, Hatzigeorgiadis meta-analysis, practical guidelines.
**Links to #3 and #8:** connects self-talk to the RPE framework (Governor) and attentional strategies (Association).
**Plotly:** No code. Pure text + figures.

---

### #11 — Mapping Your Race
**Source:** Carte interactive FR (finished, unpublished, translation only).
**Content:** Semicircles recap (brief, links back to #1), basic Folium map, start/finish markers, color-coded trace by physio variable (HR, slope...), aid station markers, elevation profile SVG overlay (branca MacroElement), AntPath animation, assembly function `build_race_map()`, remote sensing connection.
**Plotly exception:** This article keeps Folium. Plotly Mapbox does not support AntPath animation or custom SVG overlay injection. Folium is the right tool here.
**Minor overlap:** Semicircle conversion appears in both #1 and #11. Resolved with a cross-reference: *"We introduced this conversion in [Anatomy]. Here is the full picture."*

---

### #12 — Predicting Trail Performance: A (Really) Hard Problem
**Source:** Prédicteurs FR.
**Content:** Riegel 1981 (power law, limits for trail), Minetti segment-by-segment (link to #4, no re-derivation), ITRA performance index, Strava GAP model, ML approaches (gradient boosting, LSTM), the shared blind spot (fatigue accumulation), honest conclusion.
**Label note:** Capstone. References almost every previous article. Ideal for sharing on ML/data science communities.
**Plotly:** Possibly comparative charts in Plotly.

---

## Technical notes

### Plotly standardization
All notebooks switch from matplotlib to Plotly. Suggested defaults:
- Template: `plotly_white` (clean, prints well, works in dark mode with override)
- Color palette: consistent across the series (to be defined)
- Interactive hover: always include distance, altitude, and the plotted metric
- Export: each figure saved as standalone `.html` for blog embedding + static `.png` fallback

### Repository structure
Suggest maintaining `trail-lab/` on GitHub with one notebook per article, numbered to match:
```
trail-lab/
  01_anatomy_fit_file.ipynb
  02_gps_elevation_slope.ipynb
  03_central_governor/           (no notebook, article text only)
  04_minetti_model.ipynb
  05_grade_adjusted_pace.ipynb
  06_trimp_fitness_fatigue.ipynb
  07_race_physiology.ipynb
  08_association_dissociation/   (no notebook)
  09_weather_era5.ipynb
  10_self_talk/                  (no notebook)
  11_interactive_map.ipynb
  12_trail_predictors.ipynb
  lib/
    trail_analysis.py            (shared functions)
```

### Cross-references
Each article includes a "Series navigation" block at the top:
> *This is article 5 of 12 in the Trail Data series. Previous: [Why GPS Lies About Elevation]. Next: [TRIMP & the Fitness-Fatigue Model]. Full series index [here].*

And a footer:
> *The notebook for this article is available on [GitHub](link). The only cell you need to modify is the one at the top.*

---

## Summary by track

### Data · Hands-on (101)
1 → 2 → 5 → 7 → 9 → 11
*Anatomy → Slope & Walk detection → GAP & VAM → Physiology → Weather → Map*
Linear progression. Each builds on the previous DataFrame.

### Science · Deep dive (Research)
4 → 6 → 12
*Minetti → TRIMP/FFM → Predictors*
Can be read standalone, but #12 synthesizes everything.

### Mental · Deep dive (Exploration)
3 → 8 → 10
*Central Governor → Association/Dissociation → Self-Talk*
Loose trilogy. #3 sets the theoretical frame, #8 and #10 explore practical mechanisms.
