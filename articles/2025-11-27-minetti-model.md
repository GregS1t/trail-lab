---
layout: post
title: "[science · research] The Minetti Model: Energy Cost of Running on Slopes"
date: 2025-11-27
description: >
  The 2002 polynomial that powers every GAP calculator on the market. Where it comes
  from, what it captures, where it breaks, and what has happened since.
tags: [trail, physiology, biomechanics, energy, slope, modelling, Minetti, science]
categories: [trail, data, science]
related_posts: false
toc:
  sidebar: left
math: true
---

*"So, what time are you targeting for your next trail?"*

This question comes up all the time. You ask it to yourself, and if you don't, someone else does. And it turns out to be surprisingly hard to answer. I badly overestimated my pace for my last Ecotrail, for instance. Way too optimistic. So the question is: can we compute this reliably?

In trail running, estimating the energy cost of a course boils down to a simple question: how much energy does it take to cover one kilometer at +15% slope, and is that number the same at -20%?

This is precisely what Minetti and colleagues set out to quantify in 2002, in an article that has become a landmark in locomotion physiology.

---

## The paper

Let's be explicit about what we are discussing.

**Full reference:** Minetti, A. E., Moia, C., Roi, G. S., Susta, D., & Ferretti, G. (2002). *Energy cost of walking and running at extreme uphill and downhill slopes.* Journal of Applied Physiology, 93(3), 1039-1046. [DOI: 10.1152/japplphysiol.01177.2001](https://doi.org/10.1152/japplphysiol.01177.2001)

---

## Experimental protocol

### A methodological point worth noting

The experiment was conducted with 10 elite runners (the word "elite" matters here), aged $32.6 \pm 7.5$ years, weighing $61.2 \pm 5.7$ kg, with a maximal oxygen consumption ($\dot{V}_{O_2\text{max}}$) around $68.9 \pm 3.8$ mL·kg⁻¹·min⁻¹. The paper specifies they all practiced mountain endurance. Worth keeping in mind when comparing to typical trail runners.

### What they measured

The study measured the metabolic cost of walking ($C_w$) and running ($C_r$) on an inclinable treadmill, across slopes ranging from **-45% to +45%** (i.e. -24.2° to +24.2°), at various speeds. Measurements were taken at steady state, using **indirect calorimetry** (oxygen consumption).

> **Indirect calorimetry**, in a nutshell: rather than measuring the heat produced by the body (which would require a sealed chamber), you measure respiratory gases: how much O₂ is consumed and how much CO₂ is released per breath. Since aerobic combustion uses O₂ in proportion to the energy released, you can directly compute energy expenditure. In practice: a mask, a gas analyzer, and you read the output in watts or J·kg⁻¹·m⁻¹.

### The main result

The headline is a **5th-order polynomial** relating the cost of transport $C_r$ (in J·kg⁻¹·m⁻¹) to slope $i$ (expressed as a fraction, not percent):

$$C_r(i) = 155.4\,i^5 - 30.4\,i^4 - 43.3\,i^3 + 46.3\,i^2 + 19.5\,i + 3.6$$

This equation is now the most widely used function in grade adjusted pace (GAP) calculators, including Strava and most trail planning tools.

A 5th-order polynomial, yes. One could have fitted piecewise, but Minetti presumably found it simpler to fit a single curve through the data.

---

## What the data show

A few key results, taken directly from the paper:

- On **flat ground**, the running cost is $3.40 \pm 0.24$ J·kg⁻¹·m⁻¹, **independent of speed**. This is a well-established result in locomotion physiology and one of the reasons cost of transport is such a useful metric.

- Going **uphill**, $C_r$ increases roughly linearly with slope beyond ~+15%, reaching $18.93 \pm 1.74$ J·kg⁻¹·m⁻¹ at +45%, approximately **5.6 times the flat cost**.

- On a **moderate downhill**, $C_r$ decreases to a minimum around -10% to -20% ($1.73 \pm 0.36$ J·kg⁻¹·m⁻¹ at -20%), where gravity assists propulsion.

- On a **steep downhill** (beyond -20%), the cost rises again due to eccentric muscle work required for braking. This is the "J-shaped" curve that gives downhill running its peculiar energy profile.

- The **optimal trail gradient**, the one that minimizes cost per meter of elevation gain, sits around **20-30%** for both walking and running.

- In descent, the trained runners in the study showed a $C_r$ roughly **40% lower** than previously reported for sedentary subjects, highlighting the importance of specific downhill training.

One interesting comparison: the maximum speeds predicted from the model match observed competition speeds reasonably well on uphills, but overshoot badly on downhills. The authors attribute this gap to **biomechanical constraints** rather than metabolic ones, a point we will return to.

---

## Strengths of the model

**Extreme slope coverage.** The range from -45% to +45% covers nearly all gradients encountered in trail running today, including the steepest sections. In 2002, trail running was not yet the popular sport it has become, which makes this range all the more remarkable.

**Mechanical coherence.** The interpretation in terms of concentric muscular efficiency (uphill) and eccentric muscular efficiency (downhill) anchors the model in solid physics. Above +15% and below -15%, the measured mechanical efficiencies correspond to those of pure concentric and pure eccentric contraction, respectively (Minetti et al., 2002).

**Uphill robustness.** The $C_r$-slope relationship is quasi-linear on uphills above ~+15%, and the predictions agree well with observed performance in mountain running (Vernillo et al., 2017).

---

## Limitations and biases

**Only 10 runners, with a selection bias.** The measurements rest on just 10 trained runners, apparently all male. However, inter-individual variability in uphill $C_r$ is substantial. Balducci et al. (2016, 2017) showed that uphill running cost **cannot be predicted** from flat running cost, and varies considerably from one runner to another. The Minetti polynomial gives a *population average*, not an individual prediction.

**Gender, age, body mass?** The paper is silent on the gender composition of the sample. To my knowledge, no study has replicated Minetti's full protocol on a comparable female sample. The normalization by body mass makes the model theoretically mass-independent, but in steep descent, braking forces scale with mass. The age-slope interaction is also undocumented in direct extensions of the model.

**Treadmill, not trail.** All measurements were taken at steady state, at constant speed, on a treadmill. This excludes the irregularities of natural terrain: frequent direction changes, unstable surfaces (rocks, mud, snow), and the speed variability inherent to mountain running. The protocol also included no altitude effect, although oxygen availability could plausibly influence results.

**Polynomial artifact in steep descent.** The use of a 5th-order polynomial creates a well-documented artifact: for slopes steeper than roughly -55% (-29°), the formula produces **negative energy costs**, which is physically absurd. Several authors have had to correct or replace the polynomial for very steep terrain (e.g. Herzog, 2010). In practice, one simply defines a domain of validity and moves on.

**Descent: metabolism is not the whole story.** The paper itself notes that predicted maximum speeds in descent far exceed competition speeds. Trail descent is constrained by biomechanical factors (postural control, eccentric braking, fall risk) that do not appear in a purely energetic model. As Vernillo et al. (2017) emphasize in their reference review, descent relies heavily on eccentric contractions, which cause muscle damage and fatigue, and cannot be summarized by an energy balance alone.

**No cumulative fatigue.** The model gives an *instantaneous* cost per meter, measured in rested conditions. It does not account for the degradation of running economy over a prolonged effort. In ultra-trail, this degradation is well documented (Vernillo et al., 2014; Vercruyssen et al., 2016).

---

## What happened after 2002

The post-2002 literature has refined or extended the model on several fronts. A quick tour, though not necessarily exhaustive.

**Hoogkamer et al. (2014)** decomposed the cost of uphill running into three mechanical components (vertical work, work parallel to the slope, pendular movement), providing a more physically grounded framework than the phenomenological polynomial.

**Vernillo et al. (2017)** (*Sports Med.*, 47, 615-629) is the most comprehensive reference review on the biomechanics and physiology of running on slopes. It confirms the "J-shaped" $C_r$ profile on downhills and underlines the importance of biomechanical factors. A more recent review (Vernillo et al., 2025, *Frontiers in Bioengineering and Biotechnology*) updates these results, integrating new factors: shoes (notably carbon plates), stride pattern, individual profile, and cadence.

**Balducci et al. (2016, 2017)** showed that uphill running economy is independent of flat running economy, which significantly limits the applicability of the model to untested individuals.

**Lemire et al. (2021, 2022, 2023)** deepened the physiology of descent, showing in particular the central role of maximal knee extensor strength in downhill performance, a variable absent from Minetti's model.

**Besson et al. (2023)** compared elite and experienced trail runners, showing that elites have a lower $C_r$ despite similar mechanics, suggesting a role for neuromuscular characteristics not captured by the energy model alone.

---

## Does it work in trail?

For **uphill and moderate slopes** (+5% to +25%), Minetti's model remains a reliable approximation for estimating course energy cost, comparing routes, or computing a GAP. This is why it appears in most planning tools: Strava, GAP calculators, mountain race analyses.

For **downhill**, things are more nuanced. The model correctly predicts the *trend* of metabolic cost (minimum around -10/-20%, then rising), but it seriously underestimates the real difficulty on steep or technical terrain. The constraint is not caloric but biomechanical: eccentric fatigue, braking control, injury risk, all factors the model ignores.

In **ultra-trail** (80 km and beyond), using the polynomial to estimate total expenditure remains useful as an order of magnitude, but two practical corrections are needed:

1. **Add a fatigue factor to the downhill cost**, especially in the second half of the race.
2. **Do not use the formula beyond ±45% slope**, and be cautious in very steep descent where it produces meaningless values.

In short: Minetti (2002) is a solid foundation, still valid for uphills and moderate slopes, but incomplete for real-world trail descent. It would benefit from coupling with neuromuscular fatigue models for ultra-distance events.

---

## Next step: toward a data-driven approach?

The limitations of Minetti's polynomial all point in the same direction: the model is a *population average*, built in the lab, under stationary conditions. The natural alternative is to **build an individual model**, calibrated on each runner's actual training data.

The idea, concretely: from GPS traces enriched with heart rate data, it is theoretically possible to estimate the actual metabolic cost on each slope segment, regress this observed cost as a function of slope, speed, and a cumulative fatigue proxy, then compare the resulting individual curve to Minetti's polynomial. The main technical bottleneck is the estimation of energy expenditure from heart rate under non-steady-state conditions, which requires a kinetic model of $\dot{V}_{O_2}$ or a prior calibration. But it is an approachable problem once you have enough data.

This data-driven philosophy is no longer just an academic prospect. It has produced results at the highest competitive level.

### Enduraw and the 2025 UTMB victories

In August 2025, **Tom Evans** and **Ruth Croft** both won the [UTMB](https://utmb.world/) (177 km, 10,000 m D+). Behind these victories, a discreet figure: **Joseph Mestrallet**, founder of the Chamonix-based startup **[Enduraw](https://www.enduraw.co)**.

Mestrallet had been working with Ruth Croft for two years and Tom Evans for less than a year at the time of their wins, providing ultra-personalized support based on data, measurements, and algorithms. His method combines physiological testing (oxygen analysis masks, heat chambers), sensors, and effort modeling. Enduraw also has access to Strava data, with the ambition of scaling its performance algorithms to a broader audience ([source: u-Trail, 2025](https://www2.u-trail.com/joseph-mestrallet-le-specialiste-des-data-analystes-de-lentrainement-trail/)).

**Note on sources:** the information on Enduraw comes from sports journalism and podcasts, not peer-reviewed scientific publications. The exact nature of the algorithms used is not publicly documented in the academic literature.

What Enduraw does sits precisely in the continuation of Minetti's limitations: where the polynomial gives an average population cost, the data-driven approach aims to build an **individual model**, calibrated on the athlete's actual training data, integrating their personal physiological response to slope, fatigue, and race-day conditions.

For a recreational runner with access to their own GPS traces and heart rate data, the same logic applies at a smaller scale: gradually build a personal model of energy cost on slopes, rather than relying solely on a polynomial from 2002.

---

## References

- Minetti, A. E., et al. (2002). Energy cost of walking and running at extreme uphill and downhill slopes. *J. Appl. Physiol.*, 93(3), 1039-1046. [DOI](https://doi.org/10.1152/japplphysiol.01177.2001)
- Margaria, R. (1938). Sulla fisiologia e specialmente sul consumo energetico della marcia e della corsa. *Atti Accad. Naz. Lincei*, 7, 299-368.
- Vernillo, G., et al. (2017). Biomechanics and physiology of uphill and downhill running. *Sports Med.*, 47(4), 615-629. [DOI](https://doi.org/10.1007/s40279-016-0605-y)
- Vernillo, G., et al. (2014). Changes in running economy during a 65-km ultramarathon. *Eur. J. Appl. Physiol.*, 114(9), 1809-1818. [DOI](https://doi.org/10.1007/s00421-014-2908-5)
- Vercruyssen, F., et al. (2016). Changes in the energy cost of running during a 24-h ultra. *Med. Sci. Sports Exerc.*, 48(11), 2269-2275.
- Balducci, P., et al. (2016). Comparison of level and graded treadmill tests to evaluate endurance mountain runners. *J. Sports Sci. Med.*, 15, 239-246.
- Hoogkamer, W., Taboga, P., & Kram, R. (2014). Applying the cost of generating force hypothesis to uphill running. *PeerJ*, 2, e482. [DOI](https://doi.org/10.7717/peerj.482)
- Besson, T., et al. (2023). Elite vs. experienced male and female trail runners: comparing running economy, biomechanics, strength, and power. *J. Strength Cond. Res.*, 37, 1470-1478.
- Lemire, M., et al. (2021). Physiological factors determining downhill running performance. *Front. Physiol.*, 12, 748895. [DOI](https://doi.org/10.3389/fphys.2021.748895)
- Herzog, W. (2010). The biomechanics of downhill running. In: *Downhill Running and Eccentric Loading*. Routledge.

---

> **Disclaimer:** I am a research engineer, not a sports physiologist. What you read here is the logbook of a curious trail runner who enjoys understanding his data, not a lecture. Sources are there so you can check for yourself.
