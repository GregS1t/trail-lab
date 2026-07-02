---
layout: post
title: "[science · research] TRIMP & the Fitness-Fatigue Model: Quantifying Training Load"
date: 2026-01-22
description: >
  The Banister TRIMP, the fitness-fatigue model, and the Coggan CTL/ATL/TSB.
  What they measure, how they relate, and where they break.
tags: [trail, training, load, modelling, heart-rate, TRIMP, FFM, CTL, ATL, TSB, science]
categories: [trail, data, science]
related_posts: false
toc:
  sidebar: left
math: true
---

*"After a run, your watch gives you a 'load' score. On Garmin it is called acute load, on Suunto it is Training Load. Behind these labels, the same model has been running for fifty years: Banister's TRIMP."*

Still driven by the need to understand what my watch computes, I went digging into the literature. This article is the result.

For a French-speaking audience, I highly recommend the work of Cyril Forester on the Courir Mieux podcast, particularly his episode on [training load in trail running](https://courir-mieux.fr/charge-dentrainement-trail).

---

## Why quantify training load?

When you train, you deliberately apply a biological stress.

Too little: no adaptation. Too much: overtraining, injury. The productive zone sits between the two, and it is narrow and variable.

We need an indicator of the **total dose** absorbed, not just duration, not just intensity, but the combination of both.

Two types of load are distinguished in the literature:

**External load** is what you do: distance, elevation, speed. It is independent of your physiological state on the day.

**Internal load** is what your body experiences: cardiac response, blood lactate, hormonal disruption. This is what drives adaptation.

TRIMP targets internal load by relying on heart rate, an accessible proxy for physiological effort.

---

## The Banister TRIMP (1975 / 1991)

The foundational references:

- Banister et al. (1975). *A system model of training for athletic performance*. Aust. J. Sports Med., 7, 57-61.
- Banister (1991). *Modeling Elite Athletic Performance*. In: Physiological Testing of Elite Athletes. Human Kinetics.

Eric Banister proposed quantifying each session with a Training IMPulse:

$$\mathrm{TRIMP} = D \times \Delta\mathrm{HR} \times y$$

The three terms:

**$D$**: session duration in minutes.

**$\Delta\mathrm{HR}$**: fraction of heart rate reserve used:

$$\Delta\mathrm{HR} = \frac{\overline{\mathrm{HR}} - \mathrm{HR_{rest}}}{\mathrm{HR_{max}} - \mathrm{HR_{rest}}}$$

**$y$**: exponential weighting factor, calibrated on the HR-lactate relationship observed during an incremental test:

$$y = a \cdot e^{b \cdot \Delta\mathrm{HR}}$$

Banister (1991, cited in Morton et al., 1990) provides two sets of constants:

| Sex | $a$ | $b$ |
|---|---|---|
| Male | 0.64 | 1.92 |
| Female | 0.86 | 1.67 |

The non-linearity of $y$ is the key insight: going from 60% to 80% of heart rate reserve costs proportionally much more than going from 40% to 60%. One hour at 80% HRR produces far more TRIMP than one hour at 60%, consistent with what is observed on blood lactate.

<div class="plotly-container">
  <iframe src="{{ site.baseurl }}/assets/img/blog/2026-01_trimp_fitness_fatigue/banister_curve.html"
          width="100%" height="350" frameborder="0" scrolling="no">
  </iframe>
</div>

**Numerical example.** Male athlete: HR$_\text{rest}$ = 45 bpm, HR$_\text{max}$ = 190 bpm.

| Session | Duration | Mean HR | $\Delta$HR | $y$ | TRIMP |
|---|---|---|---|---|---|
| Long run | 90 min | 140 bpm | 0.66 | 1.81 | ~107 |
| Threshold | 45 min | 170 bpm | 0.86 | 2.94 | ~113 |

Two very different sessions can produce a similar training load, even though one is twice as long. That is already more informative than counting kilometers.

---

## The Fitness-Fatigue Model (Morton, Fitz-Clarke & Banister, 1990)

The real innovation goes beyond TRIMP as a score. Banister's team proposed a **dynamic model**: each training dose $w(t)$ simultaneously produces two exponentially decaying responses.

The positive component (*fitness*):

$$g(t) = g(t-1)\,e^{-1/\tau_1} + w(t)\left(1 - e^{-1/\tau_1}\right)$$

The negative component (*fatigue*):

$$h(t) = h(t-1)\,e^{-1/\tau_2} + w(t)\left(1 - e^{-1/\tau_2}\right)$$

Predicted performance:

$$\hat{p}(t) = p_0 + k_1\,g(t) - k_2\,h(t)$$

The four parameters have a direct physiological interpretation:

**$\tau_1 \approx 49$-$50$ days** (Morton et al., 1990): how long fitness persists without training.

**$\tau_2 \approx 11$ days**: how long fatigue persists. Much shorter than fitness.

**$k_1 < k_2$**: the immediate amplitude of fatigue exceeds that of fitness. Training degrades before it improves, which matches every runner's experience.

The concept of **peak form** emerges naturally: by reducing $w(t)$ (tapering), fatigue drops fast ($\tau_2 = 11$ days) while fitness persists ($\tau_1 = 49$ days). There exists a moment when predicted performance is maximal.

---

## Coggan's simplification: CTL, ATL, TSB

Andy Coggan simplified the model by removing the gain factors $k_1, k_2$ and reformulating both components as **exponentially weighted moving averages**:

$$\mathrm{CTL}(t) = \mathrm{CTL}(t-1)\,e^{-1/42} + \mathrm{TSS}(t)\left(1 - e^{-1/42}\right)$$

$$\mathrm{ATL}(t) = \mathrm{ATL}(t-1)\,e^{-1/7} + \mathrm{TSS}(t)\left(1 - e^{-1/7}\right)$$

$$\mathrm{TSB}(t) = \mathrm{CTL}(t) - \mathrm{ATL}(t)$$

TSS (*Training Stress Score*) replaces TRIMP: it is normalized so that one hour at threshold power (or pace) equals 100 points.

The three Performance Management Chart curves:

**CTL** (*Chronic Training Load*): baseline fitness, built over 42 days. Rises slowly, drops slowly. The accumulated "fitness capital."

**ATL** (*Acute Training Load*): recent fatigue, over 7 days. Very reactive.

**TSB** (*Training Stress Balance*): "freshness." Positive = rested, negative = loaded.

**Practical thresholds** from Allen & Coggan (2010) and Friel (via TrainingPeaks):

| TSB | Interpretation |
|---|---|
| +15 to +25 | Optimal race form |
| 0 to +15 | Well recovered, ready for hard training |
| -10 to -30 | Productive training zone |
| < -30 | Overload, risk of overreaching |

For an ultra, the goal is to arrive with a slightly positive TSB (form) without sacrificing too much CTL (fitness). **The ideal taper maximizes CTL minus ATL on race day.**

---

## Alternatives: Edwards (1993) and Lucia et al. (2003)

Banister's TRIMP requires the session's mean HR. Two alternatives use a zone-based approach instead.

**Edwards (1993)** splits the session into five 10%-wide zones of HR$_\text{max}$ and weights them linearly:

$$\mathrm{TRIMP_{Edwards}} = \sum_{z=1}^{5} t_z \times c_z$$

with $c_z \in \{1, 2, 3, 4, 5\}$. Simple, but the thresholds (50-60%, 60-70%, etc.) and coefficients are **arbitrary**: there is no physiological evidence that zone 5 represents five times the stress of zone 1.

**Lucia et al. (2003)** anchor the zones on **ventilatory thresholds** (VT1, VT2), which are physiologically grounded:

$$\mathrm{TRIMP_{Lucia}} = t_1 + 2\,t_2 + 3\,t_3$$

An improvement, since VT1 and VT2 delimit distinct metabolic domains. However, the coefficients 1/2/3 remain arbitrary, and determining the thresholds requires a laboratory test, which limits accessibility.

---

## Limits of the model

### Mean HR is blind to intervals

This is the most documented limitation. Two one-hour sessions with the same mean HR produce the same TRIMP. Yet a continuous session at 145 bpm and a session of 6 x 5 min at 175 bpm / 5 min recovery at 115 bpm do not stress the same systems.

Interval sessions generate more lactate disruption, more neuromuscular stress, and more complex VO$_2$ kinetics. Computing TRIMP from instantaneous HR partially corrects this bias, but HR itself has limitations at high intensity (inertia, thermal drift).

### The weighting coefficients are not individualized

The constants $a$ and $b$ were calibrated on a small sample and only distinguish sex. Manzi et al. proposed an **iTRIMP** that adjusts the weighting to each athlete's personal HR-speed relationship, showing better dose-response correlation in running and cycling.

### The nature of effort is not captured

Running 2 hours downhill on technical terrain, with heavy eccentric and neuromuscular stress, can produce a modest TRIMP while the real load is high. Same for short hill reps, strength-endurance work, or heat. External and internal load diverge, and HR alone does not capture everything.

### The time constants are fixed, not individually validated

42 days for CTL, 7 days for ATL: these are defaults, not physiological measurements. Coggan himself notes that they vary with the athlete, age, type of load, and recommends adjusting them on personal history once enough data is available.

### The model does not predict absolute performance

CTL and TSB are **relative** indicators specific to each athlete. A CTL of 80 for a recreational runner represents a very different load than a CTL of 80 for a national-level athlete. The full Banister model, with all four parameters, requires regular field performance measurements to be calibrated, which is rarely done in practice.

---

## Going further: data-driven approaches

The models above have parameters **fixed a priori**. As discussed in [The Minetti Model]({% post_url 2025-11-27-minetti-model %}), the natural next step is to **estimate them on your own data**, by observing how your performance responds to load.

**Parameter fitting by optimization.** If you regularly measure performance on a reproducible field test (fixed-loop time trial, VMA, allure at target HR), it is possible to fit $k_1, k_2, \tau_1, \tau_2$ by minimizing the error between the model and your actual measurements.

**Local estimation of $\tau$.** A prolonged rest period (forced rest, off-season break) is an opportunity: by observing the decay of your CTL or performance over that period, you can fit an exponential and estimate your personal $\tau_1$. If your fitness drops much faster than the standard 42 days, your time constant is shorter, and your taper should be too.

**Non-linear models and ML.** Recent work explores time-varying parameters (Busso, 2003), recurrent networks, and Kalman filters for modeling the load-performance relationship. Promising, but data-hungry. We may come back to this.

---

## References

- Banister EW, Calvert TW, Savage MV, Bach T (1975). *A system model of training for athletic performance*. Aust. J. Sports Med., 7, 57-61.
- Banister EW (1991). *Modeling Elite Athletic Performance*. In: MacDougall JD, Wenger HA, Green HJ (eds.), Physiological Testing of Elite Athletes. Human Kinetics.
- Morton RH, Fitz-Clarke JR, Banister EW (1990). *Modeling human performance in running*. J. Appl. Physiol., 69(3), 1171-1177. [DOI](https://doi.org/10.1152/jappl.1990.69.3.1171)
- Allen H, Coggan A (2010). *Training and Racing with a Power Meter*. VeloPress.
- Coggan A (2023). *The Science of the TrainingPeaks Performance Manager*. [trainingpeaks.com](https://www.trainingpeaks.com/learn/articles/the-science-of-the-performance-manager/)
- Edwards S (1993). *The Heart Rate Monitor Book*. Polar Electro Oy.
- Lucia A, Hoyos J, Santalla A, Earnest C, Chicharro JL (2003). Tour de France versus Vuelta a Espana: which is harder? *Med. Sci. Sports Exerc.*, 35(5), 872-878. [DOI](https://doi.org/10.1249/01.MSS.0000064999.82036.B4)
- Busso T (2003). *Variable dose-response relationship between exercise training and performance*. Med. Sci. Sports Exerc., 35(7), 1188-1195. [DOI](https://doi.org/10.1249/01.MSS.0000074465.13621.37)
- Desgorces FD, Senegas X, Garcia J, Decker L, Noirez P (2007). *Methods to quantify intermittent exercises*. Appl. Physiol. Nutr. Metab., 32(4), 762-769. [DOI](https://doi.org/10.1139/H07-037)

---

> **Disclaimer:** I am a research engineer, not an exercise physiologist. What you read here is the logbook of a curious trail runner who likes understanding his data. Not a lecture. The references are there so you can check for yourself.
