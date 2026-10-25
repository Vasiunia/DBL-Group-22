# Slides — RQ 12.1: From seasonal to extreme flooding in South Sudan

Figures are in `outputs/figures/`. Each slide lists its figure under **Graph**.
Wording rule: "associated with" and "precedes", never "drives" or "causes", unless a causal test supports it.

---

## Slide 1 — Research question
**Title:** When does normal flooding become extreme, and can we see it coming from upstream?

- South Sudan floods every year, which is normal and even needed for crops and cattle
- Since 2020 the floods have not receded: the Sudd has been flooded on a record scale for years
- **RQ 12.1:** Which hydro-climatic factors are associated with the transition from seasonal to extreme flooding,
  and can upstream water-level anomalies give early indication?

**Graph:** none, or a simple map of the Nile from Lake Victoria → Lake Albert → Sudd → Malakal

---

## Slide 2 — Data
| Source | What we use it for |
|---|---|
| NASA MODIS/VIIRS flood masks (2003–2025) | Target: flood extent in South Sudan, split into *recurring* and *unusual* |
| Lake Albert altimetry (DAHITI) | Upstream water level |
| ERA5 rainfall & runoff | Lakes basin, Ethiopian highlands (Sobat), South Sudan |
| AgERA5 evapotranspiration | Local water balance (rain − ET) |
| DFO discharge (Malakal) | Local river flow |

- Lake Victoria and Kyoga data stop in 2002, so **Lake Albert** is our only upstream lake gauge

**Graph:** none (table only)

---

## Slide 3 — Methodology: causal assumptions first
Before modelling, we wrote down how we think the system works:

```
Lakes-basin rain ──► Lake Albert level ──► White Nile ──► Sudd floods
Ethiopian rain ────► Sobat river ─────────────────────► Eastern / Upper Nile floods
Regional climate ──► Local South Sudan rain ──────────► Floods everywhere
```

- **If Lake Albert matters, three things must hold:**
  1. The signal appears **only downstream** of the lake (Sudd), not in the west or east
  2. The extremes are **not explained by local rain**
  3. The results **do not depend on how "extreme" is defined**
- We test all three (slide 9)

**Graph:** draw this diagram

---

## Slide 4 — Methodology: models and testing
- **Target:** an *extreme week* is one where unusual flood extent exceeds the 90th percentile of 2003–2019 (408 km²)
- **Drivers:** weekly anomalies relative to the normal value for that week of the year, summed over 1–52 weeks.
  Only past information is used
- **Models:** a GRU (recurrent neural network) over 52 weeks of history predicts P(extreme) 2–26 weeks ahead.
  It is compared with climatology, persistence ("still flooded"), logistic regression, gradient boosting and an ensemble
- **Validation:** leave-one-year-out, plus a forward test (train 2003–2018, test 2019–2025)
- **Uncertainty:** 95% confidence intervals from a bootstrap that resamples whole years,
  because weeks within a year are not independent

**Graph:** none

---

## Slide 5 — What happened: a regime shift
- Before 2019, unusual flooding appeared only in short seasonal peaks
- From 2020, unusual flooding is **about 50× larger and lasts year-round**
- Lake Albert rose **about 3 m above normal** at the same time
- The growth is concentrated in the **Sudd**, downstream of the lake. The west and east keep pulsing as before

**Graph:** video `outputs/animation/south_sudan_floods_lake_albert_2003_2025.mp4`
(Insert → Video → This Device; set Playback → Start: Automatically). Fallback still: `frame_2023-09.png`.
Static backup: `01_flood_regime_and_upstream_timeline.png`

*Say while it plays:* "Watch the gauge on the right and the black line at the bottom. Until 2019 the map pulses
with the seasons. In 2020 the lake rises 3 metres, and the Sudd turns orange and stays orange, while the west and
east do not."

---

## Slide 6 — Which factors? Two regimes
- **2003–2019:** extremes are seasonal peaks after wet periods in the Ethiopian highlands
  - Sobat rain predicts an extreme 13 weeks later with **AUC 0.73 (CI 0.60–0.83)**
  - Lake Albert shows **no positive signal** (0.35, CI 0.22–0.48)
- **2020–2025:** extremes are year-round and co-occur with a high Lake Albert,
  while local rain was **below normal** (−2.6σ, CI −3.4 to −1.3)

**Graph:** `08_single_driver_early_warning_skill.png`
*(backup: `02_lagged_driver_correlations.png`)*

---

## Slide 7 — Can we predict extremes?
- The ensemble reaches **AUC 0.96 at 2 weeks and 0.91 at 26 weeks** (leave-one-year-out)
- At short leads it is no better than persistence, but at **13 and 26 weeks it clearly beats persistence**
  (Brier skill +0.16 and +0.27, CIs exclude zero)
- The GRU alone is not better than simpler models. The ensemble is the most reliable

**Graph:** `10_model_skill_uncertainty.png` (or `04a_skill_by_lead_loyo_models.png`)

---

## Slide 8 — Honest check: does upstream data add skill?
- With or without upstream inputs, the GRU's skill differs by **−0.09 to +0.07, and every CI includes zero**
- A simple **time trend** separates extreme weeks as well as Lake Albert does (AUC 0.84 vs 0.82)
- The reason is that there was **one** lake-driven event (2020–25). Statistically that is n = 1
- Before an extreme onset, the drivers move in the expected direction, but the **CIs include zero**

**Graph:** `11_surrogate_test_lake_albert.png`
*(backup: `03b_onset_composites_ci.png`)*

---

## Slide 9 — Causal tests: is the Lake Albert link real?
1. **Spatial control:** unusual flooding grew **37× in the Sudd** (downstream of Lake Albert),
   **3× in the west** and **0.8× in the east** (neither downstream)
2. **Local-rain alternative ruled out:** the 2020s extremes happened in locally **dry** conditions
3. **Not a labelling artefact:** using total flood extent or same-month thresholds gives the same result
   (Lake Albert AUC 0.75–0.79)
- **Conclusion:** the pattern is specific to the White Nile corridor, as the causal diagram predicts.
  This supports the Lake Albert → White Nile → Sudd route, but it is not proof

**Graph:** `12_spatial_negative_control.png`
*(backup: `13_target_sensitivity.png`)*

---

## Slide 10 — Could we have predicted 2020?
- Models trained on 2003–2018 **missed the 2022–25 extremes**
- A simple physical model with a positive Lake Albert effect **also missed them**:
  the free model even learned a *negative* effect (the lake was high in 2003–05 without extremes)
- **The limit is the data, not the model:** the past contained no lake-driven extreme
- A threshold of **Lake Albert > +1 m** fired 13 weeks before the 2020 regime, and only twice in 2003–2018.
  We chose it *after* seeing 2020, so it is a hypothesis to monitor, not a validated rule

**Graph:** `06_forward_test_2019_2025.png` and `09_forward_test_physical_logit.png` side by side

---

## Slide 11 — SLE: Social, Legal, Ethical
**Social**
- Floods hit livelihoods: crops, cattle and access to health facilities.
  Weeks of warning let humanitarian actors prepare
- Flood extent measures area, not people: it says nothing about who is affected or how vulnerable they are

**Legal**
- Respect the data licences and attribution rules (Copernicus, NASA, DAHITI, DFO)
- The Nile is transboundary: upstream data and lake releases (Uganda) are outside South Sudan's control

**Ethical**
- False alarms erode trust, and missed events cost lives. Report probabilities with confidence intervals
- Our upstream evidence rests on **one event**, so overclaiming could misdirect resources.
  The model is **not validated for operational use**
- "Unusual" is a satellite classification, not a measure of damage. No personal data is used

**Graph:** none (or icons)

---

## Slide 12 — Conclusion
- **Factors (robust):** before 2019, extremes followed Ethiopian-highland and local rain.
  Since 2020, persistent extremes are confined to the White Nile corridor and co-occur with a high Lake Albert,
  not with local rain
- **Early warning (plausible, not proven):** Lake Albert is the best candidate indicator.
  It is on the physical route, spatially specific and robust to the target definition,
  but the evidence rests on a single event
- **Prediction:** the ensemble adds real skill 13–26 weeks ahead, but no model can foresee a regime it has never seen
- **Next step:** monitor Lake Albert (and Lake Victoria releases) as a hypothesis-driven trigger,
  and validate on future events

**Graph:** none, or a small crop of fig 01

---

### Backup slides (for questions)
- `05_gru_feature_group_importance.png`: what the GRU relies on. It understates Lake Albert because the
  flood-state features carry overlapping information
- `04b_skill_by_lead_loyo_ablations.png`: skill of each GRU variant
- `13_target_sensitivity.png`: results under other definitions of "extreme"
- Mediation check: lake-basin runoff is **not** associated with Sudd floods (r ≈ 0), while Lake Albert is
  (partial r 0.54, CI 0.23–0.72, after controlling for runoff). So the lake carries information beyond recent
  rainfall, probably multi-year storage and outflow
