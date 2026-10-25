# RQ 12.1: From seasonal to extreme flooding in South Sudan, and early warning from upstream

Group 22 · JBG060 Capstone Data Challenge · Eindhoven University of Technology & Zero Hunger Lab

## 1. Introduction and overview

**What it does.** The notebook `extreme_flood_early_warning/RQ12_1_full_pipeline.ipynb` is the complete data analysis
pipeline behind our poster and report. Running it reproduces **every result, table and figure**. All the code is
inside the notebook; there are no separate code files.

It answers two questions:
- **SQ1.** Which hydro-climatic factors are associated with the shift from normal seasonal flooding to extreme
  floods in South Sudan, and can upstream water-level anomalies give early indication?
- **SQ3.** Which ADM3 areas (payams) should be prioritised for anticipatory humanitarian action when flood hazard,
  exposure and vulnerability are combined?

**Main features**
- **Data cleaning and EDA:**
  - an inventory of all sources,
  - every cleaning decision with the evidence from the data,
  - coverage, flood-cycle, threshold, correlation and data-quality figures.
- **Forecasting:** a GRU recurrent neural network predicts the probability of an "extreme flood week" 2–26 weeks
  ahead. It is benchmarked against climatology, persistence, logistic regression, gradient boosting and an
  ensemble, with leave-one-year-out and forward (train ≤ 2018) validation.
- **Robustness checks:**
  - year-block bootstrap confidence intervals,
  - a surrogate test,
  - a spatial negative control (Sudd vs west/east),
  - alternative target definitions,
  - a physically constrained model.
- **Longer record from open data:**
  - Landsat water maps 1984–2021,
  - Lake Victoria/Kyoga levels 1992–2026,
  - the Indian Ocean Dipole,
  - documented flood events (GLIDE, Dartmouth Flood Observatory).
- **One interpretable model per question:** odds ratios (factors) and early-warning skill by lead time.
- **SQ3 priority index:** the geometric mean of hazard, exposure and vulnerability for 512 payams, with the map.
- **Communication:** the poster figures and an animation of the floods and lake level, 2003–2025.

**Code organisation (inside the notebook)**

| Section | Content |
|---|---|
| 0a / 0b | package installation; settings, input checks, helpers |
| 1 | download of the open external data |
| 2 | dataset construction from the raw data |
| 2a / 2b | shared setup (features, target, models); **data cleaning and EDA** |
| 3 | GRU and benchmark training and evaluation |
| 4–8 | physical model, robustness, longer record, documented events, one model per question |
| 9–10 | animation, SQ1 poster figure, SQ3 priority map |
| 11 / 12 | answers (every number read from the result files) and a check that every output exists |

Each step's code sits in its own `%%module` cell (a separate namespace, so steps cannot overwrite each other's
variables). It is followed by a cell that runs it and shows its results.

**Folder structure**
```text
raw_data/                                 the course data folder (not included in this zip)
├── README.md                             this file
├── requirements.txt                      exact package versions
├── flood_masks/, rainfall and runoff/, evapotranspiration/, Water levels lakes/,
│   Darthmouth Flood Observatory/, Administrative boundaries/, farmland/   INPUT: course raw data
├── flood_driver_analysis/data/, QA_results/      INPUT: earlier team outputs (payams, county IPC)
├── external/                             INPUT: open external data (included; re-downloaded by step 1 if missing)
└── extreme_flood_early_warning/
    ├── RQ12_1_full_pipeline.ipynb        THE PIPELINE (all code)
    ├── data/                             intermediate data written by the notebook (regional series, caches)
    └── outputs/
        ├── tables/                       all result tables (.csv/.parquet)
        ├── figures/                      all report figures (eda_01–05, 01–17)
        ├── poster/                       poster figures (SQ1, SQ3)
        ├── models/                       trained GRU and tabular models
        └── animation/                    MP4 of floods and Lake Albert, 2003–2025
```

## 2. Installation

**Requirements**
- **Python 3.12.** Tested with 3.12.4 on macOS (Intel).
- The packages in `requirements.txt` (exact versions).
- About 2 GB of disk space.
- An internet connection, only for step 1 and step 6 the first time. No accounts are needed.
- **Optional:** `ffmpeg` for the MP4 animation (`brew install ffmpeg` / `apt install ffmpeg`). Without it, the video is
  skipped and everything else runs.

**Steps**
1. Unzip this file **into the course `raw_data` folder**, so you get `raw_data/README.md`,
   `raw_data/requirements.txt`, `raw_data/external/` and `raw_data/extreme_flood_early_warning/`.
2. Check that the raw-data sub-folders listed above exist (the course data from SURFdrive). The notebook stops
   with a clear message if any are missing.
3. Create an environment and install everything at once:
   ```bash
   cd raw_data
   python3.12 -m venv .venv-rq121
   source .venv-rq121/bin/activate          # Windows: .venv-rq121\Scripts\activate
   pip install -r requirements.txt
   ```
   Alternatively, open the notebook, set `INSTALL = True` in cell 0a, run that cell and restart the kernel.

## 3. Usage

```bash
cd raw_data/extreme_flood_early_warning
jupyter lab RQ12_1_full_pipeline.ipynb      # then: Kernel → Restart Kernel and Run All Cells
```
Or run it without opening Jupyter:
```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 RQ12_1_full_pipeline.ipynb
```

**Settings (cell 0b)**

| Setting | Default | Effect |
|---|---|---|
| `FORCE_RECOMPUTE` | `False` | `True` recomputes every step, even if its results exist |
| `MAKE_VIDEO` | `True` | `False` skips the MP4 animation |

**What to expect**
- **With the results included in this zip,** every step is skipped and the notebook only shows the results. That
  takes about 1 minute.
- **A full recomputation** (`FORCE_RECOMPUTE = True`) takes about 2 hours: ~10 min dataset, ~45 min GRU, ~35 min
  first Landsat read, the rest a few minutes.
- **The last two cells** print the answers to SQ1 and SQ3 and a ✓/✗ checklist of all expected outputs.

**Example outputs**
- `outputs/tables/final_q1_odds_ratios.csv`: odds ratio per driver with 95% CI.
- `outputs/tables/sq3_adm3_priority_index.csv`: priority score and class for all 512 payams.
- `outputs/figures/eda_01_data_coverage.png`: data coverage per source.

## 4. Troubleshooting and FAQ

- **"missing input data" assertion in cell 0b.** A course raw-data folder is not where the notebook expects it.
  Make sure the notebook's folder sits directly inside `raw_data/`.
- **The kernel crashes (segmentation fault) during model training.** PyTorch and scikit-learn use different OpenMP
  libraries. Cell 0b sets `OMP_NUM_THREADS=1` *before* any import. Restart the kernel and run from the top.
- **`pyarrow requires NumPy 2.0` or torch/numpy import errors.** Install the exact versions in `requirements.txt`.
  PyTorch 2.2.2 needs numpy < 2, and newer pyarrow needs numpy ≥ 2.
- **Step 6 is slow or stops with an HTTP error.** It reads Landsat maps from the JRC server. Re-run the cell:
  finished years are cached in `data/_jrc/`.
- **No video.** `ffmpeg` is not installed. Everything else still runs.
- **Results differ slightly after a full recomputation.** GRU training and bootstraps use fixed seeds, but tiny
  differences between hardware and library builds are possible. Conclusions do not depend on them.
- **Windows.** All loops are plain Python (no multiprocessing), so the notebook runs on Windows too, but it was
  tested on macOS.

## 5. Credits and acknowledgements

**Data**
- NASA LANCE MODIS/VIIRS flood product (MCDWD).
- ECMWF ERA5 and AgERA5 (Copernicus Climate Change Service).
- DAHITI lake altimetry (DGFI-TUM).
- Hydroweb, and NASA GSFC G-REALM (<https://earth.gsfc.nasa.gov/gwm/lake/Index>).
- Dartmouth Flood Observatory.
- JRC Global Surface Water (Pekel et al., 2016; <https://global-surface-water.appspot.com>).
- NOAA PSL Indian Ocean Dipole and Niño 3.4 indices.
- GLIDE and DFO event lists via the Humanitarian Data Exchange (<https://data.humdata.org>).
- OCHA/HDX administrative boundaries, ICPAC cattle density, IPC food-security analyses, WorldPop.
- **Poster font:** Source Sans 3 (Adobe, SIL Open Font License).

**Libraries:** NumPy, pandas, xarray, netCDF4, Dask, rasterio, GeoPandas, Shapely, PyArrow, SciPy, scikit-learn,
PyTorch, Matplotlib, Jupyter.

**Code:** written by the group, with the help of an AI assistant (see the AI-use statement in the report). The
raw-data loaders follow the course repository `JBG060-2026` (`processing_data/loading.py`).

**Literature for the 1960s case:**
- Sutcliffe, J. V., & Brown, E. (2018). Water losses from the Sudd. *Hydrological Sciences Journal*, 63(4), 527–541.
- Sutcliffe, J. V., & Parks, Y. P. (1999). *The Hydrology of the Nile*. IAHS.

## 6. Legal and ethical considerations

- **For education and research only.** The pipeline is **not validated for operational humanitarian decisions**.
  Its outputs should support, not replace, human judgement and local knowledge.
- **Respect each data provider's licence and attribution terms.** The raw course data is not redistributed in this
  zip.
- **No personal data is processed.** All data is aggregated, at the level of pixels, payams or counties.
- **Known biases and limitations:**
  - **Satellite flood maps miss water under vegetation and cloud,** so flooding in the Sudd is underestimated.
  - **"Unusual" is a classification, not damage.**
  - **The lake–flood link rests on one large event in the satellite era,** plus historical and documented events.
    It is an association, not proven causation.
  - **Food insecurity is only available per county,** so all payams in a county share one value.
  - **Equal weights for hazard, exposure and vulnerability in SQ3 are a value choice.**

  Low data confidence in poorly measured areas must not be read as low need.
