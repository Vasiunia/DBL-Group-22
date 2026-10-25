"""
Research question 12.1 — data cleaning and exploratory data analysis (EDA).

Runs after build_dataset.py (and download_external.py for the long lake records). Documents every source,
every cleaning decision with the number that justifies it, and the basic patterns in the data.

Outputs
  outputs/tables/eda_data_inventory.csv      one row per variable: period, observations, % missing 2003-2025
  outputs/tables/eda_cleaning_decisions.csv  what was done, why, and the evidence from the data
  outputs/figures/eda_01_data_coverage.png   which source covers which years
  outputs/figures/eda_02_flood_seasonality.png  yearly flood cycle, 2003-19 vs 2020-25
  outputs/figures/eda_03_extreme_threshold.png  distribution of weekly unusual flood area and the 408 km² threshold
  outputs/figures/eda_04_driver_correlations.png correlations between the weekly driver anomalies
  outputs/figures/eda_05_data_quality.png    flood-mask observation days, altimetry gaps, Malakal discharge artefact

Run from the raw_data folder:  python extreme_flood_early_warning/eda.py
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import train_models as T

matplotlib.use("Agg")
C = T.C
OUT_T, OUT_F = T.OUT_T, T.OUT_F
EXT = T.HERE.parent / "external"

VARIABLES = {  # column in daily_series.parquet -> (source, description)
    "unusual_px_ssd": ("NASA MODIS/VIIRS flood masks", "unusual flood pixels, South Sudan"),
    "recurring_px_ssd": ("NASA MODIS/VIIRS flood masks", "recurring (normal) flood pixels, South Sudan"),
    "lake_albert": ("DAHITI altimetry", "Lake Albert water level"),
    "lake_victoria": ("Hydroweb altimetry (old file)", "Lake Victoria water level"),
    "lake_kyoga": ("Hydroweb altimetry (old file)", "Lake Kyoga water level"),
    "q_malakal": ("Dartmouth Flood Observatory", "White Nile discharge at Malakal (station 100205)"),
    "q_ethiopia": ("Dartmouth Flood Observatory", "SW Ethiopia discharge (station 1505, Omo-Turkana basin)"),
    "tp_ssd": ("ERA5", "rainfall, South Sudan mean"),
    "tp_lakes": ("ERA5", "rainfall, equatorial-lakes basin"),
    "tp_sobat": ("ERA5", "rainfall, Ethiopian highlands / Sobat"),
    "ro_ssd": ("ERA5", "runoff, South Sudan mean"),
    "et_ssd": ("AgERA5", "reference evapotranspiration, South Sudan mean"),
}


def load():
    d = pd.read_parquet(T.HERE / "data" / "daily_series.parquet")
    d.index = pd.to_datetime(d.index)
    raw_flood = pd.read_parquet(T.HERE / "data" / "_flood.parquet")      # days with at least one detection
    raw_flood.index = pd.to_datetime(raw_flood.index)
    grealm = None
    p = T.HERE / "data" / "external_lakes_grealm.parquet"
    if p.exists():
        grealm = pd.read_parquet(p)
        grealm.index = pd.to_datetime(grealm.index)
    return d, raw_flood, grealm


def inventory(d, raw_flood, grealm):
    win = (d.index >= T.START) & (d.index <= T.END)
    rows = []
    for col, (src, desc) in VARIABLES.items():
        s = d[col]
        if col.endswith("_px_ssd"):        # zero-filled in build_dataset; real observations = days with detections
            obs = raw_flood.index[(raw_flood.index >= "2000-01-01") & (raw_flood.index <= "2025-12-31")]
            first, last, n = obs.min(), obs.max(), len(obs)
            miss = 1 - raw_flood[(raw_flood.index >= T.START) & (raw_flood.index <= T.END)].shape[0] / win.sum()
        else:
            o = s.dropna()
            first, last, n = (o.index.min(), o.index.max(), len(o)) if len(o) else (pd.NaT, pd.NaT, 0)
            miss = 1 - s[win].notna().mean()
        rows.append(dict(variable=col, source=src, description=desc, first=first.date() if pd.notna(first) else None,
                         last=last.date() if pd.notna(last) else None, observations=n,
                         pct_days_missing_2003_2025=round(100 * miss, 1)))
    if grealm is not None:
        for c in grealm:
            o = grealm[c].dropna()
            rows.append(dict(variable=f"{c} (G-REALM)", source="NASA G-REALM altimetry (external)",
                             description=f"Lake {c.capitalize()} water level, 10-day", first=o.index.min().date(),
                             last=o.index.max().date(), observations=len(o),
                             pct_days_missing_2003_2025=round(100 * (1 - len(o[(o.index >= T.START) & (o.index <= T.END)]) * 10
                                                              / win.sum()), 1)))
    inv = pd.DataFrame(rows)
    inv.to_csv(OUT_T / "eda_data_inventory.csv", index=False)
    return inv


def cleaning(d, raw_flood, w):
    days = raw_flood.groupby(raw_flood.index.year).size()
    alb = d["lake_albert"].dropna()
    gaps = alb.index.to_series().diff().dt.days.dropna()
    q = d["q_malakal"].groupby(d.index.year).mean()
    pre = d.loc["2003":"2019"]
    flood_both = ((d["unusual_px_ssd"] > 0) & (d["recurring_px_ssd"] > 0)).mean()
    rows = [
        ("Study period starts in 2003",
         "Flood masks before 2003 have too few observation days (MODIS Aqua launched mid-2002).",
         f"days with detections: 2000 = {days.get(2000, 0)}, 2001 = {days.get(2001, 0)}, 2002 = {days.get(2002, 0)}, "
         f"2003–2025 median = {int(days.loc[2003:2025].median())}"),
        ("Daily flood counts aggregated to the weekly maximum",
         "Single days are often cloud-covered; the maximum of the 3-day composites in a week is robust to gaps.",
         f"days without any detection in 2003–2025: {100 - 100 * len(raw_flood.loc['2003':'2025']) / len(pd.date_range('2003-01-01', '2025-12-31')):.0f}%"),
        ("Pixel counts converted to km²", "0.0020833° pixels at ~7°N.", f"1 pixel = {T.PIXEL_KM2} km²"),
        ("Clipped to South Sudan", "Flood-mask tiles also cover Sudan, Ethiopia and CAR.", "point-in-polygon on the admin-0 boundary"),
        ("Old Lake Victoria / Kyoga files not used after 2002",
         "The course files stop in 2002; replaced by NASA G-REALM (1992–2026) in the extended analysis.",
         f"last Victoria observation in course file: {d['lake_victoria'].dropna().index.max().date()}"),
        ("Lake levels: weekly median, gaps filled forward only (max 26 weeks)",
         "Altimetry passes are irregular; forward filling never uses future values (no look-ahead).",
         f"Lake Albert: {len(alb)} observations, median gap {gaps.median():.0f} days, longest gap {gaps.max():.0f} days"),
        ("Malakal discharge kept only as a local feature, flagged",
         "Satellite-derived discharge drops during the biggest floods (saturation artefact).",
         f"yearly mean m³/s: 2019 = {q.get(2019):.0f}, 2022 = {q.get(2022):.0f}, 2023 = {q.get(2023):.0f} "
         f"(while flooding peaked)"),
        ("DFO station 1505 not counted as 'upstream'",
         "It lies at 4.9°N 36.0°E in the Omo–Turkana basin, which does not drain to the Nile.", "station metadata"),
        ("ERA5 hourly accumulations summed to daily totals", "ERA5 'tp' and 'ro' are hourly accumulations in metres.",
         f"South Sudan mean rainfall {pre['tp_ssd'].mean() * 365:.0f} mm/yr (2003–19), plausible"),
        ("Anomalies relative to the week-of-year climatology of 2003–2019",
         "Removes the seasonal cycle; 2003–2019 is the 'normal regime' reference.", "±2-week smoothed climatology"),
        ("Extreme week = unusual area above the 2003–2019 90th percentile", "Defines 'extreme' relative to the normal regime.",
         f"threshold = {w['unusual_km2_ssd'][(w.index.year >= 2003) & (w.index.year <= 2019)].quantile(0.9):.0f} km²"),
        ("Recurring and unusual pixels counted separately", "A pixel is classified once per day by the product.",
         f"days with both types present: {100 * flood_both:.0f}%"),
    ]
    out = pd.DataFrame(rows, columns=["decision", "why", "evidence_from_data"])
    out.to_csv(OUT_T / "eda_cleaning_decisions.csv", index=False)
    return out


def fig_coverage(d, raw_flood, grealm):
    series = [("Flood masks (MODIS/VIIRS)", raw_flood.index),
              ("ERA5 rain & runoff", d["tp_ssd"].dropna().index),
              ("AgERA5 evapotranspiration", d["et_ssd"].dropna().index),
              ("DFO discharge", d["q_malakal"].dropna().index),
              ("Lake Albert (DAHITI)", d["lake_albert"].dropna().index),
              ("Lake Victoria (course file)", d["lake_victoria"].dropna().index)]
    if grealm is not None:
        series.append(("Lake Victoria (G-REALM)", grealm["victoria"].dropna().index))
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, (lab, idx) in enumerate(series):
        yrs = pd.Series(1, index=idx).groupby(idx.year).size()
        for y, n in yrs.items():   # shade = how complete that year is
            ax.barh(i, 1, left=y - 0.5, height=0.6, color=C["blue"], alpha=float(np.clip(n / yrs.max(), 0.25, 1)),
                    edgecolor="none")
    ax.axvspan(T.REF[0] - 0.5, 2025.5, color=C["aqua"], alpha=0.08, lw=0)
    ax.set_yticks(range(len(series)), [s[0] for s in series])
    ax.invert_yaxis()
    ax.set_xlim(1991.5, 2026.5)
    ax.set_title("Data coverage per source (shade = observations per year; band = study period 2003–2025)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT_F / "eda_01_data_coverage.png", dpi=160)
    plt.close(fig)


def fig_seasonality(w):
    woy = w.index.isocalendar().week.clip(upper=52)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, (lab, (a, b), c) in zip(axes, [("2003–2019 (normal regime)", (2003, 2019), C["blue"]),
                                           ("2020–2025", (2020, 2025), C["orange"])]):
        m = (w.index.year >= a) & (w.index.year <= b)
        for col, cc, name in [("recurring_km2_ssd", C["blue"], "recurring (normal)"), ("unusual_km2_ssd", C["orange"], "unusual")]:
            g = w.loc[m, col].groupby(woy[m])
            ax.plot(g.median().index, g.median(), color=cc, label=f"{name}, median")
            ax.fill_between(g.median().index, g.quantile(0.25), g.quantile(0.75), color=cc, alpha=0.15, lw=0)
        ax.set_title(lab)
        ax.set_xlabel("Week of year")
        ax.set_yscale("log")
    axes[0].set_ylabel("Weekly flood area, km² (log)")
    axes[0].legend(fontsize=9)
    fig.suptitle("Yearly flood cycle: median and interquartile range", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_F / "eda_02_flood_seasonality.png", dpi=160)
    plt.close(fig)


def fig_threshold(w, thr):
    m = (w.index >= T.START) & (w.index <= T.END)
    v = w.loc[m, "unusual_km2_ssd"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    bins = np.logspace(-1, np.log10(v.max() + 1), 50)
    ref = (w.index.year <= 2019)[m]
    ax.hist(v[ref] + 0.1, bins=bins, color=C["blue"], alpha=0.7, label="2003–2019")
    ax.hist(v[~ref] + 0.1, bins=bins, color=C["orange"], alpha=0.6, label="2020–2025")
    ax.axvline(thr, color=C["ink"], ls="--", lw=1.5)
    ax.text(thr * 1.1, ax.get_ylim()[1] * 0.9, f"extreme threshold\n{thr:.0f} km²", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Weekly unusual flood area, km² (log)")
    ax.set_ylabel("Number of weeks")
    ax.legend(fontsize=9)
    ax.set_title("Distribution of weekly unusual flood area")
    ax = axes[1]
    share = w.loc[m, "extreme"].groupby(w.index[m].year).mean()
    ax.bar(share.index, share.values, color=[C["orange"] if y >= 2020 else C["blue"] for y in share.index])
    ax.set_ylabel("Share of extreme weeks")
    ax.set_title("Extreme weeks per year")
    fig.tight_layout()
    fig.savefig(OUT_F / "eda_03_extreme_threshold.png", dpi=160)
    plt.close(fig)


def fig_correlations(w, f):
    cols = {"albert_anom": "Lake Albert level", "albert_d26": "Lake Albert 26-wk rise", "tp_lakes_a52": "Lake-basin rain 52 wk",
            "ro_lakes_a52": "Lake-basin runoff 52 wk", "tp_sobat_a13": "Sobat rain 13 wk", "q_ethiopia_anom": "SW Ethiopia discharge",
            "tp_ssd_a4": "SSD rain 4 wk", "tp_ssd_a26": "SSD rain 26 wk", "ro_ssd_a13": "SSD runoff 13 wk",
            "wb_ssd_a26": "SSD rain − ET 26 wk", "et_ssd_a4": "SSD ET 4 wk", "q_malakal_anom": "Malakal discharge",
            "log_unusual": "log unusual flood area"}
    m = (f.index >= T.START) & (f.index <= T.END)
    cm = f.loc[m, list(cols)].rename(columns=cols).corr(method="spearman")
    cm.to_csv(OUT_T / "eda_driver_correlations.csv")
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", [C["blue"], "#f0efec", C["red"]])
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(cm.values, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cm)), cm.columns, rotation=55, ha="right", fontsize=9)
    ax.set_yticks(range(len(cm)), cm.index, fontsize=9)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, f"{cm.values[i, j]:.1f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(cm.values[i, j]) > 0.6 else C["ink"])
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman r (weekly, 2003–2025)")
    ax.set_title("Correlations between driver anomalies (near-duplicates are not used together in Q1)")
    fig.tight_layout()
    fig.savefig(OUT_F / "eda_04_driver_correlations.png", dpi=160)
    plt.close(fig)


def fig_quality(d, raw_flood, w):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax = axes[0]
    days = raw_flood.groupby(raw_flood.index.year).size().loc[2000:2025]
    ax.bar(days.index, days.values, color=[C["muted"] if y < 2003 else C["blue"] for y in days.index])
    ax.set_title("Flood masks: days with detections")
    ax.set_ylabel("Days per year")
    ax = axes[1]
    gaps = d["lake_albert"].dropna().index.to_series().diff().dt.days.dropna()
    ax.hist(gaps.clip(upper=120), bins=30, color=C["violet"])
    ax.set_title("Lake Albert altimetry: days between passes")
    ax.set_xlabel("Gap (days, capped at 120)")
    ax = axes[2]
    m = (w.index >= "2015-01-01") & (w.index <= T.END)
    ax.plot(w.index[m], w["q_malakal"][m], color=C["ink2"], lw=1, label="Malakal discharge (m³/s)")
    ax.set_ylabel("m³/s")
    hi = w["unusual_km2_sudd"][m] > w["unusual_km2_sudd"][m].quantile(0.75)
    for t in w.index[m][hi]:
        ax.axvspan(t, t + pd.Timedelta(days=7), color=C["orange"], alpha=0.15, lw=0)
    ax.set_title("Malakal discharge falls while the Sudd floods\n(orange = top-25% Sudd flood weeks)")
    fig.tight_layout()
    fig.savefig(OUT_F / "eda_05_data_quality.png", dpi=160)
    plt.close(fig)


def main():
    d, raw_flood, grealm = load()
    w, f, thr = T.build_weekly()
    pd.set_option("display.width", 220)
    inv = inventory(d, raw_flood, grealm)
    print(inv.to_string(index=False))
    cl = cleaning(d, raw_flood, w)
    print("\n" + cl.to_string(index=False))
    fig_coverage(d, raw_flood, grealm)
    fig_seasonality(w)
    fig_threshold(w, thr)
    fig_correlations(w, f)
    fig_quality(d, raw_flood, w)
    print("\nEDA figures and tables written")


if __name__ == "__main__":
    main()
