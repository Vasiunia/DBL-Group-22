"""
Sub-question 3: integrate flood hazard, exposure and vulnerability at ADM3 (payam) level to identify areas to
prioritise for anticipatory humanitarian action.

  Hazard        share of the payam flooded at the yearly peak, mean 2021-2025 (MODIS/VIIRS flood masks),
                log-scaled so that 30 % or more flooded scores 1
  Exposure      people (population_est, team's payam table) and cattle (ICPAC cattle raster)
  Vulnerability share of people in IPC Phase 3+ (county level, latest analysis)
  Priority      geometric mean of the three percentile scores (INFORM-style: high only when all three are high)

Outputs: outputs/tables/sq3_adm3_priority_index.csv, outputs/poster/sq3_priority_map.png (+ transparent)
Needs the flood grid cache from make_animation.py. Run from the raw_data folder with a Python that has rasterio:
  .venv/bin/python extreme_flood_early_warning/poster_sq3.py
"""
import glob
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib import font_manager as fm
from matplotlib.patches import Patch
from rasterio.mask import mask
from shapely.geometry import Point

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OUT_T = HERE / "outputs" / "tables"
OUT_P = HERE / "outputs" / "poster"
CELL, LON0, LAT0 = 0.04, 23.9, 3.3

for f in glob.glob(str(BASE / "external" / "fonts" / "ss3" / "OTF" / "SourceSans3-*.otf")):
    fm.fontManager.addfont(f)

NAVY = "#13294b"
CLASSES = [("Highest priority (top 10%)", NAVY), ("High (next 15%)", "#2a7f8f"),
           ("Moderate (next 25%)", "#9cc9cf"), ("Lower (bottom 50%)", "#eef2f4")]


def hazard(payams):
    z = np.load(HERE / "data" / "_flood_grid_monthly_frac.npz")
    tot = np.clip(z["unusual"].astype(float) + z["recurring"].astype(float), 0, 1)
    yrs = [tot[(y - 2003) * 12:(y - 2003 + 1) * 12].max(axis=0) for y in range(2021, 2026)]
    h = np.mean(yrs, axis=0)
    ny, nx = h.shape
    xc = LON0 + (np.arange(nx) + 0.5) * CELL
    yc = LAT0 + (np.arange(ny) + 0.5) * CELL
    X, Y = np.meshgrid(xc, yc)
    pts = gpd.GeoDataFrame({"h": h.ravel()}, geometry=gpd.points_from_xy(X.ravel(), Y.ravel()), crs="EPSG:4326")
    j = gpd.sjoin(pts, payams[["adm3_pcode", "geometry"]], predicate="within")
    out = j.groupby("adm3_pcode")["h"].mean()
    # payams smaller than a grid cell: nearest cell to the payam's representative point
    for _, r in payams[~payams.adm3_pcode.isin(out.index)].iterrows():
        p = r.geometry.representative_point()
        out[r.adm3_pcode] = h[int((p.y - LAT0) / CELL), int((p.x - LON0) / CELL)]
    return out


def cattle(payams):
    vals = {}
    with rasterio.open(BASE / "farmland" / "geonode__cattle_gha.tif") as src:
        for _, r in payams.iterrows():
            try:
                a, _ = mask(src, [r.geometry], crop=True, all_touched=True, filled=False)
                v = float(np.nansum(np.clip(a.compressed(), 0, None)))
            except ValueError:
                v = 0.0
            vals[r.adm3_pcode] = v
    return pd.Series(vals)


def build():
    payams = gpd.read_file(BASE / "flood_driver_analysis" / "data" / "payam_flood_predictions.geojson")
    payams = payams[["adm3_pcode", "adm3_name", "adm2_name", "adm1_name", "geometry"]]
    summ = pd.read_csv(BASE / "flood_driver_analysis" / "data" / "payam_flood_predictions_annual_summary.csv")
    ipc = pd.read_csv(BASE / "QA_results" / "county_ipc_flood_analysis.csv")[["adm2_name", "p3plus_pct_curr"]]
    df = payams.merge(summ[["adm3_pcode", "population_est"]], on="adm3_pcode").merge(ipc, on="adm2_name", how="left")
    df["hazard_flooded_share"] = df.adm3_pcode.map(hazard(payams))
    # the cattle raster is ~9 km; small payams that touch the same cell share it, so use density x area
    cat = cattle(payams)
    df["cattle"] = df.adm3_pcode.map(cat)
    pr = lambda s: s.rank(pct=True)
    # hazard on an absolute (log) scale: 0 % flooded -> 0, >= 30 % flooded -> 1. A rank would give payams that
    # barely flood a mid score, because most payams hardly flood at all.
    df["H"] = np.clip(np.log1p(df.hazard_flooded_share * 100) / np.log1p(30), 0, 1)
    df["E"] = (pr(df.population_est) + pr(df.cattle)) / 2
    df["E"] = pr(df.E)
    df["V"] = pr(df.p3plus_pct_curr)
    df["priority"] = (df.H * df.E * df.V) ** (1 / 3)
    q = df.priority.rank(pct=True, ascending=False)
    df["priority_class"] = np.select([q <= 0.10, q <= 0.25, q <= 0.50], [c[0] for c in CLASSES[:3]], CLASSES[3][0])
    df["rank"] = df.priority.rank(ascending=False).astype(int)
    df.drop(columns="geometry").sort_values("rank").to_csv(OUT_T / "sq3_adm3_priority_index.csv", index=False)
    return df


def figure(df):
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({"font.family": "Source Sans 3", "font.size": 15})
    ssd = df.dissolve()
    adm1 = df.dissolve(by="adm1_name")
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_axes([0.0, 0.0, 0.66, 1.0])
    colors = dict(CLASSES)
    df.plot(ax=ax, color=df.priority_class.map(colors), edgecolor="white", lw=0.25)
    adm1.boundary.plot(ax=ax, color="#5b6b7d", lw=0.6)
    ssd.boundary.plot(ax=ax, color=NAVY, lw=1.2)
    # label the counties with the most highest-priority payams (one label per county)
    top = df[df.priority_class == CLASSES[0][0]]
    counties = top.dissolve(by="adm2_name")
    from matplotlib import patheffects as pe
    halo = [pe.withStroke(linewidth=3.5, foreground="white")]
    nudge = {"Rubkona": (-0.35, 0.15), "Fangak": (0.0, 0.25), "Ayod": (0.15, 0.0), "Uror": (0.25, 0.0),
             "Twic East": (0.0, -0.15), "Duk": (-0.35, 0.0)}
    for name, (dx, dy) in nudge.items():
        if name not in counties.index:
            continue
        p = counties.loc[name].geometry.representative_point()
        ax.text(p.x + dx, p.y + dy, name, fontsize=14.5, fontweight="bold", color=NAVY, ha="center", va="center",
                path_effects=halo, zorder=10)
    ax.legend(handles=[Patch(fc=c, ec="#9aa5b1", lw=0.6, label=l) for l, c in CLASSES], loc="lower left",
              fontsize=13.5, frameon=False, title="Priority for anticipatory action", title_fontsize=14.5,
              alignment="left")
    ax.set_axis_off()
    ax.set_aspect("equal")

    minis = [("H", "Hazard", "flooded share of area, 2021–25", "Blues"),
             ("E", "Exposure", "people and cattle", "Greens"),
             ("V", "Vulnerability", "people in food crisis (IPC 3+)", "Oranges")]
    for i, (col, title, sub, cmap) in enumerate(minis):
        a = fig.add_axes([0.70, 0.68 - i * 0.32, 0.28, 0.28])
        df.plot(ax=a, column=col, cmap=cmap, vmin=0, vmax=1, edgecolor="none")
        ssd.boundary.plot(ax=a, color="#5b6b7d", lw=0.6)
        a.set_axis_off()
        a.set_aspect("equal")
        a.text(0.0, 1.0, title, transform=a.transAxes, fontsize=15.5, fontweight="bold", color=NAVY, va="bottom")
        a.text(0.0, 0.93, sub, transform=a.transAxes, fontsize=12.5, color="#4a5563", va="bottom")
    for name, kw in [("sq3_priority_map", dict(facecolor="white")), ("sq3_priority_map_transparent", dict(transparent=True))]:
        fig.savefig(OUT_P / f"{name}.png", dpi=300, bbox_inches="tight", **kw)
    plt.close(fig)


def main():
    df = build()
    top10 = df[df.priority_class == CLASSES[0][0]]
    print(df.sort_values("rank")[["rank", "adm3_name", "adm2_name", "adm1_name", "hazard_flooded_share",
                                  "population_est", "cattle", "p3plus_pct_curr", "priority"]].head(15).round(3).to_string(index=False))
    print(f"\nHighest-priority payams: {len(top10)} of {len(df)}; people living there: "
          f"{top10.population_est.sum():,.0f} ({top10.population_est.sum() / df.population_est.sum():.0%} of the total)")
    print("by state:", top10.adm1_name.value_counts().to_dict())
    figure(df)
    print("saved", OUT_P / "sq3_priority_map.png")


if __name__ == "__main__":
    main()
