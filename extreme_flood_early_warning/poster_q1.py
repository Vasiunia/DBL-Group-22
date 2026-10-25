"""
Poster graphics for sub-question 1, classic science-poster style (white, Arial, boxed axes), as two separate
images with no running text (the text is written in Canva):
which factors turn South Sudan's normal yearly floods into extreme floods?

Needs the outputs of robustness.py, final_model.py and make_animation.py (flood grid cache).
Run from the raw_data folder:  python extreme_flood_early_warning/poster_q1.py
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely import contains_xy
from shapely.geometry import box

import train_models as T
import robustness as R

OUT = T.HERE / "outputs" / "poster"
OUT.mkdir(parents=True, exist_ok=True)
BASE = T.HERE.parent

# blues and greens
SUDD = "#08519c"        # Sudd / flooded land
WEST = "#31a354"
EAST = "#74c476"
LAKE = "#4292c6"
RIVER = "#2171b5"
LAND = "#f2f2f2"
GREY = "#555555"

CELL, LON0, LON1, LAT0, LAT1 = 0.04, 23.9, 36.1, 3.3, 12.4     # flood grid written by make_animation.py


def data():
    rf = R.regional_flood()
    yearly = (rf.resample("W-SUN").max() * T.PIXEL_KM2).resample("YE").mean()
    yearly.index = yearly.index.year
    yearly = yearly.loc[2003:2025]
    _, f, _ = T.build_weekly()
    albert = f["albert_anom"].ffill().resample("MS").mean()["2003":"2025"]
    od = pd.read_csv(T.OUT_T / "final_q1_odds_ratios.csv")
    od = od[od.period.str.startswith("2003–2025")].set_index("driver")
    mag = pd.read_csv(T.OUT_T / "causal_spatial_magnitude.csv")
    mag = mag[mag.extent == "unusual"].set_index("region")
    z = np.load(T.HERE / "data" / "_flood_grid_monthly_frac.npz")
    flood2023 = z["unusual"][(2023 - 2003) * 12:(2024 - 2003) * 12].astype(float).max(axis=0)
    return yearly, albert, od, mag, flood2023


def draw_map(ax, flood2023):
    ssd = gpd.read_file(BASE / "Administrative boundaries" / "ssd_admin0.geojson")
    geom = ssd.geometry.union_all()
    ssd.plot(ax=ax, color=LAND, edgecolor="black", lw=0.9, zorder=1)
    nx, ny = flood2023.shape[1], flood2023.shape[0]
    xc = LON0 + (np.arange(nx) + 0.5) * CELL
    yc = LAT0 + (np.arange(ny) + 0.5) * CELL
    inside = contains_xy(geom, *np.meshgrid(xc, yc))
    rgba = np.zeros((ny, nx, 4))
    rgba[..., :3] = plt.matplotlib.colors.to_rgb(SUDD)
    rgba[..., 3] = np.clip(flood2023 / 0.3, 0, 1) ** 0.75 * inside
    ax.imshow(rgba, extent=(LON0, LON1, LAT0, LAT1), origin="lower", interpolation="nearest", zorder=2)
    sudd = box(29.5, 6.0, 32.5, 10.0).intersection(geom)
    ax.plot(*sudd.exterior.xy, color="black", lw=1.4, ls="--", zorder=3)
    lakes = [
        [(31.8, -1.0), (31.7, -1.8), (32.0, -2.4), (32.6, -2.6), (33.2, -2.4), (33.9, -2.1), (34.5, -1.6),
         (34.8, -1.0), (34.4, -0.4), (34.0, 0.0), (33.4, 0.35), (32.8, 0.3), (32.3, 0.0), (31.8, -0.4)],
        [(30.25, 1.1), (30.6, 1.1), (31.1, 1.55), (31.55, 2.1), (31.45, 2.3), (31.0, 2.0), (30.5, 1.5)],
        [(32.4, 1.45), (32.9, 1.35), (33.4, 1.55), (33.3, 1.75), (32.8, 1.6), (32.5, 1.7)],
    ]
    for pts in lakes:
        ax.add_patch(plt.Polygon(pts, closed=True, fc=LAKE, ec="none", zorder=3))
    nile = np.array([(33.2, 0.42), (32.95, 1.3), (32.3, 2.15), (31.45, 2.25), (31.8, 3.0), (32.06, 3.6), (31.6, 4.85),
                     (31.56, 6.2), (31.0, 7.4), (30.5, 8.6), (30.45, 9.5), (31.66, 9.53), (32.0, 10.6), (32.4, 12.45)])
    ax.plot(nile[:, 0], nile[:, 1], color=RIVER, lw=2.2, zorder=4)
    for i in [4, 8]:
        ax.annotate("", xy=nile[i + 1], xytext=nile[i], zorder=5,
                    arrowprops=dict(arrowstyle="-|>", color=RIVER, lw=0, mutation_scale=18))
    for txt, (x, y), kw in [("Lake\nVictoria", (33.25, -1.25), dict(ha="center", color="white", fontweight="bold")), ("Lake Albert", (29.9, 1.6), dict(ha="right")),
                            ("Sudd", (29.55, 10.5), dict(fontweight="bold")),
                            ("SOUTH SUDAN", (25.6, 7.3), dict(color=GREY)), ("UGANDA", (30.4, 0.1), dict(color=GREY)),
                            ("Juba", (31.75, 4.85), dict(fontsize=11))]:
        ax.text(x, y, txt, **{"fontsize": 12.5, "zorder": 6, "va": "center", **kw})
    ax.plot(31.58, 4.85, "o", ms=4, color="black", zorder=6)
    ax.set_xlim(24.0, 36.6)
    ax.set_ylim(-3.1, 12.6)
    ax.set_aspect("equal")
    ax.set_xticks([25, 30, 35], ["25°E", "30°E", "35°E"])
    ax.set_yticks([0, 5, 10], ["0°", "5°N", "10°N"])
    ax.tick_params(labelsize=11)
    ax.legend(handles=[Patch(color=SUDD, label="Flooded land (normally dry), 2023"),
                       Patch(color=LAKE, label="Lakes"),
                       Line2D([], [], color=RIVER, lw=2.2, label="White Nile (flows north)"),
                       Line2D([], [], color="black", lw=1.4, ls="--", label="Sudd study area")],
              loc="lower left", fontsize=11, frameon=True, edgecolor="black", fancybox=False, framealpha=1)


def main():
    yearly, albert, od, mag, flood2023 = data()
    plt.rcParams.update(plt.rcParamsDefault)          # drop the project chart style: plain, boxed, white
    plt.rcParams.update({"font.family": "Arial", "font.size": 14, "axes.linewidth": 1.0, "axes.grid": False,
                         "xtick.direction": "out", "ytick.direction": "out", "axes.facecolor": "white"})

    # ---- part 1: map ----
    fig, axm = plt.subplots(figsize=(7.2, 8.6))
    draw_map(axm, flood2023)
    fig.tight_layout()
    save(fig, "q1_part1_map")

    # ---- part 2: flooding per region (top) and Lake Albert (bottom), shared years ----
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 7.6), sharex=True, gridspec_kw=dict(height_ratios=[1.25, 1],
                                                                                        hspace=0.12))
    x = yearly.index.values
    ax.plot(x, yearly["unusual_px_sudd"], color=SUDD, lw=2.8, marker="o", ms=4.5, label="Sudd")
    ax.plot(x, yearly["unusual_px_west"], color=WEST, lw=1.8, marker="s", ms=4, label="West")
    ax.plot(x, yearly["unusual_px_east"], color=EAST, lw=1.8, marker="^", ms=4, ls="--", label="East")
    ax.set_ylabel("Flooded area (km²)")
    ax.set_ylim(0, None)
    ax.legend(loc="upper left", frameon=True, edgecolor="black", fancybox=False)
    xm = albert.index.year + (albert.index.month - 0.5) / 12
    ax2.plot(xm, albert.values, color=LAKE, lw=1.8)
    ax2.fill_between(xm, 0, albert.values, where=albert.values > 0, color=LAKE, alpha=0.25, lw=0)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_ylabel("Lake Albert level\n(m above normal)")
    ax2.set_xlabel("Year")
    ax2.set_xlim(2002.5, 2025.5)
    for a in (ax, ax2):
        a.axvline(2019.5, color=GREY, lw=1, ls=":")
    fig.align_ylabels()
    fig.tight_layout()
    save(fig, "q1_part2_charts")
    combined(yearly, albert, flood2023)
    print("saved to", OUT)


def combined(yearly, albert, flood2023):
    """Map + both charts in one roughly square image for the poster's SQ1 column (A0), in the poster font."""
    import glob
    from matplotlib import font_manager as fm
    for f in glob.glob(str(BASE / "external" / "fonts" / "ss3" / "OTF" / "SourceSans3-*.otf")):
        fm.fontManager.addfont(f)
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({"font.family": "Source Sans 3", "font.size": 18, "axes.linewidth": 1.1,
                         "axes.facecolor": "white", "xtick.labelsize": 16, "ytick.labelsize": 16})
    fig = plt.figure(figsize=(12, 12.4))
    axm = fig.add_axes([0.0, 0.22, 0.45, 0.78])
    draw_map(axm, flood2023)
    axm.get_legend().remove()
    for t in axm.texts:
        t.set_fontsize(t.get_fontsize() * 1.3)
    axm.tick_params(labelsize=15)
    fig.legend(handles=[Patch(color=SUDD, label="Flooded land (normally dry), 2023"),
                        Patch(color=LAKE, label="Lakes in Uganda"),
                        Line2D([], [], color=RIVER, lw=2.4, label="White Nile (flows north)"),
                        Line2D([], [], color="black", lw=1.4, ls="--", label="Sudd study area")],
               loc="lower left", bbox_to_anchor=(0.02, 0.0), fontsize=16.5, frameon=False, handlelength=1.6)
    ax = fig.add_axes([0.58, 0.56, 0.41, 0.42])
    ax2 = fig.add_axes([0.58, 0.08, 0.41, 0.40], sharex=ax)
    x = yearly.index.values
    ax.plot(x, yearly["unusual_px_sudd"], color=SUDD, lw=3.2, marker="o", ms=5.5, label="Sudd")
    ax.plot(x, yearly["unusual_px_west"], color=WEST, lw=2.2, marker="s", ms=5, label="West")
    ax.plot(x, yearly["unusual_px_east"], color=EAST, lw=2.2, marker="^", ms=5, ls="--", label="East")
    ax.set_ylabel("Flooded area (km²)")
    ax.set_ylim(0, None)
    ax.legend(loc="upper left", frameon=True, edgecolor="black", fancybox=False, fontsize=16.5)
    ax.tick_params(labelbottom=False)
    xm = albert.index.year + (albert.index.month - 0.5) / 12
    ax2.plot(xm, albert.values, color=LAKE, lw=2.2)
    ax2.fill_between(xm, 0, albert.values, where=albert.values > 0, color=LAKE, alpha=0.25, lw=0)
    ax2.axhline(0, color="black", lw=0.9)
    ax2.set_ylabel("Lake Albert level (m above normal)")
    ax2.set_xlabel("Year")
    ax2.set_xlim(2002.5, 2025.5)
    ax2.set_xticks([2005, 2010, 2015, 2020, 2025])
    for a in (ax, ax2):
        a.axvline(2019.5, color=GREY, lw=1.2, ls=":")
    save(fig, "sq1_combined")


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(OUT / f"{name}_transparent.png", dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
