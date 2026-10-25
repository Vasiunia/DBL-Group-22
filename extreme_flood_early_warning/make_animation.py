"""
Research question 12.1 — animation: South Sudan's floods and the Lake Albert level, monthly 2003-2025.

  Map (left):    flood-mask detections per month on a 0.04° grid (~4.4 km): recurring (normal seasonal) in blue,
                 unusual in orange. The Sudd, West and East regions of robustness.py are outlined.
  Gauge (right): Lake Albert water-level anomaly vs. the 2003-19 mean, with the +1 m hindsight threshold.
  Strip (bottom): unusual flood extent per region (same series as robustness.py), drawn up to the current month.

Output: outputs/animation/south_sudan_floods_lake_albert_2003_2025.mp4 (1920x1080, H.264, PowerPoint-compatible)
        plus poster frames as PNG.

Run from the raw_data folder:  python extreme_flood_early_warning/make_animation.py   (~3 min)
"""
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Patch
from shapely import contains_xy
from shapely.geometry import box

matplotlib.use("Agg")

import train_models as T
import robustness as R

BASE = T.HERE.parent
OUT = T.HERE / "outputs" / "animation"
OUT.mkdir(parents=True, exist_ok=True)

CELL = 0.04                                   # grid cell size in degrees
LON0, LON1, LAT0, LAT1 = 23.9, 36.1, 3.3, 12.4
NX, NY = int(round((LON1 - LON0) / CELL)), int(round((LAT1 - LAT0) / CELL))
MONTHS = pd.date_range("2003-01-01", "2025-12-01", freq="MS")
FPS, HOLD = 10, 40                            # 276 months at 10 fps + 4 s hold on the last frame

COL = dict(recurring="#2a78d6", unusual="#eb6834", lake="#4a3aa7", land="#f3f2ee", border="#9a988f",
           ink=T.C["ink"], ink2=T.C["ink2"], muted=T.C["muted"], grid=T.C["grid"], surface=T.C["surface"])
REGION_STYLE = {  # strip and outline encoding: emphasis on the Sudd, the region downstream of Lake Albert
    "sudd": dict(color=COL["ink"], lw=2.6, ls="-", label="Sudd / White Nile (downstream of Lake Albert)"),
    "west": dict(color="#8a8780", lw=1.8, ls="-", label="West: Bahr el Ghazal"),
    "east": dict(color="#8a8780", lw=1.8, ls=(0, (4, 2)), label="East: Sobat–Pibor"),
}
RGB = {k: matplotlib.colors.to_rgb(COL[k]) for k in ["recurring", "unusual"]}
TOWNS = {"Juba": (31.58, 4.85), "Bor": (31.56, 6.21), "Malakal": (31.66, 9.53), "Wau": (27.99, 7.70)}


def caption(t):
    if t.year <= 2019:
        return "2003–2019", "Floods pulse with the seasons.\nLake Albert stays below +1 m."
    if t.year == 2020:
        return "2020", "Lake Albert climbs to\n+3 m above normal."
    return "2021–2025", "The Sudd stays flooded, even though\nlocal rain is below normal."


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
PX_PER_CELL = (CELL / 0.0020833) ** 2      # flood-mask pixels per grid cell (~369)
FULL = 0.30                                 # cell share flooded that is drawn at full colour


def flood_grid():
    """(n_months, NY, NX) float16 arrays: share of each cell flooded, as the highest daily pixel count in the month
    (the same 'max of 3-day composites' logic as the weekly target). Area-faithful, unlike 'any pixel in cell'."""
    p = T.HERE / "data" / "_flood_grid_monthly_frac.npz"
    if p.exists():
        z = np.load(p)
        return z["recurring"].astype(np.float32), z["unusual"].astype(np.float32)
    grids = {}
    for typ in ["recurring", "unusual"]:
        g = np.zeros((len(MONTHS), NY, NX), np.float16)
        for year in range(2003, 2026):
            d = pd.concat([pd.read_parquet(BASE / "flood_masks" / f"compact_{typ}" / f"flood_events_{t}_{year}.parquet",
                                           columns=["date", "lat", "lon"]) for t in ["h20v08", "h21v08"]])
            d = d[(d.lon >= LON0) & (d.lon < LON1) & (d.lat >= LAT0) & (d.lat < LAT1)]
            dt = pd.to_datetime(d["date"])
            doy = dt.dt.dayofyear.values - 1
            ix = ((d["lon"].values - LON0) / CELL).astype(int)
            iy = ((d["lat"].values - LAT0) / CELL).astype(int)
            daily = np.zeros((366, NY, NX), np.uint16)
            np.add.at(daily, (doy, iy, ix), 1)
            days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
            for mth in range(1, 13):
                sel = np.where(days.month == mth)[0]
                g[(year - 2003) * 12 + mth - 1] = np.minimum(daily[sel].max(0) / PX_PER_CELL, 1)
            print(f"  gridding {typ} {year}")
        grids[typ] = g
    np.savez_compressed(p, **grids)
    return grids["recurring"].astype(np.float32), grids["unusual"].astype(np.float32)


def strength(frac):
    """Monotone colour strength: ~0.08 at 1 % flooded, ~0.44 at 10 %, 1 at >= 30 %."""
    return np.clip(frac / FULL, 0, 1) ** 0.75


def load():
    ssd = gpd.read_file(BASE / "Administrative boundaries" / "ssd_admin0.geojson")
    adm1 = gpd.read_file(BASE / "Administrative boundaries" / "ssd_admin1.geojson")
    geom = ssd.geometry.union_all()
    xc = LON0 + (np.arange(NX) + 0.5) * CELL
    yc = LAT0 + (np.arange(NY) + 0.5) * CELL
    inside = contains_xy(geom, *np.meshgrid(xc, yc))

    rec, unu = flood_grid()
    rec, unu = rec * inside, unu * inside

    w, f, _ = T.build_weekly()
    lake = f["albert_anom"].ffill().resample("MS").mean().reindex(MONTHS)

    rf = R.regional_flood()
    rf = (rf.resample("W-SUN").max() * T.PIXEL_KM2).resample("MS").mean().reindex(MONTHS).fillna(0)
    strip = pd.DataFrame({r: rf[f"unusual_px_{r}"] for r in REGION_STYLE})
    return ssd, adm1, geom, rec, unu, lake, strip


# --------------------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------------------
def build_figure(ssd, adm1, geom, lake, strip):
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": COL["ink"], "axes.labelcolor": COL["ink2"],
                         "xtick.color": COL["muted"], "ytick.color": COL["muted"]})
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=COL["surface"])
    fig.text(0.035, 0.945, "South Sudan's floods and the Lake Albert level, 2003–2025",
             fontsize=27, fontweight="bold", color=COL["ink"])
    fig.text(0.035, 0.910, "Monthly satellite flood detections (NASA MODIS/VIIRS) · Lake Albert satellite altimetry (DAHITI)",
             fontsize=14, color=COL["ink2"])

    # ---- map ----
    axm = fig.add_axes([0.035, 0.30, 0.60, 0.58])
    axm.set_facecolor(COL["surface"])
    ssd.plot(ax=axm, color=COL["land"], edgecolor="none", zorder=0)
    adm1.boundary.plot(ax=axm, color="#d6d4cc", lw=0.6, zorder=1)
    rgba = np.zeros((NY, NX, 4))
    im = axm.imshow(rgba, extent=(LON0, LON1, LAT0, LAT1), origin="lower", interpolation="nearest", zorder=2)
    ssd.boundary.plot(ax=axm, color=COL["border"], lw=1.2, zorder=3)
    for r, (a, b, c, d) in R.FLOOD_REGIONS.items():
        shp = box(c, a, d, b).intersection(geom)
        st = REGION_STYLE[r]
        for poly in getattr(shp, "geoms", [shp]):
            x, y = poly.exterior.xy
            axm.plot(x, y, color=st["color"], lw=2.2 if r == "sudd" else 1.5, ls=st["ls"], zorder=4)
    axm.text(29.6, 10.62, "SUDD", fontsize=15, fontweight="bold", color=COL["ink"], zorder=6)
    axm.text(29.6, 10.40, "downstream of Lake Albert", fontsize=11.5, color=COL["ink"], zorder=6)
    axm.text(24.35, 10.75, "WEST", fontsize=13, fontweight="bold", color=COL["ink2"], zorder=6)
    axm.text(24.35, 10.52, "Bahr el Ghazal · not downstream", fontsize=10.5, color=COL["ink2"], zorder=6)
    axm.text(33.3, 10.95, "EAST", fontsize=13, fontweight="bold", color=COL["ink2"], zorder=6)
    axm.text(33.3, 10.72, "Sobat–Pibor · not downstream", fontsize=10.5, color=COL["ink2"], zorder=6)
    for name, (x, y) in TOWNS.items():
        left = name in ("Malakal", "Bor")       # keep labels off the region outlines
        axm.plot(x, y, "o", ms=5, mfc="white", mec=COL["ink"], mew=1.2, zorder=7)
        axm.text(x - 0.15 if left else x + 0.12, y, name, fontsize=10.5, color=COL["ink"], zorder=7,
                 va="center", ha="right" if left else "left",
                 bbox=dict(boxstyle="round,pad=0.15", fc=COL["surface"], ec="none", alpha=0.75))
    # The White Nile (Bahr el Jebel) enters South Sudan from Uganda at Nimule (3.6 N, 32.06 E)
    axm.annotate("White Nile from\nLake Albert (Uganda)", xy=(32.06, 3.75), xytext=(33.25, 3.55),
                 fontsize=11, color=COL["lake"], fontweight="bold", va="center", zorder=7,
                 arrowprops=dict(arrowstyle="-|>", color=COL["lake"], lw=2, mutation_scale=16))
    axm.set_xlim(LON0, LON1)
    axm.set_ylim(LAT0, LAT1)
    axm.set_aspect("equal")
    axm.axis("off")
    # Colour key: hue = flood type, strength = share of the 4-km cell flooded
    axk = fig.add_axes([0.105, 0.335, 0.12, 0.06])
    fr = np.linspace(0, 0.4, 200)
    rows = ["recurring", "unusual"]              # origin="lower": row 0 is drawn at the bottom
    key = np.zeros((2, len(fr), 4))
    for row, typ in enumerate(rows):
        key[row, :, :3] = RGB[typ]
        key[row, :, 3] = strength(fr)
    axk.imshow(key, aspect="auto", extent=(0, 40, 0, 2), origin="lower", interpolation="nearest")
    axk.set_yticks([0.5, 1.5], ["Normal seasonal", "Unusual"], fontsize=11.5)
    axk.set_xticks([0, 10, 20, 30, 40], ["0", "10", "20", "30", "40 %"], fontsize=10.5)
    axk.tick_params(length=0)
    for s in axk.spines.values():
        s.set_visible(False)
    axk.set_xlabel("share of each 4-km cell flooded", fontsize=10.5, color=COL["ink2"], labelpad=2)
    axk.set_title("Flooding", fontsize=12.5, fontweight="bold", loc="left", color=COL["ink"], pad=4)

    date_txt = fig.text(0.665, 0.815, "", fontsize=46, fontweight="bold", color=COL["ink"])
    phase_txt = fig.text(0.665, 0.765, "", fontsize=17, fontweight="bold", color=COL["ink2"])
    cap_txt = fig.text(0.665, 0.745, "", fontsize=17, color=COL["ink"], va="top", linespacing=1.35)

    # ---- gauge ----
    axg = fig.add_axes([0.715, 0.30, 0.055, 0.30])
    axg.set_facecolor("#ebe9e2")
    axg.set_xlim(0, 1)
    axg.set_ylim(-1, 4.0)
    bar = axg.bar(0.5, 0, width=1.0, bottom=0, color=COL["lake"], zorder=2)[0]
    axg.axhline(0, color=COL["ink2"], lw=1.2, zorder=3)
    axg.axhline(1, color=COL["ink"], lw=1.6, ls=(0, (5, 3)), zorder=3)
    axg.text(1.15, 1, "+1 m\nhindsight\nthreshold", fontsize=11, color=COL["ink"], va="center")
    axg.set_yticks([-1, 0, 1, 2, 3], ["−1 m", "0", "+1 m", "+2 m", "+3 m"], fontsize=12)
    axg.set_xticks([])
    for s in axg.spines.values():
        s.set_visible(False)
    axg.set_title("Lake Albert level\nvs. 2003–19 mean", fontsize=13.5, color=COL["ink"], loc="center", pad=12,
                  fontweight="bold")
    val_txt = axg.text(0.5, 0, "", ha="center", fontsize=15, fontweight="bold", color=COL["lake"], zorder=4)

    # ---- lake history, same y-scale as the gauge ----
    axh = fig.add_axes([0.835, 0.30, 0.135, 0.30], sharey=axg)
    axh.set_facecolor(COL["surface"])
    (lake_line,) = axh.plot([], [], color=COL["lake"], lw=2.2)
    axh.axhline(0, color=COL["ink2"], lw=1)
    axh.axhline(1, color=COL["ink"], lw=1.2, ls=(0, (5, 3)))
    axh.set_xlim(MONTHS[0], MONTHS[-1] + pd.offsets.MonthEnd(1))
    axh.xaxis.set_major_locator(matplotlib.dates.YearLocator(10, month=1, day=1))
    axh.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    axh.tick_params(labelleft=False, labelsize=11, length=0)
    axh.grid(axis="y", color=COL["grid"], lw=0.7)
    axh.grid(axis="x", visible=False)
    for s in ["top", "right", "left"]:
        axh.spines[s].set_visible(False)
    axh.spines["bottom"].set_color("#c3c2b7")
    axh.set_title("Lake Albert since 2003", fontsize=13.5, color=COL["ink"], loc="left", pad=12, fontweight="bold")

    # ---- strip ----
    axs = fig.add_axes([0.07, 0.065, 0.87, 0.17])
    axs.set_facecolor(COL["surface"])
    lines = {}
    for r, st in REGION_STYLE.items():
        (lines[r],) = axs.plot([], [], color=st["color"], lw=st["lw"], ls=st["ls"], label=st["label"])
    cursor = axs.axvline(MONTHS[0], color=COL["ink2"], lw=1.2)
    axs.set_xlim(MONTHS[0], MONTHS[-1] + pd.offsets.MonthEnd(1))
    axs.set_ylim(0, strip.values.max() * 1.08)
    axs.grid(axis="y", color=COL["grid"], lw=0.7)
    axs.grid(axis="x", visible=False)
    for s in ["top", "right"]:
        axs.spines[s].set_visible(False)
    axs.spines["left"].set_color("#c3c2b7")
    axs.spines["bottom"].set_color("#c3c2b7")
    axs.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axs.tick_params(labelsize=11.5)
    axs.set_ylabel("km²", fontsize=12)
    axs.set_title("Unusual flood extent by region (monthly mean of weekly maxima)", fontsize=13.5, loc="left",
                  fontweight="bold", color=COL["ink"])
    axs.legend(loc="upper left", fontsize=11.5, frameon=False, ncol=3)

    return fig, dict(im=im, date=date_txt, phase=phase_txt, cap=cap_txt, bar=bar, val=val_txt,
                     lines=lines, cursor=cursor, lake=lake_line)


def update(i, A, rec, unu, lake, strip):
    k = min(i, len(MONTHS) - 1)
    t = MONTHS[k]
    # "over" compositing: unusual on top of recurring, each with area-faithful opacity
    a_r, a_u = strength(rec[k])[..., None], strength(unu[k])[..., None]
    c_r, c_u = np.array(RGB["recurring"]), np.array(RGB["unusual"])
    alpha = a_u + a_r * (1 - a_u)
    rgb = (c_u * a_u + c_r * a_r * (1 - a_u)) / np.where(alpha > 0, alpha, 1)
    A["im"].set_data(np.concatenate([rgb, alpha], axis=2))
    A["date"].set_text(t.strftime("%B %Y"))
    ph, cap = caption(t)
    A["phase"].set_text(ph)
    A["cap"].set_text(cap)
    v = lake.iloc[k]
    A["bar"].set_height(v)
    A["val"].set_text("0.0 m" if abs(v) < 0.05 else f"{v:+.1f} m".replace("-", "−"))
    A["val"].set_y(max(v, 0) + 0.08)
    A["lake"].set_data(MONTHS[:k + 1], lake.values[:k + 1])
    for r, ln in A["lines"].items():
        ln.set_data(MONTHS[:k + 1], strip[r].values[:k + 1])
    A["cursor"].set_xdata([t, t])


def main():
    ssd, adm1, geom, rec, unu, lake, strip = load()
    assert len(lake) == len(MONTHS) and lake.notna().all(), "Lake Albert series has gaps"
    fig, A = build_figure(ssd, adm1, geom, lake, strip)

    # Poster frames for slides and for checking
    for t in ["2010-09-01", "2019-09-01", "2020-12-01", "2023-09-01"]:
        update(MONTHS.get_loc(pd.Timestamp(t)), A, rec, unu, lake, strip)
        fig.savefig(OUT / f"frame_{t[:7]}.png", dpi=100, facecolor=COL["surface"])

    out = OUT / "south_sudan_floods_lake_albert_2003_2025.mp4"
    writer = FFMpegWriter(fps=FPS, codec="libx264", bitrate=-1,
                          extra_args=["-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow", "-movflags", "+faststart"])
    n = len(MONTHS) + HOLD
    with writer.saving(fig, str(out), dpi=100):
        for i in range(n):
            update(i, A, rec, unu, lake, strip)
            writer.grab_frame(facecolor=COL["surface"])
            if i % 50 == 0:
                print(f"  frame {i}/{n}")
    plt.close(fig)
    print("Saved", out)


if __name__ == "__main__":
    main()
