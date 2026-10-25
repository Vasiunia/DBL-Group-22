"""
Research question 12.1 — Step 5: open external data (no registration needed) that extends the record before 2003.

  1. Lake Victoria & Kyoga water levels, 1992-2026 (NASA GSFC G-REALM, 10-day altimetry)
     https://earth.gsfc.nasa.gov/gwm/lake/Index  -> lake.10.tar.gz, files lake000314 (Victoria), lake000398 (Kyoga)
     Downloaded to raw_data/external/grealm/.
  2. Surface water per year, 1984-2021, 30 m Landsat (JRC Global Surface Water, YearlyClassification v1.4)
     https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GSWE/YearlyClassification/LATEST/tiles/
     Classes 2 = seasonal water, 3 = permanent water. In the downloaded tiles observed dry land is 0, so the
     share of unobserved pixels is not available; only water area (2 + 3) is used.
     Read remotely (windowed) and aggregated to 0.01° cell counts; only the South Sudan window is transferred.
  3. Climate indices (NOAA PSL): Indian Ocean Dipole (DMI, HadISST) and ENSO (Nino 3.4 anomaly).

Writes extreme_flood_early_warning/data/external_*.parquet.
Run from the raw_data folder with the project venv:  .venv/bin/python extreme_flood_early_warning/external_data.py
"""
import io
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from shapely import contains_xy

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OUT = HERE / "data"
GREALM = BASE / "external" / "grealm"
JRC = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GSWE/YearlyClassification/LATEST/tiles"

# Same regions as robustness.py, capped at 10 N (edge of the JRC tile row used here)
REGIONS = {"sudd": (6.0, 10.0, 29.5, 32.5), "west": (4.0, 10.0, 24.0, 29.5), "east": (5.5, 10.0, 32.5, 36.0)}
AGG = 40                                   # 40 x 0.00025° = 0.01° output cells
LON0, LON1, LAT0, LAT1 = 24.0, 36.0, 4.0, 10.0


# --------------------------------------------------------------------------------------
def grealm(lake_file):
    rows = []
    with open(lake_file, encoding="latin-1") as f:
        for line in f:
            p = line.split()
            if len(p) >= 6 and p[2].isdigit() and len(p[2]) == 8:
                h = float(p[5])
                if abs(h) < 900:                          # 999.99 = missing
                    rows.append((pd.Timestamp(p[2]), h))
    s = pd.DataFrame(rows, columns=["date", "h"]).groupby("date")["h"].median()
    return s


def lakes():
    out = pd.DataFrame({"victoria": grealm(GREALM / "lake000314.10d.2.txt"),
                        "kyoga": grealm(GREALM / "lake000398.10d.2.txt")})
    out.index.name = "date"
    return out


# --------------------------------------------------------------------------------------
def jrc_year(year):
    """Class counts per 0.01° cell over lon 24-36, lat 4-10 for one year: array (600, 1200, 4)."""
    p = OUT / "_jrc" / f"{year}.npy"
    if p.exists():
        return year, np.load(p)
    ny, nx = int((LAT1 - LAT0) * 100), int((LON1 - LON0) * 100)
    res = np.zeros((ny, nx, 4), np.uint16)
    for col, (tlon0, lo, hi) in {"0000800000": (20.0, 24.0, 30.0), "0000840000": (30.0, 30.0, 36.0)}.items():
        sep = "_" if year == 2021 else "-"           # the 2021 tiles use a different file-name separator
        url = f"/vsicurl/{JRC}/yearlyClassification{year}/yearlyClassification{year}{sep}0000280000-{col}.tif"
        c0, c1 = int((lo - tlon0) / 0.00025), int((hi - tlon0) / 0.00025)
        rows = int((10.0 - LAT0) / 0.00025)
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIRANGE="YES",
                          GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="3"), rasterio.open(url) as r:
            for r0 in range(0, rows, 4000):
                a = r.read(1, window=Window(c0, r0, c1 - c0, 4000))
                a = a.reshape(4000 // AGG, AGG, (c1 - c0) // AGG, AGG)
                ox = int((lo - LON0) * 100)
                # output rows are south -> north
                oy1 = ny - r0 // AGG
                for k in range(4):
                    cnt = (a == k).sum(axis=(1, 3)).astype(np.uint16)[::-1]
                    res[oy1 - cnt.shape[0]:oy1, ox:ox + cnt.shape[1], k] = cnt
    (OUT / "_jrc").mkdir(exist_ok=True)
    np.save(p, res)
    print(f"  JRC {year} done", flush=True)
    return year, res


def jrc():
    ssd = gpd.read_file(BASE / "Administrative boundaries" / "ssd_admin0.geojson").geometry.union_all()
    xc = LON0 + (np.arange(int((LON1 - LON0) * 100)) + 0.5) / 100
    yc = LAT0 + (np.arange(int((LAT1 - LAT0) * 100)) + 0.5) / 100
    X, Y = np.meshgrid(xc, yc)
    inside = contains_xy(ssd, X, Y)
    with ThreadPoolExecutor(4) as ex:
        data = dict(ex.map(jrc_year, range(1984, 2022)))
    rows = []
    px_km2 = (0.00025 * 111.32) ** 2 * np.cos(np.deg2rad(Y))       # km² per 30 m pixel, by latitude
    for year, a in sorted(data.items()):
        for reg, (la0, la1, lo0, lo1) in REGIONS.items():
            m = inside & (Y >= la0) & (Y < la1) & (X >= lo0) & (X < lo1)
            # In the downloaded yearly tiles, observed dry land is stored as 0 (checked on Lake Turkana and its
            # desert shore), so unobserved pixels cannot be separated; classes 2 and 3 (water) are unambiguous.
            area = lambda k: float((a[..., k] * px_km2)[m].sum())
            rows.append(dict(year=year, region=reg, seasonal_km2=area(2), permanent_km2=area(3),
                             water_km2=area(2) + area(3)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
def noaa(url, name):
    """Read a NOAA PSL monthly table; uses the copy in raw_data/external/noaa/ if present (the server is slow)."""
    local = BASE / "external" / "noaa" / url.rsplit("/", 1)[1]
    txt = local.read_text() if local.exists() else urllib.request.urlopen(url, timeout=120).read().decode()
    rows = []
    for line in txt.splitlines():
        p = line.split()
        if len(p) == 13 and p[0].isdigit():
            for m, v in enumerate(p[1:], 1):
                v = float(v)
                if v > -99:
                    rows.append((pd.Timestamp(int(p[0]), m, 1), v))
    return pd.Series(dict(rows), name=name)


def main():
    lk = lakes()
    lk.to_parquet(OUT / "external_lakes_grealm.parquet")
    print("Lakes:", lk.index.min().date(), "to", lk.index.max().date(), len(lk), "obs")

    idx = pd.concat([noaa("https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data", "iod_dmi"),
                     noaa("https://psl.noaa.gov/data/correlation/nina34.anom.data", "nino34")], axis=1)
    idx.index.name = "month"
    idx.to_parquet(OUT / "external_climate_indices.parquet")
    print("Climate indices:", idx.index.min().date(), "to", idx.index.max().date())

    j = jrc()
    j.to_parquet(OUT / "external_jrc_surface_water.parquet")
    print(j.pivot(index="year", columns="region", values=["water_km2", "permanent_km2"]).round(0).to_string())


if __name__ == "__main__":
    main()
