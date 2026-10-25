"""
Research question 12.1 — Step 1: build the hydro-climatic dataset.

Extracts daily, region-aggregated time series from the raw data folder and saves them to
extreme_flood_early_warning/data/daily_series.parquet:

  Flood regime (MODIS/VIIRS MCDWD 3-day composites, South Sudan only)
    - recurring_px_{region} / unusual_px_{region}: flood pixels detected per day
  Upstream water levels (satellite altimetry)
    - lake_victoria, lake_kyoga, lake_albert: water level [m], on observation days only
  River discharge (Dartmouth Flood Observatory, satellite-derived)
    - q_malakal (station 100205, 9.6N 31.6E, White Nile at the Sobat confluence)
    - q_ethiopia (station 1505, 4.9N 36.0E, south-west Ethiopian highlands)
  ERA5 precipitation / runoff [mm/day], area means over hydrologically meaningful regions
  AgERA5 reference evapotranspiration [mm/day], area means

Run from the raw_data folder:  python extreme_flood_early_warning/build_dataset.py
"""
import glob
import io
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely import contains_xy

BASE = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)
YEARS = range(2000, 2026)

# Boxes as (lat_min, lat_max, lon_min, lon_max)
REGIONS = {
    # Equatorial lakes plateau: Lake Victoria/Kyoga/Albert catchment, source of the White Nile
    "lakes": (-3.0, 3.5, 29.0, 35.5),
    # Sudd wetland and the Bahr el Jebel / White Nile floodplain (Jonglei, Unity, Lakes, Upper Nile)
    "sudd": (6.0, 10.0, 29.5, 32.5),
    # Baro-Akobo-Sobat headwaters in the south-west Ethiopian highlands
    "sobat": (5.5, 9.5, 33.5, 36.5),
    # Bahr el Ghazal rain-fed tributaries (western South Sudan)
    "ghazal": (5.0, 10.0, 24.0, 29.5),
}


def south_sudan_geom():
    return gpd.read_file(BASE / "Administrative boundaries" / "ssd_admin0.geojson").geometry.union_all()


def grid_masks(lat, lon, ssd):
    lon2, lat2 = np.meshgrid(lon, lat)
    masks = {"ssd": contains_xy(ssd, lon2, lat2)}
    for name, (a, b, c, d) in REGIONS.items():
        masks[name] = (lat2 >= a) & (lat2 <= b) & (lon2 >= c) & (lon2 <= d)
    return masks


# --------------------------------------------------------------------------------------
# Flood masks
# --------------------------------------------------------------------------------------
def flood_series(ssd):
    sudd = REGIONS["sudd"]
    rows = []
    for typ in ["recurring", "unusual"]:
        for year in YEARS:
            parts = [
                pd.read_parquet(BASE / "flood_masks" / f"compact_{typ}" / f"flood_events_{t}_{year}.parquet",
                                columns=["date", "lat", "lon"])
                for t in ["h20v08", "h21v08"]
            ]
            d = pd.concat(parts, ignore_index=True)
            d = d[contains_xy(ssd, d["lon"].values, d["lat"].values)]
            in_sudd = (d["lat"].between(sudd[0], sudd[1]) & d["lon"].between(sudd[2], sudd[3]))
            g = pd.DataFrame({
                f"{typ}_px_ssd": d.groupby("date").size(),
                f"{typ}_px_sudd": d[in_sudd].groupby("date").size(),
            })
            rows.append(g)
            print(f"  flood masks {typ} {year}: {len(d):,} pixels in South Sudan")
    out = pd.concat(rows).groupby(level=0).sum()
    out.index = pd.to_datetime(out.index)
    return out


# --------------------------------------------------------------------------------------
# Lakes and discharge
# --------------------------------------------------------------------------------------
def lake_series():
    missions = {'TOPEX', 'JASON-1', 'JASON-2', 'JASON-3', 'J1', 'J2', 'J3',
                'S3A', 'S3B', 'S6A', 'ENVISAT', 'ERS-1', 'ERS-2'}
    cols = ['mission', 'cycle', 'date', 'hour', 'minute', 'height_wrt_ref', 'height_err', 'backscatter_ku',
            'wet_tropo_corr', 'iono_corr', 'dry_tropo_corr', 'mode1', 'mode2', 'ice_flag',
            'height_egm2008', 'data_source_flag']
    out = {}
    for name in ["victoria", "Kyoga"]:
        with open(BASE / "Water levels lakes" / f"water_level_{name}.txt", encoding="latin-1") as f:
            lines = [ln.strip() for ln in f if ln.strip() and ln.split()[0] in missions]
        df = pd.read_csv(io.StringIO("\n".join(lines)), sep=r"\s+", names=cols,
                         na_values=['999.99', '99.999', '9999.99'])
        df = df.dropna(subset=["height_wrt_ref"])
        df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
        out[f"lake_{name.lower()}"] = df.groupby("date")["height_wrt_ref"].median()
    ds = xr.open_dataset(BASE / "Water levels lakes" / "water_level_altimetry_Albert.nc")
    alb = pd.Series(ds["water_level"].values, index=pd.to_datetime(ds["datetime"].values))
    out["lake_albert"] = alb.groupby(alb.index.normalize()).median()
    return pd.DataFrame(out)


def discharge_series():
    out = {}
    for col, sid in [("q_malakal", 100205), ("q_ethiopia", 1505)]:
        df = pd.read_csv(BASE / "Darthmouth Flood Observatory" / f"{sid}_discharge.csv", parse_dates=["Date"])
        out[col] = df.set_index("Date")["Discharge (m3/s)"]
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------
# ERA5 precipitation & runoff (hourly -> daily, area mean)
# --------------------------------------------------------------------------------------
def era5_year(year):
    ssd = south_sudan_geom()
    ds = xr.open_dataset(BASE / "rainfall and runoff" / f"ERA5_{year}.nc")
    masks = grid_masks(ds.latitude.values, ds.longitude.values, ssd)
    times = pd.to_datetime(ds.valid_time.values)
    res = {}
    for var in ["tp", "ro"]:
        arr = ds[var].values  # (time, lat, lon), metres per hour
        arr = np.nan_to_num(arr, nan=0.0)
        for reg, m in masks.items():
            s = pd.Series(arr[:, m].mean(axis=1) * 1000.0, index=times)  # mm
            res[f"{var}_{reg}"] = s.resample("1D").sum()
    ds.close()
    print(f"  ERA5 {year} done")
    return pd.DataFrame(res)


# --------------------------------------------------------------------------------------
# AgERA5 reference evapotranspiration (one file per day)
# --------------------------------------------------------------------------------------
_ET_MASKS = None


def et_file(path):
    global _ET_MASKS
    with xr.open_dataset(path) as ds:
        if _ET_MASKS is None:
            _ET_MASKS = grid_masks(ds.lat.values, ds.lon.values, south_sudan_geom())
        a = ds["ReferenceET_PenmanMonteith_FAO56"].values[0]
        t = pd.Timestamp(ds["time"].values[0])
    return t, {f"et_{k}": float(np.nanmean(a[m])) for k, m in _ET_MASKS.items() if k in ("ssd", "sudd", "lakes")}


def et_series():
    files = sorted(glob.glob(str(BASE / "evapotranspiration" / "ET_*" / "*.nc")))
    with ProcessPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(et_file, files, chunksize=64))
    return pd.DataFrame({t: v for t, v in res}).T.sort_index()


def main():
    ssd = south_sudan_geom()
    cache = {}
    for name, fn in [("flood", lambda: flood_series(ssd)), ("lakes", lake_series),
                     ("discharge", discharge_series), ("et", et_series)]:
        p = OUT / f"_{name}.parquet"
        if not p.exists():
            print(f"Extracting {name} ...")
            fn().to_parquet(p)
        cache[name] = pd.read_parquet(p)

    p = OUT / "_era5.parquet"
    if not p.exists():
        print("Extracting ERA5 ...")
        with ProcessPoolExecutor(max_workers=4) as ex:
            pd.concat(list(ex.map(era5_year, YEARS))).sort_index().to_parquet(p)
    cache["era5"] = pd.read_parquet(p)

    idx = pd.date_range("2000-01-01", "2025-12-31", freq="D")
    daily = pd.concat([v.reindex(idx) for v in cache.values()], axis=1)
    # Days without any detection in a year that has data are true zeros for the flood counts
    for c in [c for c in daily if c.endswith("_px_ssd") or c.endswith("_px_sudd")]:
        daily[c] = daily[c].fillna(0)
    daily.index.name = "date"
    daily.to_parquet(OUT / "daily_series.parquet")
    print(daily.describe().T.to_string())


if __name__ == "__main__":
    main()
