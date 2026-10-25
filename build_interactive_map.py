#!/usr/bin/env python3
"""
build_interactive_map.py

Generates a standalone, fully interactive web map mapping:
1. Parquet Flood Events (Unusual & Recurring Floods from MCDWD VIIRS/MODIS)
2. NetCDF Rainfall & Runoff Grid (ECMWF ERA5)
3. NetCDF Evapotranspiration Grid (AgERA5 FAO-56 Penman-Monteith)
4. NetCDF Lake Altimetry Station (Lake Albert - DAHITI)
5. Dartmouth River Discharge Stations (Nile Basin, Sudan & South Sudan)
6. Administrative Boundaries (Sudan country boundary, South Sudan states & counties)

Points are clustered using Leaflet.markercluster, allowing the user to view regional
density clusters and zoom in/spiderfy down to exact individual points with rich popups.
"""

import os
import sys
import json
import glob
import time
import argparse
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow as pa
import pandas as pd
import xarray as xr
import numpy as np
import geopandas as gpd
from shapely.geometry import Point


def parse_args():
    parser = argparse.ArgumentParser(description="Build interactive Sudan & South Sudan NetCDF + Parquet map")
    parser.add_argument("--year", type=int, default=2024, help="Target year for flood masks and climate data (2000-2025)")
    parser.add_argument("--output", type=str, default="sudan_interactive_map.html", help="Output HTML file path")
    parser.add_argument("--max_unusual_pts", type=int, default=18000, help="Max unusual flood points to render")
    parser.add_argument("--max_recurring_pts", type=int, default=10000, help="Max recurring flood points to render")
    return parser.parse_args()


def load_boundaries():
    print("-> Loading and simplifying administrative boundaries...")
    admin1_path = Path("Administrative boundaries/ssd_admin1.geojson")
    admin2_path = Path("Administrative boundaries/ssd_admin2.geojson")
    sdn_path = Path("Administrative boundaries/sdn_admin0.geojson")

    admin1 = gpd.read_file(admin1_path)[['adm1_name', 'geometry']]
    admin2 = gpd.read_file(admin2_path)[['adm1_name', 'adm2_name', 'area_sqkm', 'geometry']]
    
    # Simplify geometries slightly for fast web rendering (<300KB)
    admin1['geometry'] = admin1.geometry.simplify(0.005)
    admin2['geometry'] = admin2.geometry.simplify(0.005)

    sdn_json = None
    if sdn_path.exists():
        sdn = gpd.read_file(sdn_path)[['name', 'geometry']]
        sdn['geometry'] = sdn.geometry.simplify(0.008)
        sdn_json = json.loads(sdn.to_json())

    admin1_json = json.loads(admin1.to_json())
    admin2_json = json.loads(admin2.to_json())

    return admin1, admin2, admin1_json, admin2_json, sdn_json


def load_lake_station():
    print("-> Loading NetCDF Lake Albert altimetry station...")
    lake_file = Path("Water levels lakes/water_level_altimetry_Albert.nc")
    if not lake_file.exists():
        return None

    ds = xr.open_dataset(lake_file)
    lat = float(ds.attrs.get('latitude', 1.6807))
    lon = float(ds.attrs.get('longitude', 30.9210))
    target_name = str(ds.attrs.get('target_name', 'Lake Albert'))
    source = str(ds.attrs.get('source', 'DAHITI'))
    institution = str(ds.attrs.get('institution', 'DGFI-TUM'))
    
    wl_series = ds['water_level'].values
    mean_wl = float(np.nanmean(wl_series))
    min_wl = float(np.nanmin(wl_series))
    max_wl = float(np.nanmax(wl_series))
    latest_wl = float(wl_series[-1]) if len(wl_series) > 0 else mean_wl
    ds.close()

    return {
        "name": target_name,
        "lat": round(lat, 4),
        "lon": round(lon, 4),
        "type": "lake_station",
        "latest_wl": round(latest_wl, 2),
        "mean_wl": round(mean_wl, 2),
        "min_wl": round(min_wl, 2),
        "max_wl": round(max_wl, 2),
        "source": source,
        "institution": institution,
        "file": str(lake_file)
    }


def load_dartmouth_stations():
    print("-> Loading Dartmouth Flood Observatory river discharge stations...")
    info_path = Path("Darthmouth Flood Observatory/information.xlsx")
    if not info_path.exists():
        return []

    df = pd.read_excel(info_path)
    stations = []
    for _, row in df.iterrows():
        lat = float(row['latitude'])
        lon = float(row['longitude'])
        st_nr = int(row['station nr'])
        area_id = int(row['area id'])
        country = str(row['Country'])
        
        # Check if discharge file exists
        q_file = Path(f"Darthmouth Flood Observatory/{area_id}_discharge.csv")
        mean_q = None
        if q_file.exists():
            try:
                df_q = pd.read_csv(q_file)
                mean_q = round(float(df_q['Discharge'].mean()), 1)
            except Exception:
                pass

        stations.append({
            "name": f"DFO Gauge {st_nr} (Area {area_id})",
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "station_nr": st_nr,
            "area_id": area_id,
            "country": country,
            "mean_discharge_m3s": mean_q,
            "type": "river_station"
        })
    return stations


def load_era5_rainfall_runoff(year: int):
    print(f"-> Loading NetCDF ERA5 rainfall and runoff for year {year}...")
    fpath = Path(f"rainfall and runoff/ERA5_{year}.nc")
    if not fpath.exists():
        print(f"   WARNING: {fpath} not found.")
        return []

    ds = xr.open_dataset(fpath)
    # Subset to Sudan and South Sudan: Lat [3.5, 22.5], Lon [23.0, 38.0]
    sub = ds.sel(latitude=slice(22.5, 3.5), longitude=slice(23.0, 38.0))
    
    # Annual total precipitation in mm (tp in meters * 1000)
    tp_annual = (sub['tp'].sum(dim='valid_time') * 1000).values
    ro_annual = (sub['ro'].sum(dim='valid_time') * 1000).values
    
    # Peak daily precipitation
    tp_daily_max = (sub['tp'].resample(valid_time='1D').sum().max(dim='valid_time') * 1000).values
    
    lats = sub['latitude'].values
    lons = sub['longitude'].values
    ds.close()

    points = []
    # Subsample step: every 2nd grid point (0.5 degree grid resolution, ~55km)
    step_lat = 2
    step_lon = 2
    for i in range(0, len(lats), step_lat):
        for j in range(0, len(lons), step_lon):
            p_mm = float(tp_annual[i, j])
            r_mm = float(ro_annual[i, j])
            pk_mm = float(tp_daily_max[i, j])
            
            if np.isnan(p_mm):
                continue
                
            points.append({
                "lat": round(float(lats[i]), 4),
                "lon": round(float(lons[j]), 4),
                "type": "era5",
                "year": year,
                "annual_precip_mm": round(p_mm, 1),
                "annual_runoff_mm": round(r_mm, 1),
                "peak_daily_mm": round(pk_mm, 1),
                "file": f"ERA5_{year}.nc"
            })

    print(f"   Extracted {len(points)} ERA5 climate grid points across Sudan/South Sudan")
    return points


def load_agera5_evapotranspiration(year: int):
    print(f"-> Loading NetCDF AgERA5 Evapotranspiration for year {year}...")
    et_dir = Path(f"evapotranspiration/ET_{year}")
    if not et_dir.exists():
        print(f"   WARNING: {et_dir} not found.")
        return []

    files = sorted(et_dir.glob("*.nc"))
    if not files:
        print(f"   WARNING: No .nc files in {et_dir}")
        return []

    # Sample 12 representative monthly dates to compute mean annual reference ET rapidly
    num_samples = min(12, len(files))
    step = max(1, len(files) // num_samples)
    sampled_files = files[::step][:num_samples]

    datasets = []
    for f in sampled_files:
        try:
            ds = xr.open_dataset(f)
            sub = ds['ReferenceET_PenmanMonteith_FAO56'].sel(lat=slice(22.5, 3.5), lon=slice(23.0, 38.0)).squeeze()
            datasets.append(sub)
            ds.close()
        except Exception:
            continue

    if not datasets:
        return []

    et_mean = xr.concat(datasets, dim='sample').mean(dim='sample')
    lats = et_mean['lat'].values
    lons = et_mean['lon'].values
    vals = et_mean.values

    points = []
    # Subsample step: every 4th grid point (0.4 degree resolution)
    step_lat = 4
    step_lon = 4
    for i in range(0, len(lats), step_lat):
        for j in range(0, len(lons), step_lon):
            val = float(vals[i, j])
            if np.isnan(val) or val <= 0:
                continue
            points.append({
                "lat": round(float(lats[i]), 4),
                "lon": round(float(lons[j]), 4),
                "type": "agera5",
                "year": year,
                "mean_et_mm_day": round(val, 2),
                "est_annual_et_mm": round(val * 365, 0),
                "file": f"AgERA5_{year}"
            })

    print(f"   Extracted {len(points)} AgERA5 ET grid points across Sudan/South Sudan")
    return points


def load_parquet_floods(year: int, max_unusual: int = 18000, max_recurring: int = 10000):
    print(f"-> Loading Parquet flood masks for year {year} (VIIRS/MODIS MCDWD)...")
    tiles = ['h20v08', 'h21v08']

    # --- 1. RECURRING FLOODS ---
    rec_tables = []
    for tile in tiles:
        fpath = Path(f"flood_masks/compact_recurring/flood_events_{tile}_{year}.parquet")
        if fpath.exists():
            tbl = pq.read_table(fpath, columns=['lat', 'lon', 'date'])
            tbl = tbl.set_column(tbl.schema.get_field_index('date'), 'date', pc.cast(tbl['date'], pa.string()))
            rec_tables.append(tbl)

    rec_pts = []
    if rec_tables:
        t_rec = pa.concat_tables(rec_tables)
        grp_rec = t_rec.group_by(['lat', 'lon']).aggregate([('lat', 'count'), ('date', 'min'), ('date', 'max')])
        df_rec = grp_rec.to_pandas()
        df_rec.rename(columns={'lat_count': 'days', 'date_min': 'first_date', 'date_max': 'last_date'}, inplace=True)
        print(f"   Found {len(df_rec)} unique recurring flood locations in {year}")

        # Stratified sampling across a ~2.5km spatial grid to preserve entire geographic footprint
        # Sort by days descending so each grid cell keeps its most persistent flood point
        df_rec.sort_values('days', ascending=False, inplace=True)
        df_rec['grid_lat'] = (df_rec['lat'] * 40).round() / 40
        df_rec['grid_lon'] = (df_rec['lon'] * 40).round() / 40
        strat_rec = df_rec.groupby(['grid_lat', 'grid_lon']).head(1)
        if len(strat_rec) > max_recurring:
            strat_rec = strat_rec.iloc[:max_recurring]

        for _, row in strat_rec.iterrows():
            rec_pts.append({
                "lat": round(float(row['lat']), 4),
                "lon": round(float(row['lon']), 4),
                "type": "recurring",
                "year": year,
                "days": int(row['days']),
                "first_date": str(row['first_date']),
                "last_date": str(row['last_date'])
            })
        print(f"   Selected {len(rec_pts)} representative recurring flood points across all clusters")

    # --- 2. UNUSUAL FLOODS ---
    unu_tables = []
    for tile in tiles:
        fpath = Path(f"flood_masks/compact_unusual/flood_events_{tile}_{year}.parquet")
        if fpath.exists():
            tbl = pq.read_table(fpath, columns=['lat', 'lon', 'date'])
            tbl = tbl.set_column(tbl.schema.get_field_index('date'), 'date', pc.cast(tbl['date'], pa.string()))
            unu_tables.append(tbl)

    unu_pts = []
    if unu_tables:
        t_unu = pa.concat_tables(unu_tables)
        grp_unu = t_unu.group_by(['lat', 'lon']).aggregate([('lat', 'count'), ('date', 'min'), ('date', 'max')])
        df_unu = grp_unu.to_pandas()
        df_unu.rename(columns={'lat_count': 'days', 'date_min': 'first_date', 'date_max': 'last_date'}, inplace=True)
        print(f"   Found {len(df_unu)} unique unusual flood locations in {year}")

        # Stratified sampling across a ~2.5km grid, picking the most severely flooded pixel in each grid cell
        df_unu.sort_values('days', ascending=False, inplace=True)
        df_unu['grid_lat'] = (df_unu['lat'] * 40).round() / 40
        df_unu['grid_lon'] = (df_unu['lon'] * 40).round() / 40
        strat_unu = df_unu.groupby(['grid_lat', 'grid_lon']).head(1)
        if len(strat_unu) > max_unusual:
            strat_unu = strat_unu.iloc[:max_unusual]

        for _, row in strat_unu.iterrows():
            unu_pts.append({
                "lat": round(float(row['lat']), 4),
                "lon": round(float(row['lon']), 4),
                "type": "unusual",
                "year": year,
                "days": int(row['days']),
                "first_date": str(row['first_date']),
                "last_date": str(row['last_date'])
            })
        print(f"   Selected {len(unu_pts)} representative unusual flood points across all clusters")

    return rec_pts, unu_pts


def attach_administrative_info(points, admin2_gdf):
    print(f"-> Attaching County & State information to {len(points)} points via spatial join...")
    if not points:
        return points

    df_pts = pd.DataFrame(points)
    gdf_pts = gpd.GeoDataFrame(
        df_pts,
        geometry=gpd.points_from_xy(df_pts['lon'], df_pts['lat']),
        crs="EPSG:4326"
    )

    joined = gpd.sjoin(
        gdf_pts,
        admin2_gdf[['adm1_name', 'adm2_name', 'geometry']],
        how='left',
        predicate='within'
    )

    enriched_points = []
    for _, row in joined.iterrows():
        p = row.to_dict()
        p.pop('geometry', None)
        p.pop('index_right', None)
        
        # State and county names
        state = row['adm1_name'] if pd.notna(row['adm1_name']) else "Cross-border / Nile Basin"
        county = row['adm2_name'] if pd.notna(row['adm2_name']) else "Nile Basin"
        p['state'] = state
        p['county'] = county
        enriched_points.append(p)

    return enriched_points


def generate_html_map(output_path, all_points, admin1_json, admin2_json, sdn_json, year):
    print(f"-> Generating interactive HTML dashboard at {output_path}...")

    # Separate points by dataset type for clean client-side handling
    unusual_pts = [p for p in all_points if p['type'] == 'unusual']
    recurring_pts = [p for p in all_points if p['type'] == 'recurring']
    era5_pts = [p for p in all_points if p['type'] == 'era5']
    agera5_pts = [p for p in all_points if p['type'] == 'agera5']
    stations_pts = [p for p in all_points if p['type'] in ['lake_station', 'river_station']]

    # Metrics summary for HUD
    total_pts = len(all_points)
    max_days_flood = max([p.get('days', 0) for p in unusual_pts + recurring_pts], default=0)
    max_precip = max([p.get('annual_precip_mm', 0) for p in era5_pts], default=0)
    mean_precip = np.mean([p.get('annual_precip_mm', 0) for p in era5_pts]) if era5_pts else 0

    points_json_str = json.dumps({
        "unusual": unusual_pts,
        "recurring": recurring_pts,
        "era5": era5_pts,
        "agera5": agera5_pts,
        "stations": stations_pts
    })

    admin1_str = json.dumps(admin1_json)
    admin2_str = json.dumps(admin2_json)
    sdn_str = json.dumps(sdn_json) if sdn_json else "null"

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Sudan & South Sudan: NetCDF & Parquet Interactive Big-Data Map</title>
  
  <!-- Google Fonts -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  
  <!-- Leaflet & MarkerCluster CSS -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css" />
  <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css" />

  <style>
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}
    body {{
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      height: 100vh;
      width: 100vw;
      overflow: hidden;
      background: #0f172a;
      color: #0f172a;
    }}
    #map {{
      height: 100%;
      width: 100%;
      z-index: 1;
    }}

    /* Floating Dashboard Panel */
    .dashboard-panel {{
      position: absolute;
      top: 16px;
      left: 16px;
      width: 380px;
      max-height: calc(100vh - 32px);
      background: rgba(255, 255, 255, 0.94);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border-radius: 16px;
      box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(0, 0, 0, 0.06);
      z-index: 1000;
      overflow-y: auto;
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      display: flex;
      flex-direction: column;
    }}
    .dashboard-panel.collapsed {{
      transform: translateX(-400px);
    }}
    .panel-toggle-btn {{
      position: absolute;
      top: 16px;
      left: 16px;
      z-index: 999;
      background: #ffffff;
      border: 1px solid rgba(0, 0, 0, 0.1);
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
      border-radius: 12px;
      padding: 10px 14px;
      font-size: 13px;
      font-weight: 700;
      color: #0f172a;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .panel-toggle-btn:hover {{
      background: #f8fafc;
    }}

    .panel-header {{
      padding: 20px 20px 14px 20px;
      border-bottom: 1px solid #e2e8f0;
      position: relative;
    }}
    .panel-header h1 {{
      font-size: 18px;
      font-weight: 800;
      color: #0f172a;
      line-height: 1.3;
      letter-spacing: -0.02em;
    }}
    .panel-header p {{
      font-size: 12px;
      color: #64748b;
      margin-top: 4px;
      font-weight: 500;
    }}
    .close-btn {{
      position: absolute;
      top: 18px;
      right: 18px;
      background: #f1f5f9;
      border: none;
      width: 28px;
      height: 28px;
      border-radius: 50%;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 16px;
      color: #475569;
    }}
    .close-btn:hover {{
      background: #e2e8f0;
      color: #0f172a;
    }}

    .panel-body {{
      padding: 18px 20px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }}

    /* Stat Cards */
    .stat-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }}
    .stat-card {{
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 12px;
      padding: 10px 12px;
    }}
    .stat-card .val {{
      font-size: 18px;
      font-weight: 800;
      color: #0f172a;
    }}
    .stat-card .label {{
      font-size: 11px;
      font-weight: 600;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-top: 2px;
    }}

    /* Section Headers */
    .section-title {{
      font-size: 12px;
      font-weight: 700;
      color: #475569;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}

    /* Layer Checklist */
    .layer-item {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 9px 12px;
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 10px;
      margin-bottom: 6px;
      cursor: pointer;
      transition: all 0.15s ease;
    }}
    .layer-item:hover {{
      border-color: #cbd5e1;
      background: #f8fafc;
    }}
    .layer-item.active {{
      background: #f1f5f9;
      border-color: #94a3b8;
    }}
    .layer-left {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .color-dot {{
      width: 14px;
      height: 14px;
      border-radius: 4px;
      flex-shrink: 0;
    }}
    .layer-title {{
      font-size: 13px;
      font-weight: 600;
      color: #1e293b;
    }}
    .layer-tag {{
      font-size: 10px;
      font-weight: 700;
      padding: 2px 6px;
      border-radius: 4px;
      margin-left: 6px;
    }}
    .tag-parquet {{
      background: #fef3c7;
      color: #92400e;
    }}
    .tag-nc {{
      background: #e0e7ff;
      color: #3730a3;
    }}
    .tag-geojson {{
      background: #dcfce7;
      color: #166534;
    }}
    .layer-count {{
      font-size: 12px;
      font-weight: 700;
      color: #64748b;
    }}

    /* Filter Controls */
    .filter-btn-group {{
      display: flex;
      gap: 6px;
      background: #f1f5f9;
      padding: 4px;
      border-radius: 10px;
    }}
    .filter-btn {{
      flex: 1;
      border: none;
      background: transparent;
      padding: 7px 6px;
      font-size: 11px;
      font-weight: 600;
      color: #475569;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.15s ease;
    }}
    .filter-btn.active {{
      background: #ffffff;
      color: #0f172a;
      box-shadow: 0 2px 4px rgba(0,0,0,0.06);
      font-weight: 700;
    }}

    /* Dropdown Selector */
    select.custom-select {{
      width: 100%;
      padding: 9px 12px;
      border-radius: 10px;
      border: 1px solid #cbd5e1;
      background: #ffffff;
      font-family: inherit;
      font-size: 13px;
      font-weight: 600;
      color: #1e293b;
      cursor: pointer;
      outline: none;
    }}
    select.custom-select:focus {{
      border-color: #3b82f6;
      box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15);
    }}

    /* Search Input */
    .search-input {{
      width: 100%;
      padding: 9px 12px;
      border-radius: 10px;
      border: 1px solid #cbd5e1;
      font-family: inherit;
      font-size: 13px;
      outline: none;
    }}
    .search-input:focus {{
      border-color: #3b82f6;
      box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15);
    }}

    /* Custom Leaflet Cluster Badges */
    .marker-cluster-unusual {{
      background-color: rgba(225, 29, 72, 0.25);
    }}
    .marker-cluster-unusual div {{
      background: linear-gradient(135deg, #e11d48, #be123c);
      color: #ffffff;
      font-weight: 800;
      font-size: 12px;
      box-shadow: 0 4px 10px rgba(225, 29, 72, 0.5);
    }}

    .marker-cluster-recurring {{
      background-color: rgba(2, 132, 199, 0.25);
    }}
    .marker-cluster-recurring div {{
      background: linear-gradient(135deg, #0284c7, #0369a1);
      color: #ffffff;
      font-weight: 800;
      font-size: 12px;
      box-shadow: 0 4px 10px rgba(2, 132, 199, 0.5);
    }}

    .marker-cluster-era5 {{
      background-color: rgba(79, 70, 229, 0.25);
    }}
    .marker-cluster-era5 div {{
      background: linear-gradient(135deg, #4f46e5, #4338ca);
      color: #ffffff;
      font-weight: 800;
      font-size: 12px;
      box-shadow: 0 4px 10px rgba(79, 70, 229, 0.5);
    }}

    .marker-cluster-agera5 {{
      background-color: rgba(217, 119, 6, 0.25);
    }}
    .marker-cluster-agera5 div {{
      background: linear-gradient(135deg, #d97706, #b45309);
      color: #ffffff;
      font-weight: 800;
      font-size: 12px;
      box-shadow: 0 4px 10px rgba(217, 119, 6, 0.5);
    }}

    .marker-cluster-stations {{
      background-color: rgba(5, 150, 105, 0.25);
    }}
    .marker-cluster-stations div {{
      background: linear-gradient(135deg, #059669, #047857);
      color: #ffffff;
      font-weight: 800;
      font-size: 12px;
      box-shadow: 0 4px 10px rgba(5, 150, 105, 0.5);
    }}

    .custom-pin-icon {{
      background: transparent !important;
      border: none !important;
    }}

    /* Styled Leaflet Popups */
    .leaflet-popup-content-wrapper {{
      border-radius: 14px;
      box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.2), 0 10px 10px -5px rgba(0, 0, 0, 0.1);
      padding: 0;
      overflow: hidden;
    }}
    .leaflet-popup-content {{
      margin: 0;
      line-height: 1.4;
      font-family: 'Plus Jakarta Sans', sans-serif;
      min-width: 250px;
    }}
    .popup-header {{
      padding: 12px 16px;
      color: #ffffff;
      font-weight: 700;
      font-size: 13px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}
    .popup-header.unusual {{
      background: linear-gradient(135deg, #e11d48, #be123c);
    }}
    .popup-header.recurring {{
      background: linear-gradient(135deg, #0284c7, #0369a1);
    }}
    .popup-header.era5 {{
      background: linear-gradient(135deg, #4f46e5, #4338ca);
    }}
    .popup-header.agera5 {{
      background: linear-gradient(135deg, #d97706, #b45309);
    }}
    .popup-header.station {{
      background: linear-gradient(135deg, #059669, #047857);
    }}

    .popup-body {{
      padding: 14px 16px;
    }}
    .popup-title {{
      font-size: 15px;
      font-weight: 800;
      color: #0f172a;
      margin-bottom: 8px;
    }}
    .popup-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    .popup-table td {{
      padding: 4px 0;
      border-bottom: 1px solid #f1f5f9;
    }}
    .popup-table td.key {{
      color: #64748b;
      font-weight: 600;
      width: 45%;
    }}
    .popup-table td.val {{
      color: #0f172a;
      font-weight: 700;
      text-align: right;
    }}
    .popup-footer {{
      padding: 8px 16px;
      background: #f8fafc;
      border-top: 1px solid #e2e8f0;
      font-size: 11px;
      color: #64748b;
      font-weight: 500;
    }}

    /* Legend Box (Bottom Right) */
    .map-legend {{
      position: absolute;
      bottom: 24px;
      right: 24px;
      background: rgba(255, 255, 255, 0.92);
      backdrop-filter: blur(12px);
      padding: 12px 16px;
      border-radius: 12px;
      box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
      z-index: 1000;
      font-size: 12px;
      font-weight: 600;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .legend-row {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .legend-color {{
      width: 12px;
      height: 12px;
      border-radius: 3px;
    }}
  </style>
</head>
<body>

  <button class="panel-toggle-btn" id="openPanelBtn">
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M3 12h18M3 6h18M3 18h18"/></svg>
    Dashboard Controls
  </button>

  <div class="dashboard-panel" id="dashboardPanel">
    <div class="panel-header">
      <button class="close-btn" id="closePanelBtn">&times;</button>
      <h1>Sudan & South Sudan</h1>
      <p>Interactive NetCDF & Parquet Big-Data Map (Year {year})</p>
    </div>

    <div class="panel-body">
      <!-- Quick Stats -->
      <div class="stat-grid">
        <div class="stat-card">
          <div class="val" id="statTotalPts">{total_pts:,}</div>
          <div class="label">Total Points</div>
        </div>
        <div class="stat-card">
          <div class="val" id="statUnusualPts">{len(unusual_pts):,}</div>
          <div class="label">Unusual Floods</div>
        </div>
        <div class="stat-card">
          <div class="val" id="statMeanPrecip">{mean_precip:.0f} mm</div>
          <div class="label">Mean Rainfall (NC)</div>
        </div>
        <div class="stat-card">
          <div class="val" id="statMaxDays">{max_days_flood} d</div>
          <div class="label">Max Flood Days</div>
        </div>
      </div>

      <!-- Quick Fly to County / Hotspot -->
      <div>
        <div class="section-title">Jump to Region / Hotspot</div>
        <select class="custom-select" id="quickFlySelect">
          <option value="">-- Choose Hotspot --</option>
          <option value="9.26,29.83,9">Bentiu & Rubkona (Catastrophic Floods)</option>
          <option value="8.16,30.25,9">Mayendit & Leer (Sudd Wetland Corridor)</option>
          <option value="9.08,30.88,9">Fangak (Zeraf River Confluence)</option>
          <option value="9.53,31.65,9">Malakal & Baliet (Sobat / White Nile)</option>
          <option value="6.20,31.55,9">Bor Floodplain (Jonglei)</option>
          <option value="4.85,31.60,10">Juba Capital (White Nile)</option>
          <option value="7.70,27.99,9">Wau (Bahr el Ghazal Basin)</option>
          <option value="15.50,32.55,9">Khartoum (Sudan - Main Confluence)</option>
          <option value="1.68,30.92,10">Lake Albert (DAHITI Altimetry Station)</option>
          <option value="9.5,30.5,6">Reset: South Sudan View</option>
          <option value="13.0,30.5,5">Reset: Entire Sudan & South Sudan</option>
        </select>
      </div>

      <!-- Filter Parquet Floods by Inundation Persistence -->
      <div>
        <div class="section-title">Filter Inundation Persistence</div>
        <div class="filter-btn-group" id="persistenceFilter">
          <button class="filter-btn active" data-min="1">All</button>
          <button class="filter-btn" data-min="5">&ge; 5 Days</button>
          <button class="filter-btn" data-min="15">&ge; 15 Days</button>
          <button class="filter-btn" data-min="30">&ge; 30 Days</button>
        </div>
      </div>

      <!-- Dataset Layers -->
      <div>
        <div class="section-title">
          <span>Active Datasets</span>
          <span style="font-size: 10px; color: #94a3b8;">TOGGLE LAYERS</span>
        </div>

        <div class="layer-item active" id="toggleUnusual">
          <div class="layer-left">
            <div class="color-dot" style="background: #e11d48;"></div>
            <div>
              <span class="layer-title">Unusual Floods</span>
              <span class="layer-tag tag-parquet">PARQUET</span>
            </div>
          </div>
          <span class="layer-count" id="countUnusual">{len(unusual_pts):,}</span>
        </div>

        <div class="layer-item active" id="toggleRecurring">
          <div class="layer-left">
            <div class="color-dot" style="background: #0284c7;"></div>
            <div>
              <span class="layer-title">Recurring Floods</span>
              <span class="layer-tag tag-parquet">PARQUET</span>
            </div>
          </div>
          <span class="layer-count" id="countRecurring">{len(recurring_pts):,}</span>
        </div>

        <div class="layer-item active" id="toggleERA5">
          <div class="layer-left">
            <div class="color-dot" style="background: #4f46e5;"></div>
            <div>
              <span class="layer-title">ERA5 Rain & Runoff</span>
              <span class="layer-tag tag-nc">NETCDF</span>
            </div>
          </div>
          <span class="layer-count">{len(era5_pts):,}</span>
        </div>

        <div class="layer-item active" id="toggleAgERA5">
          <div class="layer-left">
            <div class="color-dot" style="background: #d97706;"></div>
            <div>
              <span class="layer-title">AgERA5 Evapotranspiration</span>
              <span class="layer-tag tag-nc">NETCDF</span>
            </div>
          </div>
          <span class="layer-count">{len(agera5_pts):,}</span>
        </div>

        <div class="layer-item active" id="toggleStations">
          <div class="layer-left">
            <div class="color-dot" style="background: #059669;"></div>
            <div>
              <span class="layer-title">Lake & River Gauges</span>
              <span class="layer-tag tag-nc">NC + DFO</span>
            </div>
          </div>
          <span class="layer-count">{len(stations_pts)}</span>
        </div>

        <div class="layer-item active" id="toggleAdmin2">
          <div class="layer-left">
            <div class="color-dot" style="background: #10b981;"></div>
            <div>
              <span class="layer-title">South Sudan Counties</span>
              <span class="layer-tag tag-geojson">GEOJSON</span>
            </div>
          </div>
          <span class="layer-count">79</span>
        </div>

        <div class="layer-item active" id="toggleSudan">
          <div class="layer-left">
            <div class="color-dot" style="background: #64748b;"></div>
            <div>
              <span class="layer-title">Sudan Country Border</span>
              <span class="layer-tag tag-geojson">GEOJSON</span>
            </div>
          </div>
          <span class="layer-count">1</span>
        </div>
      </div>

      <!-- Basemap Selector -->
      <div>
        <div class="section-title">Base Map Style</div>
        <select class="custom-select" id="basemapSelect">
          <option value="osm">OpenStreetMap Standard</option>
          <option value="satellite" selected>Esri World Imagery (Satellite)</option>
          <option value="positron">CartoDB Positron (Light Minimal)</option>
          <option value="dark">CartoDB Dark Matter (Night Mode)</option>
        </select>
      </div>

    </div>
  </div>

  <!-- Legend Box -->
  <div class="map-legend">
    <div style="font-weight: 800; font-size: 11px; text-transform: uppercase; color: #475569; margin-bottom: 2px;">Data Points</div>
    <div class="legend-row">
      <div class="legend-color" style="background: #e11d48;"></div>
      <span>Parquet Unusual Flood</span>
    </div>
    <div class="legend-row">
      <div class="legend-color" style="background: #0284c7;"></div>
      <span>Parquet Recurring Flood</span>
    </div>
    <div class="legend-row">
      <div class="legend-color" style="background: #4f46e5;"></div>
      <span>NetCDF ERA5 Rain/Runoff</span>
    </div>
    <div class="legend-row">
      <div class="legend-color" style="background: #d97706;"></div>
      <span>NetCDF AgERA5 ET</span>
    </div>
    <div class="legend-row">
      <div class="legend-color" style="background: #059669;"></div>
      <span>NetCDF Lake / River Station</span>
    </div>
  </div>

  <!-- Map Container -->
  <div id="map"></div>

  <!-- Leaflet & MarkerCluster JS -->
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js"></script>

  <script>
    // 1. EMBEDDED DATA FROM PYTHON PIPELINE
    const dataPoints = {points_json_str};
    const admin1GeoJSON = {admin1_str};
    const admin2GeoJSON = {admin2_str};
    const sdnGeoJSON = {sdn_str};

    // 2. INITIALIZE MAP CENTERED ON SUDAN & SOUTH SUDAN
    const map = L.map('map', {{
      center: [9.5, 31.0],
      zoom: 6,
      zoomControl: false,
      attributionControl: true
    }});

    L.control.zoom({{ position: 'bottomright' }}).addTo(map);
    L.control.scale({{ position: 'bottomleft', imperial: false }}).addTo(map);

    // 3. TILE LAYERS
    const baseLayers = {{
      satellite: L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
        attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
        maxZoom: 18
      }}),
      osm: L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 19
      }}),
      positron: L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{y}}/{{x}}{{r}}.png', {{
        attribution: '&copy; CartoDB &copy; OpenStreetMap contributors',
        maxZoom: 19
      }}),
      dark: L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{y}}/{{x}}{{r}}.png', {{
        attribution: '&copy; CartoDB &copy; OpenStreetMap contributors',
        maxZoom: 19
      }})
    }};

    baseLayers.satellite.addTo(map);

    // Switch Basemap Handler
    document.getElementById('basemapSelect').addEventListener('change', function(e) {{
      const selected = e.target.value;
      Object.values(baseLayers).forEach(layer => map.removeLayer(layer));
      if (baseLayers[selected]) {{
        baseLayers[selected].addTo(map);
      }}
    }});

    // 4. BOUNDARY LAYERS
    let admin2Layer = null;
    let admin1Layer = null;
    let sdnLayer = null;

    if (admin2GeoJSON) {{
      admin2Layer = L.geoJSON(admin2GeoJSON, {{
        style: {{
          color: '#10b981',
          weight: 1.2,
          opacity: 0.6,
          fillColor: '#10b981',
          fillOpacity: 0.04
        }},
        onEachFeature: function(feature, layer) {{
          const p = feature.properties;
          layer.bindTooltip(`<strong>${{p.adm2_name || 'County'}}</strong><br><span style="font-size:11px;color:#64748b">${{p.adm1_name || 'State'}}</span>`, {{
            sticky: true
          }});
        }}
      }}).addTo(map);
    }}

    if (sdnGeoJSON) {{
      sdnLayer = L.geoJSON(sdnGeoJSON, {{
        style: {{
          color: '#64748b',
          weight: 2,
          opacity: 0.8,
          fillColor: '#64748b',
          fillOpacity: 0.02,
          dashArray: '4, 4'
        }},
        onEachFeature: function(feature, layer) {{
          layer.bindTooltip('<strong>Republic of the Sudan</strong>', {{ sticky: true }});
        }}
      }}).addTo(map);
    }}

    // 5. CUSTOM CLUSTERING & MARKER BUILDERS
    function createClusterGroup(className) {{
      return L.markerClusterGroup({{
        chunkedLoading: true,
        maxClusterRadius: 48,
        spiderfyOnMaxZoom: true,
        showCoverageOnHover: false,
        zoomToBoundsOnClick: true,
        iconCreateFunction: function(cluster) {{
          const count = cluster.getChildCount();
          let size = 'small';
          if (count > 100) size = 'medium';
          if (count > 1000) size = 'large';
          return L.divIcon({{
            html: `<div><span>${{count.toLocaleString()}}</span></div>`,
            className: `marker-cluster ${{className}} marker-cluster-${{size}}`,
            iconSize: L.point(40, 40)
          }});
        }}
      }});
    }}

    const clusterUnusual = createClusterGroup('marker-cluster-unusual');
    const clusterRecurring = createClusterGroup('marker-cluster-recurring');
    const clusterERA5 = createClusterGroup('marker-cluster-era5');
    const clusterAgERA5 = createClusterGroup('marker-cluster-agera5');
    const clusterStations = createClusterGroup('marker-cluster-stations');

    // Marker Icon Generator
    function makePinIcon(color, svgPath) {{
      return L.divIcon({{
        className: 'custom-pin-icon',
        html: `<svg width="24" height="24" viewBox="0 0 24 24" fill="none">
          <circle cx="12" cy="12" r="9" fill="${{color}}" stroke="#ffffff" stroke-width="2" />
          <circle cx="12" cy="12" r="4" fill="#ffffff" />
        </svg>`,
        iconSize: [24, 24],
        iconAnchor: [12, 12],
        popupAnchor: [0, -12]
      }});
    }}

    const iconUnusual = makePinIcon('#e11d48');
    const iconRecurring = makePinIcon('#0284c7');
    const iconERA5 = makePinIcon('#4f46e5');
    const iconAgERA5 = makePinIcon('#d97706');
    const iconStation = makePinIcon('#059669');

    // 6. POPULATE CLUSTER GROUPS
    let currentMinDays = 1;

    function buildUnusualMarkers(minDays = 1) {{
      clusterUnusual.clearLayers();
      const markers = [];
      dataPoints.unusual.forEach(p => {{
        if (p.days < minDays) return;
        const m = L.marker([p.lat, p.lon], {{ icon: iconUnusual }});
        m.bindPopup(`
          <div class="popup-header unusual">
            <span>&#9888;&#65039; UNUSUAL FLOOD EVENT</span>
            <span style="font-size:10px;opacity:0.9">MCDWD VIIRS/MODIS</span>
          </div>
          <div class="popup-body">
            <div class="popup-title">${{p.county || 'South Sudan County'}}</div>
            <table class="popup-table">
              <tr><td class="key">State</td><td class="val">${{p.state || 'N/A'}}</td></tr>
              <tr><td class="key">Flood Duration</td><td class="val" style="color:#e11d48;font-weight:800;">${{p.days}} Days Detected</td></tr>
              <tr><td class="key">Active Window</td><td class="val">${{p.first_date}} &rarr; ${{p.last_date}}</td></tr>
              <tr><td class="key">Coordinates</td><td class="val">${{p.lat.toFixed(4)}}&deg;N, ${{p.lon.toFixed(4)}}&deg;E</td></tr>
              <tr><td class="key">Year Analyzed</td><td class="val">${{p.year}}</td></tr>
            </table>
          </div>
          <div class="popup-footer">Source: Parquet compact_unusual flood_events</div>
        `);
        markers.push(m);
      }});
      clusterUnusual.addLayers(markers);
      document.getElementById('countUnusual').innerText = markers.length.toLocaleString();
    }}

    function buildRecurringMarkers(minDays = 1) {{
      clusterRecurring.clearLayers();
      const markers = [];
      dataPoints.recurring.forEach(p => {{
        if (p.days < minDays) return;
        const m = L.marker([p.lat, p.lon], {{ icon: iconRecurring }});
        m.bindPopup(`
          <div class="popup-header recurring">
            <span>&#127754; RECURRING FLOOD EVENT</span>
            <span style="font-size:10px;opacity:0.9">MCDWD VIIRS/MODIS</span>
          </div>
          <div class="popup-body">
            <div class="popup-title">${{p.county || 'South Sudan County'}}</div>
            <table class="popup-table">
              <tr><td class="key">State</td><td class="val">${{p.state || 'N/A'}}</td></tr>
              <tr><td class="key">Flood Duration</td><td class="val" style="color:#0284c7;font-weight:800;">${{p.days}} Days Detected</td></tr>
              <tr><td class="key">Active Window</td><td class="val">${{p.first_date}} &rarr; ${{p.last_date}}</td></tr>
              <tr><td class="key">Coordinates</td><td class="val">${{p.lat.toFixed(4)}}&deg;N, ${{p.lon.toFixed(4)}}&deg;E</td></tr>
              <tr><td class="key">Classification</td><td class="val">Regular Seasonal Inundation</td></tr>
            </table>
          </div>
          <div class="popup-footer">Source: Parquet compact_recurring flood_events</div>
        `);
        markers.push(m);
      }});
      clusterRecurring.addLayers(markers);
      document.getElementById('countRecurring').innerText = markers.length.toLocaleString();
    }}

    // ERA5 Grid Points
    const era5Markers = [];
    dataPoints.era5.forEach(p => {{
      const m = L.marker([p.lat, p.lon], {{ icon: iconERA5 }});
      m.bindPopup(`
        <div class="popup-header era5">
          <span>&#127783;&#65039; ERA5 CLIMATE GRID POINT</span>
          <span style="font-size:10px;opacity:0.9">ECMWF NetCDF</span>
        </div>
        <div class="popup-body">
          <div class="popup-title">${{p.county || 'Sudan / Nile Basin'}}</div>
          <table class="popup-table">
            <tr><td class="key">Annual Precipitation</td><td class="val" style="color:#4f46e5;font-weight:800;">${{p.annual_precip_mm.toLocaleString()}} mm</td></tr>
            <tr><td class="key">Annual Runoff</td><td class="val">${{p.annual_runoff_mm.toLocaleString()}} mm</td></tr>
            <tr><td class="key">Peak Daily Rainfall</td><td class="val">${{p.peak_daily_mm}} mm/day</td></tr>
            <tr><td class="key">State / Province</td><td class="val">${{p.state || 'Nile Basin'}}</td></tr>
            <tr><td class="key">Coordinates</td><td class="val">${{p.lat.toFixed(4)}}&deg;N, ${{p.lon.toFixed(4)}}&deg;E</td></tr>
            <tr><td class="key">Resolution</td><td class="val">0.25&deg; &times; 0.25&deg; (~28 km)</td></tr>
          </table>
        </div>
        <div class="popup-footer">Source: NetCDF rainfall and runoff/${{p.file}}</div>
      `);
      era5Markers.push(m);
    }});
    clusterERA5.addLayers(era5Markers);

    // AgERA5 ET Grid Points
    const agera5Markers = [];
    dataPoints.agera5.forEach(p => {{
      const m = L.marker([p.lat, p.lon], {{ icon: iconAgERA5 }});
      m.bindPopup(`
        <div class="popup-header agera5">
          <span>&#9728;&#65039; AgERA5 EVAPOTRANSPIRATION</span>
          <span style="font-size:10px;opacity:0.9">Copernicus C3S NetCDF</span>
        </div>
        <div class="popup-body">
          <div class="popup-title">${{p.county || 'Sudan / Nile Basin'}}</div>
          <table class="popup-table">
            <tr><td class="key">Mean Reference ET</td><td class="val" style="color:#d97706;font-weight:800;">${{p.mean_et_mm_day}} mm/day</td></tr>
            <tr><td class="key">Estimated Annual ET</td><td class="val">${{p.est_annual_et_mm.toLocaleString()}} mm/year</td></tr>
            <tr><td class="key">Methodology</td><td class="val">FAO-56 Penman-Monteith</td></tr>
            <tr><td class="key">Coordinates</td><td class="val">${{p.lat.toFixed(4)}}&deg;N, ${{p.lon.toFixed(4)}}&deg;E</td></tr>
          </table>
        </div>
        <div class="popup-footer">Source: NetCDF evapotranspiration/ET_${{p.year}}</div>
      `);
      agera5Markers.push(m);
    }});
    clusterAgERA5.addLayers(agera5Markers);

    // Lake & River Stations
    const stationMarkers = [];
    dataPoints.stations.forEach(p => {{
      const m = L.marker([p.lat, p.lon], {{ icon: iconStation }});
      let html = '';
      if (p.type === 'lake_station') {{
        html = `
          <div class="popup-header station">
            <span>&#128752; SATELLITE ALTIMETRY STATION</span>
            <span style="font-size:10px;opacity:0.9">DAHITI NetCDF</span>
          </div>
          <div class="popup-body">
            <div class="popup-title">${{p.name}}</div>
            <table class="popup-table">
              <tr><td class="key">Target Lake</td><td class="val">Lake Albert (Target 85)</td></tr>
              <tr><td class="key">Latest Water Level</td><td class="val" style="color:#059669;font-weight:800;">${{p.latest_wl}} m</td></tr>
              <tr><td class="key">Mean Water Level</td><td class="val">${{p.mean_wl}} m</td></tr>
              <tr><td class="key">Historical Range</td><td class="val">${{p.min_wl}} &rarr; ${{p.max_wl}} m</td></tr>
              <tr><td class="key">Institution</td><td class="val">${{p.institution}}</td></tr>
              <tr><td class="key">Coordinates</td><td class="val">${{p.lat.toFixed(4)}}&deg;N, ${{p.lon.toFixed(4)}}&deg;E</td></tr>
            </table>
          </div>
          <div class="popup-footer">Source: NetCDF ${{p.file}}</div>
        `;
      }} else {{
        html = `
          <div class="popup-header station">
            <span>&#128207; RIVER DISCHARGE STATION</span>
            <span style="font-size:10px;opacity:0.9">DFO Gauging</span>
          </div>
          <div class="popup-body">
            <div class="popup-title">${{p.name}}</div>
            <table class="popup-table">
              <tr><td class="key">Country</td><td class="val">${{p.country}}</td></tr>
              <tr><td class="key">Station Number</td><td class="val">${{p.station_nr}}</td></tr>
              <tr><td class="key">Area ID</td><td class="val">${{p.area_id}}</td></tr>
              <tr><td class="key">Mean Discharge</td><td class="val">${{p.mean_discharge_m3s ? p.mean_discharge_m3s + ' m&sup3;/s' : 'Available'}}</td></tr>
              <tr><td class="key">Coordinates</td><td class="val">${{p.lat.toFixed(4)}}&deg;N, ${{p.lon.toFixed(4)}}&deg;E</td></tr>
            </table>
          </div>
          <div class="popup-footer">Source: Dartmouth Flood Observatory/information.xlsx</div>
        `;
      }}
      m.bindPopup(html);
      stationMarkers.push(m);
    }});
    clusterStations.addLayers(stationMarkers);

    // Initial Build
    buildUnusualMarkers(1);
    buildRecurringMarkers(1);

    // Add layers to map
    map.addLayer(clusterUnusual);
    map.addLayer(clusterRecurring);
    map.addLayer(clusterERA5);
    map.addLayer(clusterAgERA5);
    map.addLayer(clusterStations);

    // 7. FILTER BUTTONS HANDLER
    document.querySelectorAll('#persistenceFilter .filter-btn').forEach(btn => {{
      btn.addEventListener('click', function() {{
        document.querySelectorAll('#persistenceFilter .filter-btn').forEach(b => b.classList.remove('active'));
        this.classList.add('active');
        const minDays = parseInt(this.getAttribute('data-min'));
        buildUnusualMarkers(minDays);
        buildRecurringMarkers(minDays);
      }});
    }});

    // 8. LAYER TOGGLE HANDLERS
    function setupToggle(btnId, layer) {{
      const el = document.getElementById(btnId);
      el.addEventListener('click', function() {{
        if (map.hasLayer(layer)) {{
          map.removeLayer(layer);
          el.classList.remove('active');
        }} else {{
          map.addLayer(layer);
          el.classList.add('active');
        }}
      }});
    }}

    setupToggle('toggleUnusual', clusterUnusual);
    setupToggle('toggleRecurring', clusterRecurring);
    setupToggle('toggleERA5', clusterERA5);
    setupToggle('toggleAgERA5', clusterAgERA5);
    setupToggle('toggleStations', clusterStations);
    if (admin2Layer) setupToggle('toggleAdmin2', admin2Layer);
    if (sdnLayer) setupToggle('toggleSudan', sdnLayer);

    // 9. QUICK FLY SELECTOR
    document.getElementById('quickFlySelect').addEventListener('change', function(e) {{
      const val = e.target.value;
      if (!val) return;
      const parts = val.split(',');
      const lat = parseFloat(parts[0]);
      const lon = parseFloat(parts[1]);
      const zoom = parseInt(parts[2]);
      map.flyTo([lat, lon], zoom, {{ duration: 1.5 }});
    }});

    // 10. COLLAPSIBLE PANEL
    const panel = document.getElementById('dashboardPanel');
    const openBtn = document.getElementById('openPanelBtn');
    const closeBtn = document.getElementById('closePanelBtn');

    closeBtn.addEventListener('click', () => {{
      panel.classList.add('collapsed');
      openBtn.style.display = 'flex';
    }});

    openBtn.addEventListener('click', () => {{
      panel.classList.remove('collapsed');
      openBtn.style.display = 'none';
    }});
    openBtn.style.display = 'none';

  </script>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_template)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"-> SUCCESS! Map written to {output_path} ({file_size_mb:.2f} MB)")


def main():
    args = parse_args()
    year = args.year
    output = args.output
    t0 = time.time()

    print(f"============================================================")
    print(f"BUILDING INTERACTIVE MAP FOR SUDAN & SOUTH SUDAN (YEAR {year})")
    print(f"============================================================")

    # 1. Boundaries
    admin1, admin2, admin1_json, admin2_json, sdn_json = load_boundaries()

    # 2. NetCDF: Lake Albert
    lake_pt = load_lake_station()

    # 3. Dartmouth Stations
    dartmouth_pts = load_dartmouth_stations()

    station_pts = []
    if lake_pt:
        station_pts.append(lake_pt)
    station_pts.extend(dartmouth_pts)

    # 4. NetCDF: ERA5 Precipitation & Runoff
    era5_pts = load_era5_rainfall_runoff(year)

    # 5. NetCDF: AgERA5 Evapotranspiration
    agera5_pts = load_agera5_evapotranspiration(year)

    # 6. Parquet: Flood Masks (Unusual & Recurring)
    rec_pts, unu_pts = load_parquet_floods(year, args.max_unusual_pts, args.max_recurring_pts)

    # 7. Spatial Join with Admin2 to attach County and State to flood points
    all_flood_pts = rec_pts + unu_pts
    enriched_flood_pts = attach_administrative_info(all_flood_pts, admin2)

    # Re-split enriched flood points
    enriched_rec = [p for p in enriched_flood_pts if p['type'] == 'recurring']
    enriched_unu = [p for p in enriched_flood_pts if p['type'] == 'unusual']

    # Attach county/state to ERA5 & AgERA5 points
    enriched_era5 = attach_administrative_info(era5_pts, admin2)
    enriched_agera5 = attach_administrative_info(agera5_pts, admin2)

    all_points = enriched_unu + enriched_rec + enriched_era5 + enriched_agera5 + station_pts

    print(f"\nSummary of mapped points:")
    print(f"  Parquet Unusual Floods:     {len(enriched_unu):,} points")
    print(f"  Parquet Recurring Floods:   {len(enriched_rec):,} points")
    print(f"  NetCDF ERA5 Rain & Runoff:  {len(enriched_era5):,} points")
    print(f"  NetCDF AgERA5 ET:           {len(enriched_agera5):,} points")
    print(f"  Hydrological Gauges:        {len(station_pts):,} points")
    print(f"  Total Active Points:        {len(all_points):,} points")

    # 8. Generate HTML
    generate_html_map(output, all_points, admin1_json, admin2_json, sdn_json, year)

    print(f"\nCompleted in {time.time()-t0:.2f}s!")
    print(f"Open in your browser: open {output}")


if __name__ == "__main__":
    main()
