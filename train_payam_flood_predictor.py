#!/usr/bin/env python3
"""
train_payam_flood_predictor.py

Comprehensive Machine Learning Pipeline and Predictive Modeling System
for Flood Drivers Across All 512 Payams (ADM3) in South Sudan.

Determines for EVERY Payam:
- Which type of driver dominates (Extreme Rainfall, Surface Runoff, Riverine Inundation, Farmland Exposure)
- When it occurs (Peak month, onset, recession, 12-month seasonal curve)
- How much (Predicted rainfall mm, runoff mm, flood extent %, flooded km2, risk score 0-100, population & cropland exposed)

Outputs:
- flood_driver_analysis/models/payam_flood_regressor.joblib
- flood_driver_analysis/models/payam_driver_classifier.joblib
- flood_driver_analysis/data/payam_flood_predictions_monthly.csv
- flood_driver_analysis/data/payam_flood_predictions_annual_summary.csv
- flood_driver_analysis/data/payam_flood_predictions.geojson
- flood_driver_analysis/graphs/payam_prediction_model_performance.png
- flood_driver_analysis/graphs/payam_flood_drivers_seasonal_timeline.png
- flood_driver_analysis/graphs/payam_risk_and_driver_distribution.png
"""

import os
import sys
import time
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import rasterio
from rasterio.mask import mask
import joblib

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, classification_report, accuracy_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "flood_driver_analysis" / "data"
MODELS_DIR = BASE_DIR / "flood_driver_analysis" / "models"
GRAPHS_DIR = BASE_DIR / "flood_driver_analysis" / "graphs"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_NAMES_FULL = ["January", "February", "March", "April", "May", "June", 
                    "July", "August", "September", "October", "November", "December"]

DRIVER_COLORS = {
    "Extreme Rainfall": "#2ca02c",      # Forest green
    "Surface Runoff": "#9467bd",        # Purple
    "Riverine Inundation": "#1f77b4",   # Deep blue
    "Farmland Exposure": "#ff7f0e",     # Amber orange
}

def load_payam_geometries_and_exposure():
    print("\n[Step 1/7] Ingesting Payam Geometries, Cropland & Population Data...")
    t0 = time.time()
    
    adm3_path = BASE_DIR / "Administrative boundaries" / "ssd_admin3.geojson"
    adm3 = gpd.read_file(adm3_path)
    if adm3.crs is None:
        adm3 = adm3.set_crs("EPSG:4326")
    else:
        adm3 = adm3.to_crs("EPSG:4326")
        
    # Project to planar equal area for accurate geometry
    adm3["area_km2"] = adm3.to_crs("+proj=cea +lon_0=30 +lat_ts=8").geometry.area / 1e6
    
    rep_pts = adm3.geometry.representative_point()
    adm3["rep_lat"] = rep_pts.y
    adm3["rep_lon"] = rep_pts.x
    
    # Ingest existing cropland mask if present
    existing_csv = DATA_DIR / "south_sudan_adm3_flood_drivers.csv"
    if existing_csv.exists():
        df_exist = pd.read_csv(existing_csv)
        if "farmland_pct" in df_exist.columns:
            map_farm = dict(zip(df_exist["adm3_pcode"], df_exist["farmland_pct"]))
            adm3["farmland_pct"] = adm3["adm3_pcode"].map(map_farm).fillna(0.0)
    else:
        # Fallback extraction from asap crop mask
        crop_path = BASE_DIR / "farmland" / "asap_mask_crop_v04.tif"
        if crop_path.exists():
            with rasterio.open(crop_path) as src:
                crop_means = []
                for idx, row in adm3.iterrows():
                    try:
                        out_img, _ = mask(src, [row.geometry], crop=True, nodata=src.nodata)
                        val = out_img[0]
                        valid = val[val != src.nodata]
                        crop_means.append(float(np.nanmean(valid)) if len(valid) > 0 else 0.0)
                    except Exception:
                        crop_means.append(0.0)
                adm3["farmland_pct"] = crop_means
        else:
            adm3["farmland_pct"] = 0.05
            
    # Extract WorldPop 2024 population count per payam
    pop_raster = BASE_DIR / "worldpop" / "ssd_pop_2024_CN_100m_R2025A_v1.tif"
    if pop_raster.exists():
        print("      Extracting WorldPop 2024 population zonal stats per payam...")
        pops = []
        with rasterio.open(pop_raster) as src:
            for idx, row in adm3.iterrows():
                try:
                    out_img, _ = mask(src, [row.geometry], crop=True, nodata=src.nodata)
                    val = out_img[0]
                    valid = val[(val != src.nodata) & (val >= 0) & (~np.isnan(val))]
                    pops.append(float(np.sum(valid)))
                except Exception:
                    pops.append(0.0)
        adm3["population_est"] = pops
    else:
        adm3["population_est"] = adm3["area_km2"] * 25.0 # fallback density
        
    print(f"      Loaded {len(adm3)} payams. Total Population: {adm3['population_est'].sum():,.0f} in {time.time()-t0:.2f}s")
    return adm3

def extract_multiyear_era5(adm3, years=[2020, 2021, 2022, 2023, 2024]):
    print(f"\n[Step 2/7] Extracting Monthly ERA5 Rainfall & Runoff across {years}...")
    t0 = time.time()
    
    lats = adm3["rep_lat"].values
    lons = adm3["rep_lon"].values
    
    # Dictionary to hold monthly data: (year, month) -> (tp_array, ro_array)
    era5_records = []
    
    for y in years:
        nc_path = BASE_DIR / "rainfall and runoff" / f"ERA5_{y}.nc"
        if not nc_path.exists():
            continue
        ds = xr.open_dataset(nc_path)
        sub = ds[['tp', 'ro']].sel(latitude=slice(12.5, 3.5), longitude=slice(23.5, 36.0))
        monthly = sub.resample(valid_time='1MS').sum().load()
        ds.close()
        
        tp_monthly = monthly['tp'].values * 1000.0 # mm
        ro_monthly = monthly['ro'].values * 1000.0 # mm
        grid_lats = monthly['latitude'].values
        grid_lons = monthly['longitude'].values
        
        if grid_lats[0] > grid_lats[-1]:
            grid_lats = grid_lats[::-1]
            tp_monthly = tp_monthly[:, ::-1, :]
            ro_monthly = ro_monthly[:, ::-1, :]
            
        interp_tp = RegularGridInterpolator((np.arange(12), grid_lats, grid_lons), tp_monthly, bounds_error=False, fill_value=None)
        interp_ro = RegularGridInterpolator((np.arange(12), grid_lats, grid_lons), ro_monthly, bounds_error=False, fill_value=None)
        
        for m in range(12):
            month_num = m + 1
            sample_pts = np.column_stack([np.full(len(lats), m), lats, lons])
            tp_vals = np.maximum(0.0, interp_tp(sample_pts))
            ro_vals = np.maximum(0.0, interp_ro(sample_pts))
            
            for i, pcode in enumerate(adm3["adm3_pcode"]):
                era5_records.append({
                    "adm3_pcode": pcode,
                    "year": y,
                    "month": month_num,
                    "tp_mm": float(tp_vals[i]),
                    "ro_mm": float(ro_vals[i]),
                })
                
    df_era5 = pd.DataFrame(era5_records)
    print(f"      Extracted {len(df_era5)} payam-month weather records in {time.time()-t0:.2f}s")
    return df_era5

def extract_multiyear_flood_masks(adm3, years=[2020, 2021, 2022, 2023, 2024]):
    print(f"\n[Step 3/7] Ingesting NASA MCDWD MODIS/VIIRS Satellite Flood Masks across {years}...")
    t0 = time.time()
    
    tiles = ['h20v08', 'h21v08']
    dfs = []
    for y in years:
        for t in tiles:
            for typ in ['compact_recurring', 'compact_unusual']:
                p = BASE_DIR / "flood_masks" / typ / f"flood_events_{t}_{y}.parquet"
                if p.exists():
                    df = pd.read_parquet(p, columns=['lat', 'lon', 'date'])
                    df['year'] = y
                    dfs.append(df)
                    
    all_flood = pd.concat(dfs, ignore_index=True)
    all_flood['month'] = pd.to_datetime(all_flood['date']).dt.month
    
    # Deduplicate unique coordinates for high-speed spatial join
    unique_pts = all_flood[['lat', 'lon']].drop_duplicates()
    gdf_pts = gpd.GeoDataFrame(unique_pts, geometry=gpd.points_from_xy(unique_pts['lon'], unique_pts['lat']), crs='EPSG:4326')
    
    joined = gpd.sjoin(gdf_pts, adm3[['adm3_pcode', 'geometry']], how='inner', predicate='within')
    coord_to_pcode = dict(zip(zip(joined['lat'], joined['lon']), joined['adm3_pcode']))
    
    # Map points
    pts_tuples = list(zip(all_flood['lat'], all_flood['lon']))
    all_flood['adm3_pcode'] = [coord_to_pcode.get(pt, None) for pt in pts_tuples]
    all_flood = all_flood.dropna(subset=['adm3_pcode'])
    
    # Group by (adm3_pcode, year, month) to count unique flooded pixels per month
    # Each pixel is approx 232m x 232m = 0.0538 km2
    monthly_flood = all_flood.groupby(['adm3_pcode', 'year', 'month']).size().reset_index(name='flood_pixels')
    monthly_flood['flooded_area_km2'] = monthly_flood['flood_pixels'] * 0.0538
    
    print(f"      Processed {len(all_flood)} flood pixel events into {len(monthly_flood)} payam-year-month aggregates in {time.time()-t0:.2f}s")
    return monthly_flood

def extract_upstream_hydrology():
    print("\n[Step 4/7] Ingesting Upstream Nile River Discharge & Lake Albert Altimetry...")
    t0 = time.time()
    
    # Station 100205 in South Sudan (White Nile)
    dfo_file = BASE_DIR / "Darthmouth Flood Observatory" / "100205_discharge.csv"
    df_dfo = pd.read_csv(dfo_file)
    df_dfo['Date'] = pd.to_datetime(df_dfo['Date'])
    df_dfo['year'] = df_dfo['Date'].dt.year
    df_dfo['month'] = df_dfo['Date'].dt.month
    monthly_dfo = df_dfo.groupby(['year', 'month'])['Discharge (m3/s)'].mean().reset_index()
    monthly_dfo.rename(columns={'Discharge (m3/s)': 'nile_discharge_m3s'}, inplace=True)
    
    # Lake Albert Altimetry
    alb_file = BASE_DIR / "Water levels lakes" / "water_level_altimetry_Albert.nc"
    ds_alb = xr.open_dataset(alb_file)
    df_alb = ds_alb.to_dataframe().reset_index()
    ds_alb.close()
    df_alb['datetime'] = pd.to_datetime(df_alb['datetime'])
    df_alb['year'] = df_alb['datetime'].dt.year
    df_alb['month'] = df_alb['datetime'].dt.month
    monthly_alb = df_alb.groupby(['year', 'month'])['water_level'].mean().reset_index()
    monthly_alb.rename(columns={'water_level': 'lake_albert_level_m'}, inplace=True)
    
    hydro = pd.merge(monthly_dfo, monthly_alb, on=['year', 'month'], how='outer')
    # Fill missing values with overall monthly mean
    for col in ['nile_discharge_m3s', 'lake_albert_level_m']:
        month_means = hydro.groupby('month')[col].transform('mean')
        hydro[col] = hydro[col].fillna(month_means).fillna(hydro[col].mean())
        
    print(f"      Hydrology loaded: Nile discharge & Lake Albert water levels in {time.time()-t0:.2f}s")
    return hydro

def build_modeling_dataset(adm3, df_era5, df_flood, df_hydro):
    print("\n[Step 5/7] Building Master Feature Set and Engineering Multi-Driver Signals...")
    t0 = time.time()
    
    # Merge ERA5 with Flood
    df = pd.merge(df_era5, df_flood, on=['adm3_pcode', 'year', 'month'], how='left')
    df['flood_pixels'] = df['flood_pixels'].fillna(0.0)
    df['flooded_area_km2'] = df['flooded_area_km2'].fillna(0.0)
    
    # Merge with Payam properties
    payam_props = adm3[['adm3_pcode', 'adm3_name', 'adm2_name', 'adm1_name', 'area_km2', 'farmland_pct', 'population_est', 'rep_lat', 'rep_lon']]
    df = pd.merge(df, payam_props, on='adm3_pcode', how='left')
    
    # Inundation fraction
    df['flood_extent_pct'] = (df['flooded_area_km2'] / df['area_km2']) * 100.0
    df['flood_extent_pct'] = np.clip(df['flood_extent_pct'], 0.0, 100.0)
    
    # Merge with upstream hydrology
    df = pd.merge(df, df_hydro, on=['year', 'month'], how='left')
    df['nile_discharge_m3s'] = df['nile_discharge_m3s'].fillna(df['nile_discharge_m3s'].mean())
    df['lake_albert_level_m'] = df['lake_albert_level_m'].fillna(df['lake_albert_level_m'].mean())
    
    # Sort for lag calculations
    df = df.sort_values(['adm3_pcode', 'year', 'month']).reset_index(drop=True)
    
    # Add lagged precipitation & runoff
    df['tp_lag1'] = df.groupby('adm3_pcode')['tp_mm'].shift(1).fillna(df['tp_mm'])
    df['tp_lag2'] = df.groupby('adm3_pcode')['tp_mm'].shift(2).fillna(df['tp_lag1'])
    df['ro_lag1'] = df.groupby('adm3_pcode')['ro_mm'].shift(1).fillna(df['ro_mm'])
    
    # Seasonal cyclical harmonics
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12.0)
    
    # Multi-criteria Driver Scores (0-100 scale)
    # 1. Rainfall Driver Score: Percentile rank of current + lagged rainfall
    rain_composite = 0.7 * df['tp_mm'] + 0.3 * df['tp_lag1']
    df['rainfall_driver_score'] = rain_composite.rank(pct=True) * 100.0
    
    # 2. Surface Runoff Driver Score: Percentile rank of runoff
    runoff_composite = 0.75 * df['ro_mm'] + 0.25 * df['ro_lag1']
    df['runoff_driver_score'] = runoff_composite.rank(pct=True) * 100.0
    
    # 3. Fluvial / Riverine Inundation Score: Satellite flood extent + Nile discharge signal
    inundation_composite = 0.8 * df['flood_extent_pct'].rank(pct=True) + 0.2 * df['nile_discharge_m3s'].rank(pct=True)
    df['inundation_driver_score'] = inundation_composite.rank(pct=True) * 100.0
    
    # 4. Farmland Exposure Score: Cropland fraction intersected with hazard level
    hazard_env = (df['rainfall_driver_score'] + df['runoff_driver_score'] + df['inundation_driver_score']) / 3.0
    farm_composite = df['farmland_pct'].rank(pct=True) * 0.6 + hazard_env * 0.4
    df['farmland_driver_score'] = farm_composite.rank(pct=True) * 100.0
    
    # Determine Dominant Driver
    driver_scores = df[['rainfall_driver_score', 'runoff_driver_score', 'inundation_driver_score', 'farmland_driver_score']]
    driver_names = ['Extreme Rainfall', 'Surface Runoff', 'Riverine Inundation', 'Farmland Exposure']
    
    max_idx = np.argmax(driver_scores.values, axis=1)
    df['primary_driver'] = [driver_names[i] for i in max_idx]
    
    # Composite Flood Risk Score (0-100)
    # Hazard components + exposure
    df['flood_risk_score'] = (
        0.35 * df['inundation_driver_score'] +
        0.25 * df['rainfall_driver_score'] +
        0.20 * df['runoff_driver_score'] +
        0.20 * df['farmland_driver_score']
    )
    df['flood_risk_score'] = np.clip(df['flood_risk_score'], 0.0, 100.0)
    
    # Risk Level Categorization
    def categorize_risk(score):
        if score < 25.0:
            return "Low"
        elif score < 50.0:
            return "Moderate"
        elif score < 70.0:
            return "High"
        elif score < 85.0:
            return "Very High"
        else:
            return "Critical"
            
    df['risk_level'] = df['flood_risk_score'].apply(categorize_risk)
    
    # Population and Farmland exposed
    df['population_at_risk'] = (df['population_est'] * (df['flood_risk_score'] / 100.0) * (df['flood_extent_pct'].clip(1.0, 100.0) / 100.0)).round().astype(int)
    # Save cached panel dataset
    panel_cache = DATA_DIR / "payam_panel_dataset_2020_2024.parquet"
    df.to_parquet(panel_cache, index=False)
    print(f"      Cached panel dataset to {panel_cache}")
    print(f"      Master panel dataset constructed with {len(df):,} observations in {time.time()-t0:.2f}s")
    return df

def train_and_validate_models(df):
    print("\n[Step 6/7] Training & Cross-Validating Machine Learning Predictors...")
    t0 = time.time()
    
    feature_cols = [
        'tp_mm', 'ro_mm', 'tp_lag1', 'tp_lag2', 'ro_lag1',
        'nile_discharge_m3s', 'lake_albert_level_m',
        'rep_lat', 'rep_lon', 'area_km2', 'farmland_pct', 'population_est',
        'sin_month', 'cos_month', 'month'
    ]
    
    # Use standard numpy float arrays to ensure 100% robust multiprocessing in joblib
    X_mat = np.asarray(df[feature_cols].values, dtype=np.float64)
    y_reg = np.asarray(df['flood_extent_pct'].values, dtype=np.float64)
    y_risk = np.asarray(df['flood_risk_score'].values, dtype=np.float64)
    y_clf = np.asarray(df['primary_driver'].astype(str).values, dtype=str)
    
    # 1. Regressor for Flood Extent
    print("      Training Random Forest Regressor for Inundation Extent & Risk...")
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_split=4, random_state=42, n_jobs=-1)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(rf_reg, X_mat, y_reg, cv=kf, scoring='r2', n_jobs=-1)
    cv_mae = -cross_val_score(rf_reg, X_mat, y_reg, cv=kf, scoring='neg_mean_absolute_error', n_jobs=-1)
    
    rf_reg.fit(X_mat, y_reg)
    y_pred = rf_reg.predict(X_mat)
    r2_train = r2_score(y_reg, y_pred)
    rmse_train = np.sqrt(mean_squared_error(y_reg, y_pred))
    mae_train = mean_absolute_error(y_reg, y_pred)
    
    print(f"      [Regression Results] 5-Fold CV R2: {cv_r2.mean():.3f} (+/- {cv_r2.std():.3f}) | CV MAE: {cv_mae.mean():.3f}%")
    print(f"                           Train R2: {r2_train:.3f} | Train RMSE: {rmse_train:.3f}% | Train MAE: {mae_train:.3f}%")
    
    # 2. Classifier for Dominant Driver
    print("      Training Random Forest Classifier for Dominant Flood Driver...")
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_split=4, random_state=42, n_jobs=-1)
    cv_acc = cross_val_score(rf_clf, X_mat, y_clf, cv=kf, scoring='accuracy', n_jobs=-1)
    
    rf_clf.fit(X_mat, y_clf)
    y_clf_pred = rf_clf.predict(X_mat)
    acc_train = accuracy_score(y_clf, y_clf_pred)
    
    print(f"      [Classification Results] 5-Fold CV Accuracy: {cv_acc.mean()*100:.2f}% (+/- {cv_acc.std()*100:.2f}%)")
    print(f"                               Train Accuracy: {acc_train*100:.2f}%")
    
    # Save models
    joblib.dump(rf_reg, MODELS_DIR / "payam_flood_regressor.joblib")
    joblib.dump(rf_clf, MODELS_DIR / "payam_driver_classifier.joblib")
    print("      Saved models to flood_driver_analysis/models/")
    
    # Feature importances
    imp_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance_Reg': rf_reg.feature_importances_,
        'Importance_Clf': rf_clf.feature_importances_
    }).sort_values('Importance_Reg', ascending=False)
    
    return rf_reg, rf_clf, feature_cols, imp_df, {
        'cv_r2_mean': float(cv_r2.mean()),
        'cv_r2_std': float(cv_r2.std()),
        'cv_mae_mean': float(cv_mae.mean()),
        'cv_acc_mean': float(cv_acc.mean()),
        'cv_acc_std': float(cv_acc.std()),
        'r2_train': float(r2_train),
        'rmse_train': float(rmse_train),
        'acc_train': float(acc_train)
    }

def generate_predictions_and_exports(adm3, df_panel, rf_reg, rf_clf, feature_cols):
    print("\n[Step 7/7] Generating Climatological Monthly Predictions & Annual Summaries for 512 Payams...")
    t0 = time.time()
    
    # Climatological baseline: Mean by (adm3_pcode, month)
    monthly_clim = df_panel.groupby(['adm3_pcode', 'month']).agg({
        'adm3_name': 'first',
        'adm2_name': 'first',
        'adm1_name': 'first',
        'area_km2': 'first',
        'farmland_pct': 'first',
        'population_est': 'first',
        'rep_lat': 'first',
        'rep_lon': 'first',
        'tp_mm': 'mean',
        'ro_mm': 'mean',
        'tp_lag1': 'mean',
        'tp_lag2': 'mean',
        'ro_lag1': 'mean',
        'nile_discharge_m3s': 'mean',
        'lake_albert_level_m': 'mean',
        'sin_month': 'first',
        'cos_month': 'first',
        'rainfall_driver_score': 'mean',
        'runoff_driver_score': 'mean',
        'inundation_driver_score': 'mean',
        'farmland_driver_score': 'mean',
    }).reset_index()
    
    # Model inference on climatological features
    X_clim = np.asarray(monthly_clim[feature_cols].values, dtype=np.float64)
    monthly_clim['predicted_flood_pct'] = np.clip(rf_reg.predict(X_clim), 0.0, 100.0)
    monthly_clim['predicted_flood_km2'] = (monthly_clim['area_km2'] * (monthly_clim['predicted_flood_pct'] / 100.0)).round(2)
    monthly_clim['primary_driver'] = rf_clf.predict(X_clim)
    
    # Calibrated monthly risk score
    monthly_clim['flood_risk_score'] = (
        0.35 * monthly_clim['inundation_driver_score'] +
        0.25 * monthly_clim['rainfall_driver_score'] +
        0.20 * monthly_clim['runoff_driver_score'] +
        0.20 * monthly_clim['farmland_driver_score']
    )
    monthly_clim['flood_risk_score'] = np.clip(monthly_clim['flood_risk_score'], 0.0, 100.0).round(1)
    
    def cat_risk(s):
        if s < 25.0: return "Low"
        elif s < 50.0: return "Moderate"
        elif s < 70.0: return "High"
        elif s < 85.0: return "Very High"
        else: return "Critical"
    monthly_clim['risk_level'] = monthly_clim['flood_risk_score'].apply(cat_risk)
    
    monthly_clim['month_name'] = monthly_clim['month'].apply(lambda m: MONTH_NAMES[m-1])
    monthly_clim['predicted_rainfall_mm'] = monthly_clim['tp_mm'].round(1)
    monthly_clim['predicted_runoff_mm'] = monthly_clim['ro_mm'].round(1)
    monthly_clim['predicted_flood_pct'] = monthly_clim['predicted_flood_pct'].round(2)
    
    monthly_clim['population_at_risk'] = (monthly_clim['population_est'] * (monthly_clim['flood_risk_score'] / 100.0) * (monthly_clim['predicted_flood_pct'].clip(0.5, 100.0) / 100.0)).round().astype(int)
    monthly_clim['farmland_km2_at_risk'] = ((monthly_clim['area_km2'] * (monthly_clim['farmland_pct'] / 100.0)) * (monthly_clim['predicted_flood_pct'] / 100.0)).round(2)
    
    # Save monthly dataset (512 payams x 12 months = 6,144 rows)
    out_monthly_csv = DATA_DIR / "payam_flood_predictions_monthly.csv"
    monthly_cols = [
        'adm3_pcode', 'adm3_name', 'adm2_name', 'adm1_name',
        'month', 'month_name',
        'primary_driver', 'flood_risk_score', 'risk_level',
        'predicted_flood_pct', 'predicted_flood_km2',
        'predicted_rainfall_mm', 'predicted_runoff_mm',
        'rainfall_driver_score', 'runoff_driver_score', 'inundation_driver_score', 'farmland_driver_score',
        'population_at_risk', 'farmland_km2_at_risk'
    ]
    monthly_clim[monthly_cols].to_csv(out_monthly_csv, index=False)
    print(f"      Saved monthly predictions: {out_monthly_csv} ({len(monthly_clim)} rows)")
    
    # Build Annual Summary (1 row per payam)
    summary_records = []
    
    for pcode, grp in monthly_clim.groupby('adm3_pcode'):
        first_row = grp.iloc[0]
        
        # Peak risk month
        peak_idx = grp['flood_risk_score'].idxmax()
        peak_row = grp.loc[peak_idx]
        peak_month = int(peak_row['month'])
        
        # High risk window (months where risk score >= 50 or top 3 months)
        high_risk_months = grp[grp['flood_risk_score'] >= 45.0]['month'].tolist()
        if not high_risk_months:
            high_risk_months = grp.nlargest(3, 'flood_risk_score')['month'].tolist()
            
        onset_m = min(high_risk_months)
        end_m = max(high_risk_months)
        window_str = f"{MONTH_NAMES[onset_m-1]} - {MONTH_NAMES[end_m-1]}"
        
        # Dominant driver overall across the high-risk season
        season_grp = grp[grp['month'].isin(high_risk_months)]
        top_driver_annual = season_grp['primary_driver'].mode()[0] if len(season_grp) > 0 else peak_row['primary_driver']
        
        # Early warning lead time based on driver
        if top_driver_annual == "Riverine Inundation":
            lead_time = "30 - 60 days (Lake Victoria & Albert upstream wave tracking)"
            gov_action = "Reinforce perimeter dykes; clear drainage trunk canals; preposition riverine boats and water purification in Bentiu/Fangak."
            cit_action = "Relocate cattle herds to higher ground; store grain in elevated storage platforms; identify community evacuation dyke routes."
        elif top_driver_annual == "Extreme Rainfall":
            lead_time = "3 - 7 days (Synoptic convective forecasting)"
            gov_action = "Clear urban runoff conduits; issue radio storm flash-flood advisories; ready culvert unblocking teams."
            cit_action = "Clear domestic ditches; reinforce mud and thatch walls; safeguard drinking water sources from contamination."
        elif top_driver_annual == "Surface Runoff":
            lead_time = "5 - 14 days (Catchment saturation monitoring)"
            gov_action = "Inspect bridge culverts and unpaved road embankments; establish temporary catchment retaining bunds."
            cit_action = "Avoid low-lying stream crossings during rainfall; prepare barrier bags around homestead perimeters."
        else: # Farmland Exposure
            lead_time = "15 - 30 days (Seasonal agricultural flood outlook)"
            gov_action = "Distribute early-maturing and flood-tolerant rice/sorghum seeds; establish crop loss contingency reserves."
            cit_action = "Harvest mature crops before peak inundation; move livestock to upland dry pastures."
            
        summary_records.append({
            'adm3_pcode': pcode,
            'adm3_name': first_row['adm3_name'],
            'adm2_name': first_row['adm2_name'],
            'adm1_name': first_row['adm1_name'],
            'area_km2': round(first_row['area_km2'], 1),
            'population_est': int(first_row['population_est']),
            'farmland_pct': round(first_row['farmland_pct'], 2),
            'annual_primary_driver': top_driver_annual,
            'max_flood_risk_score': round(peak_row['flood_risk_score'], 1),
            'max_risk_level': peak_row['risk_level'],
            'peak_risk_month': peak_month,
            'peak_month_name': MONTH_NAMES[peak_month-1],
            'flood_onset_month': onset_m,
            'flood_recession_month': end_m,
            'high_risk_window': window_str,
            'peak_flood_pct': round(grp['predicted_flood_pct'].max(), 2),
            'peak_flood_km2': round(grp['predicted_flood_km2'].max(), 1),
            'annual_rainfall_mm': round(grp['predicted_rainfall_mm'].sum(), 1),
            'annual_runoff_mm': round(grp['predicted_runoff_mm'].sum(), 1),
            'peak_population_at_risk': int(grp['population_at_risk'].max()),
            'peak_farmland_km2_at_risk': round(grp['farmland_km2_at_risk'].max(), 2),
            'early_warning_lead_time': lead_time,
            'government_action': gov_action,
            'citizen_action': cit_action,
            'rep_lat': first_row['rep_lat'],
            'rep_lon': first_row['rep_lon']
        })
        
    df_summary = pd.DataFrame(summary_records).sort_values('max_flood_risk_score', ascending=False)
    out_summary_csv = DATA_DIR / "payam_flood_predictions_annual_summary.csv"
    df_summary.to_csv(out_summary_csv, index=False)
    print(f"      Saved annual summary: {out_summary_csv} ({len(df_summary)} rows)")
    
    # Generate GeoJSON with embedded monthly profile
    print("      Exporting full GIS GeoJSON with embedded seasonal curves...")
    adm3_geo = adm3[['adm3_pcode', 'geometry']].merge(df_summary, on='adm3_pcode', how='left')
    
    # Build monthly profile dict
    monthly_dict = {}
    for pcode, grp in monthly_clim.groupby('adm3_pcode'):
        monthly_dict[pcode] = grp[['month', 'predicted_flood_pct', 'predicted_rainfall_mm', 'predicted_runoff_mm', 'flood_risk_score', 'primary_driver']].to_dict(orient='records')
        
    adm3_geo['monthly_profile_json'] = adm3_geo['adm3_pcode'].apply(lambda c: json.dumps(monthly_dict.get(c, [])))
    out_geojson = DATA_DIR / "payam_flood_predictions.geojson"
    adm3_geo.to_file(out_geojson, driver="GeoJSON")
    print(f"      Saved spatial GeoJSON: {out_geojson}")
    
    return monthly_clim, df_summary

def plot_model_evaluations(df_panel, imp_df, metrics, monthly_clim, df_summary):
    print("\n--- Generating High-Resolution Model Performance & Seasonal Visualizations ---")
    
    # 1. Model Performance Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    fig.patch.set_facecolor("#fcfcfc")
    
    # 1A. Feature Importances
    ax = axes[0, 0]
    imp_sub = imp_df.sort_values('Importance_Reg', ascending=True).tail(10)
    colors = ['#1f77b4' if 'tp' in f or 'ro' in f else '#ff7f0e' if 'lat' in f or 'lon' in f or 'area' in f else '#2ca02c' for f in imp_sub['Feature']]
    bars = ax.barh(imp_sub['Feature'], imp_sub['Importance_Reg'], color=colors, edgecolor='none', height=0.65)
    ax.set_title("Machine Learning Feature Importances (Inundation Regressor)", fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel("Relative Importance (Gini)", fontsize=11)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    # 1B. Predicted vs Observed Scatter / Residual
    ax = axes[0, 1]
    sample_eval = df_panel.sample(min(3000, len(df_panel)), random_state=42)
    sc = ax.scatter(sample_eval['flood_extent_pct'], sample_eval['flood_risk_score'], c=sample_eval['month'], cmap='viridis', alpha=0.5, s=20)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Calendar Month (1-12)", fontsize=10)
    ax.plot([0, 100], [0, 100], 'r--', lw=1.5, label='1:1 Parity')
    ax.set_title(f"Risk Score vs Satellite Inundation Extent (R²={metrics['cv_r2_mean']:.2f})", fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel("Observed Satellite Flood Extent (%)", fontsize=11)
    ax.set_ylabel("Composite Model Flood Risk Score (0-100)", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='lower right')
    
    # 1C. National Seasonal Timeline of Drivers
    ax = axes[1, 0]
    monthly_drivers = monthly_clim.groupby(['month', 'primary_driver']).size().unstack(fill_value=0)
    monthly_drivers.index = MONTH_NAMES
    monthly_drivers.plot(kind='bar', stacked=True, ax=ax, color=[DRIVER_COLORS.get(col, '#333') for col in monthly_drivers.columns], width=0.75)
    ax.set_title("National Seasonality: Number of Payams by Dominant Driver", fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel("Month of Year", fontsize=11)
    ax.set_ylabel("Number of Payams (out of 512)", fontsize=11)
    ax.legend(title="Primary Driver", frameon=True, facecolor='white', framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.setp(ax.get_xticklabels(), rotation=0)
    
    # 1D. State-Level High Risk Payams Distribution
    ax = axes[1, 1]
    state_risk = df_summary.groupby(['adm1_name', 'annual_primary_driver']).size().unstack(fill_value=0)
    state_risk['total'] = state_risk.sum(axis=1)
    state_risk = state_risk.sort_values('total', ascending=True).drop(columns='total')
    state_risk.plot(kind='barh', stacked=True, ax=ax, color=[DRIVER_COLORS.get(col, '#333') for col in state_risk.columns], width=0.7)
    ax.set_title("Dominant Drivers by State (ADM1)", fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel("Number of Payams", fontsize=11)
    ax.set_ylabel("State", fontsize=11)
    ax.legend(title="Primary Driver", frameon=True, facecolor='white', framealpha=0.9)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    perf_path = GRAPHS_DIR / "payam_prediction_model_performance.png"
    plt.savefig(perf_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"      Saved performance figure: {perf_path}")
    
    # 2. Detailed Seasonal Timeline Chart
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True, dpi=300)
    fig.patch.set_facecolor("#ffffff")
    
    # Climatological monthly means across the country
    nat_month = monthly_clim.groupby('month').agg({
        'predicted_rainfall_mm': 'mean',
        'predicted_runoff_mm': 'mean',
        'predicted_flood_pct': 'mean',
        'flood_risk_score': 'mean'
    }).reset_index()
    
    # 2A. Rainfall & Runoff
    ax = axes[0]
    ax.plot(MONTH_NAMES, nat_month['predicted_rainfall_mm'], color='#2ca02c', marker='o', lw=2.5, label='Mean Monthly Rainfall (mm)')
    ax.set_ylabel("Precipitation (mm)", color='#2ca02c', fontsize=11, fontweight='bold')
    ax.tick_params(axis='y', labelcolor='#2ca02c')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    ax2 = ax.twinx()
    ax2.plot(MONTH_NAMES, nat_month['predicted_runoff_mm'], color='#9467bd', marker='s', lw=2.5, linestyle='--', label='Mean Surface Runoff (mm)')
    ax2.set_ylabel("Runoff (mm)", color='#9467bd', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#9467bd')
    ax.set_title("South Sudan Seasonal Hydrometeorological Cycle (ERA5 Reanalysis Climatology)", fontsize=13, fontweight='bold')
    
    # 2B. Inundation Extent & Flood Risk Score
    ax = axes[1]
    ax.bar(MONTH_NAMES, nat_month['predicted_flood_pct'], color='#1f77b4', alpha=0.7, width=0.55, label='Mean Inundation Extent (% of Payam Area)')
    ax.set_ylabel("Flooded Extent (%)", color='#1f77b4', fontsize=11, fontweight='bold')
    ax.tick_params(axis='y', labelcolor='#1f77b4')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    ax3 = ax.twinx()
    ax3.plot(MONTH_NAMES, nat_month['flood_risk_score'], color='#d62728', marker='^', lw=2.5, label='National Mean Flood Risk Score')
    ax3.set_ylabel("Risk Score (0-100)", color='#d62728', fontsize=11, fontweight='bold')
    ax3.tick_params(axis='y', labelcolor='#d62728')
    ax.set_title("Lagged Flood Inundation & Composite Risk Wave (NASA MCDWD Satellite Detections)", fontsize=13, fontweight='bold')
    
    # 2C. Breakdown of Payam Peak Months
    ax = axes[2]
    peak_counts = df_summary['peak_risk_month'].value_counts().reindex(range(1, 13), fill_value=0)
    bars = ax.bar(MONTH_NAMES, peak_counts.values, color='#ff7f0e', edgecolor='#d95f02', width=0.6)
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.annotate(f"{int(h)}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_title("Flood Peak Month Distribution across all 512 Payams", fontsize=13, fontweight='bold')
    ax.set_xlabel("Peak Month", fontsize=11, fontweight='bold')
    ax.set_ylabel("Number of Payams Peaking", fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    time_path = GRAPHS_DIR / "payam_flood_drivers_seasonal_timeline.png"
    plt.savefig(time_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"      Saved seasonal timeline figure: {time_path}")
    
    # 3. Spatial & Risk Distribution Chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    
    # Driver pie chart
    ax = axes[0]
    driver_totals = df_summary['annual_primary_driver'].value_counts()
    colors = [DRIVER_COLORS.get(k, '#999') for k in driver_totals.index]
    wedges, texts, autotexts = ax.pie(driver_totals, labels=driver_totals.index, autopct='%1.1f%%',
                                      startangle=140, colors=colors, textprops={'fontsize': 11},
                                      wedgeprops=dict(width=0.6, edgecolor='white', linewidth=2))
    for at in autotexts:
        at.set_color('white')
        at.set_fontweight('bold')
    ax.set_title("Dominant Flood Drivers Distribution (512 Payams)", fontsize=14, fontweight='bold', pad=15)
    
    # Risk Level distribution
    ax = axes[1]
    risk_order = ["Low", "Moderate", "High", "Very High", "Critical"]
    risk_colors = ["#2ca02c", "#ffbb78", "#ff7f0e", "#d62728", "#7f0000"]
    risk_counts = df_summary['max_risk_level'].value_counts().reindex(risk_order, fill_value=0)
    bars = ax.bar(risk_order, risk_counts, color=risk_colors, edgecolor='none', width=0.6)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{int(h)} ({h/len(df_summary)*100:.1f}%)", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_title("Peak Annual Flood Risk Tiers across South Sudan", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Risk Category", fontsize=11, fontweight='bold')
    ax.set_ylabel("Number of Payams", fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    dist_path = GRAPHS_DIR / "payam_risk_and_driver_distribution.png"
    plt.savefig(dist_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"      Saved risk distribution figure: {dist_path}")

def main():
    print("=" * 80)
    print("SOUTH SUDAN PAYAM-LEVEL FLOOD PREDICTION MODEL TRAINING PIPELINE")
    print("=" * 80)
    
    adm3 = load_payam_geometries_and_exposure()
    panel_cache = DATA_DIR / "payam_panel_dataset_2020_2024.parquet"
    if panel_cache.exists():
        print(f"\n[Step 2-5/7] Loading pre-extracted panel dataset from {panel_cache}...")
        df_panel = pd.read_parquet(panel_cache)
    else:
        df_era5 = extract_multiyear_era5(adm3, years=[2020, 2021, 2022, 2023, 2024])
        df_flood = extract_multiyear_flood_masks(adm3, years=[2020, 2021, 2022, 2023, 2024])
        df_hydro = extract_upstream_hydrology()
        df_panel = build_modeling_dataset(adm3, df_era5, df_flood, df_hydro)
    
    rf_reg, rf_clf, feature_cols, imp_df, metrics = train_and_validate_models(df_panel)
    monthly_clim, df_summary = generate_predictions_and_exports(adm3, df_panel, rf_reg, rf_clf, feature_cols)
    plot_model_evaluations(df_panel, imp_df, metrics, monthly_clim, df_summary)
    
    print("\n" + "=" * 80)
    print("MODEL TRAINING & PREDICTION GENERATION COMPLETED SUCCESSFULLY!")
    print(f"Total Payams Modeled: {len(df_summary)}")
    print(f"Monthly Predictions Generated: {len(monthly_clim)}")
    print(f"Regression 5-Fold CV R2: {metrics['cv_r2_mean']:.3f}")
    print(f"Classification 5-Fold CV Accuracy: {metrics['cv_acc_mean']*100:.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()
