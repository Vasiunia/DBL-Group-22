#!/usr/bin/env python3
"""
generate_flood_driver_graphs.py

Comprehensive analysis and cartographic generation of top flood drivers across all
512 ADM3 (Payam) regions in South Sudan.

Drivers evaluated:
1. Farmland Exposure: Agricultural cropland density / fraction (ASAP FAO/EC Crop Mask)
2. Satellite Flood Masks: Satellite-detected flood inundation extent & recurrence (NASA MCDWD VIIRS/MODIS)
3. Extreme Rainfall: Annual accumulated precipitation (ECMWF ERA5 Reanalysis)
4. Surface Runoff: Overland surface runoff accumulation (ECMWF ERA5 Reanalysis)

Outputs:
- graphs/south_sudan_adm3_top_flood_drivers_map.png
- graphs/south_sudan_adm3_all_drivers_comparison.png
- graphs/south_sudan_adm3_drivers_breakdown.png
- graphs/south_sudan_adm3_executive_dashboard.png
- south_sudan_adm3_flood_drivers_map.html
- data/south_sudan_adm3_flood_drivers.csv
- data/south_sudan_adm3_flood_drivers.geojson
"""

import os
import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.mask import mask
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import folium
from folium import plugins

# ==============================================================================
# 1. SETUP & DIRECTORIES
# ==============================================================================
BASE_DIR = Path(".")
OUTPUT_DIR = BASE_DIR / "flood_driver_analysis"
GRAPHS_DIR = OUTPUT_DIR / "graphs"
DATA_DIR = OUTPUT_DIR / "data"

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
GRAPHS_DIR.mkdir(exist_ok=True, parents=True)
DATA_DIR.mkdir(exist_ok=True, parents=True)

# Also prepare artifact directory for viewing if available
ARTIFACT_DIR = Path("/Users/varvaratolmach/.gemini/antigravity-cli/brain/daf9fdb2-48b5-45a5-82c7-0fbeecbe15c4")
if ARTIFACT_DIR.exists():
    (ARTIFACT_DIR / "graphs").mkdir(exist_ok=True, parents=True)

# Colors and styling for drivers
DRIVER_PALETTE = {
    "Flood Masks (Inundation)": "#1f78b4",  # Ocean blue
    "Farmland Exposure":        "#ff7f00",  # Harvest amber / orange
    "Extreme Rainfall":         "#33a02c",  # Forest green
    "Surface Runoff":           "#6a3d9a",  # Deep purple
}

DRIVER_LABELS_SHORT = {
    "Flood Masks (Inundation)": "Flood Masks",
    "Farmland Exposure":        "Farmland",
    "Extreme Rainfall":         "Rainfall",
    "Surface Runoff":           "Runoff",
}

DRIVER_ICONS = {
    "Flood Masks (Inundation)": "🌊",
    "Farmland Exposure":        "🌾",
    "Extreme Rainfall":         "🌧️",
    "Surface Runoff":           "💧",
}

# ==============================================================================
# 2. DATA INGESTION & EXTRACTION
# ==============================================================================
def load_and_process_data():
    print("=" * 80)
    print("SOUTH SUDAN ADM3 FLOOD DRIVER ANALYSIS")
    print("=" * 80)
    
    # 2.1 Load ADM3 Boundaries
    print("\n[1/5] Loading Administrative Boundaries...")
    t0 = time.time()
    adm3_path = BASE_DIR / "Administrative boundaries" / "ssd_admin3.geojson"
    adm1_path = BASE_DIR / "Administrative boundaries" / "ssd_admin1.geojson"
    
    adm3 = gpd.read_file(adm3_path)
    if adm3.crs is None:
        adm3 = adm3.set_crs("EPSG:4326")
    else:
        adm3 = adm3.to_crs("EPSG:4326")
        
    adm1 = gpd.read_file(adm1_path)
    if adm1.crs is None:
        adm1 = adm1.set_crs("EPSG:4326")
    else:
        adm1 = adm1.to_crs("EPSG:4326")
        
    # Calculate planar area in km2
    adm3["area_km2"] = adm3.to_crs("+proj=cea +lon_0=30 +lat_ts=8").geometry.area / 1e6
    adm1["area_km2"] = adm1.to_crs("+proj=cea +lon_0=30 +lat_ts=8").geometry.area / 1e6
    print(f"      Loaded {len(adm3)} ADM3 regions (Payams) and {len(adm1)} States in {time.time()-t0:.2f}s")
    print(f"      Total country area: {adm3['area_km2'].sum():,.1f} km²")
    
    total_bounds = adm3.total_bounds # [minx, miny, maxx, maxy]
    
    # 2.2 Farmland / Cropland Mask
    print("\n[2/5] Extracting Farmland / Cropland Density (ASAP FAO/EC)...")
    t0 = time.time()
    crop_path = BASE_DIR / "farmland" / "asap_mask_crop_v04.tif"
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
    print(f"      Farmland zonal statistics computed in {time.time()-t0:.2f}s")
    print(f"      Cropland % range: min={adm3['farmland_pct'].min():.2f}%, "
          f"median={adm3['farmland_pct'].median():.2f}%, max={adm3['farmland_pct'].max():.2f}%")
    
    # 2.3 ERA5 Rainfall & Runoff
    print("\n[3/5] Extracting ERA5 Precipitation & Runoff (2024)...")
    t0 = time.time()
    era5_path = BASE_DIR / "rainfall and runoff" / "ERA5_2024.nc"
    ds = xr.open_dataset(era5_path)
    sub = ds.sel(latitude=slice(total_bounds[3] + 0.5, total_bounds[1] - 0.5),
                 longitude=slice(total_bounds[0] - 0.5, total_bounds[2] + 0.5))
    
    # Total annual sum in mm (meters * 1000)
    annual_tp = (sub['tp'].sum(dim='valid_time') * 1000.0).values
    annual_ro = (sub['ro'].sum(dim='valid_time') * 1000.0).values
    lats = sub['latitude'].values
    lons = sub['longitude'].values
    ds.close()
    
    if lats[0] > lats[-1]:
        lats_asc = lats[::-1]
        annual_tp = annual_tp[::-1, :]
        annual_ro = annual_ro[::-1, :]
    else:
        lats_asc = lats
        
    interp_tp = RegularGridInterpolator((lats_asc, lons), annual_tp, bounds_error=False, fill_value=None)
    interp_ro = RegularGridInterpolator((lats_asc, lons), annual_ro, bounds_error=False, fill_value=None)
    
    rep_pts = adm3.geometry.representative_point()
    pts_coords = np.column_stack([rep_pts.y, rep_pts.x])
    adm3["rainfall_mm"] = interp_tp(pts_coords)
    adm3["runoff_mm"] = interp_ro(pts_coords)
    print(f"      ERA5 climate fields sampled in {time.time()-t0:.2f}s")
    print(f"      Rainfall range: {adm3['rainfall_mm'].min():.1f} mm to {adm3['rainfall_mm'].max():.1f} mm (mean: {adm3['rainfall_mm'].mean():.1f} mm)")
    print(f"      Runoff range:   {adm3['runoff_mm'].min():.1f} mm to {adm3['runoff_mm'].max():.1f} mm (mean: {adm3['runoff_mm'].mean():.1f} mm)")
    
    # 2.4 Satellite Flood Masks
    print("\n[4/5] Aggregating Satellite Flood Masks (MCDWD VIIRS/MODIS 2024)...")
    t0 = time.time()
    tiles = ["h20v08", "h21v08"]
    rec_parts, unu_parts = [], []
    for tile in tiles:
        p_rec = BASE_DIR / "flood_masks" / "compact_recurring" / f"flood_events_{tile}_2024.parquet"
        p_unu = BASE_DIR / "flood_masks" / "compact_unusual" / f"flood_events_{tile}_2024.parquet"
        if p_rec.exists():
            rec_parts.append(pd.read_parquet(p_rec, columns=["lat", "lon"]))
        if p_unu.exists():
            unu_parts.append(pd.read_parquet(p_unu, columns=["lat", "lon"]))
            
    df_rec = pd.concat(rec_parts, ignore_index=True)
    df_unu = pd.concat(unu_parts, ignore_index=True)
    
    # Unique flooded pixels (232m x 232m = ~0.0538 km2 per pixel)
    all_unique_pts = pd.concat([df_rec.drop_duplicates(), df_unu.drop_duplicates()]).drop_duplicates().reset_index(drop=True)
    gdf_flood = gpd.GeoDataFrame(
        all_unique_pts,
        geometry=gpd.points_from_xy(all_unique_pts["lon"], all_unique_pts["lat"]),
        crs="EPSG:4326"
    )
    
    joined_flood = gpd.sjoin(gdf_flood, adm3[["adm3_pcode", "geometry"]], how="inner", predicate="within")
    pixel_counts = joined_flood.groupby("adm3_pcode").size().rename("flood_pixels")
    
    adm3 = adm3.merge(pixel_counts, on="adm3_pcode", how="left")
    adm3["flood_pixels"] = adm3["flood_pixels"].fillna(0)
    adm3["flooded_area_km2"] = adm3["flood_pixels"] * 0.0538
    adm3["flood_pct"] = (adm3["flooded_area_km2"] / adm3["area_km2"]) * 100.0
    print(f"      Satellite flood masks aggregated in {time.time()-t0:.2f}s")
    print(f"      Payam flooded area range: 0% to {adm3['flood_pct'].max():.1f}% (mean: {adm3['flood_pct'].mean():.2f}%)")
    
    # 2.5 Compute Percentile Ranks & Determine Top Flood Driver
    print("\n[5/5] Normalizing Drivers & Determining Top Flood Driver for Each ADM3 Region...")
    drivers = ["farmland_pct", "flood_pct", "rainfall_mm", "runoff_mm"]
    driver_name_map = {
        "farmland_pct": "Farmland Exposure",
        "flood_pct":    "Flood Masks (Inundation)",
        "rainfall_mm":  "Extreme Rainfall",
        "runoff_mm":    "Surface Runoff",
    }
    
    for d in drivers:
        adm3[f"pctile_{d}"] = adm3[d].rank(pct=True) * 100.0
        adm3[f"z_{d}"] = (adm3[d] - adm3[d].mean()) / adm3[d].std()
        
    pctile_cols = [f"pctile_{d}" for d in drivers]
    adm3["top_driver"] = adm3[pctile_cols].idxmax(axis=1).map(lambda c: driver_name_map[c.replace("pctile_", "")])
    adm3["top_driver_score"] = adm3[pctile_cols].max(axis=1)
    
    # Summary of classification
    driver_counts = adm3["top_driver"].value_counts()
    print("\n--- TOP FLOOD DRIVERS SUMMARY ACROSS 512 ADM3 REGIONS ---")
    for driver, count in driver_counts.items():
        pct = (count / len(adm3)) * 100.0
        icon = DRIVER_ICONS.get(driver, "")
        print(f"  {icon} {driver:25s}: {count:3d} payams ({pct:5.1f}%)")
        
    return adm3, adm1

# ==============================================================================
# 3. HIGH-RESOLUTION CARTOGRAPHIC PLOTTING
# ==============================================================================

def add_scale_bar(ax, length_km=200, location=(0.06, 0.08)):
    """Draws a professional map scale bar in kilometers."""
    # Approximate 1 degree latitude = 111 km at 8°N
    # 1 degree longitude = 111 * cos(8°) ≈ 110 km
    km_per_deg = 110.0
    length_deg = length_km / km_per_deg
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    x0 = xlim[0] + (xlim[1] - xlim[0]) * location[0]
    y0 = ylim[0] + (ylim[1] - ylim[0]) * location[1]
    
    # Draw bar
    ax.plot([x0, x0 + length_deg], [y0, y0], color="#222222", lw=3.5, solid_capstyle='butt')
    ax.plot([x0, x0], [y0 - 0.08, y0 + 0.08], color="#222222", lw=2)
    ax.plot([x0 + length_deg, x0 + length_deg], [y0 - 0.08, y0 + 0.08], color="#222222", lw=2)
    ax.plot([x0 + length_deg/2, x0 + length_deg/2], [y0 - 0.05, y0 + 0.05], color="#222222", lw=1.5)
    
    ax.text(x0, y0 + 0.15, "0", ha='center', va='bottom', fontsize=8, fontweight='bold', color="#222222")
    ax.text(x0 + length_deg/2, y0 + 0.15, f"{length_km//2}", ha='center', va='bottom', fontsize=8, fontweight='bold', color="#222222")
    ax.text(x0 + length_deg, y0 + 0.15, f"{length_km} km", ha='center', va='bottom', fontsize=8, fontweight='bold', color="#222222")


def add_north_arrow(ax, location=(0.06, 0.22)):
    """Draws a stylish North Arrow on the map."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    x = xlim[0] + (xlim[1] - xlim[0]) * location[0]
    y = ylim[0] + (ylim[1] - ylim[0]) * location[1]
    
    arrow_size = 0.35
    ax.annotate(
        'N', xy=(x, y + arrow_size * 0.8), xytext=(x, y - arrow_size * 0.4),
        arrowprops=dict(facecolor='#222222', edgecolor='#222222', width=2.5, headwidth=9),
        ha='center', va='center', fontsize=11, fontweight='bold', color='#222222'
    )


def plot_primary_top_driver_map(adm3, adm1, out_path):
    """
    Creates Figure 1: South Sudan ADM3 Top Flood Drivers Choropleth Map with Legend on the side.
    """
    print(f"\n-> Generating Primary Top Flood Driver Map: {out_path.name}...")
    
    fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f7f9fb')
    
    # Assign color column
    adm3['color'] = adm3['top_driver'].map(DRIVER_PALETTE)
    
    # 1. Plot ADM3 regions
    adm3.plot(
        ax=ax,
        color=adm3['color'],
        edgecolor='#ffffff',
        linewidth=0.25,
        alpha=0.92
    )
    
    # 2. Plot ADM1 State boundaries for regional structure
    adm1.boundary.plot(
        ax=ax,
        edgecolor='#2c3e50',
        linewidth=1.2,
        linestyle='-',
        alpha=0.85
    )
    
    # 3. Label key States with subtle labels
    for _, row in adm1.iterrows():
        centroid = row.geometry.representative_point()
        state_name = row['adm1_name'].replace("Western ", "W. ").replace("Eastern ", "E. ").replace("Central ", "C. ").replace("Northern ", "N. ")
        ax.text(
            centroid.x, centroid.y, state_name,
            fontsize=8, fontweight='bold', color='#1a252f',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.6)
        )
        
    # Scale bar and North arrow
    add_scale_bar(ax, length_km=200, location=(0.05, 0.05))
    add_north_arrow(ax, location=(0.05, 0.16))
    
    # Set map limits
    bounds = adm3.total_bounds
    pad_x = (bounds[2] - bounds[0]) * 0.04
    pad_y = (bounds[3] - bounds[1]) * 0.04
    ax.set_xlim(bounds[0] - pad_x, bounds[2] + pad_x)
    ax.set_ylim(bounds[1] - pad_y, bounds[3] + pad_y)
    
    # Remove standard axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#cccccc')
        spine.set_linewidth(0.8)
        
    # Title & Subtitle
    ax.set_title(
        "South Sudan: Top Flood Drivers by ADM3 Region (Payams)",
        fontsize=18, fontweight='bold', pad=20, color='#111111', loc='left'
    )
    plt.suptitle(
        "Dominant flood risk and hazard mechanism for all 512 Payams | Relative percentile rank among national indicators",
        fontsize=11, color='#555555', x=0.125, y=0.93, ha='left'
    )
    
    # Create Comprehensive Legend on the Side
    driver_counts = adm3['top_driver'].value_counts()
    legend_patches = []
    
    for driver_name, color in DRIVER_PALETTE.items():
        count = driver_counts.get(driver_name, 0)
        pct = (count / len(adm3)) * 100.0
        label_text = f"{driver_name}\n({count} Payams, {pct:.1f}%)"
        patch = mpatches.Patch(facecolor=color, edgecolor='#333333', linewidth=0.5, label=label_text)
        legend_patches.append(patch)
        
    state_line = Line2D([0], [0], color='#2c3e50', lw=1.5, label="State Boundary (ADM1)")
    legend_patches.append(state_line)
    
    # Place Legend on the Right Side
    leg = ax.legend(
        handles=legend_patches,
        loc='center left',
        bbox_to_anchor=(1.01, 0.5),
        frameon=True,
        facecolor='#ffffff',
        edgecolor='#b0bec5',
        fontsize=10.5,
        title="TOP FLOOD DRIVER\n(Dominant Factor)",
        title_fontsize=12,
        labelspacing=1.3,
        handlelength=1.6,
        handleheight=1.2,
        borderpad=1.0
    )
    leg.get_title().set_fontweight('bold')
    leg.get_title().set_color('#111111')
    
    # Data Sources Footer Annotation
    footer_text = (
        "Data Sources:\n"
        "• Administrative Boundaries: UN OCHA HDX (512 Payams)\n"
        "• Flood Masks: NASA VIIRS/MODIS 3-day NRT (MCDWD)\n"
        "• Rainfall & Runoff: ECMWF ERA5 Reanalysis (0.25° grid, 2024)\n"
        "• Farmland Exposure: ASAP European Commission Crop Mask (1/4 km²)"
    )
    fig.text(0.81, 0.12, footer_text, fontsize=8, color='#666666', va='bottom', ha='left',
             bbox=dict(boxstyle='square,pad=0.6', facecolor='#fafafa', edgecolor='#e0e0e0', lw=0.8))
             
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   -> Saved: {out_path}")
    
    # Copy to artifact dir if exists
    if ARTIFACT_DIR.exists():
        import shutil
        shutil.copy2(out_path, ARTIFACT_DIR / "graphs" / out_path.name)


def plot_four_component_driver_maps(adm3, adm1, out_path):
    """
    Creates Figure 2: 4-Panel Map showing spatial distribution of each individual driver.
    """
    print(f"\n-> Generating 4-Panel Individual Driver Maps: {out_path.name}...")
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    
    panels = [
        {
            "ax": axes[0, 0],
            "col": "farmland_pct",
            "title": "A) Farmland Exposure (Cropland %)",
            "subtitle": "Mean crop area fraction per payam (ASAP)",
            "cmap": "YlOrBr",
            "unit": "% Crop Area",
            "vmax": min(30.0, adm3['farmland_pct'].quantile(0.98))
        },
        {
            "ax": axes[0, 1],
            "col": "flood_pct",
            "title": "B) Satellite Flood Masks (Flooded Area %)",
            "subtitle": "Satellite-observed inundation extent (VIIRS/MODIS)",
            "cmap": "Blues",
            "unit": "% Area Inundated",
            "vmax": min(40.0, adm3['flood_pct'].quantile(0.96))
        },
        {
            "ax": axes[1, 0],
            "col": "rainfall_mm",
            "title": "C) Annual Precipitation (Rainfall mm)",
            "subtitle": "Total annual rainfall depth (ERA5 2024)",
            "cmap": "YlGnBu",
            "unit": "mm / year",
            "vmax": adm3['rainfall_mm'].quantile(0.98)
        },
        {
            "ax": axes[1, 1],
            "col": "runoff_mm",
            "title": "D) Surface Runoff Accumulation (mm)",
            "subtitle": "Overland runoff water depth (ERA5 2024)",
            "cmap": "Purples",
            "unit": "mm / year",
            "vmax": adm3['runoff_mm'].quantile(0.96)
        }
    ]
    
    for p in panels:
        ax = p["ax"]
        ax.set_facecolor('#f8fafc')
        
        # Plot choropleth
        adm3.plot(
            column=p["col"],
            ax=ax,
            cmap=p["cmap"],
            vmax=p["vmax"],
            edgecolor='#ffffff',
            linewidth=0.2,
            legend=True,
            legend_kwds={
                'label': p["unit"],
                'orientation': 'vertical',
                'shrink': 0.7,
                'pad': 0.02,
                'fraction': 0.046
            }
        )
        
        # State boundary overlay
        adm1.boundary.plot(ax=ax, edgecolor='#333333', linewidth=0.7, alpha=0.75)
        
        ax.set_title(p["title"], fontsize=13, fontweight='bold', pad=22, color='#111111', loc='left')
        ax.text(0.01, 1.025, p["subtitle"], transform=ax.transAxes, fontsize=9.5, color='#4b5563')
        
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('#d1d5db')
            spine.set_linewidth(0.6)
            
    fig.suptitle(
        "South Sudan: Spatial Distribution of the Four Primary Flood & Risk Drivers (ADM3 Resolution)",
        fontsize=17, fontweight='bold', y=0.98, color='#111111'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   -> Saved: {out_path}")
    
    if ARTIFACT_DIR.exists():
        import shutil
        shutil.copy2(out_path, ARTIFACT_DIR / "graphs" / out_path.name)


def plot_statistical_breakdown(adm3, out_path):
    """
    Creates Figure 3: National & State Breakdown Charts of Top Flood Drivers.
    """
    print(f"\n-> Generating Statistical Breakdown Graphs: {out_path.name}...")
    
    fig = plt.figure(figsize=(16, 10), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], hspace=0.3, wspace=0.25)
    
    ax_pie = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_state = fig.add_subplot(gs[1, :])
    
    # 1. National Donut Chart
    driver_counts = adm3['top_driver'].value_counts()
    colors = [DRIVER_PALETTE[d] for d in driver_counts.index]
    
    wedges, texts, autotexts = ax_pie.pie(
        driver_counts.values,
        labels=[f"{d}" for d in driver_counts.index],
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2),
        pctdistance=0.75,
        textprops=dict(fontsize=10, fontweight='bold')
    )
    for at in autotexts:
        at.set_color('white')
        at.set_fontsize(10)
    ax_pie.set_title("A) National Share of Dominant Flood Drivers\n(Across 512 ADM3 Payams)",
                     fontsize=12, fontweight='bold', pad=12)
    
    # 2. National Driver Count Bar Chart
    bars = ax_bar.bar(
        [DRIVER_LABELS_SHORT[d] for d in driver_counts.index],
        driver_counts.values,
        color=colors,
        edgecolor='#333333',
        linewidth=0.8,
        width=0.55
    )
    ax_bar.set_ylabel("Number of Payams", fontsize=11, fontweight='bold')
    ax_bar.set_title("B) Payam Count by Top Flood Driver", fontsize=12, fontweight='bold', pad=12)
    ax_bar.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars:
        h = bar.get_height()
        ax_bar.text(
            bar.get_x() + bar.get_width()/2., h + 3,
            f"{int(h)} ({h/len(adm3)*100:.1f}%)",
            ha='center', va='bottom', fontsize=9.5, fontweight='bold'
        )
    ax_bar.set_ylim(0, max(driver_counts.values) * 1.18)
    
    # 3. Stacked Horizontal Bar Chart by State
    crosstab = pd.crosstab(adm3["adm1_name"], adm3["top_driver"])
    # Reindex columns to match palette order
    crosstab = crosstab.reindex(columns=list(DRIVER_PALETTE.keys()), fill_value=0)
    # Sort states by total payams
    crosstab["total"] = crosstab.sum(axis=1)
    crosstab = crosstab.sort_values(by="total", ascending=True)
    crosstab_pct = crosstab.drop(columns=["total"]).div(crosstab["total"], axis=0) * 100.0
    
    y_pos = np.arange(len(crosstab))
    left = np.zeros(len(crosstab))
    
    for driver in DRIVER_PALETTE.keys():
        values = crosstab_pct[driver].values
        ax_state.barh(
            y_pos, values, left=left, color=DRIVER_PALETTE[driver],
            edgecolor='white', height=0.65, label=driver
        )
        # Add text labels on segments if wide enough
        for i, val in enumerate(values):
            if val >= 10.0:
                ax_state.text(
                    left[i] + val/2.0, y_pos[i], f"{val:.0f}%",
                    ha='center', va='center', color='white', fontsize=8.5, fontweight='bold'
                )
        left += values
        
    ax_state.set_yticks(y_pos)
    state_labels = [f"{st} (n={crosstab.loc[st, 'total']})" for st in crosstab.index]
    ax_state.set_yticklabels(state_labels, fontsize=10, fontweight='bold')
    ax_state.set_xlabel("Percentage of Payams within State (%)", fontsize=11, fontweight='bold')
    ax_state.set_xlim(0, 100)
    ax_state.set_title("C) State-by-State Composition of Top Flood Drivers (10 States + Abyei)",
                       fontsize=13, fontweight='bold', pad=12)
    ax_state.grid(axis='x', linestyle='--', alpha=0.5)
    
    ax_state.legend(
        loc='upper center', bbox_to_anchor=(0.5, -0.15),
        ncol=4, frameon=True, fontsize=10, title="Top Flood Driver",
        title_fontsize=11
    )
    
    fig.suptitle(
        "South Sudan: Top Flood Driver Statistical Distributions & Regional Heterogeneity",
        fontsize=16, fontweight='bold', y=0.98
    )
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   -> Saved: {out_path}")
    
    if ARTIFACT_DIR.exists():
        import shutil
        shutil.copy2(out_path, ARTIFACT_DIR / "graphs" / out_path.name)


def plot_executive_dashboard(adm3, adm1, out_path):
    """
    Creates Figure 4: Executive 2-Column Dashboard (Large Map + Analytical Breakdown).
    """
    print(f"\n-> Generating Executive Dashboard: {out_path.name}...")
    
    fig = plt.figure(figsize=(20, 11), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    
    # 2 columns: left is map (width 60%), right is charts (width 40%)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 0.9], height_ratios=[1, 1], wspace=0.18, hspace=0.25)
    
    ax_map = fig.add_subplot(gs[:, 0])
    ax_state = fig.add_subplot(gs[0, 1])
    ax_box = fig.add_subplot(gs[1, 1])
    
    # --- LEFT: MAP ---
    ax_map.set_facecolor('#f7f9fb')
    adm3['color'] = adm3['top_driver'].map(DRIVER_PALETTE)
    adm3.plot(ax=ax_map, color=adm3['color'], edgecolor='#ffffff', linewidth=0.25, alpha=0.92)
    adm1.boundary.plot(ax=ax_map, edgecolor='#1e293b', linewidth=1.1, linestyle='-', alpha=0.85)
    
    add_scale_bar(ax_map, length_km=200, location=(0.05, 0.05))
    add_north_arrow(ax_map, location=(0.05, 0.16))
    
    bounds = adm3.total_bounds
    pad_x = (bounds[2] - bounds[0]) * 0.03
    pad_y = (bounds[3] - bounds[1]) * 0.03
    ax_map.set_xlim(bounds[0] - pad_x, bounds[2] + pad_x)
    ax_map.set_ylim(bounds[1] - pad_y, bounds[3] + pad_y)
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    for spine in ax_map.spines.values():
        spine.set_color('#cccccc')
    
    ax_map.set_title("South Sudan ADM3 Top Flood Drivers Map", fontsize=15, fontweight='bold', pad=10, loc='left')
    
    # Map Legend inside/adjacent
    legend_patches = [
        mpatches.Patch(facecolor=DRIVER_PALETTE[d], edgecolor='#333333', linewidth=0.5,
                       label=f"{d} ({adm3['top_driver'].value_counts().get(d, 0)})")
        for d in DRIVER_PALETTE.keys()
    ]
    ax_map.legend(
        handles=legend_patches,
        loc='lower right',
        frameon=True,
        facecolor='#ffffff',
        edgecolor='#b0bec5',
        fontsize=9.5,
        title="Top Driver (Payams)",
        title_fontsize=10.5,
        borderpad=0.8
    )
    
    # --- RIGHT TOP: STATE COMPOSITION ---
    crosstab = pd.crosstab(adm3["adm1_name"], adm3["top_driver"]).reindex(columns=list(DRIVER_PALETTE.keys()), fill_value=0)
    crosstab["total"] = crosstab.sum(axis=1)
    crosstab = crosstab.sort_values(by="total", ascending=True)
    crosstab_pct = crosstab.drop(columns=["total"]).div(crosstab["total"], axis=0) * 100.0
    
    y_pos = np.arange(len(crosstab))
    left = np.zeros(len(crosstab))
    for driver in DRIVER_PALETTE.keys():
        vals = crosstab_pct[driver].values
        ax_state.barh(y_pos, vals, left=left, color=DRIVER_PALETTE[driver], edgecolor='white', height=0.65)
        left += vals
    ax_state.set_yticks(y_pos)
    ax_state.set_yticklabels(crosstab.index, fontsize=8.5, fontweight='bold')
    ax_state.set_xlabel("Payam Percentage (%)", fontsize=9.5, fontweight='bold')
    ax_state.set_title("State-by-State Flood Driver Dominance", fontsize=12, fontweight='bold', pad=8)
    ax_state.grid(axis='x', linestyle='--', alpha=0.5)
    
    # --- RIGHT BOTTOM: METRIC DISTRIBUTIONS ---
    # Boxplot of driver percentiles to show discrimination
    pctile_data = [
        adm3[adm3["top_driver"] == d]["top_driver_score"].values
        for d in DRIVER_PALETTE.keys()
    ]
    bp = ax_box.boxplot(
        pctile_data,
        patch_artist=True,
        tick_labels=[DRIVER_LABELS_SHORT[d] for d in DRIVER_PALETTE.keys()],
        medianprops=dict(color='black', lw=1.5),
        boxprops=dict(lw=1.0),
        whiskerprops=dict(lw=1.0),
        capprops=dict(lw=1.0),
        flierprops=dict(marker='o', markersize=3, alpha=0.4)
    )
    for patch, driver in zip(bp['boxes'], DRIVER_PALETTE.keys()):
        patch.set_facecolor(DRIVER_PALETTE[driver])
        patch.set_alpha(0.85)
        
    ax_box.set_ylabel("Dominant Factor Percentile Rank (0-100)", fontsize=9.5, fontweight='bold')
    ax_box.set_title("Severity & Dominance Intensity by Top Driver", fontsize=12, fontweight='bold', pad=8)
    ax_box.grid(axis='y', linestyle='--', alpha=0.5)
    ax_box.set_ylim(40, 102)
    
    fig.suptitle(
        "Executive Dashboard: South Sudan ADM3 Flood Driver Geography & Regional Profiles",
        fontsize=17, fontweight='bold', y=0.98
    )
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   -> Saved: {out_path}")
    
    if ARTIFACT_DIR.exists():
        import shutil
        shutil.copy2(out_path, ARTIFACT_DIR / "graphs" / out_path.name)


# ==============================================================================
# 4. INTERACTIVE FOLIUM HTML MAP GENERATION
# ==============================================================================

def generate_interactive_map(adm3, adm1, out_path):
    """
    Builds an interactive Leaflet/Folium web map with payam hover, click popups,
    side legend, and layer toggle.
    """
    print(f"\n-> Generating Standalone Interactive Map: {out_path.name}...")
    
    # Compute center of South Sudan
    center_lat = 7.5
    center_lon = 30.0
    
    # Initialize Folium Map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6.5,
        tiles="CartoDB positron",
        control_scale=True
    )
    
    # Select clean JSON-serializable columns
    keep_cols = [
        'adm3_name', 'adm2_name', 'adm1_name', 'area_km2', 'top_driver', 'top_driver_score',
        'farmland_pct', 'flood_pct', 'rainfall_mm', 'runoff_mm', 'geometry'
    ]
    adm3_simple = adm3[keep_cols].copy()
    adm3_simple['area_km2'] = adm3_simple['area_km2'].round(1)
    adm3_simple['top_driver_score'] = adm3_simple['top_driver_score'].round(1)
    adm3_simple['farmland_pct'] = adm3_simple['farmland_pct'].round(2)
    adm3_simple['flood_pct'] = adm3_simple['flood_pct'].round(2)
    adm3_simple['rainfall_mm'] = adm3_simple['rainfall_mm'].round(1)
    adm3_simple['runoff_mm'] = adm3_simple['runoff_mm'].round(1)
    adm3_simple['geometry'] = adm3_simple.geometry.simplify(0.003)
    
    adm1_simple = adm1[['adm1_name', 'geometry']].copy()
    adm1_simple['geometry'] = adm1_simple.geometry.simplify(0.005)
    
    # Style function for top driver
    def top_driver_style(feature):
        driver = feature['properties']['top_driver']
        color = DRIVER_PALETTE.get(driver, "#888888")
        return {
            'fillColor': color,
            'color': '#ffffff',
            'weight': 0.4,
            'fillOpacity': 0.82
        }
        
    def highlight_style(feature):
        return {
            'color': '#000000',
            'weight': 2.5,
            'fillOpacity': 0.95
        }
        
    # Top Driver Layer
    top_driver_layer = folium.GeoJson(
        adm3_simple,
        name="Top Flood Driver",
        style_function=top_driver_style,
        highlight_function=highlight_style,
        tooltip=folium.GeoJsonTooltip(
            fields=['adm3_name', 'adm2_name', 'adm1_name', 'top_driver', 'top_driver_score'],
            aliases=['Payam (ADM3):', 'County (ADM2):', 'State (ADM1):', 'Top Flood Driver:', 'Percentile Score:'],
            localize=True,
            sticky=False,
            labels=True,
            style="""
                background-color: #ffffff;
                border: 2px solid #2c3e50;
                border-radius: 4px;
                box-shadow: 3px 3px rgba(0,0,0,0.15);
                font-family: sans-serif;
                font-size: 12px;
                padding: 8px;
            """
        ),
        popup=folium.GeoJsonPopup(
            fields=['adm3_name', 'adm2_name', 'adm1_name', 'area_km2', 'top_driver',
                    'farmland_pct', 'flood_pct', 'rainfall_mm', 'runoff_mm'],
            aliases=['Payam:', 'County:', 'State:', 'Area (km²):', 'Top Flood Driver:',
                     'Farmland Crop Cover (%):', 'Flood Mask Inundation (%):', 'Annual Rainfall (mm):', 'Surface Runoff (mm):'],
            localize=True,
            labels=True,
            style="""
                font-family: sans-serif;
                font-size: 12px;
                width: 250px;
            """
        )
    ).add_to(m)
    
    # State Boundaries Layer
    folium.GeoJson(
        adm1_simple,
        name="State Boundaries (ADM1)",
        style_function=lambda f: {
            'fillColor': 'none',
            'color': '#1a252f',
            'weight': 1.8,
            'fillOpacity': 0.0
        }
    ).add_to(m)
    
    # Custom HTML Responsive Legend on the Side
    legend_html = f"""
    <div style="
        position: fixed; 
        bottom: 30px; right: 30px; width: 280px; height: auto;
        background-color: white; border:2px solid #b0bec5; border-radius: 8px;
        z-index:9999; font-size:12px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        padding: 14px 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.18);
        line-height: 1.5;
    ">
        <div style="font-size: 13px; font-weight: bold; margin-bottom: 8px; color: #111; border-bottom: 1px solid #ddd; padding-bottom: 5px;">
            South Sudan ADM3 Flood Drivers
        </div>
        <div style="margin-bottom: 6px;">
            <i style="background: {DRIVER_PALETTE['Flood Masks (Inundation)']}; width: 16px; height: 16px; display: inline-block; margin-right: 8px; border-radius: 3px; vertical-align: middle;"></i>
            <b>Flood Masks</b>: Inundation extent ({adm3['top_driver'].value_counts().get('Flood Masks (Inundation)', 0)})
        </div>
        <div style="margin-bottom: 6px;">
            <i style="background: {DRIVER_PALETTE['Farmland Exposure']}; width: 16px; height: 16px; display: inline-block; margin-right: 8px; border-radius: 3px; vertical-align: middle;"></i>
            <b>Farmland</b>: Crop exposure ({adm3['top_driver'].value_counts().get('Farmland Exposure', 0)})
        </div>
        <div style="margin-bottom: 6px;">
            <i style="background: {DRIVER_PALETTE['Extreme Rainfall']}; width: 16px; height: 16px; display: inline-block; margin-right: 8px; border-radius: 3px; vertical-align: middle;"></i>
            <b>Rainfall</b>: Precipitation ({adm3['top_driver'].value_counts().get('Extreme Rainfall', 0)})
        </div>
        <div style="margin-bottom: 6px;">
            <i style="background: {DRIVER_PALETTE['Surface Runoff']}; width: 16px; height: 16px; display: inline-block; margin-right: 8px; border-radius: 3px; vertical-align: middle;"></i>
            <b>Runoff</b>: Surface runoff ({adm3['top_driver'].value_counts().get('Surface Runoff', 0)})
        </div>
        <div style="border-top: 1px solid #eee; margin-top: 8px; padding-top: 6px; font-size: 10px; color: #666;">
            512 Payams | Click any region for detailed metric breakdown.
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Layer control
    folium.LayerControl(position='topright').add_to(m)
    
    m.save(str(out_path))
    print(f"   -> Saved Interactive Map: {out_path}")


# ==============================================================================
# 5. DATA EXPORT (CSV & GEOJSON)
# ==============================================================================

def export_data(adm3):
    print("\n-> Exporting Structured Datasets...")
    
    export_cols = [
        "adm3_name", "adm3_pcode", "adm2_name", "adm1_name", "area_km2",
        "farmland_pct", "flood_pct", "rainfall_mm", "runoff_mm",
        "pctile_farmland_pct", "pctile_flood_pct", "pctile_rainfall_mm", "pctile_runoff_mm",
        "top_driver", "top_driver_score"
    ]
    
    csv_out = DATA_DIR / "south_sudan_adm3_flood_drivers.csv"
    adm3[export_cols].to_csv(csv_out, index=False)
    print(f"   -> Saved CSV: {csv_out}")
    
    geojson_out = DATA_DIR / "south_sudan_adm3_flood_drivers.geojson"
    adm3[export_cols + ["geometry"]].to_file(geojson_out, driver="GeoJSON")
    print(f"   -> Saved GeoJSON: {geojson_out}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    start_time = time.time()
    
    # 1. Ingest & process
    adm3, adm1 = load_and_process_data()
    
    # 2. Generate Figure 1: Primary Map
    fig1_path = GRAPHS_DIR / "south_sudan_adm3_top_flood_drivers_map.png"
    plot_primary_top_driver_map(adm3, adm1, fig1_path)
    
    # 3. Generate Figure 2: 4-Panel Component Maps
    fig2_path = GRAPHS_DIR / "south_sudan_adm3_all_drivers_comparison.png"
    plot_four_component_driver_maps(adm3, adm1, fig2_path)
    
    # 4. Generate Figure 3: Statistical Breakdown
    fig3_path = GRAPHS_DIR / "south_sudan_adm3_drivers_breakdown.png"
    plot_statistical_breakdown(adm3, fig3_path)
    
    # 5. Generate Figure 4: Executive Dashboard
    fig4_path = GRAPHS_DIR / "south_sudan_adm3_executive_dashboard.png"
    plot_executive_dashboard(adm3, adm1, fig4_path)
    
    # 6. Generate Interactive Web Map
    html_path = OUTPUT_DIR / "south_sudan_adm3_flood_drivers_map.html"
    generate_interactive_map(adm3, adm1, html_path)
    
    # 7. Export CSV & GeoJSON
    export_data(adm3)
    
    print("\n" + "=" * 80)
    print(f"ALL GRAPHS AND MAPS SUCCESSFULLY GENERATED IN {time.time()-start_time:.1f}s!")
    print(f"Directory: {OUTPUT_DIR.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
