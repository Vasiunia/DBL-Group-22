#!/usr/bin/env python3
"""
build_predictor_app.py

Compiles South Sudan payam prediction datasets into a standalone, interactive,
high-performance web application: south_sudan_payam_flood_predictor.html.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("flood_driver_analysis/data")
MONTH_CSV = DATA_DIR / "payam_flood_predictions_monthly.csv"
SUMMARY_CSV = DATA_DIR / "payam_flood_predictions_annual_summary.csv"
GEOJSON_FILE = DATA_DIR / "payam_flood_predictions.geojson"
OUTPUT_HTML = Path("south_sudan_payam_flood_predictor.html")

def build_html_application():
    print("Building South Sudan Payam Flood & Driver Predictor Web Application...")
    
    df_m = pd.read_csv(MONTH_CSV)
    df_s = pd.read_csv(SUMMARY_CSV)
    
    # Load GeoJSON to get simplified boundary features
    with open(GEOJSON_FILE, "r") as f:
        geo_data = json.load(f)
        
    # Build compact dictionary for all 512 payams
    payams_dict = {}
    for _, row in df_s.iterrows():
        pcode = row['adm3_pcode']
        payams_dict[pcode] = {
            "pcode": pcode,
            "name": row['adm3_name'],
            "county": row['adm2_name'],
            "state": row['adm1_name'],
            "area_km2": float(row['area_km2']),
            "pop": int(row['population_est']),
            "farm_pct": float(row['farmland_pct']),
            "driver": row['annual_primary_driver'],
            "risk_score": float(row['max_flood_risk_score']),
            "risk_level": row['max_risk_level'],
            "peak_month": int(row['peak_risk_month']),
            "peak_month_name": row['peak_month_name'],
            "window": row['high_risk_window'],
            "peak_flood_pct": float(row['peak_flood_pct']),
            "peak_flood_km2": float(row['peak_flood_km2']),
            "annual_rain": float(row['annual_rainfall_mm']),
            "annual_ro": float(row['annual_runoff_mm']),
            "peak_pop_risk": int(row['peak_population_at_risk']),
            "peak_farm_risk": float(row['peak_farmland_km2_at_risk']),
            "lead_time": row['early_warning_lead_time'],
            "gov_action": row['government_action'],
            "cit_action": row['citizen_action'],
            "lat": float(row['rep_lat']),
            "lon": float(row['rep_lon'])
        }
        
    # Attach 12-month series
    monthly_data = {}
    for pcode, grp in df_m.groupby('adm3_pcode'):
        grp_sorted = grp.sort_values('month')
        monthly_data[pcode] = []
        for _, mrow in grp_sorted.iterrows():
            monthly_data[pcode].append({
                "m": int(mrow['month']),
                "name": mrow['month_name'],
                "driver": mrow['primary_driver'],
                "risk": float(mrow['flood_risk_score']),
                "level": mrow['risk_level'],
                "flood_pct": float(mrow['predicted_flood_pct']),
                "flood_km2": float(mrow['predicted_flood_km2']),
                "rain": float(mrow['predicted_rainfall_mm']),
                "ro": float(mrow['predicted_runoff_mm']),
                "rain_s": float(mrow['rainfall_driver_score']),
                "ro_s": float(mrow['runoff_driver_score']),
                "inund_s": float(mrow['inundation_driver_score']),
                "farm_s": float(mrow['farmland_driver_score']),
                "pop_risk": int(mrow['population_at_risk'])
            })
            
    payams_json = json.dumps(payams_dict)
    monthly_json = json.dumps(monthly_data)
    
    # We also embed a simplified GeoJSON for Leaflet
    # To keep the file lightweight, extract properties and geometries
    geo_features = []
    for feat in geo_data['features']:
        pcode = feat['properties'].get('adm3_pcode')
        if pcode in payams_dict:
            # retain only essential properties in the geojson layer
            feat['properties'] = {
                'pcode': pcode,
                'name': payams_dict[pcode]['name'],
                'county': payams_dict[pcode]['county'],
                'state': payams_dict[pcode]['state'],
                'driver': payams_dict[pcode]['driver'],
                'risk': payams_dict[pcode]['risk_score'],
                'level': payams_dict[pcode]['risk_level'],
                'peak_m': payams_dict[pcode]['peak_month_name'],
                'flood_pct': payams_dict[pcode]['peak_flood_pct'],
                'lat': payams_dict[pcode]['lat'],
                'lon': payams_dict[pcode]['lon']
            }
            geo_features.append(feat)
            
    simplified_geojson = json.dumps({"type": "FeatureCollection", "features": geo_features})
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>South Sudan Payam-Level Flood & Driver Prediction System</title>
    <!-- Leaflet CSS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #1e293b;
            --bg-hover: #334155;
            --border: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent-blue: #38bdf8;
            --accent-green: #22c55e;
            --accent-purple: #a855f7;
            --accent-amber: #f59e0b;
            --accent-red: #ef4444;
            --accent-critical: #dc2626;
            
            --driver-rain: #22c55e;
            --driver-runoff: #a855f7;
            --driver-inundation: #0284c7;
            --driver-farm: #f59e0b;
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        
        body {{
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.5;
            overflow-x: hidden;
        }}
        
        /* Header */
        header {{
            background: rgba(15, 23, 42, 0.95);
            border-bottom: 1px solid var(--border);
            padding: 1rem 2rem;
            position: sticky;
            top: 0;
            z-index: 1000;
            backdrop-filter: blur(10px);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .logo-group {{
            display: flex;
            align-items: center;
            gap: 1rem;
        }}
        
        .flag-icon {{
            font-size: 2rem;
        }}
        
        .header-title h1 {{
            font-size: 1.35rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #fff;
        }}
        
        .header-title p {{
            font-size: 0.8rem;
            color: var(--text-secondary);
        }}
        
        .header-badges {{
            display: flex;
            gap: 0.75rem;
            align-items: center;
        }}
        
        .stat-badge {{
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid var(--border);
            padding: 0.4rem 0.8rem;
            border-radius: 8px;
            font-size: 0.75rem;
            display: flex;
            flex-direction: column;
            align-items: flex-end;
        }}
        
        .stat-badge span:first-child {{
            color: var(--text-muted);
            text-transform: uppercase;
            font-size: 0.65rem;
            font-weight: 600;
        }}
        
        .stat-badge span:last-child {{
            color: var(--accent-blue);
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }}
        
        /* Main Layout */
        .dashboard-container {{
            display: grid;
            grid-template-columns: 420px 1fr;
            height: calc(100vh - 75px);
            overflow: hidden;
        }}
        
        /* Sidebar */
        .sidebar {{
            background: var(--bg-secondary);
            border-right: 1px solid var(--border);
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
            padding: 1.25rem;
        }}
        
        /* Search Box */
        .search-card {{
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }}
        
        .search-input-wrapper {{
            position: relative;
        }}
        
        .search-input-wrapper input {{
            width: 100%;
            background: #0f172a;
            border: 1px solid #475569;
            color: #fff;
            padding: 0.65rem 1rem 0.65rem 2.25rem;
            border-radius: 8px;
            font-size: 0.9rem;
            outline: none;
            transition: all 0.2s;
        }}
        
        .search-input-wrapper input:focus {{
            border-color: var(--accent-blue);
            box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2);
        }}
        
        .search-icon {{
            position: absolute;
            left: 0.75rem;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-muted);
            font-size: 0.9rem;
        }}
        
        .autocomplete-dropdown {{
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            background: #1e293b;
            border: 1px solid #475569;
            border-radius: 8px;
            max-height: 220px;
            overflow-y: auto;
            z-index: 1000;
            display: none;
            margin-top: 4px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
        }}
        
        .autocomplete-item {{
            padding: 0.6rem 1rem;
            cursor: pointer;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .autocomplete-item:hover {{
            background: #334155;
        }}
        
        .filter-row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
        }}
        
        select {{
            background: #0f172a;
            border: 1px solid #475569;
            color: #fff;
            padding: 0.5rem 0.75rem;
            border-radius: 6px;
            font-size: 0.8rem;
            outline: none;
        }}
        
        /* Payam Profile Card */
        .profile-card {{
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.25rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }}
        
        .profile-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }}
        
        .profile-title h2 {{
            font-size: 1.35rem;
            font-weight: 700;
            color: #fff;
        }}
        
        .profile-subtitle {{
            font-size: 0.8rem;
            color: var(--text-secondary);
        }}
        
        .driver-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.35rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }}
        
        .driver-rain {{ background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }}
        .driver-runoff {{ background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }}
        .driver-inundation {{ background: rgba(2, 132, 199, 0.15); color: #38bdf8; border: 1px solid rgba(2, 132, 199, 0.3); }}
        .driver-farm {{ background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }}
        
        /* Core Q&A Grid */
        .answers-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 0.75rem;
        }}
        
        .qa-box {{
            background: rgba(30, 41, 59, 0.5);
            border-left: 3px solid var(--accent-blue);
            padding: 0.75rem;
            border-radius: 0 8px 8px 0;
        }}
        
        .qa-title {{
            font-size: 0.7rem;
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            margin-bottom: 0.3rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }}
        
        .qa-value {{
            font-size: 0.95rem;
            font-weight: 600;
            color: #fff;
        }}
        
        .qa-sub {{
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 0.2rem;
        }}
        
        /* Risk Meter */
        .risk-gauge-wrapper {{
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
            margin-top: 0.25rem;
        }}
        
        .gauge-bar-bg {{
            height: 10px;
            background: #334155;
            border-radius: 5px;
            overflow: hidden;
            position: relative;
        }}
        
        .gauge-bar-fill {{
            height: 100%;
            border-radius: 5px;
            transition: width 0.6s ease;
        }}
        
        .gauge-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 0.7rem;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
        }}
        
        /* Metrics 2x2 */
        .metrics-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
        }}
        
        .metric-cell {{
            background: rgba(15, 23, 42, 0.5);
            padding: 0.6rem;
            border-radius: 6px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }}
        
        .metric-cell .label {{
            font-size: 0.65rem;
            color: var(--text-muted);
            text-transform: uppercase;
        }}
        
        .metric-cell .val {{
            font-size: 1rem;
            font-weight: 700;
            color: #fff;
            font-family: 'JetBrains Mono', monospace;
        }}
        
        /* Tabs for Action Guidance */
        .action-tabs {{
            display: flex;
            border-bottom: 1px solid var(--border);
            margin-top: 0.5rem;
        }}
        
        .action-tab {{
            padding: 0.4rem 0.8rem;
            font-size: 0.75rem;
            font-weight: 600;
            cursor: pointer;
            color: var(--text-muted);
            border-bottom: 2px solid transparent;
        }}
        
        .action-tab.active {{
            color: var(--accent-blue);
            border-color: var(--accent-blue);
        }}
        
        .action-content {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            padding-top: 0.5rem;
            line-height: 1.4;
        }}
        
        /* Right Main Content */
        .main-stage {{
            display: grid;
            grid-template-rows: 1fr 340px;
            overflow: hidden;
            position: relative;
        }}
        
        #map-container {{
            width: 100%;
            height: 100%;
            background: #090d16;
            position: relative;
        }}
        
        /* Map Floating Controls */
        .map-floating-overlay {{
            position: absolute;
            top: 15px;
            right: 15px;
            z-index: 800;
            background: rgba(15, 23, 42, 0.9);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.75rem 1rem;
            backdrop-filter: blur(8px);
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
        }}
        
        .overlay-title {{
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            color: var(--text-secondary);
        }}
        
        .month-slider-wrapper {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}
        
        .month-slider-wrapper input[type="range"] {{
            width: 160px;
            accent-color: var(--accent-blue);
        }}
        
        .month-badge {{
            background: var(--accent-blue);
            color: #0f172a;
            font-weight: 700;
            font-size: 0.8rem;
            padding: 0.2rem 0.6rem;
            border-radius: 4px;
            min-width: 45px;
            text-align: center;
        }}
        
        .map-legend {{
            display: flex;
            gap: 0.75rem;
            font-size: 0.7rem;
            margin-top: 0.25rem;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 0.3rem;
        }}
        
        .legend-color {{
            width: 10px;
            height: 10px;
            border-radius: 2px;
        }}
        
        /* Bottom Analytics Panel */
        .bottom-panel {{
            background: var(--bg-secondary);
            border-top: 1px solid var(--border);
            padding: 1rem 1.5rem;
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 1.5rem;
            overflow-y: auto;
        }}
        
        .chart-box {{
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.75rem 1rem;
            display: flex;
            flex-direction: column;
            height: 100%;
        }}
        
        .chart-box h3 {{
            font-size: 0.85rem;
            font-weight: 700;
            color: #fff;
            margin-bottom: 0.5rem;
            display: flex;
            justify-content: space-between;
        }}
        
        .scenario-box {{
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.75rem 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }}
        
        .scenario-box h3 {{
            font-size: 0.85rem;
            font-weight: 700;
            color: var(--accent-amber);
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }}
        
        .slider-group {{
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }}
        
        .slider-label {{
            display: flex;
            justify-content: space-between;
            font-size: 0.75rem;
            color: var(--text-secondary);
        }}
        
        .slider-label span:last-child {{
            font-weight: 700;
            color: var(--accent-blue);
            font-family: 'JetBrains Mono', monospace;
        }}
        
        .slider-group input[type="range"] {{
            width: 100%;
            accent-color: var(--accent-amber);
        }}
        
        /* Leaflet Popups */
        .leaflet-popup-content-wrapper {{
            background: #1e293b;
            color: #f8fafc;
            border: 1px solid #475569;
            border-radius: 8px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
        }}
        
        .leaflet-popup-tip {{
            background: #1e293b;
        }}
        
        /* Table Modal & Rankings */
        .view-table-btn {{
            background: #334155;
            color: #fff;
            border: 1px solid #475569;
            padding: 0.4rem 0.8rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }}
        
        .view-table-btn:hover {{
            background: #475569;
        }}
        
        /* Scrollbar */
        ::-webkit-scrollbar {{
            width: 6px;
            height: 6px;
        }}
        ::-webkit-scrollbar-track {{
            background: rgba(15, 23, 42, 0.5);
        }}
        ::-webkit-scrollbar-thumb {{
            background: #475569;
            border-radius: 3px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: #64748b;
        }}
    </style>
</head>
<body>

    <header>
        <div class="logo-group">
            <div class="flag-icon">🇸🇸</div>
            <div class="header-title">
                <h1>South Sudan Payam-Level Flood & Driver Predictor</h1>
                <p>National Operational Intelligence Platform for Government Disaster Management, UN OCHA & Citizens</p>
            </div>
        </div>
        
        <div class="header-badges">
            <div class="stat-badge">
                <span>Total Payams Modeled</span>
                <span>512 ADM3</span>
            </div>
            <div class="stat-badge">
                <span>Model Reliability</span>
                <span>CV R² 0.773 | Acc 83.1%</span>
            </div>
            <div class="stat-badge">
                <span>Peak Flood Window</span>
                <span>Aug - Nov</span>
            </div>
        </div>
    </header>

    <div class="dashboard-container">
        <!-- LEFT SIDEBAR: Search & Payam Profile -->
        <div class="sidebar">
            <!-- Search & Filters -->
            <div class="search-card">
                <div class="search-input-wrapper">
                    <span class="search-icon">🔍</span>
                    <input type="text" id="payam-search" placeholder="Search Payam (e.g. Kurwai, Bentiu, Bor, Juba)..." autocomplete="off">
                    <div class="autocomplete-dropdown" id="autocomplete-list"></div>
                </div>
                
                <div class="filter-row">
                    <select id="state-filter">
                        <option value="ALL">All States (10 + 3 Areas)</option>
                    </select>
                    <select id="driver-filter">
                        <option value="ALL">All Drivers</option>
                        <option value="Extreme Rainfall">Extreme Rainfall</option>
                        <option value="Surface Runoff">Surface Runoff</option>
                        <option value="Riverine Inundation">Riverine Inundation</option>
                        <option value="Farmland Exposure">Farmland Exposure</option>
                    </select>
                </div>
            </div>

            <!-- Payam Deep Dive Intelligence Card -->
            <div class="profile-card" id="payam-card">
                <div class="profile-header">
                    <div class="profile-title">
                        <h2 id="card-name">Kurwai</h2>
                        <div class="profile-subtitle" id="card-location">Ayod County, Jonglei State (SS030202)</div>
                    </div>
                    <div class="driver-badge driver-inundation" id="card-driver-badge">
                        <span>🌊</span> <span id="card-driver-name">Riverine Inundation</span>
                    </div>
                </div>

                <!-- 3 Core User Questions -->
                <div class="answers-grid">
                    <!-- 1. Which Driver -->
                    <div class="qa-box" style="border-left-color: #38bdf8;">
                        <div class="qa-title">
                            <span>1. Which Type of Driver?</span>
                        </div>
                        <div class="qa-value" id="card-which-driver">Riverine Inundation (Sudd Wetland Inflow)</div>
                        <div class="qa-sub" id="card-driver-desc">Driven by upstream White Nile discharge, Lake Victoria outflow, and chronic swamp drainage stagnation.</div>
                    </div>

                    <!-- 2. When -->
                    <div class="qa-box" style="border-left-color: #f59e0b;">
                        <div class="qa-title">
                            <span>2. When (Timing & Season)?</span>
                        </div>
                        <div class="qa-value" id="card-when-timing">Peak Risk: August (Critical Window: Mar - Dec)</div>
                        <div class="qa-sub" id="card-lead-time">Early Warning Lead Time: 30 - 60 days (Upstream basin tracking)</div>
                    </div>

                    <!-- 3. How Much -->
                    <div class="qa-box" style="border-left-color: #ef4444;">
                        <div class="qa-title">
                            <span>3. How Much (Severity & Magnitude)?</span>
                        </div>
                        <div class="risk-gauge-wrapper">
                            <div style="display:flex; justify-content:space-between; font-size:0.85rem; font-weight:700;">
                                <span id="card-risk-level" style="color: #dc2626;">CRITICAL RISK TIER</span>
                                <span id="card-risk-score">90.2 / 100</span>
                            </div>
                            <div class="gauge-bar-bg">
                                <div class="gauge-bar-fill" id="card-gauge-fill" style="width: 90.2%; background: #dc2626;"></div>
                            </div>
                            <div class="gauge-labels">
                                <span>Low</span>
                                <span>Moderate</span>
                                <span>High</span>
                                <span>Very High</span>
                                <span>Critical</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Quantitative Metrics 2x2 -->
                <div class="metrics-grid">
                    <div class="metric-cell">
                        <div class="label">Peak Inundation Extent</div>
                        <div class="val" id="card-flood-pct" style="color: #38bdf8;">57.3%</div>
                        <div class="qa-sub" id="card-flood-km2">1,361.2 km² submerged</div>
                    </div>
                    <div class="metric-cell">
                        <div class="label">Annual Precipitation</div>
                        <div class="val" id="card-rain-val" style="color: #4ade80;">689.4 mm</div>
                        <div class="qa-sub" id="card-runoff-val">Runoff: 31.0 mm</div>
                    </div>
                    <div class="metric-cell">
                        <div class="label">Population Exposed</div>
                        <div class="val" id="card-pop-exposed" style="color: #fb7185;">26,386</div>
                        <div class="qa-sub" id="card-pop-total">out of 51,036 residents</div>
                    </div>
                    <div class="metric-cell">
                        <div class="label">Cropland at Risk</div>
                        <div class="val" id="card-farm-exposed" style="color: #fbbf24;">0.0 km²</div>
                        <div class="qa-sub" id="card-farm-density">Cropland density: 0.0%</div>
                    </div>
                </div>

                <!-- Action Protocols -->
                <div>
                    <div class="action-tabs">
                        <div class="action-tab active" id="tab-gov" onclick="switchTab('gov')">🏛️ Government / UN Action</div>
                        <div class="action-tab" id="tab-cit" onclick="switchTab('cit')">👥 Citizen & Community Safety</div>
                    </div>
                    <div class="action-content" id="action-content-text">
                        Reinforce perimeter dykes; clear drainage trunk canals; preposition riverine boats and water purification in Bentiu/Fangak.
                    </div>
                </div>
            </div>
        </div>

        <!-- RIGHT STAGE: Interactive Map + Analytics -->
        <div class="main-stage">
            <!-- Map Container -->
            <div id="map-container">
                <!-- Floating Controls -->
                <div class="map-floating-overlay">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span class="overlay-title">Seasonal Month Animation</span>
                        <span class="month-badge" id="slider-month-label">Aug</span>
                    </div>
                    
                    <div class="month-slider-wrapper">
                        <span style="font-size:0.75rem; color:var(--text-muted);">Jan</span>
                        <input type="range" id="month-slider" min="1" max="12" value="8" step="1">
                        <span style="font-size:0.75rem; color:var(--text-muted);">Dec</span>
                    </div>

                    <div style="display:flex; justify-content:space-between; align-items:center; margin-top:4px;">
                        <span class="overlay-title">Color Mode</span>
                        <select id="map-color-mode" style="padding:0.2rem 0.5rem; font-size:0.75rem;">
                            <option value="driver">By Dominant Driver</option>
                            <option value="risk">By Flood Risk Score</option>
                        </select>
                    </div>

                    <div class="map-legend" id="map-legend">
                        <div class="legend-item"><div class="legend-color" style="background:#0284c7;"></div><span>Inundation</span></div>
                        <div class="legend-item"><div class="legend-color" style="background:#22c55e;"></div><span>Rainfall</span></div>
                        <div class="legend-item"><div class="legend-color" style="background:#a855f7;"></div><span>Runoff</span></div>
                        <div class="legend-item"><div class="legend-color" style="background:#f59e0b;"></div><span>Farmland</span></div>
                    </div>
                </div>
            </div>

            <!-- Bottom Analytics & Scenario Simulator -->
            <div class="bottom-panel">
                <!-- 12-Month Seasonality Chart -->
                <div class="chart-box">
                    <h3>
                        <span>📈 12-Month Seasonality Profile (<span id="chart-payam-name">Kurwai</span>)</span>
                        <span style="font-size:0.75rem; font-weight:normal; color:var(--text-muted);">Risk Score (0-100) vs Rainfall & Runoff (mm)</span>
                    </h3>
                    <div style="flex:1; position:relative; min-height:180px;">
                        <canvas id="seasonality-chart"></canvas>
                    </div>
                </div>

                <!-- What-if Scenario Simulator -->
                <div class="scenario-box">
                    <h3>⚡ "What-If" Scenario Simulator</h3>
                    <p style="font-size:0.75rem; color:var(--text-secondary);">Simulate climate anomalies and stress-test disaster response in real-time.</p>
                    
                    <div class="slider-group">
                        <div class="slider-label">
                            <span>Extreme Rainfall Anomaly</span>
                            <span id="val-rain-sim">0%</span>
                        </div>
                        <input type="range" id="sim-rain-slider" min="-50" max="50" value="0" step="5">
                    </div>

                    <div class="slider-group">
                        <div class="slider-label">
                            <span>Upstream Nile Flood Surge</span>
                            <span id="val-surge-sim">Baseline</span>
                        </div>
                        <input type="range" id="sim-surge-slider" min="0" max="3" value="0" step="1">
                    </div>

                    <div style="margin-top:auto; font-size:0.75rem; background:rgba(30,41,59,0.7); padding:0.4rem 0.6rem; border-radius:6px; border:1px solid rgba(255,255,255,0.05);">
                        <span style="color:var(--accent-amber); font-weight:600;">Simulation Outcome:</span>
                        <span id="sim-outcome-text" style="color:#fff;">Operating under historical climatological baseline.</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Leaflet JS -->
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

    <script>
        // Data injected from Python pipeline
        const PAYAMS = {payams_json};
        const MONTHLY = {monthly_json};
        const GEOJSON_DATA = {simplified_geojson};

        const MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
        const DRIVER_COLORS = {{
            "Extreme Rainfall": "#22c55e",
            "Surface Runoff": "#a855f7",
            "Riverine Inundation": "#0284c7",
            "Farmland Exposure": "#f59e0b"
        }};
        const DRIVER_ICONS = {{
            "Extreme Rainfall": "🌧️",
            "Surface Runoff": "💧",
            "Riverine Inundation": "🌊",
            "Farmland Exposure": "🌾"
        }};

        let currentPayam = "SS030202"; // Kurwai default
        let currentMonth = 8; // August default
        let currentTab = "gov";
        let map, geojsonLayer;
        let chartInstance;
        
        let simRainAnomaly = 0;
        let simSurgeLevel = 0;

        // Initialize Application
        document.addEventListener("DOMContentLoaded", () => {{
            populateFilters();
            initMap();
            initChart();
            selectPayam(currentPayam);
            setupEventListeners();
        }});

        function populateFilters() {{
            const stateSet = new Set();
            for (let p in PAYAMS) {{
                stateSet.add(PAYAMS[p].state);
            }}
            const stateFilter = document.getElementById("state-filter");
            Array.from(stateSet).sort().forEach(state => {{
                const opt = document.createElement("option");
                opt.value = state;
                opt.textContent = state;
                stateFilter.appendChild(opt);
            }});
        }}

        function initMap() {{
            map = L.map('map-container', {{
                center: [7.5, 30.0],
                zoom: 6.5,
                minZoom: 5.5,
                maxZoom: 12,
                zoomControl: false
            }});

            L.control.zoom({{ position: 'topleft' }}).addTo(map);

            // Dark styled CartoDB tile layer
            L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
                attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
                subdomains: 'abcd',
                maxZoom: 19
            }}).addTo(map);

            renderGeoJsonLayer();
        }}

        function getPayamColor(pcode) {{
            const mode = document.getElementById("map-color-mode").value;
            const p = PAYAMS[pcode];
            if (!p) return "#475569";

            const mData = MONTHLY[pcode]?.find(m => m.m === currentMonth);
            const driver = mData ? mData.driver : p.driver;
            let risk = mData ? mData.risk : p.risk_score;

            // Apply simulation shifts
            if (simRainAnomaly !== 0) {{
                risk = Math.min(100, Math.max(0, risk + simRainAnomaly * 0.15));
            }}
            if (simSurgeLevel > 0 && (driver === "Riverine Inundation" || p.state === "Unity" || p.state === "Jonglei")) {{
                risk = Math.min(100, risk + simSurgeLevel * 6);
            }}

            if (mode === "driver") {{
                return DRIVER_COLORS[driver] || "#64748b";
            }} else {{
                // By risk
                if (risk >= 85) return "#dc2626"; // Critical
                if (risk >= 70) return "#ea580c"; // Very High
                if (risk >= 50) return "#f59e0b"; // High
                if (risk >= 25) return "#eab308"; // Moderate
                return "#22c55e"; // Low
            }}
        }}

        function renderGeoJsonLayer() {{
            if (geojsonLayer) {{
                map.removeLayer(geojsonLayer);
            }}

            geojsonLayer = L.geoJSON(GEOJSON_DATA, {{
                style: function(feature) {{
                    const pcode = feature.properties.pcode;
                    const isSelected = (pcode === currentPayam);
                    return {{
                        fillColor: getPayamColor(pcode),
                        weight: isSelected ? 3 : 0.8,
                        opacity: 1,
                        color: isSelected ? '#ffffff' : 'rgba(255, 255, 255, 0.25)',
                        fillOpacity: isSelected ? 0.85 : 0.65
                    }};
                }},
                onEachFeature: function(feature, layer) {{
                    const pcode = feature.properties.pcode;
                    const p = PAYAMS[pcode];

                    layer.on({{
                        mouseover: function(e) {{
                            const l = e.target;
                            l.setStyle({{
                                weight: 2.5,
                                color: '#38bdf8',
                                fillOpacity: 0.9
                            }});
                            l.bringToFront();
                        }},
                        mouseout: function(e) {{
                            geojsonLayer.resetStyle(e.target);
                        }},
                        click: function() {{
                            selectPayam(pcode);
                        }}
                    }});

                    const popupContent = `
                        <div style="font-family:'Inter',sans-serif; min-width:180px;">
                            <div style="font-weight:700; font-size:1rem; margin-bottom:2px;">${{p.name}}</div>
                            <div style="font-size:0.75rem; color:#94a3b8; margin-bottom:6px;">${{p.county}}, ${{p.state}}</div>
                            <div style="font-size:0.8rem; margin-bottom:4px;"><b>Top Driver:</b> ${{p.driver}}</div>
                            <div style="font-size:0.8rem; margin-bottom:4px;"><b>Peak Month:</b> ${{p.peak_month_name}}</div>
                            <div style="font-size:0.8rem;"><b>Risk Score:</b> <span style="font-weight:700; color:${{p.risk_score>=70?'#ef4444':'#f59e0b'}};">${{p.risk_score.toFixed(1)}}/100</span></div>
                        </div>
                    `;
                    layer.bindPopup(popupContent);
                }}
            }}).addTo(map);
        }}

        function initChart() {{
            const ctx = document.getElementById('seasonality-chart').getContext('2d');
            chartInstance = new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: MONTH_NAMES,
                    datasets: [
                        {{
                            label: 'Flood Risk Score (0-100)',
                            data: [],
                            borderColor: '#ef4444',
                            backgroundColor: 'rgba(239, 68, 68, 0.1)',
                            borderWidth: 2.5,
                            fill: true,
                            tension: 0.3,
                            yAxisID: 'y'
                        }},
                        {{
                            label: 'Predicted Inundation (%)',
                            data: [],
                            borderColor: '#38bdf8',
                            backgroundColor: 'transparent',
                            borderWidth: 2,
                            borderDash: [5, 5],
                            tension: 0.3,
                            yAxisID: 'y'
                        }},
                        {{
                            label: 'Monthly Rainfall (mm)',
                            data: [],
                            borderColor: '#22c55e',
                            backgroundColor: 'rgba(34, 197, 94, 0.2)',
                            type: 'bar',
                            yAxisID: 'y1'
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {{
                        mode: 'index',
                        intersect: false
                    }},
                    plugins: {{
                        legend: {{
                            labels: {{
                                color: '#94a3b8',
                                font: {{ size: 10, family: 'Inter' }}
                            }}
                        }},
                        tooltip: {{
                            backgroundColor: '#0f172a',
                            titleColor: '#fff',
                            bodyColor: '#cbd5e1',
                            borderColor: '#334155',
                            borderWidth: 1
                        }}
                    }},
                    scales: {{
                        x: {{
                            grid: {{ color: 'rgba(255, 255, 255, 0.05)' }},
                            ticks: {{ color: '#94a3b8', font: {{ size: 10 }} }}
                        }},
                        y: {{
                            type: 'linear',
                            display: true,
                            position: 'left',
                            min: 0,
                            max: 100,
                            grid: {{ color: 'rgba(255, 255, 255, 0.05)' }},
                            ticks: {{ color: '#ef4444', font: {{ size: 10 }} }},
                            title: {{ display: true, text: 'Risk Score / Flood %', color: '#ef4444', font: {{ size: 9 }} }}
                        }},
                        y1: {{
                            type: 'linear',
                            display: true,
                            position: 'right',
                            min: 0,
                            grid: {{ drawOnChartArea: false }},
                            ticks: {{ color: '#22c55e', font: {{ size: 10 }} }},
                            title: {{ display: true, text: 'Precipitation (mm)', color: '#22c55e', font: {{ size: 9 }} }}
                        }}
                    }}
                }}
            }});
        }}

        function selectPayam(pcode) {{
            currentPayam = pcode;
            const p = PAYAMS[pcode];
            if (!p) return;

            document.getElementById("card-name").textContent = p.name;
            document.getElementById("chart-payam-name").textContent = p.name;
            document.getElementById("card-location").textContent = `${{p.county}} County, ${{p.state}} (${{p.pcode}})`;
            
            // Driver Badge
            const badge = document.getElementById("card-driver-badge");
            badge.className = `driver-badge driver-${{p.driver === 'Extreme Rainfall' ? 'rain' : p.driver === 'Surface Runoff' ? 'runoff' : p.driver === 'Riverine Inundation' ? 'inundation' : 'farm'}}`;
            badge.innerHTML = `<span>${{DRIVER_ICONS[p.driver]}}</span> <span>${{p.driver}}</span>`;
            
            // Which Driver Description
            document.getElementById("card-which-driver").textContent = `${{p.driver}}`;
            let driverDesc = "";
            if (p.driver === "Riverine Inundation") {{
                driverDesc = "Sudd wetland expansion & White Nile overflow with multi-month surface water stagnation.";
            }} else if (p.driver === "Extreme Rainfall") {{
                driverDesc = "High-intensity convective cloudbursts generating acute pluvial water-logging.";
            }} else if (p.driver === "Surface Runoff") {{
                driverDesc = "Rapid overland drainage accumulation and catchment runoff saturation.";
            }} else {{
                driverDesc = "High cropland concentration vulnerable to seasonal water inundation.";
            }}
            document.getElementById("card-driver-desc").textContent = driverDesc;

            // When Timing
            document.getElementById("card-when-timing").textContent = `Peak Risk: ${{p.peak_month_name}} (Critical Window: ${{p.window}})`;
            document.getElementById("card-lead-time").textContent = `Early Warning Horizon: ${{p.lead_time}}`;

            // How Much Risk
            let risk = p.risk_score;
            let floodPct = p.peak_flood_pct;
            let rainVal = p.annual_rain;
            
            // Apply simulation
            if (simRainAnomaly !== 0) {{
                risk = Math.min(100, Math.max(0, risk + simRainAnomaly * 0.15));
                floodPct = Math.min(100, Math.max(0, floodPct * (1 + simRainAnomaly/100)));
                rainVal = rainVal * (1 + simRainAnomaly/100);
            }}
            if (simSurgeLevel > 0 && (p.driver === "Riverine Inundation" || p.state === "Unity" || p.state === "Jonglei")) {{
                risk = Math.min(100, risk + simSurgeLevel * 6);
                floodPct = Math.min(100, floodPct + simSurgeLevel * 5);
            }}

            const riskLevel = (risk >= 85) ? "CRITICAL" : (risk >= 70) ? "VERY HIGH" : (risk >= 50) ? "HIGH" : (risk >= 25) ? "MODERATE" : "LOW";
            const riskColor = (risk >= 85) ? "#dc2626" : (risk >= 70) ? "#ea580c" : (risk >= 50) ? "#f59e0b" : "#22c55e";

            document.getElementById("card-risk-level").textContent = `${{riskLevel}} RISK TIER`;
            document.getElementById("card-risk-level").style.color = riskColor;
            document.getElementById("card-risk-score").textContent = `${{risk.toFixed(1)}} / 100`;
            document.getElementById("card-gauge-fill").style.width = `${{risk}}%`;
            document.getElementById("card-gauge-fill").style.backgroundColor = riskColor;

            // Metrics
            document.getElementById("card-flood-pct").textContent = `${{floodPct.toFixed(1)}}%`;
            document.getElementById("card-flood-km2").textContent = `${{(p.area_km2 * (floodPct/100)).toFixed(1)}} km² submerged`;
            document.getElementById("card-rain-val").textContent = `${{rainVal.toFixed(1)}} mm`;
            document.getElementById("card-runoff-val").textContent = `Runoff: ${{p.annual_ro.toFixed(1)}} mm`;
            
            const popExposed = Math.round(p.pop * (risk/100) * (Math.max(0.5, floodPct)/100));
            document.getElementById("card-pop-exposed").textContent = popExposed.toLocaleString();
            document.getElementById("card-pop-total").textContent = `out of ${{p.pop.toLocaleString()}} residents`;

            const farmExposed = ((p.area_km2 * (p.farm_pct/100)) * (floodPct/100)).toFixed(2);
            document.getElementById("card-farm-exposed").textContent = `${{farmExposed}} km²`;
            document.getElementById("card-farm-density").textContent = `Cropland density: ${{p.farm_pct.toFixed(1)}}%`;

            // Action guidance
            document.getElementById("action-content-text").textContent = (currentTab === "gov") ? p.gov_action : p.cit_action;

            // Update Chart
            updateChart(pcode);

            // Re-style map feature
            if (geojsonLayer) {{
                geojsonLayer.eachLayer(layer => {{
                    const isSelected = (layer.feature.properties.pcode === pcode);
                    if (isSelected) {{
                        layer.setStyle({{ weight: 3, color: '#ffffff', fillOpacity: 0.9 }});
                        layer.bringToFront();
                    }} else {{
                        geojsonLayer.resetStyle(layer);
                    }}
                }});
            }}
        }}

        function updateChart(pcode) {{
            const mList = MONTHLY[pcode];
            if (!mList || !chartInstance) return;

            const riskScores = mList.map(d => {{
                let r = d.risk;
                if (simRainAnomaly !== 0) r = Math.min(100, Math.max(0, r + simRainAnomaly * 0.15));
                if (simSurgeLevel > 0 && (d.driver === "Riverine Inundation")) r = Math.min(100, r + simSurgeLevel * 6);
                return r;
            }});

            const floodPcts = mList.map(d => {{
                let f = d.flood_pct;
                if (simRainAnomaly !== 0) f = Math.min(100, Math.max(0, f * (1 + simRainAnomaly/100)));
                if (simSurgeLevel > 0 && (d.driver === "Riverine Inundation")) f = Math.min(100, f + simSurgeLevel * 5);
                return f;
            }});

            const rainVals = mList.map(d => {{
                return (simRainAnomaly !== 0) ? (d.rain * (1 + simRainAnomaly/100)) : d.rain;
            }});

            chartInstance.data.datasets[0].data = riskScores;
            chartInstance.data.datasets[1].data = floodPcts;
            chartInstance.data.datasets[2].data = rainVals;
            chartInstance.update();
        }}

        function switchTab(tab) {{
            currentTab = tab;
            document.getElementById("tab-gov").className = `action-tab ${{tab === 'gov' ? 'active' : ''}}`;
            document.getElementById("tab-cit").className = `action-tab ${{tab === 'cit' ? 'active' : ''}}`;
            const p = PAYAMS[currentPayam];
            if (p) {{
                document.getElementById("action-content-text").textContent = (tab === "gov") ? p.gov_action : p.cit_action;
            }}
        }}

        function setupEventListeners() {{
            // Search Autocomplete
            const searchInput = document.getElementById("payam-search");
            const dropdown = document.getElementById("autocomplete-list");

            searchInput.addEventListener("input", function() {{
                const val = this.value.trim().toLowerCase();
                dropdown.innerHTML = "";
                if (!val) {{
                    dropdown.style.display = "none";
                    return;
                }}

                const matches = [];
                for (let p in PAYAMS) {{
                    const payam = PAYAMS[p];
                    if (payam.name.toLowerCase().includes(val) || payam.county.toLowerCase().includes(val)) {{
                        matches.push(payam);
                        if (matches.length >= 10) break;
                    }}
                }}

                if (matches.length > 0) {{
                    dropdown.style.display = "block";
                    matches.forEach(m => {{
                        const item = document.createElement("div");
                        item.className = "autocomplete-item";
                        item.innerHTML = `
                            <div>
                                <span style="font-weight:600; color:#fff;">${{m.name}}</span>
                                <span style="font-size:0.75rem; color:#94a3b8; margin-left:6px;">(${{m.county}}, ${{m.state}})</span>
                            </div>
                            <span style="font-size:0.7rem; color:${{DRIVER_COLORS[m.driver]}};">${{m.driver}}</span>
                        `;
                        item.addEventListener("click", () => {{
                            searchInput.value = m.name;
                            dropdown.style.display = "none";
                            selectPayam(m.pcode);
                            // Pan map to payam centroid
                            map.flyTo([m.lat, m.lon], 9);
                        }});
                        dropdown.appendChild(item);
                    }});
                }} else {{
                    dropdown.style.display = "none";
                }}
            }});

            document.addEventListener("click", function(e) {{
                if (!searchInput.contains(e.target) && !dropdown.contains(e.target)) {{
                    dropdown.style.display = "none";
                }}
            }});

            // Month Slider
            const monthSlider = document.getElementById("month-slider");
            const monthLabel = document.getElementById("slider-month-label");
            monthSlider.addEventListener("input", function() {{
                currentMonth = parseInt(this.value);
                monthLabel.textContent = MONTH_NAMES[currentMonth - 1];
                renderGeoJsonLayer();
            }});

            // Map Color Mode
            document.getElementById("map-color-mode").addEventListener("change", function() {{
                const legend = document.getElementById("map-legend");
                if (this.value === "driver") {{
                    legend.innerHTML = `
                        <div class="legend-item"><div class="legend-color" style="background:#0284c7;"></div><span>Inundation</span></div>
                        <div class="legend-item"><div class="legend-color" style="background:#22c55e;"></div><span>Rainfall</span></div>
                        <div class="legend-item"><div class="legend-color" style="background:#a855f7;"></div><span>Runoff</span></div>
                        <div class="legend-item"><div class="legend-color" style="background:#f59e0b;"></div><span>Farmland</span></div>
                    `;
                }} else {{
                    legend.innerHTML = `
                        <div class="legend-item"><div class="legend-color" style="background:#22c55e;"></div><span>Low</span></div>
                        <div class="legend-item"><div class="legend-color" style="background:#eab308;"></div><span>Mod</span></div>
                        <div class="legend-item"><div class="legend-color" style="background:#f59e0b;"></div><span>High</span></div>
                        <div class="legend-item"><div class="legend-color" style="background:#ea580c;"></div><span>V.High</span></div>
                        <div class="legend-item"><div class="legend-color" style="background:#dc2626;"></div><span>Critical</span></div>
                    `;
                }}
                renderGeoJsonLayer();
            }});

            // Scenario Sliders
            const rainSlider = document.getElementById("sim-rain-slider");
            const rainVal = document.getElementById("val-rain-sim");
            const surgeSlider = document.getElementById("sim-surge-slider");
            const surgeVal = document.getElementById("val-surge-sim");
            const outcomeText = document.getElementById("sim-outcome-text");

            function updateSimulation() {{
                simRainAnomaly = parseFloat(rainSlider.value);
                simSurgeLevel = parseInt(surgeSlider.value);

                rainVal.textContent = (simRainAnomaly > 0 ? `+${{simRainAnomaly}}%` : `${{simRainAnomaly}}%`);
                const surgeLabels = ["Baseline", "+0.5m Surge", "+1.0m Severe Surge", "+1.5m Record Surge"];
                surgeVal.textContent = surgeLabels[simSurgeLevel];

                let desc = "";
                if (simRainAnomaly === 0 && simSurgeLevel === 0) {{
                    desc = "Operating under historical climatological baseline.";
                }} else {{
                    desc = `Simulating ${{simRainAnomaly > 0 ? '+' + simRainAnomaly + '% rainfall deluges' : simRainAnomaly < 0 ? simRainAnomaly + '% drought/deficit' : ''}} ${{simSurgeLevel > 0 ? 'with ' + surgeLabels[simSurgeLevel] + ' from Lake Victoria' : ''}}. Peak inundation & exposed population recalculated.`;
                }}
                outcomeText.textContent = desc;

                selectPayam(currentPayam);
                renderGeoJsonLayer();
            }}

            rainSlider.addEventListener("input", updateSimulation);
            surgeSlider.addEventListener("input", updateSimulation);
        }}
    </script>
</body>
</html>
"""
    
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"Web application successfully compiled to: {OUTPUT_HTML} (Size: {len(html_content):,} bytes)")

if __name__ == "__main__":
    build_html_application()
