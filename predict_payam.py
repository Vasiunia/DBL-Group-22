#!/usr/bin/env python3
"""
predict_payam.py

Command-Line Interface and Query Engine for South Sudan Payam-Level Flood Driver Predictions.
Enables Government Officials, Humanitarian Responders, and Citizens to predict:
- In WHICH payam
- WHICH TYPE of driver
- WHEN (timing, peak month, seasonal window)
- HOW MUCH (magnitude of rainfall, runoff, inundation %, flooded km2, risk score, population exposed)

Usage Examples:
    python predict_payam.py --payam "Kurwai"
    python predict_payam.py --payam "Juba Town" --month 8
    python predict_payam.py --state "Unity" --month 10 --top 10
    python predict_payam.py --driver "Riverine Inundation" --top 15
    python predict_payam.py --simulate --payam "Pilieny" --rain-anomaly 25
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("flood_driver_analysis/data")
MONTH_CSV = DATA_DIR / "payam_flood_predictions_monthly.csv"
SUMMARY_CSV = DATA_DIR / "payam_flood_predictions_annual_summary.csv"

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_NAMES_FULL = ["January", "February", "March", "April", "May", "June", 
                    "July", "August", "September", "October", "November", "December"]

DRIVER_ICONS = {
    "Extreme Rainfall": "🌧️",
    "Surface Runoff": "💧",
    "Riverine Inundation": "🌊",
    "Farmland Exposure": "🌾",
}

RISK_COLORS = {
    "Low": "\033[92m",       # Green
    "Moderate": "\033[93m",  # Yellow
    "High": "\033[91m",      # Light Red
    "Very High": "\033[95m",  # Magenta
    "Critical": "\033[41m\033[97m", # Red background, white text
}
RESET_COLOR = "\033[0m"
BOLD = "\033[1m"

def load_data():
    if not MONTH_CSV.exists() or not SUMMARY_CSV.exists():
        print(f"Error: Prediction datasets not found in {DATA_DIR}. Please run train_payam_flood_predictor.py first.")
        sys.exit(1)
    df_m = pd.read_csv(MONTH_CSV)
    df_s = pd.read_csv(SUMMARY_CSV)
    return df_m, df_s

def format_risk_badge(level, score):
    color = RISK_COLORS.get(level, "")
    return f"{color}[{level.upper()} - {score:.1f}/100]{RESET_COLOR}"

def print_payam_profile(payam_name, month_num=None, rain_anomaly=0.0):
    df_m, df_s = load_data()
    
    # Case-insensitive match
    matches = df_s[df_s['adm3_name'].str.lower() == payam_name.lower()]
    if len(matches) == 0:
        # Partial match
        matches = df_s[df_s['adm3_name'].str.lower().str.contains(payam_name.lower())]
        
    if len(matches) == 0:
        print(f"No payam found matching '{payam_name}'. Available payams include: Juba Town, Malakal, Bentiu, Kurwai, Bor, etc.")
        return
        
    row = matches.iloc[0]
    pcode = row['adm3_pcode']
    p_months = df_m[df_m['adm3_pcode'] == pcode].sort_values('month').copy()
    
    # Apply scenario anomaly if provided
    if rain_anomaly != 0.0:
        p_months['predicted_rainfall_mm'] *= (1.0 + rain_anomaly / 100.0)
        p_months['predicted_runoff_mm'] *= (1.0 + (rain_anomaly * 1.2) / 100.0)
        p_months['rainfall_driver_score'] = np.clip(p_months['rainfall_driver_score'] * (1.0 + rain_anomaly/100.0), 0, 100)
        p_months['flood_risk_score'] = np.clip(p_months['flood_risk_score'] + rain_anomaly * 0.15, 0, 100)
        p_months['predicted_flood_pct'] = np.clip(p_months['predicted_flood_pct'] * (1.0 + rain_anomaly/100.0), 0, 100)
        p_months['predicted_flood_km2'] = (row['area_km2'] * (p_months['predicted_flood_pct'] / 100.0)).round(2)
    
    print("\n" + "=" * 78)
    print(f"{BOLD}PAYAM FLOOD PREDICTION INTELLIGENCE CARD{RESET_COLOR}")
    print("=" * 78)
    print(f"📍 {BOLD}Payam (ADM3):{RESET_COLOR}  {row['adm3_name']} (P-Code: {row['adm3_pcode']})")
    print(f"🏛️  {BOLD}County (ADM2):{RESET_COLOR} {row['adm2_name']}   |   {BOLD}State (ADM1):{RESET_COLOR} {row['adm1_name']}")
    print(f"📐 {BOLD}Area:{RESET_COLOR}          {row['area_km2']:.1f} km²   |   {BOLD}Est. Population:{RESET_COLOR} {row['population_est']:,}")
    print(f"🌾 {BOLD}Cropland Density:{RESET_COLOR} {row['farmland_pct']:.2f}%")
    print("-" * 78)
    
    if rain_anomaly != 0.0:
        print(f"⚠️  {BOLD}SIMULATED SCENARIO:{RESET_COLOR} Rainfall Anomaly of {rain_anomaly:+.1f}% applied.")
        print("-" * 78)
        
    print(f"{BOLD}1. WHICH TYPE OF DRIVER?{RESET_COLOR}")
    driver = row['annual_primary_driver']
    icon = DRIVER_ICONS.get(driver, "⚠️")
    print(f"   Primary Dominant Driver: {icon} {BOLD}{driver}{RESET_COLOR}")
    print(f"   Driver Characteristics:  ", end="")
    if driver == "Riverine Inundation":
        print("Sudd wetland expansion & White Nile overflow with multi-month surface water stagnation.")
    elif driver == "Extreme Rainfall":
        print("High-intensity convective cloudbursts generating acute pluvial water-logging.")
    elif driver == "Surface Runoff":
        print("Rapid overland drainage accumulation and catchment runoff saturation.")
    else:
        print("High cropland concentration vulnerable to seasonal water inundation.")
        
    print(f"\n{BOLD}2. WHEN (TIMING & SEASONAL PROFILE)?{RESET_COLOR}")
    print(f"   Peak Risk Month:       {BOLD}{row['peak_month_name']} (Month {row['peak_risk_month']}){RESET_COLOR}")
    print(f"   High Risk Window:      {BOLD}{row['high_risk_window']}{RESET_COLOR}")
    print(f"   Early Warning Horizon: {row['early_warning_lead_time']}")
    
    print(f"\n{BOLD}3. HOW MUCH (MAGNITUDE & IMPACT)?{RESET_COLOR}")
    peak_risk = p_months['flood_risk_score'].max()
    risk_level = "Critical" if peak_risk >= 85 else "Very High" if peak_risk >= 70 else "High" if peak_risk >= 50 else "Moderate" if peak_risk >= 25 else "Low"
    print(f"   Peak Flood Risk Score: {format_risk_badge(risk_level, peak_risk)}")
    print(f"   Peak Flooded Extent:   {BOLD}{p_months['predicted_flood_pct'].max():.2f}%{RESET_COLOR} ({p_months['predicted_flood_km2'].max():.1f} km²)")
    print(f"   Annual Rainfall:       {p_months['predicted_rainfall_mm'].sum():.1f} mm  |  Annual Runoff: {p_months['predicted_runoff_mm'].sum():.1f} mm")
    print(f"   Peak Population Exposed: {BOLD}{int(row['population_est'] * (peak_risk/100.0) * (p_months['predicted_flood_pct'].max().clip(0.5, 100.0)/100.0)):,}{RESET_COLOR} people")
    print(f"   Peak Farmland Exposed:   {BOLD}{((row['area_km2'] * (row['farmland_pct']/100.0)) * (p_months['predicted_flood_pct'].max()/100.0)):.2f}{RESET_COLOR} km²")
    
    print("-" * 78)
    print(f"{BOLD}12-MONTH PREDICTIVE BREAKDOWN:{RESET_COLOR}")
    print("Month | Primary Driver      | Risk Score | Level     | Flood % | Rainfall(mm) | Runoff(mm)")
    print("------+---------------------+------------+-----------+---------+--------------+-----------")
    for _, mrow in p_months.iterrows():
        is_peak = (mrow['month'] == row['peak_risk_month'])
        star = "★ " if is_peak else "  "
        m_name = mrow['month_name']
        m_driver = mrow['primary_driver']
        m_score = mrow['flood_risk_score']
        m_level = mrow['risk_level']
        m_flood = mrow['predicted_flood_pct']
        m_rain = mrow['predicted_rainfall_mm']
        m_ro = mrow['predicted_runoff_mm']
        print(f"{star}{m_name:3s} | {m_driver:19s} | {m_score:10.1f} | {m_level:9s} | {m_flood:7.2f} | {m_rain:12.1f} | {m_ro:9.1f}")
        
    print("-" * 78)
    print(f"{BOLD}ACTION PROTOCOLS:{RESET_COLOR}")
    print(f"🏛️  {BOLD}Government / UN Response:{RESET_COLOR}\n   {row['government_action']}")
    print(f"👥 {BOLD}Citizen & Community Guidance:{RESET_COLOR}\n   {row['citizen_action']}")
    print("=" * 78 + "\n")

def list_top_payams(state=None, driver=None, month=None, top_n=10):
    df_m, df_s = load_data()
    
    if month is not None:
        sub = df_m[df_m['month'] == month].copy()
        if state:
            sub = sub[sub['adm1_name'].str.lower() == state.lower()]
        if driver:
            sub = sub[sub['primary_driver'].str.lower() == driver.lower()]
            
        sub = sub.sort_values('flood_risk_score', ascending=False).head(top_n)
        
        print("\n" + "=" * 90)
        print(f"{BOLD}TOP {len(sub)} HIGHEST FLOOD RISK PAYAMS FOR {MONTH_NAMES_FULL[month-1].upper()}{RESET_COLOR}")
        if state: print(f"Filtered by State: {state.title()}")
        if driver: print(f"Filtered by Driver: {driver}")
        print("=" * 90)
        print("Rank | Payam        | County       | State            | Dominant Driver     | Risk Score | Flood % | Rain(mm)")
        print("-----+--------------+--------------+------------------+---------------------+------------+---------+---------")
        for i, (_, r) in enumerate(sub.iterrows(), 1):
            print(f"{i:4d} | {r['adm3_name']:12s} | {r['adm2_name']:12s} | {r['adm1_name']:16s} | {r['primary_driver']:19s} | {r['flood_risk_score']:10.1f} | {r['predicted_flood_pct']:7.2f} | {r['predicted_rainfall_mm']:7.1f}")
        print("=" * 90 + "\n")
        
    else:
        sub = df_s.copy()
        if state:
            sub = sub[sub['adm1_name'].str.lower() == state.lower()]
        if driver:
            sub = sub[sub['annual_primary_driver'].str.lower() == driver.lower()]
            
        sub = sub.sort_values('max_flood_risk_score', ascending=False).head(top_n)
        
        print("\n" + "=" * 95)
        print(f"{BOLD}TOP {len(sub)} HIGHEST ANNUAL FLOOD RISK PAYAMS ACROSS SOUTH SUDAN{RESET_COLOR}")
        if state: print(f"Filtered by State: {state.title()}")
        if driver: print(f"Filtered by Driver: {driver}")
        print("=" * 95)
        print("Rank | Payam        | County       | State            | Primary Driver      | Peak Month | Risk Score | Flood % | Area km²")
        print("-----+--------------+--------------+------------------+---------------------+------------+------------+---------+---------")
        for i, (_, r) in enumerate(sub.iterrows(), 1):
            print(f"{i:4d} | {r['adm3_name']:12s} | {r['adm2_name']:12s} | {r['adm1_name']:16s} | {r['annual_primary_driver']:19s} | {r['peak_month_name']:10s} | {r['max_flood_risk_score']:10.1f} | {r['peak_flood_pct']:7.2f} | {r['peak_flood_km2']:7.1f}")
        print("=" * 95 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="South Sudan Payam-Level Flood Driver Predictive Modeling CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python predict_payam.py --payam "Kurwai"
    python predict_payam.py --payam "Juba Town"
    python predict_payam.py --state "Unity" --top 10
    python predict_payam.py --driver "Extreme Rainfall" --top 10
    python predict_payam.py --month 9 --top 15
    python predict_payam.py --simulate --payam "Bentiu" --rain-anomaly 25
        """
    )
    parser.add_argument("--payam", type=str, help="Name of the Payam (ADM3) to inspect")
    parser.add_argument("--state", type=str, help="Filter by State (ADM1)")
    parser.add_argument("--driver", type=str, help="Filter by dominant driver type")
    parser.add_argument("--month", type=int, choices=range(1, 13), help="Calendar month (1-12)")
    parser.add_argument("--top", type=int, default=10, help="Number of top payams to display (default: 10)")
    parser.add_argument("--simulate", action="store_true", help="Enable scenario simulation mode")
    parser.add_argument("--rain-anomaly", type=float, default=0.0, help="Precipitation anomaly percentage for what-if simulation (e.g., 20 for +20%)")
    
    args = parser.parse_args()
    
    if args.payam:
        print_payam_profile(args.payam, month_num=args.month, rain_anomaly=args.rain_anomaly)
    elif args.state or args.driver or args.month:
        list_top_payams(state=args.state, driver=args.driver, month=args.month, top_n=args.top)
    else:
        # Default: Show national summary and top 10 payams
        list_top_payams(top_n=10)
        print("💡 Tip: Specify --payam \"<Name>\" to view full monthly driver breakdown and action guidance.")

if __name__ == "__main__":
    main()
