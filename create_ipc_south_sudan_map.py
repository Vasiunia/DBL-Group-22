"""
Create an interactive IPC Acute Food Insecurity map for South Sudan.

Output:
    ipc_south_sudan.html

Requirements:
    pip install pandas geopandas folium openpyxl
"""

from pathlib import Path
from difflib import get_close_matches
import re

import pandas as pd
import geopandas as gpd
import folium


# ============================================================
# SETTINGS — normally only change these
# ============================================================

# Latest uploaded IPC analysis
IPC_FILE = Path("./IPC/IPC_SS_Acute_Food_Insecurity_September_2025_2026-06-11.xlsx")

# "Current", "Projected 1", or "Projected 2"
PERIOD = "Current"

ADMIN1_FILE = Path("./Administrative boundaries/ssd_admin1.geojson")
ADMIN2_FILE = Path("./Administrative boundaries/ssd_admin2.geojson")
ADMIN3_FILE = Path("./Administrative boundaries/ssd_admin3.geojson")

OUTPUT_FILE = "ipc_south_sudan.html"


# ============================================================
# IPC colour scale
# ============================================================

PHASE_COLORS = {
    1: "#A6D96A",  # green
    2: "#FFFFBF",  # yellow
    3: "#FDAE61",  # orange
    4: "#D7191C",  # red
    5: "#7F0000",  # dark red
}

PHASE_NAMES = {
    1: "Phase 1 - Minimal",
    2: "Phase 2 - Stressed",
    3: "Phase 3 - Crisis",
    4: "Phase 4 - Emergency",
    5: "Phase 5 - Catastrophe/Famine",
}


# ============================================================
# Helpers
# ============================================================

def normalize_name(name):
    """
    Normalize county names so small differences such as:
    'Kajo-keji' vs 'Kajo Keji' or 'Yei county' vs 'Yei'
    are easier to match.
    """
    if pd.isna(name):
        return ""

    name = str(name).lower().strip()
    name = re.sub(r"\bcounty\b", "", name)
    name = name.replace("-", " ")
    name = re.sub(r"[^a-z0-9 ]", "", name)
    name = re.sub(r"\s+", " ", name).strip()

    return name


def parse_phase(value):
    """Convert P1, P2, ... P5 to integer 1...5."""
    if pd.isna(value):
        return None

    match = re.search(r"([1-5])", str(value))
    return int(match.group(1)) if match else None


def format_population(value):
    if pd.isna(value):
        return "No data"
    return f"{int(round(value)):,}"


def format_percent(value):
    if pd.isna(value):
        return "No data"
    return f"{float(value) * 100:.0f}%"


# ============================================================
# Load IPC data
# ============================================================

def load_ipc(ipc_file, period="Current"):
    df = pd.read_excel(ipc_file, sheet_name="Population Table")

    # In these IPC files, county rows begin with two spaces.
    county_mask = df["Area Name"].astype(str).str.startswith("  ")
    counties = df.loc[county_mask].copy()

    # Keep clean county name.
    counties["County"] = counties["Area Name"].str.strip()
    counties["join_name"] = counties["County"].map(normalize_name)

    phase_col = f"{period} - Phase"
    if phase_col not in counties.columns:
        available = [
            c.replace(" - Phase", "")
            for c in counties.columns
            if c.endswith(" - Phase")
        ]
        raise ValueError(
            f"{phase_col!r} does not exist.\n"
            f"Available periods: {available}"
        )

    counties["IPC_Phase"] = counties[phase_col].map(parse_phase)
    counties["IPC_Label"] = counties["IPC_Phase"].map(PHASE_NAMES)

    # Period dates.
    from_col = f"{period} - From Date"
    thru_col = f"{period} - Thru Date"

    counties["From_Date"] = pd.to_datetime(
        counties[from_col], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    counties["Thru_Date"] = pd.to_datetime(
        counties[thru_col], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    # Population information for popup/tooltip.
    est_pop_col = f"{period} - Est Pop"
    if est_pop_col in counties.columns:
        counties["Estimated Population"] = counties[est_pop_col].map(format_population)
    else:
        counties["Estimated Population"] = "No data"

    for phase in range(1, 6):
        pop_col = f"{period} - Phase {phase}"
        pct_col = f"{period} - Phase {phase} %"

        if pop_col in counties.columns:
            pop_text = counties[pop_col].map(format_population)
        else:
            pop_text = pd.Series("No data", index=counties.index)

        if pct_col in counties.columns:
            pct_text = counties[pct_col].map(format_percent)
        else:
            pct_text = pd.Series("No data", index=counties.index)

        counties[f"Phase {phase} population"] = (
            pop_text + " (" + pct_text + ")"
        )

    phase3_col = f"{period} - Phase 3+"
    phase3_pct_col = f"{period} - Phase 3+ %"

    if phase3_col in counties.columns and phase3_pct_col in counties.columns:
        counties["Phase 3+ population"] = (
            counties[phase3_col].map(format_population)
            + " ("
            + counties[phase3_pct_col].map(format_percent)
            + ")"
        )
    else:
        counties["Phase 3+ population"] = "No data"

    return counties


# ============================================================
# Join IPC counties to geographic boundaries
# ============================================================

def join_to_boundaries(ipc_counties, admin2):
    admin2 = admin2.copy()

    if admin2.crs is None:
        admin2 = admin2.set_crs("EPSG:4326")
    else:
        admin2 = admin2.to_crs("EPSG:4326")

    if "adm2_name" not in admin2.columns:
        raise ValueError(
            "Expected county-name column 'adm2_name' in ssd_admin2.geojson.\n"
            f"Columns found: {admin2.columns.tolist()}"
        )

    admin2["join_name"] = admin2["adm2_name"].map(normalize_name)

    # Optional manual fixes can be added here if a name differs between sources.
    # Format: "ipc normalized name": "boundary normalized name"
    MANUAL_ALIASES = {
        # IPC calls this "Wau (Rural Only)", while the boundary file uses "Wau".
        "wau rural only": "wau",
    }

    ipc_counties = ipc_counties.copy()
    ipc_counties["join_name"] = ipc_counties["join_name"].replace(MANUAL_ALIASES)

    merged = admin2.merge(
        ipc_counties.drop(columns=["Area Name"], errors="ignore"),
        on="join_name",
        how="left",
        suffixes=("", "_ipc"),
    )

    # Print counties that did not match, plus useful suggestions.
    boundary_names = set(admin2["join_name"])
    ipc_names = set(ipc_counties["join_name"])

    unmatched_ipc = sorted(ipc_names - boundary_names)
    unmatched_map = sorted(boundary_names - ipc_names)

    if unmatched_ipc:
        print("\nIPC counties not matched to boundary names:")
        for name in unmatched_ipc:
            suggestions = get_close_matches(
                name, list(boundary_names), n=3, cutoff=0.55
            )
            print(f"  {name!r} -> suggestions: {suggestions}")

    if unmatched_map:
        print("\nBoundary counties without an IPC match:")
        for name in unmatched_map:
            suggestions = get_close_matches(
                name, list(ipc_names), n=3, cutoff=0.55
            )
            print(f"  {name!r} -> suggestions: {suggestions}")

    return merged


# ============================================================
# Create map
# ============================================================

def create_map(ipc_file, period="Current", output_file="ipc_south_sudan.html"):
    ipc = load_ipc(ipc_file, period)

    admin2 = gpd.read_file(ADMIN2_FILE)
    admin1 = gpd.read_file(ADMIN1_FILE)

    # In ssd_admin2, Akoka is included within Baliet county.
    # If ssd_admin3 is available, extract Akoka payam as a separate county polygon
    # so all 79 IPC counties can be matched and rendered.
    if ADMIN3_FILE.exists():
        admin3 = gpd.read_file(ADMIN3_FILE)
        akoka_payam = admin3[admin3["adm3_name"] == "Akoka"]
        baliet_matches = admin2[admin2["adm2_name"] == "Baliet"]
        if not akoka_payam.empty and not baliet_matches.empty:
            akoka_geom = akoka_payam.geometry.iloc[0]
            baliet_row = baliet_matches.iloc[0].copy()
            akoka_row = baliet_row.copy()
            akoka_row["adm2_name"] = "Akoka"
            akoka_row["geometry"] = akoka_geom
            admin2.loc[admin2["adm2_name"] == "Baliet", "geometry"] = (
                baliet_row["geometry"].difference(akoka_geom)
            )
            admin2 = pd.concat(
                [admin2, gpd.GeoDataFrame([akoka_row], crs=admin2.crs)],
                ignore_index=True,
            )

    merged = join_to_boundaries(ipc, admin2)

    if admin1.crs is None:
        admin1 = admin1.set_crs("EPSG:4326")
    else:
        admin1 = admin1.to_crs("EPSG:4326")

    # Map centre from South Sudan geometry.
    minx, miny, maxx, maxy = admin1.total_bounds
    centre = [(miny + maxy) / 2, (minx + maxx) / 2]

    m = folium.Map(
        location=centre,
        zoom_start=6,
        tiles=None,
        control_scale=True,
    )

    # Clean Mapbox-like basemap with no API token needed.
    folium.TileLayer(
        "CartoDB positron",
        name="Light basemap",
        control=True,
        show=True,
    ).add_to(m)

    folium.TileLayer(
        "OpenStreetMap",
        name="OpenStreetMap",
        control=True,
        show=False,
    ).add_to(m)

    # Prepare text columns for the tooltip.
    merged["Map State"] = merged["adm1_name"].fillna("Unknown")
    merged["Map County"] = merged["adm2_name"]
    merged["Map Phase"] = merged["IPC_Label"].fillna("No classification / no match")
    merged["Map Period"] = (
        merged["From_Date"].fillna("?")
        + " to "
        + merged["Thru_Date"].fillna("?")
    )

    for col in [
        "Estimated Population",
        "Phase 1 population",
        "Phase 2 population",
        "Phase 3 population",
        "Phase 4 population",
        "Phase 5 population",
        "Phase 3+ population",
    ]:
        if col not in merged:
            merged[col] = "No data"
        merged[col] = merged[col].fillna("No data")

    # Folium serializes every non-geometry column in the GeoDataFrame.
    # The merged IPC table still contains raw pandas Timestamp objects, which
    # are not JSON serializable. Keep only the fields actually needed by the map.
    map_columns = [
        "geometry",
        "adm1_name",
        "adm2_name",
        "IPC_Phase",
        "Map State",
        "Map County",
        "Map Phase",
        "Map Period",
        "Estimated Population",
        "Phase 1 population",
        "Phase 2 population",
        "Phase 3 population",
        "Phase 4 population",
        "Phase 5 population",
        "Phase 3+ population",
    ]

    map_gdf = merged[[c for c in map_columns if c in merged.columns]].copy()

    # Make sure the phase contains plain Python-friendly values.
    map_gdf["IPC_Phase"] = map_gdf["IPC_Phase"].apply(
        lambda x: None if pd.isna(x) else int(x)
    )

    # County polygons coloured by IPC phase.
    def style_function(feature):
        phase = feature["properties"].get("IPC_Phase")

        try:
            phase = int(phase)
        except (TypeError, ValueError):
            phase = None

        return {
            "fillColor": PHASE_COLORS.get(phase, "#BDBDBD"),
            "color": "#555555",
            "weight": 0.8,
            "fillOpacity": 0.78,
        }

    def highlight_function(feature):
        return {
            "weight": 2.5,
            "color": "#111111",
            "fillOpacity": 0.9,
        }

    county_layer = folium.GeoJson(
        map_gdf,
        name=f"IPC {period}",
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "Map State",
                "Map County",
                "Map Phase",
                "Map Period",
                "Estimated Population",
                "Phase 3+ population",
            ],
            aliases=[
                "State:",
                "County:",
                "IPC classification:",
                "Period:",
                "Estimated population:",
                "Population in Phase 3+:",
            ],
            localize=True,
            sticky=False,
            labels=True,
            style="""
                background-color: white;
                color: #222;
                font-family: Arial;
                font-size: 13px;
                padding: 8px;
            """,
        ),
        popup=folium.GeoJsonPopup(
            fields=[
                "Map State",
                "Map County",
                "Map Phase",
                "Map Period",
                "Estimated Population",
                "Phase 1 population",
                "Phase 2 population",
                "Phase 3 population",
                "Phase 4 population",
                "Phase 5 population",
                "Phase 3+ population",
            ],
            aliases=[
                "State:",
                "County:",
                "IPC classification:",
                "Period:",
                "Estimated population:",
                "Phase 1:",
                "Phase 2:",
                "Phase 3:",
                "Phase 4:",
                "Phase 5:",
                "Phase 3+:",
            ],
            localize=True,
            max_width=450,
        ),
    )

    county_layer.add_to(m)

    # State borders on top of county colours.
    # We set interactive=False so that this overlay does NOT intercept mouse hovers
    # and clicks, allowing the user to interact seamlessly with the counties underneath.
    admin1_map = admin1[["adm1_name", "geometry"]].copy()

    folium.GeoJson(
        admin1_map,
        name="State boundaries",
        interactive=False,
        style_function=lambda feature: {
            "fill": False,
            "fillOpacity": 0,
            "color": "#111111",
            "weight": 2.0,
        },
    ).add_to(m)

    # Fit exactly to South Sudan.
    m.fit_bounds([[miny, minx], [maxy, maxx]])

    # Title.
    valid_dates = ipc[["From_Date", "Thru_Date"]].dropna()
    if len(valid_dates):
        from_date = valid_dates["From_Date"].iloc[0]
        thru_date = valid_dates["Thru_Date"].iloc[0]
        date_text = f"{from_date} to {thru_date}"
    else:
        date_text = ""

    title_html = f"""
    <div style="
        position: fixed;
        top: 12px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 9999;
        background: rgba(255,255,255,0.94);
        border: 1px solid #888;
        border-radius: 4px;
        padding: 8px 14px;
        font-family: Arial;
        font-size: 17px;
        font-weight: bold;
        box-shadow: 0 1px 5px rgba(0,0,0,0.25);
    ">
        South Sudan - IPC Acute Food Insecurity ({period})<br>
        <span style="font-size: 12px; font-weight: normal;">{date_text}</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # Fixed legend.
    legend_items = ""
    for phase in range(1, 6):
        legend_items += f"""
        <div style="margin: 4px 0;">
            <span style="
                display:inline-block;
                width:18px;
                height:14px;
                background:{PHASE_COLORS[phase]};
                border:1px solid #666;
                margin-right:7px;
                vertical-align:middle;
            "></span>
            {PHASE_NAMES[phase]}
        </div>
        """

    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 35px;
        left: 20px;
        z-index: 9999;
        background: rgba(255,255,255,0.96);
        border: 1px solid #777;
        border-radius: 4px;
        padding: 10px 12px;
        font-family: Arial;
        font-size: 12px;
        box-shadow: 0 1px 5px rgba(0,0,0,0.25);
    ">
        <div style="font-weight:bold; margin-bottom:7px;">IPC Phase</div>
        {legend_items}
        <div style="margin-top:5px;">
            <span style="
                display:inline-block;
                width:18px;
                height:14px;
                background:#BDBDBD;
                border:1px solid #666;
                margin-right:7px;
                vertical-align:middle;
            "></span>
            No classification / unmatched
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(collapsed=False).add_to(m)

    m.save(output_file)

    print(f"\nMap saved to: {output_file}")
    print(f"IPC counties in spreadsheet: {len(ipc)}")
    print(
        "Mapped counties with phase:",
        int(merged["IPC_Phase"].notna().sum()),
        "/",
        len(merged),
    )

    return m


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    create_map(
        ipc_file=IPC_FILE,
        period=PERIOD,
        output_file=OUTPUT_FILE,
    )
