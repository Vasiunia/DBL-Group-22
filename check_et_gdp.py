import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import calendar


# ============================================================
# SETTINGS
# ============================================================

ET_FOLDER = Path("./evapotranspiration")
GDP_FILE = Path("./GDP/API_SSD_DS2_en_csv_v2_2529.csv")

ET_VARIABLE = "ReferenceET_PenmanMonteith_FAO56"

YEARS_ET = range(2000, 2026)


# ============================================================
# 1. EVAPOTRANSPIRATION CHECK
# ============================================================

def check_evapotranspiration():

    print("\n==============================")
    print("EVAPOTRANSPIRATION")
    print("==============================")

    yearly_results = []

    all_min = []
    all_max = []
    all_nan = []
    total_negative = 0

    first_date = None
    last_date = None

    lat_min = lat_max = None
    lon_min = lon_max = None
    lat_resolution = lon_resolution = None
    units = None

    missing_files_total = 0

    for year in YEARS_ET:

        folder = ET_FOLDER / f"ET_{year}"

        files = sorted(folder.glob("*.nc"))

        expected = 366 if calendar.isleap(year) else 365
        missing = expected - len(files)

        if missing > 0:
            missing_files_total += missing

        print(
            f"{year}: {len(files)}/{expected} files "
            f"{'✓' if len(files) == expected else 'MISSING'}"
        )

        dates = []

        # --------------------------------------
        # Check every file
        # --------------------------------------

        for file in files:

            try:
                with xr.open_dataset(file) as ds:

                    if ET_VARIABLE not in ds:
                        print(f"WARNING: variable missing in {file}")
                        continue

                    da = ds[ET_VARIABLE].squeeze()

                    values = np.asarray(da.values, dtype=float)

                    # Quality
                    finite = values[np.isfinite(values)]

                    if len(finite) > 0:
                        all_min.append(finite.min())
                        all_max.append(finite.max())

                        total_negative += np.sum(finite < 0)

                    all_nan.append(np.isnan(values).mean())

                    # Date
                    if "time" in ds:
                        date = pd.Timestamp(ds["time"].values[0])
                        dates.append(date)

                        if first_date is None or date < first_date:
                            first_date = date

                        if last_date is None or date > last_date:
                            last_date = date

                    # --------------------------------------
                    # Metadata only needs checking once
                    # --------------------------------------

                    if lat_min is None:

                        lat_name = (
                            "lat"
                            if "lat" in ds.coords
                            else "latitude"
                        )

                        lon_name = (
                            "lon"
                            if "lon" in ds.coords
                            else "longitude"
                        )

                        lats = np.asarray(ds[lat_name])
                        lons = np.asarray(ds[lon_name])

                        lat_min = float(lats.min())
                        lat_max = float(lats.max())

                        lon_min = float(lons.min())
                        lon_max = float(lons.max())

                        if len(lats) > 1:
                            lat_resolution = float(
                                np.median(np.abs(np.diff(lats)))
                            )

                        if len(lons) > 1:
                            lon_resolution = float(
                                np.median(np.abs(np.diff(lons)))
                            )

                        units = da.attrs.get(
                            "units",
                            "not specified"
                        )

            except Exception as e:
                print(f"ERROR reading {file}: {e}")

        yearly_results.append({
            "year": year,
            "expected_files": expected,
            "actual_files": len(files),
            "missing_files": max(0, missing)
        })

    # --------------------------------------
    # Final ET result
    # --------------------------------------

    et_summary = {
        "Dataset": "Evapotranspiration",
        "Source": "AgERA5",
        "Temporal resolution": "Daily",
        "Start date": first_date,
        "End date": last_date,
        "Spatial resolution": (
            f"{lat_resolution:.3f}° x "
            f"{lon_resolution:.3f}°"
            if lat_resolution is not None
            else "Unknown"
        ),
        "Coordinate system": "Latitude / Longitude",
        "Latitude coverage": f"{lat_min} to {lat_max}",
        "Longitude coverage": f"{lon_min} to {lon_max}",
        "Units": units,
        "Minimum": min(all_min) if all_min else np.nan,
        "Maximum": max(all_max) if all_max else np.nan,
        "Average missing %": (
            np.mean(all_nan) * 100
            if all_nan else np.nan
        ),
        "Negative values": total_negative,
        "Missing files": missing_files_total,
        "1/3/7/15 day analysis": "YES",
        "Flood framework": "Driver / flood hazard"
    }

    yearly_df = pd.DataFrame(yearly_results)

    print("\nET SUMMARY")
    for key, value in et_summary.items():
        print(f"{key}: {value}")

    return et_summary, yearly_df


# ============================================================
# 2. GDP CHECK
# ============================================================

def check_gdp():

    print("\n==============================")
    print("GDP")
    print("==============================")

    if not GDP_FILE.exists():
        raise FileNotFoundError(
            f"GDP file not found: {GDP_FILE}"
        )

    df = pd.read_csv(
        GDP_FILE,
        skiprows=4
    )

    # Use indicator code instead of only the name
    gdp = df[
        (df["Country Code"] == "SSD") &
        (df["Indicator Code"] == "NY.GDP.MKTP.CD")
    ]

    if len(gdp) == 0:
        raise ValueError("South Sudan GDP row not found.")

    # Get columns that represent years
    year_columns = [
        col for col in gdp.columns
        if str(col).isdigit()
    ]

    values = (
        gdp.iloc[0][year_columns]
        .apply(pd.to_numeric, errors="coerce")
    )

    values.index = values.index.astype(int)

    available = values.dropna()

    # --------------------------------------
    # Detect unusually large changes
    # --------------------------------------

    yearly_change = available.pct_change() * 100

    # This only FLAGS values for review.
    # It does NOT remove anything.
    large_changes = yearly_change[
        yearly_change.abs() > 30
    ]

    gdp_summary = {
        "Dataset": "GDP",
        "Source": "World Bank",
        "Temporal resolution": "Annual",
        "Start date": int(available.index.min()),
        "End date": int(available.index.max()),
        "Spatial resolution": "Whole country",
        "Coordinate system": "Not applicable",
        "Latitude coverage": "Not applicable",
        "Longitude coverage": "Not applicable",
        "Units": "Current US$",
        "Minimum": available.min(),
        "Maximum": available.max(),
        "Average missing %": (
            values.isna().mean() * 100
        ),
        "Negative values": int((available < 0).sum()),
        "Missing files": 0,
        "1/3/7/15 day analysis": "NO",
        "Flood framework": "Exposure / vulnerability"
    }

    gdp_values = pd.DataFrame({
        "GDP_current_USD": available,
        "year_change_percent": yearly_change
    })

    print("\nGDP VALUES")
    print(gdp_values)

    print("\nLarge year-to-year changes (>30%):")
    print(large_changes)

    return gdp_summary, gdp_values


# ============================================================
# 3. COMPARISON TABLE
# ============================================================

def make_comparison_table(et, gdp):

    comparison = pd.DataFrame({

        "Property": [
            "Source",
            "Temporal resolution",
            "Coverage",
            "Spatial resolution",
            "Coordinate system",
            "Units",
            "Missing data",
            "Negative / invalid values",
            "File completeness",
            "Can use 1/3/7/15 day windows?",
            "Role in flood analysis"
        ],

        "Evapotranspiration": [
            et["Source"],
            et["Temporal resolution"],
            f'{et["Start date"]} → {et["End date"]}',
            et["Spatial resolution"],
            et["Coordinate system"],
            et["Units"],
            f'{et["Average missing %"]:.2f}%',
            et["Negative values"],
            f'{et["Missing files"]} missing files',
            "Yes",
            "Driver / flood hazard"
        ],

        "GDP": [
            gdp["Source"],
            gdp["Temporal resolution"],
            f'{gdp["Start date"]} → {gdp["End date"]}',
            gdp["Spatial resolution"],
            gdp["Coordinate system"],
            gdp["Units"],
            f'{gdp["Average missing %"]:.2f}%',
            gdp["Negative values"],
            "GDP CSV checked",
            "No",
            "Exposure / vulnerability"
        ]
    })

    print("\n\n==========================================")
    print("FINAL DATASET COMPARISON")
    print("==========================================")

    print(comparison.to_string(index=False))

    return comparison


# ============================================================
# RUN EVERYTHING
# ============================================================

if __name__ == "__main__":

    et_summary, et_years = check_evapotranspiration()

    gdp_summary, gdp_values = check_gdp()

    comparison = make_comparison_table(
        et_summary,
        gdp_summary
    )

    # --------------------------------------
    # Save reports
    # --------------------------------------

    output = Path("./QA_results")
    output.mkdir(exist_ok=True)

    et_years.to_csv(
        output / "ET_file_completeness.csv",
        index=False
    )

    gdp_values.to_csv(
        output / "GDP_values.csv"
    )

    comparison.to_csv(
        output / "ET_GDP_comparison.csv",
        index=False
    )

    print("\nReports saved in:")
    print(output)