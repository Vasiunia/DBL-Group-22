"""
Download the open external data used by the extended analysis (no registration needed).
Already-downloaded files are skipped. Everything goes to raw_data/external/.

  grealm/   Lake Victoria & Kyoga levels 1992-today, NASA GSFC G-REALM 10-day altimetry (~325 MB bundle;
            only the two lake files are kept)
  noaa/     Indian Ocean Dipole (DMI, HadISST) and Nino 3.4 monthly indices, NOAA PSL
  events/   GLIDE disaster records (South Sudan, Sudan) and the Dartmouth Flood Observatory master list (via HDX)
  fonts/    Source Sans 3 (Adobe, SIL Open Font License), the poster font used by the poster figures

The JRC Global Surface Water maps are not downloaded here: external_data.py reads only the South Sudan window
directly from the JRC server.

Run from the raw_data folder:  python extreme_flood_early_warning/download_external.py
"""
import shutil
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
EXT = BASE / "external"

FILES = {
    "noaa/dmi.had.long.data": "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data",
    "noaa/nina34.anom.data": "https://psl.noaa.gov/data/correlation/nina34.anom.data",
    "events/ssd_glide_events.csv": "https://data.humdata.org/dataset/35c1f93c-64f2-4533-8efa-d28ed9c7753d/resource/"
                                   "74c4e896-3ddb-40c5-81b1-f1ed721fb8cc/download/ssd_glide_events.csv",
    "events/sdn_glide_events.csv": "https://data.humdata.org/dataset/4277194b-d1a7-4ab2-8fa6-ae31699cf507/resource/"
                                   "8e6bfc42-4530-46e0-8fb7-08e3362bdbeb/download/sdn_glide_events.csv",
    "events/dfo_masterlist_20190418.zip": "https://data.humdata.org/dataset/1fd855de-57c6-42b3-83e1-9cf989b0f70d/"
                                          "resource/984cc240-b2b7-4266-9f61-5715a9e10ff5/download/"
                                          "wlf_nhr_fl_dfomasterlist_20190418.zip",
}
GREALM_URL = "https://earth.gsfc.nasa.gov/gwm/zip/lake.10.tar.gz"
GREALM_KEEP = ["lake000314.10d.2.txt", "lake000398.10d.2.txt"]   # Victoria, Kyoga
FONT_URL = "https://github.com/adobe-fonts/source-sans/releases/download/3.052R/OTF-source-sans-3.052R.zip"


def fetch(url, dest, tries=3):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research data download)"})
            with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            tmp.rename(dest)
            return
        except Exception as e:  # network hiccups: retry
            print(f"  retry {i + 1}/{tries} for {url}: {e}")
            time.sleep(5)
    raise RuntimeError(f"could not download {url}")


def main():
    for rel, url in FILES.items():
        dest = EXT / rel
        if dest.exists():
            print("ok      ", rel)
            continue
        print("download", rel)
        fetch(url, dest)
    dfo = EXT / "events" / "dfo"
    if not any(dfo.glob("*.shp")):
        with zipfile.ZipFile(EXT / "events" / "dfo_masterlist_20190418.zip") as z:
            z.extractall(dfo)

    gdir = EXT / "grealm"
    if all((gdir / f).exists() for f in GREALM_KEEP):
        print("ok       grealm lake files")
    else:
        tar = gdir / "lake.10.tar.gz"
        if not tar.exists():
            print("download grealm bundle (~325 MB)")
            fetch(GREALM_URL, tar)
        with tarfile.open(tar) as t:
            for m in t.getmembers():
                if Path(m.name).name in GREALM_KEEP:
                    m.name = Path(m.name).name
                    t.extract(m, gdir)
        tar.unlink()                           # keep only the two lake files

    fdir = EXT / "fonts" / "ss3" / "OTF"
    if any(fdir.glob("SourceSans3-*.otf")):
        print("ok       fonts")
    else:
        z = EXT / "fonts" / "ss3.zip"
        print("download fonts")
        fetch(FONT_URL, z)
        with zipfile.ZipFile(z) as zz:
            zz.extractall(EXT / "fonts" / "ss3")
        z.unlink()
    print("external data ready in", EXT)


if __name__ == "__main__":
    main()
