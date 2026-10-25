"""
Research question 12.1 — Step 6: does the longer, open external record add independent events?

Chain tested: Indian Ocean Dipole (IOD) -> Lake Victoria / Kyoga level -> Sudd flooding.
Inputs: data/external_*.parquet (external_data.py) and the MODIS-based series of the main analysis.
Outputs: outputs/tables/extended_*.csv, outputs/figures/14_iod_lakes_sudd_chain.png

Run from the raw_data folder:  python extreme_flood_early_warning/extended_analysis.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import train_models as T
import robustness as R

C = T.C
D = T.HERE / "data"


def load():
    lk = pd.read_parquet(D / "external_lakes_grealm.parquet").resample("MS").median().interpolate(limit=3)
    lk = lk - lk["1993":"2018"].mean()
    ci = pd.read_parquet(D / "external_climate_indices.parquet")
    son = ci["iod_dmi"][ci.index.month.isin([9, 10, 11])]
    son = son.groupby(son.index.year).mean()
    jrc = pd.read_parquet(D / "external_jrc_surface_water.parquet").pivot(index="year", columns="region",
                                                                         values="water_km2")
    jrc[jrc.sum(axis=1) < 300] = np.nan          # years without Landsat coverage over South Sudan
    w, f, _ = T.build_weekly()
    albert = f["albert_anom"].ffill().resample("MS").mean()
    rf = R.regional_flood()
    modis = (rf.resample("W-SUN").max() * T.PIXEL_KM2).resample("YE").mean()
    modis.index = modis.index.year
    modis = modis.loc[2003:2025, [f"unusual_px_{r}" for r in ["sudd", "west", "east"]]]
    modis.columns = ["sudd", "west", "east"]
    return lk, son, jrc, albert, modis


def iod_to_lakes(lk, son):
    """For each year, IOD (Sep-Nov) vs the change in annual mean lake level from that year to the next."""
    ann = lk.groupby(lk.index.year).mean()
    rows = []
    for y in range(1993, 2025):
        if y in son.index and y + 1 in ann.index and y in ann.index:
            rows.append(dict(year=y, iod_son=son[y], victoria_rise_next_year=ann.loc[y + 1, "victoria"] - ann.loc[y, "victoria"],
                             kyoga_rise_next_year=ann.loc[y + 1, "kyoga"] - ann.loc[y, "kyoga"]))
    t = pd.DataFrame(rows)
    strong = t[t.iod_son >= 0.75]
    rest = t[t.iod_son < 0.75]
    from scipy.stats import spearmanr
    summ = dict(n_years=len(t), n_strong_iod=len(strong),
                strong_iod_years=", ".join(str(y) for y in strong.year),
                mean_victoria_rise_after_strong_iod=strong.victoria_rise_next_year.mean(),
                max_victoria_rise_after_other_years=rest.victoria_rise_next_year.max(),
                spearman_iod_vs_victoria_rise=spearmanr(t.iod_son, t.victoria_rise_next_year).statistic,
                spearman_iod_vs_kyoga_rise=spearmanr(t.iod_son, t.kyoga_rise_next_year).statistic)
    return t, summ


def episodes(lk, jrc, modis, albert):
    """High-lake episodes and what the Sudd did, with both flood records where available."""
    vic = lk["victoria"].groupby(lk.index.year).mean()
    kyo = lk["kyoga"].groupby(lk.index.year).mean()
    alb = albert.groupby(albert.index.year).mean()
    jrc_base = jrc.loc[2002:2018].mean()
    mod_base = modis.loc[2003:2019].mean()
    rows = []
    for name, yrs in [("1998–2001 (after 1997 IOD)", range(1998, 2002)),
                      ("2003–2005 (Albert moderately high)", range(2003, 2006)),
                      ("2020–2021 (after 2019 IOD)", range(2020, 2022)),
                      ("2022–2025 (lakes stay high)", range(2022, 2026))]:
        yrs = list(yrs)
        g = lambda s: s.reindex(yrs).mean()
        rows.append({
            "episode": name,
            "Victoria anomaly (m)": g(vic), "Kyoga anomaly (m)": g(kyo), "Albert anomaly (m)": g(alb),
            "Sudd water, Landsat (× 2002–18)": g(jrc["sudd"]) / jrc_base["sudd"],
            "West water, Landsat (× 2002–18)": g(jrc["west"]) / jrc_base["west"],
            "East water, Landsat (× 2002–18)": g(jrc["east"]) / jrc_base["east"],
            "Sudd unusual flood, MODIS (× 2003–19)": g(modis["sudd"]) / mod_base["sudd"],
            "West unusual flood, MODIS (× 2003–19)": g(modis["west"]) / mod_base["west"],
            "East unusual flood, MODIS (× 2003–19)": g(modis["east"]) / mod_base["east"],
        })
    return pd.DataFrame(rows)


def figure(lk, son, jrc, albert, modis):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10.5), sharex=True,
                             gridspec_kw=dict(height_ratios=[1, 1.4, 1.2, 1.2]))
    x0, x1 = pd.Timestamp("1984-01-01"), pd.Timestamp("2026-12-31")
    ax = axes[0]
    yrs = son.loc[1984:2025]
    ax.bar([pd.Timestamp(f"{y}-10-15") for y in yrs.index], yrs.values, width=250,
           color=[C["red"] if v >= 0.75 else "#c3c2b7" for v in yrs.values])
    for y in yrs[yrs >= 0.75].index:
        ax.text(pd.Timestamp(f"{y}-10-15"), yrs[y] + 0.08, str(y), ha="center", fontsize=9, color=C["ink"])
    ax.axhline(0, color=C["ink2"], lw=0.8)
    ax.set_ylim(-1.1, 1.35)
    ax.set_ylabel("°C")
    ax.set_title("1  Indian Ocean Dipole, Sep–Nov (red = strong positive, ≥ 0.75 °C)")
    ax = axes[1]
    for col, c, lab in [("victoria", C["blue"], "Lake Victoria"), ("kyoga", C["aqua"], "Lake Kyoga")]:
        ax.plot(lk.index, lk[col], color=c, lw=1.6, label=lab)
    ax.plot(albert.index, albert.values, color=C["violet"], lw=1.6, label="Lake Albert (from 2002)")
    ax.axhline(0, color=C["ink2"], lw=0.8)
    ax.set_ylabel("m vs. normal")
    ax.set_title("2  Upstream lakes, water-level anomaly (satellite altimetry)")
    ax.legend(loc="upper left", fontsize=8.5, ncol=3)
    ax = axes[2]
    for col, c, ls, lab in [("sudd", C["ink"], "-", "Sudd"), ("west", C["muted"], "-", "West"),
                            ("east", C["muted"], (0, (4, 2)), "East")]:
        s = jrc[col] / jrc.loc[2002:2018, col].mean()
        ax.plot([pd.Timestamp(f"{y}-07-01") for y in s.index], s.values, color=c, ls=ls, marker="o", ms=3.5,
                lw=2.2 if col == "sudd" else 1.5, label=lab)
    for a, b in [(1988, 1993), (1996, 1998)]:
        ax.axvspan(pd.Timestamp(f"{a}-01-01"), pd.Timestamp(f"{b}-12-31"), color="#ebe9e2", lw=0)
        ax.text(pd.Timestamp(f"{a}-01-01") + (pd.Timestamp(f"{b}-12-31") - pd.Timestamp(f"{a}-01-01")) / 2, 0.12,
                "no Landsat\ncoverage", ha="center", va="bottom", fontsize=8.5, color=C["ink2"])
    ax.axhline(1, color=C["ink2"], lw=0.8)
    ax.set_ylim(0, 2.8)
    ax.set_ylabel("× 2002–18 mean")
    ax.set_title("3a  Open-water area per year, Landsat (JRC Global Surface Water, 1984–2021)")
    ax.legend(loc="upper left", fontsize=8.5, ncol=3)
    ax = axes[3]
    for col, c, ls, lab in [("sudd", C["ink"], "-", "Sudd"), ("west", C["muted"], "-", "West"),
                            ("east", C["muted"], (0, (4, 2)), "East")]:
        s = modis[col] / modis.loc[2003:2019, col].mean()
        ax.plot([pd.Timestamp(f"{y}-07-01") for y in s.index], s.values, color=c, ls=ls, marker="o", ms=3.5,
                lw=2.2 if col == "sudd" else 1.5, label=lab)
    ax.axhline(1, color=C["ink2"], lw=0.8)
    ax.set_yscale("log")
    ax.set_ylabel("× 2003–19 mean (log)")
    ax.set_title("3b  Unusual flood area per year, MODIS/VIIRS flood masks (2003–2025)")
    ax.legend(loc="upper left", fontsize=8.5, ncol=3)
    ax.set_xlim(x0, x1)
    fig.tight_layout()
    fig.savefig(T.OUT_F / "14_iod_lakes_sudd_chain.png", dpi=170)
    plt.close(fig)


def main():
    lk, son, jrc, albert, modis = load()
    t, summ = iod_to_lakes(lk, son)
    t.to_csv(T.OUT_T / "extended_iod_vs_lake_rise.csv", index=False)
    print("IOD -> lake rise next year:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in summ.items()})
    ep = episodes(lk, jrc, modis, albert)
    ep.to_csv(T.OUT_T / "extended_high_lake_episodes.csv", index=False)
    print(ep.set_index("episode").T.round(2).to_string())
    figure(lk, son, jrc, albert, modis)


if __name__ == "__main__":
    main()
