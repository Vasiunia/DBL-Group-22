"""
Research question 12.1 — Step 8: documented flood events (independent of the satellite flood masks).

Sources (open, no registration, via the Humanitarian Data Exchange):
  - GLIDE disaster numbers, South Sudan 2011-2025 and Sudan (pre-2011 southern events)
    https://data.humdata.org/dataset/ssd-glide-events , https://data.humdata.org/dataset/sdn-glide-events
  - Dartmouth Flood Observatory master list (1985-2019)
    https://data.humdata.org/dataset/global-active-archive-of-large-flood-events-dfo
Files are in raw_data/external/events/.

Each event in what is now South Sudan is classified by the locations named in the source text:
  corridor = White Nile / Sudd corridor, downstream of the equatorial lakes (Jonglei, Bor, Unity, White Nile)
  west     = Bahr el Ghazal / Abyei, rain-fed rivers not fed by the lakes
  mixed    = names places in both
Then: was the upstream lake level higher in years with corridor floods than in years with western floods?

Run from the raw_data folder:  python extreme_flood_early_warning/event_catalog.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

import train_models as T

C = T.C
D = T.HERE / "data"

# year, month, source id, region class, the place names in the source text that decide the class, impact note
EVENTS = [
    (1996, 10, "DFO 2577", "mixed", "Bor (Jonglei, White Nile) and Tonj (Warrap)", "8,000 displaced, 102 dead"),
    (1998, 8, "DFO 2268 / FL-1998-000251-SDN", "corridor", "rivers listed incl. Jonglei and White Nile", "national event, 900,000 affected (all Sudan)"),
    (2008, 8, "DFO 3351 / FL-2008-000125-SDN", "west", "Aweil, Kuom river, Northern Bahr el Ghazal", "39,000 homeless"),
    (2008, 10, "DFO 3390 / FL-2008-000207-SDN", "unknown", "'Southern Sudan' only", "31 dead"),
    (2011, 9, "FL-2011-000142-SSD", "west", "Agok, Abyei", "flash floods, displaced population"),
    (2012, 6, "FL-2012-000128-SSD", "west", "Abyei", "seasonal rains, roads flooded"),
    (2014, 8, "FL-2014-000123-SSD", "corridor", "Bentiu (Unity)", "displacement sites flooded"),
    (2017, 9, "FL-2017-000137-SSD", "west", "Aweil North and Aweil West", "119,000 affected"),
    (2019, 6, "FL-2019-000062-SSD", "west", "Nyamlel, Aweil, Gokmachar (Northern Bahr el Ghazal)", "houses and roads destroyed"),
    (2020, 8, "FL-2020-000193-SSD", "corridor", "areas along the White Nile; Jonglei and Lakes worst", "600,000 affected"),
    (2021, 7, "FL-2021-000108-SSD", "corridor", "rising waters; GLIDE centroid 8.1 N 31.0 E (Sudd)", "90,000 affected (early season)"),
    (2024, 7, "FL-2024-000145-SSD", "mixed", "Aweil West, Jur River (west) and Panyijar (Unity)", "3,180 displaced"),
    (2025, 7, "FL-2025-000158-SSD", "corridor", "Mayendit (Unity) and Bor South (Jonglei)", "80,000 affected"),
]


def main():
    ev = pd.DataFrame(EVENTS, columns=["year", "month", "source", "region", "evidence", "impact"])
    lk = pd.read_parquet(D / "external_lakes_grealm.parquet").resample("MS").median().interpolate(limit=3)
    lk = lk - lk["1993":"2018"].mean()
    w, f, _ = T.build_weekly()
    albert = f["albert_anom"].ffill().resample("MS").mean()
    # lake level in the 6 months up to the event month (what an early-warning system would have seen)
    def before(s, y, m):
        end = pd.Timestamp(y, m, 1)
        v = s[(s.index > end - pd.DateOffset(months=6)) & (s.index <= end)]
        return v.mean() if v.notna().any() else np.nan
    ev["victoria_anom_6m_before"] = [before(lk["victoria"], y, m) for y, m in zip(ev.year, ev.month)]
    ev["kyoga_anom_6m_before"] = [before(lk["kyoga"], y, m) for y, m in zip(ev.year, ev.month)]
    ev["albert_anom_6m_before"] = [before(albert, y, m) for y, m in zip(ev.year, ev.month)]
    ev.to_csv(T.OUT_T / "event_catalog_south_sudan.csv", index=False)
    pd.set_option("display.width", 220)
    print(ev.drop(columns=["evidence", "impact"]).round(2).to_string(index=False))

    cor = ev[ev.region == "corridor"]
    wst = ev[ev.region == "west"]
    rows = []
    for col in ["victoria_anom_6m_before", "kyoga_anom_6m_before"]:
        a, b = cor[col].dropna(), wst[col].dropna()
        p = mannwhitneyu(a, b, alternative="greater").pvalue
        rows.append(dict(lake=col, corridor_n=len(a), corridor_mean=a.mean(), west_n=len(b), west_mean=b.mean(),
                         one_sided_p=p))
    res = pd.DataFrame(rows)
    res.to_csv(T.OUT_T / "event_catalog_lake_test.csv", index=False)
    print("\nLake level before corridor floods vs western floods (Mann-Whitney, one-sided):\n", res.round(3).to_string(index=False))

    # Without the 2020s (the single long regime), does the pattern still hold?
    pre = ev[ev.year < 2020]
    a = pre[pre.region == "corridor"]["victoria_anom_6m_before"].dropna()
    b = pre[pre.region == "west"]["victoria_anom_6m_before"].dropna()
    print(f"\nBefore 2020 only: corridor events n={len(a)} mean Victoria {a.mean():+.2f} m "
          f"({', '.join(f'{v:+.2f}' for v in a)}); west n={len(b)} mean {b.mean():+.2f} m")

    # 2020, 2021 and 2025 fall in one continuous high-lake period: count them as ONE independent event
    cor_ind = list(cor[cor.year < 2020]["victoria_anom_6m_before"]) + [cor[cor.year >= 2020]["victoria_anom_6m_before"].mean()]
    p_ind = mannwhitneyu(cor_ind, b, alternative="greater", method="exact").pvalue
    print(f"Independent events only (2020s counted once): corridor {[round(v, 2) for v in cor_ind]} vs west "
          f"{[round(v, 2) for v in b]} -> one-sided exact p = {p_ind:.3f}")
    pd.DataFrame([dict(test="2020s regime counted as one event", corridor_values=cor_ind, west_values=list(b),
                       one_sided_exact_p=p_ind)]).to_csv(T.OUT_T / "event_catalog_lake_test_independent.csv", index=False)

    # Figure: lake levels with documented events
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.plot(lk.index, lk["victoria"], color=C["blue"], lw=1.6, label="Lake Victoria")
    ax.plot(lk.index, lk["kyoga"], color=C["aqua"], lw=1.4, label="Lake Kyoga")
    ax.axhline(0, color=C["ink2"], lw=0.8)
    style = {"corridor": (C["red"], "o", "Flood event, White Nile / Sudd corridor"),
             "west": (C["ink2"], "s", "Flood event, west (Bahr el Ghazal / Abyei)"),
             "mixed": (C["yellow"], "D", "Flood event, mixed / unclear"),
             "unknown": (C["yellow"], "D", None)}
    done, labelled = set(), set()
    for _, r in ev.iterrows():
        c, mk, lab = style[r.region]
        t = pd.Timestamp(r.year, r.month, 15)
        lab = lab if (lab and lab not in done) else None
        done.add(lab)
        row = -1.55 if r.year not in labelled else -1.25          # second event in the same year: stack above
        ax.plot(t, row, marker=mk, ms=10, color=c, mec="white", mew=1.2, ls="none", label=lab, zorder=5)
        if r.year not in labelled:
            ax.text(t, -1.75, str(r.year), ha="center", va="top", fontsize=8, color=C["ink2"], rotation=90)
            labelled.add(r.year)
    ax.set_ylim(-2.15, 2.9)
    ax.set_ylabel("m vs. 1993–2018 mean")
    ax.set_title("Documented flood events in South Sudan (GLIDE, Dartmouth Flood Observatory) and upstream lake levels",
                 fontsize=11)
    ax.legend(loc="upper left", fontsize=8.5, ncol=2)
    fig.tight_layout()
    fig.savefig(T.OUT_F / "17_documented_flood_events_and_lakes.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
