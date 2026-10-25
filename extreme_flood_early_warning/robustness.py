"""
Research question 12.1 — Step 4: robustness, uncertainty and causal checks.

Answers four weaknesses of the main analysis:
  A. Uncertainty      — 95 % confidence intervals from a block bootstrap that resamples whole years
                        (driver AUCs, model skill, differences between models, onset composites).
  B. n = 1 problem    — phase-randomised surrogates of Lake Albert (same autocorrelation, random timing),
                        a time-trend placebo, and the within-regime (2020-25) association.
  C. Causality        — a stated causal diagram (DAG) and three tests it implies:
                        1. spatial negative control: Bahr el Ghazal (west) is not downstream of Lake Albert
                        2. the competing path "regional rain -> local rain -> floods" is closed
                        3. mediation: lake-basin rain is associated with floods only *through* Lake Albert
  D. Target choice    — repeat the key results with targets that do not use the recurring/unusual label.

Run from the raw_data folder:  python extreme_flood_early_warning/robustness.py   (~5 min; needs outputs of
train_models.py)
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from shapely import contains_xy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import train_models as T

BASE = T.HERE.parent
B = 1000
RNG = np.random.default_rng(42)
C = T.C

# Flood regions within South Sudan, as (lat_min, lat_max, lon_min, lon_max)
FLOOD_REGIONS = {
    "west": (4.0, 10.5, 24.0, 29.5),   # Bahr el Ghazal: rain-fed, NOT downstream of Lake Albert (negative control)
    "sudd": (6.0, 10.0, 29.5, 32.5),   # Bahr el Jebel / White Nile corridor: directly downstream of Lake Albert
    "east": (5.5, 10.5, 32.5, 36.0),   # Sobat / Pibor / eastern Upper Nile: fed mainly by the Ethiopian highlands
}
REGION_LABEL = {"ssd": "South Sudan (main target)", "sudd": "Sudd / White Nile corridor",
                "east": "East: Sobat–Pibor", "west": "West: Bahr el Ghazal (negative control)"}


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def regional_flood():
    p = T.HERE / "data" / "_flood_regions.parquet"
    if p.exists():
        return pd.read_parquet(p)
    ssd = gpd.read_file(BASE / "Administrative boundaries" / "ssd_admin0.geojson").geometry.union_all()
    out = []
    for typ in ["recurring", "unusual"]:
        for year in range(2002, 2026):
            d = pd.concat([pd.read_parquet(BASE / "flood_masks" / f"compact_{typ}" / f"flood_events_{t}_{year}.parquet",
                                           columns=["date", "lat", "lon"]) for t in ["h20v08", "h21v08"]])
            d = d[contains_xy(ssd, d["lon"].values, d["lat"].values)]
            g = {}
            for r, (a, b, c, e) in FLOOD_REGIONS.items():
                sel = d["lat"].between(a, b) & d["lon"].between(c, e)
                g[f"{typ}_px_{r}"] = d[sel].groupby("date").size()
            out.append(pd.DataFrame(g))
        print(f"  regional flood extraction: {typ} done")
    df = pd.concat(out).groupby(level=0).sum()
    df.index = pd.to_datetime(df.index)
    df.to_parquet(p)
    return df


def weekly_targets(w):
    """Main target plus alternatives that do not depend on the recurring/unusual label or on region."""
    rf = regional_flood().resample("W-SUN").max().reindex(w.index).fillna(0) * T.PIXEL_KM2
    ref = (w.index.year >= T.REF[0]) & (w.index.year <= T.REF[1])
    t = pd.DataFrame(index=w.index)
    t["main"] = w["extreme"]
    total = w["unusual_km2_ssd"] + w["recurring_km2_ssd"]
    t["total_extent"] = (total > total[ref].quantile(0.9)).astype(int)
    month_thr = total[ref].groupby(total.index[ref].month).quantile(0.9)
    t["total_extent_same_month"] = (total > total.index.month.map(month_thr).values).astype(int)
    for r in FLOOD_REGIONS:
        u = rf[f"unusual_px_{r}"]
        t[f"region_{r}"] = (u > u[ref].quantile(0.9)).astype(int)
    return t, rf


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def fast_auc(y, s):
    y = np.asarray(y, bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def year_blocks(index):
    years = np.asarray(index.year)
    return {y: np.where(years == y)[0] for y in np.unique(years)}


def boot_ci(stat, index, b=B):
    """Point estimate and 95 % CI of stat(idx) under a bootstrap that resamples whole calendar years."""
    blocks = year_blocks(index)
    keys = list(blocks)
    est = stat(np.arange(len(index)))
    vals = []
    for _ in range(b):
        idx = np.concatenate([blocks[k] for k in RNG.choice(keys, len(keys))])
        vals.append(stat(idx))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return est, lo, hi


def deseason(s):
    return T.woy_anomaly(s, np.ones(len(s), bool))


# --------------------------------------------------------------------------------------
# A. Uncertainty of driver AUCs and model skill
# --------------------------------------------------------------------------------------
KEY_DRIVERS = {"Lake Albert level anomaly": "albert_anom", "Equatorial-lakes runoff, 52-wk": "ro_lakes_a52",
               "Sobat/Ethiopia rain, 13-wk": "tp_sobat_a13", "South Sudan rain, 26-wk": "tp_ssd_a26",
               "South Sudan rain, 4-wk": "tp_ssd_a4"}


def driver_auc_ci(f, target, period, horizons=(0, 13, 26)):
    a, b = period
    rows = []
    for name, col in KEY_DRIVERS.items():
        for h in horizons:
            y = target.shift(-h)
            m = (f.index.year >= a) & (f.index.year <= b) & y.notna() & f[col].notna()
            yy, xx, idx = y[m].values, f[col][m].values, f.index[m]
            est, lo, hi = boot_ci(lambda i: fast_auc(yy[i], xx[i]), idx)
            rows.append(dict(driver=name, horizon=h, period=f"{a}–{b}", auc=est, ci_low=lo, ci_high=hi))
    return pd.DataFrame(rows)


def model_skill_ci():
    oof = pd.read_parquet(T.OUT_T / "oof_predictions_leave_one_year_out.parquet")
    models = ["Persistence (logistic, flood state + season)", "Logistic regression – all features",
              "Gradient boosting – all features", "GRU – all features", T.ENSEMBLE,
              "GRU – without upstream", "GRU – upstream + season only"]
    contrasts = {"Ensemble − Persistence": (T.ENSEMBLE, "Persistence (logistic, flood state + season)"),
                 "GRU all − GRU without upstream": ("GRU – all features", "GRU – without upstream"),
                 "GRU all − Persistence": ("GRU – all features", "Persistence (logistic, flood state + season)")}
    rows = []
    for h in T.HORIZONS:
        P = {m: oof.loc[m, f"p_{h}wk"] for m in models + ["Climatology"]}
        y = oof.loc["Climatology", f"y_{h}wk"]
        ok = y.notna().values
        y = y.values[ok].astype(int)
        P = {m: v.values[ok] for m, v in P.items()}
        idx = oof.loc["Climatology"].index[ok]

        def bss(m, i):
            return 1 - np.mean((P[m][i] - y[i]) ** 2) / np.mean((P["Climatology"][i] - y[i]) ** 2)

        for m in models:
            for metric, fn in [("BSS", lambda i, m=m: bss(m, i)), ("AUC", lambda i, m=m: fast_auc(y[i], P[m][i]))]:
                est, lo, hi = boot_ci(fn, idx)
                rows.append(dict(item=m, kind="model", metric=metric, horizon=h, est=est, ci_low=lo, ci_high=hi))
        for cname, (m1, m2) in contrasts.items():
            est, lo, hi = boot_ci(lambda i: bss(m1, i) - bss(m2, i), idx)
            rows.append(dict(item=cname, kind="difference", metric="BSS", horizon=h, est=est, ci_low=lo, ci_high=hi))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# B. The n = 1 problem
# --------------------------------------------------------------------------------------
def phase_surrogates(x, n):
    """Surrogates with the same power spectrum (autocorrelation) as x but random phases, i.e. random timing."""
    x = np.asarray(x, float)
    mu = x.mean()
    X = np.fft.rfft(x - mu)
    out = np.empty((n, len(x)))
    for k in range(n):
        ph = RNG.uniform(0, 2 * np.pi, len(X))
        ph[0] = 0
        if len(x) % 2 == 0:
            ph[-1] = 0
        out[k] = np.fft.irfft(np.abs(X) * np.exp(1j * ph), n=len(x)) + mu
    return out


def n1_tests(w, f, target):
    m = (f.index >= T.START) & (f.index <= T.END)
    alb = f["albert_anom"].ffill().bfill()[m]
    rows = []
    sur = phase_surrogates(alb.values, 2000)
    for h in [0, 13, 26]:
        y = target.shift(-h)[m]
        ok = y.notna().values
        obs = fast_auc(y.values[ok], alb.values[ok])
        s_auc = np.array([fast_auc(y.values[ok], s[ok]) for s in sur])
        trend = fast_auc(y.values[ok], np.arange(ok.sum()))
        rows.append(dict(horizon=h, albert_auc=obs, surrogate_p=(np.sum(s_auc >= obs) + 1) / (len(s_auc) + 1),
                         surrogate_auc_95pct=np.percentile(s_auc, 95), time_trend_placebo_auc=trend))
        if h == 0:
            sur_h0 = s_auc
    n1 = pd.DataFrame(rows)

    # Within-regime association (2020-25) and pre-shift (2003-19), de-seasonalised log extent
    ext = deseason(np.log1p(w["unusual_km2_ssd"] + w["recurring_km2_ssd"]))
    cum = f["albert_anom"].ffill().rolling(104, min_periods=52).mean()   # excess level over the past ~2 years
    within = []
    for label, (a, b) in {"2003–2019": (2003, 2019), "2020–2025": (2020, 2025)}.items():
        for xname, x in [("Lake Albert anomaly (now)", f["albert_anom"]),
                         ("Lake Albert anomaly, 2-yr mean", cum)]:
            mm = (w.index.year >= a) & (w.index.year <= b) & x.notna() & ext.notna()
            xx, ee, idx = x[mm].values, ext[mm].values, w.index[mm]
            est, lo, hi = boot_ci(lambda i: spearmanr(xx[i], ee[i]).statistic, idx)
            within.append(dict(period=label, driver=xname, spearman_r=est, ci_low=lo, ci_high=hi))
    return n1, pd.DataFrame(within), sur_h0


# --------------------------------------------------------------------------------------
# C. Causal checks
# --------------------------------------------------------------------------------------
def spatial_control(f, targets):
    rows = []
    for r in ["sudd", "east", "west"]:
        y = targets[f"region_{r}"]
        for name, col in [("Lake Albert level anomaly", "albert_anom"), ("Sobat/Ethiopia rain, 13-wk", "tp_sobat_a13"),
                          ("South Sudan rain, 26-wk", "tp_ssd_a26")]:
            for h in [0, 26]:
                yy = y.shift(-h)
                m = (f.index >= T.START) & (f.index <= T.END) & yy.notna() & f[col].notna()
                Y, X, idx = yy[m].values, f[col][m].values, f.index[m]
                est, lo, hi = boot_ci(lambda i: fast_auc(Y[i], X[i]), idx)
                rows.append(dict(region=REGION_LABEL[r], driver=name, horizon=h, auc=est, ci_low=lo, ci_high=hi))
    return pd.DataFrame(rows)


def spatial_magnitude(rf):
    """Growth of mean weekly flood extent, 2021-25 vs 2003-19, per region (with year-block bootstrap CI)."""
    rf = rf[(rf.index >= T.START) & (rf.index <= T.END)]
    late = rf.index.year >= 2021
    rows = []
    for r in ["sudd", "east", "west"]:
        for kind in ["unusual", "total"]:
            s = rf[f"unusual_px_{r}"] + (rf[f"recurring_px_{r}"] if kind == "total" else 0)
            v, lt = s.values, late

            def ratio(i):
                a, b = v[i][~lt[i]], v[i][lt[i]]
                return b.mean() / a.mean() if len(a) and len(b) else np.nan

            est, lo, hi = boot_ci(ratio, rf.index)
            rows.append(dict(region=REGION_LABEL[r], extent=kind, ratio_2021_25_vs_2003_19=est, ci_low=lo, ci_high=hi))
    return pd.DataFrame(rows)


def local_rain_path(w, f, target):
    """If the 2020s extremes came from local rain (the competing path), local rain should be above normal then."""
    rows = []
    for label, (a, b) in {"2003–2019": (2003, 2019), "2020–2025": (2020, 2025)}.items():
        for name, col in [("South Sudan rain, 26-wk", "tp_ssd_a26"), ("Sudd rain, 13-wk", "tp_sudd_a13"),
                          ("South Sudan P–ET, 26-wk", "wb_ssd_a26")]:
            m = (w.index.year >= a) & (w.index.year <= b) & (target == 1) & f[col].notna()
            x, idx = f[col][m].values, w.index[m]
            est, lo, hi = boot_ci(lambda i: np.mean(x[i]), idx)
            rows.append(dict(period=label, driver=name, mean_anomaly_in_extreme_weeks=est, ci_low=lo, ci_high=hi))
    return pd.DataFrame(rows)


def mediation(w, f):
    """Partial Spearman correlations on monthly de-seasonalised series (DAG: lake rain -> Albert -> floods)."""
    ext = deseason(np.log1p(w["unusual_km2_sudd"] + w["recurring_km2_sudd"]))
    df = pd.DataFrame({"flood": ext, "albert": f["albert_anom"], "rain": f["ro_lakes_a52"]})
    df = df[(df.index >= T.START) & (df.index <= T.END)].resample("MS").mean().dropna()
    R = df.rank()

    def pcorr(x, y, z, i):
        d = R.iloc[i]
        rx = d[x] - np.polyval(np.polyfit(d[z], d[x], 1), d[z])
        ry = d[y] - np.polyval(np.polyfit(d[z], d[y], 1), d[z])
        return np.corrcoef(rx, ry)[0, 1]

    rows = []
    for label, fn in [("Lake-basin runoff ↔ Sudd floods (total)", lambda i: R.iloc[i][["rain", "flood"]].corr().iloc[0, 1]),
                      ("Lake-basin runoff ↔ Sudd floods | Lake Albert", lambda i: pcorr("rain", "flood", "albert", i)),
                      ("Lake Albert ↔ Sudd floods (total)", lambda i: R.iloc[i][["albert", "flood"]].corr().iloc[0, 1]),
                      ("Lake Albert ↔ Sudd floods | lake-basin runoff", lambda i: pcorr("albert", "flood", "rain", i))]:
        est, lo, hi = boot_ci(fn, df.index)
        rows.append(dict(relation=label, partial_spearman=est, ci_low=lo, ci_high=hi))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# D. Target sensitivity (cheap models only: logistic regressions, leave-one-year-out)
# --------------------------------------------------------------------------------------
def target_sensitivity(w, f, targets):
    rows, shares = [], {}
    dates = w.index
    m = (dates >= T.START) & (dates <= T.END)
    for tname in ["main", "total_extent", "total_extent_same_month", "region_sudd", "region_west"]:
        t = targets[tname]
        shares[tname] = t[m].groupby(dates[m].year).mean()
        for h in [0, 26]:
            yy = t.shift(-h)
            mm = m & yy.notna()
            Y, X, idx = yy[mm].values, f["albert_anom"][mm].values, dates[mm]
            ok = ~np.isnan(X)
            est, lo, hi = boot_ci(lambda i: fast_auc(Y[ok][i], X[ok][i]), idx[ok])
            rows.append(dict(target=tname, metric=f"Albert AUC, lead {h} wk", est=est, ci_low=lo, ci_high=hi))
        # LOYO logistic regression (all features) vs persistence, 13-week lead
        y13 = t.shift(-13).values
        valid = m & ~np.isnan(y13)
        for mname, cols, Cc in [("Persistence", T.cols_for(["Current flood state", "Seasonality"]), 1.0),
                                ("Logistic – all features", T.cols_for(T.FEATURE_GROUPS), 0.1)]:
            Xa = f[cols].fillna(0).values
            oof = np.full(len(w), np.nan)
            for _, tr, te in T.make_folds(dates, valid):
                sc = StandardScaler().fit(Xa[tr])
                if len(np.unique(y13[tr])) < 2:
                    continue
                mdl = LogisticRegression(C=Cc, max_iter=5000).fit(sc.transform(Xa[tr]), y13[tr])
                oof[te] = mdl.predict_proba(sc.transform(Xa[te]))[:, 1]
            ok = valid & ~np.isnan(oof)
            rows.append(dict(target=tname, metric=f"{mname} AUC, lead 13 wk (LOYO)",
                             est=roc_auc_score(y13[ok], oof[ok]), ci_low=np.nan, ci_high=np.nan))
    return pd.DataFrame(rows), pd.DataFrame(shares)


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def fig_skill_ci(sk):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    show = [("Persistence (logistic, flood state + season)", C["ink2"]), ("Logistic regression – all features", C["aqua"]),
            ("GRU – all features", C["blue"]), (T.ENSEMBLE, C["red"])]
    ax = axes[0]
    for j, (m, c) in enumerate(show):
        s = sk[(sk.item == m) & (sk.metric == "BSS")].sort_values("horizon")
        x = np.arange(len(T.HORIZONS)) + (j - 1.5) * 0.17
        ax.errorbar(x, s["est"], yerr=[s["est"] - s["ci_low"], s["ci_high"] - s["est"]], fmt="o", ms=5,
                    color=c, ecolor=c, elinewidth=1.6, capsize=0, label=m)
    ax.set_xticks(range(len(T.HORIZONS)), [f"{h} wk" for h in T.HORIZONS])
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.set_ylabel("Brier skill score vs. climatology")
    ax.set_title("Model skill with 95 % CI (year-block bootstrap)")
    ax.legend(fontsize=8, loc="lower left")
    ax = axes[1]
    show = [("Ensemble − Persistence", C["red"]), ("GRU all − GRU without upstream", C["violet"])]
    for j, (m, c) in enumerate(show):
        s = sk[(sk.item == m)].sort_values("horizon")
        x = np.arange(len(T.HORIZONS)) + (j - 0.5) * 0.2
        ax.errorbar(x, s["est"], yerr=[s["est"] - s["ci_low"], s["ci_high"] - s["est"]], fmt="o", ms=5,
                    color=c, ecolor=c, elinewidth=1.6, capsize=0, label=m)
    ax.axhline(0, color=C["ink2"], lw=1, ls="--")
    ax.set_xticks(range(len(T.HORIZONS)), [f"{h} wk" for h in T.HORIZONS])
    ax.set_ylabel("Difference in Brier skill score")
    ax.set_title("Is the difference real? (CI crossing 0 = not shown)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(T.OUT_F / "10_model_skill_uncertainty.png", dpi=170)
    plt.close(fig)


def fig_surrogate(n1, sur_h0):
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.hist(sur_h0, bins=40, color=C["muted"], alpha=0.6, edgecolor=C["surface"])
    obs = n1.loc[n1.horizon == 0, "albert_auc"].iloc[0]
    trend = n1.loc[n1.horizon == 0, "time_trend_placebo_auc"].iloc[0]
    p = n1.loc[n1.horizon == 0, "surrogate_p"].iloc[0]
    ax.axvline(obs, color=C["violet"], lw=2)
    ax.text(obs, ax.get_ylim()[1] * 0.92, f"  Lake Albert: AUC {obs:.2f}\n  p = {p:.3f}", color=C["ink"], fontsize=9, va="top")
    ax.axvline(trend, color=C["orange"], lw=2, ls="--")
    ax.text(trend, ax.get_ylim()[1] * 0.6, f"time-trend placebo {trend:.2f}  ", color=C["ink"], fontsize=9, ha="right")
    ax.set_xlabel("AUC for extreme weeks (lead 0)")
    ax.set_ylabel("Number of surrogate series")
    ax.set_title("Could a random series with Lake Albert's persistence line up with the floods by chance?")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(T.OUT_F / "11_surrogate_test_lake_albert.png", dpi=170)
    plt.close(fig)


def fig_spatial(targets, f, sp, mag):
    order = [REGION_LABEL["sudd"], REGION_LABEL["west"], REGION_LABEL["east"]]
    cols = {REGION_LABEL["sudd"]: C["blue"], REGION_LABEL["east"]: C["yellow"], REGION_LABEL["west"]: C["orange"]}
    ylab = ["Sudd / White Nile\n(downstream of Lake Albert)", "West: Bahr el Ghazal\n(not downstream)",
            "East: Sobat–Pibor\n(not downstream)"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=True)
    ax = axes[0]
    s = mag[mag.extent == "unusual"].set_index("region").loc[order]
    for j, (reg, r) in enumerate(s.iterrows()):
        ax.barh(j, r["ratio_2021_25_vs_2003_19"], color=cols[reg], height=0.55)
        ax.errorbar(r["ratio_2021_25_vs_2003_19"], j, xerr=[[r["ratio_2021_25_vs_2003_19"] - r["ci_low"]],
                    [r["ci_high"] - r["ratio_2021_25_vs_2003_19"]]], fmt="none", ecolor=C["ink2"], elinewidth=1.4)
        ax.text(r["ci_high"] * 1.15, j, f"{r['ratio_2021_25_vs_2003_19']:.1f}×", va="center", color=C["ink"], fontsize=9.5)
    ax.set_xscale("log")
    ax.axvline(1, color=C["ink2"], lw=1, ls="--")
    ax.set_yticks(range(3), ylab)
    ax.invert_yaxis()
    ax.set_xlabel("Unusual flood extent, 2021–25 ÷ 2003–19 (log, 95 % CI)")
    ax.set_title("Where did flooding grow?")
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    s = sp[(sp.driver == "Lake Albert level anomaly") & (sp.horizon == 26)].set_index("region").loc[order]
    for j, (reg, r) in enumerate(s.iterrows()):
        ax.errorbar(r["auc"], j, xerr=[[r["auc"] - r["ci_low"]], [r["ci_high"] - r["auc"]]], fmt="o", ms=7,
                    color=cols[reg], elinewidth=1.8)
    ax.axvline(0.5, color=C["ink2"], lw=1, ls="--")
    ax.set_xlabel("AUC of Lake Albert anomaly for regional extremes, 26 weeks ahead (95 % CI)")
    ax.set_title("Does Lake Albert warn for each region?")
    ax.grid(axis="y", visible=False)
    fig.suptitle("Spatial control: the signal should appear only downstream of Lake Albert",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(T.OUT_F / "12_spatial_negative_control.png", dpi=170)
    plt.close(fig)


def fig_onsets_ci(w, f):
    ext = w["extreme"].values
    onsets = [i for i in range(26, len(ext) - 5) if ext[i] == 1 and ext[i - 8:i].sum() == 0
              and T.START <= str(w.index[i].date()) <= T.END]
    show = [("Lake Albert level anomaly (m)", "albert_anom", C["violet"]),
            ("Equatorial-lakes rain, 52-wk (σ)", "tp_lakes_a52", C["blue"]),
            ("Sobat/Ethiopia rain, 13-wk (σ)", "tp_sobat_a13", C["yellow"]),
            ("South Sudan rain, 4-wk (σ)", "tp_ssd_a4", C["orange"])]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharex=True)
    lags = np.arange(-26, 5)
    for ax, (name, col, c) in zip(axes, show):
        M = np.array([f[col].values[i - 26:i + 5] for i in onsets])
        mean = np.nanmean(M, 0)
        boots = np.array([np.nanmean(M[RNG.integers(0, len(M), len(M))], 0) for _ in range(B)])
        lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
        ax.fill_between(lags, lo, hi, color=c, alpha=0.2, lw=0)
        ax.plot(lags, mean, color=c)
        ax.axvline(0, color=C["ink2"], lw=1, ls="--")
        ax.axhline(0, color="#c3c2b7", lw=1)
        ax.set_title(name, fontsize=9.5)
        ax.set_xlabel("Weeks to onset")
    fig.suptitle(f"Driver anomalies before extreme-flood onsets, mean and 95 % CI (n = {len(onsets)} onsets)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(T.OUT_F / "03b_onset_composites_ci.png", dpi=170)
    plt.close(fig)


def fig_target_sensitivity(ts, shares):
    labels = {"main": "Main: unusual extent > P90", "total_extent": "Total extent > P90 (no label)",
              "total_extent_same_month": "Total extent > P90 of same month",
              "region_sudd": "Sudd only", "region_west": "West only (negative control)"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    ax = axes[0]
    cols = [C["ink2"], C["blue"], C["aqua"], C["violet"], C["orange"]]
    for (t, s), c in zip(shares.items(), cols):
        ax.plot(s.index, s.values, color=c, marker="o", ms=3, label=labels[t])
    ax.set_ylabel("Share of extreme weeks per year")
    ax.set_title("Does the regime shift depend on how 'extreme' is defined?")
    ax.legend(fontsize=8, loc="upper left")
    ax = axes[1]
    s = ts[ts.metric == "Albert AUC, lead 26 wk"].reset_index(drop=True)
    for j, r in s.iterrows():
        ax.errorbar(r["est"], j, xerr=[[r["est"] - r["ci_low"]], [r["ci_high"] - r["est"]]], fmt="o", ms=6,
                    color=cols[j], elinewidth=1.8)
    ax.set_yticks(range(len(s)), [labels[t] for t in s["target"]])
    ax.axvline(0.5, color=C["ink2"], lw=1, ls="--")
    ax.set_xlabel("AUC of Lake Albert anomaly, 26 weeks ahead (95 % CI)")
    ax.set_title("Upstream signal under each target")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(T.OUT_F / "13_target_sensitivity.png", dpi=170)
    plt.close(fig)


def main():
    w, f, _ = T.build_weekly()
    targets, rf = weekly_targets(w)
    pd.set_option("display.width", 200)

    print("\n=== A. Driver AUC with 95 % CI ===")
    auc = pd.concat([driver_auc_ci(f, targets["main"], p) for p in [(2003, 2025), (2003, 2019)]])
    auc.to_csv(T.OUT_T / "ci_driver_auc.csv", index=False)
    print(auc.round(2).to_string(index=False))

    print("\n=== A. Model skill with 95 % CI ===")
    sk = model_skill_ci()
    sk.to_csv(T.OUT_T / "ci_model_skill.csv", index=False)
    print(sk[sk.metric == "BSS"].pivot_table(index="item", columns="horizon", values=["est", "ci_low", "ci_high"])
          .round(2).to_string())
    fig_skill_ci(sk)
    fig_onsets_ci(w, f)

    print("\n=== B. n = 1: surrogate test and placebo ===")
    n1, within, sur_h0 = n1_tests(w, f, targets["main"])
    n1.to_csv(T.OUT_T / "n1_surrogate_and_placebo.csv", index=False)
    within.to_csv(T.OUT_T / "n1_within_regime_association.csv", index=False)
    print(n1.round(3).to_string(index=False))
    print(within.round(2).to_string(index=False))
    fig_surrogate(n1, sur_h0)

    print("\n=== C. Causal checks ===")
    sp = spatial_control(f, targets)
    sp.to_csv(T.OUT_T / "causal_spatial_negative_control.csv", index=False)
    print(sp.round(2).to_string(index=False))
    mag = spatial_magnitude(rf)
    mag.to_csv(T.OUT_T / "causal_spatial_magnitude.csv", index=False)
    print(mag.round(1).to_string(index=False))
    fig_spatial(targets, f, sp, mag)
    lr = local_rain_path(w, f, targets["main"])
    lr.to_csv(T.OUT_T / "causal_local_rain_path.csv", index=False)
    print(lr.round(2).to_string(index=False))
    med = mediation(w, f)
    med.to_csv(T.OUT_T / "causal_mediation.csv", index=False)
    print(med.round(2).to_string(index=False))

    print("\n=== D. Target sensitivity ===")
    ts, shares = target_sensitivity(w, f, targets)
    ts.to_csv(T.OUT_T / "target_sensitivity.csv", index=False)
    shares.to_csv(T.OUT_T / "target_sensitivity_yearly_share.csv")
    print(ts.round(2).to_string(index=False))
    print(shares.round(2).to_string())
    fig_target_sensitivity(ts, shares)


if __name__ == "__main__":
    main()
