"""
Research question 12.1 — Step 7: one model per sub-question.

Q1  Which hydro-climatic factors are associated with the transition from normal to extreme flooding?
    -> Interpretable logistic regression on flood-season weeks (Jul-Dec) only, so it contrasts extreme flood
       seasons with normal ones rather than wet vs dry season. Standardised drivers -> odds ratio per +1 SD,
       95 % CI from a year-block bootstrap. Fitted on 2003-2019 (seasonal regime) and on 2003-2025.

Q2  Can upstream water-level anomalies provide an early indication of these events?
    -> Out-of-sample (leave-one-year-out) skill at 4-52 weeks ahead of
         local-only  vs  upstream-only  vs  local + upstream, and
         persistence vs  persistence + upstream (does upstream add anything beyond "it is flooded now"?)
       with year-block bootstrap CIs on the skill and on the differences.

Upstream = Lake Albert (DAHITI), Lake Victoria & Kyoga (NASA G-REALM), lake-basin runoff, Sobat rain,
           Indian Ocean Dipole (NOAA). All features use only information available at week t.

Run from the raw_data folder:  python extreme_flood_early_warning/final_model.py   (~5 min)
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import train_models as T
import robustness as R

C = T.C
D = T.HERE / "data"
LEADS = [4, 13, 26, 39, 52]


# --------------------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------------------
def features():
    w, f, thr = T.build_weekly()
    ref = (w.index.year >= T.REF[0]) & (w.index.year <= T.REF[1])
    lk = pd.read_parquet(D / "external_lakes_grealm.parquet")
    lk = lk.resample("W-SUN").median().reindex(w.index).ffill(limit=4)       # causal: last observation
    for c in ["victoria", "kyoga"]:
        f[f"{c}_anom"] = lk[c] - lk[c][ref].mean()
    ci = pd.read_parquet(D / "external_climate_indices.parquet")["iod_dmi"]
    # a month's DMI is published after the month ends -> shift by one month before use
    iod3 = ci.rolling(3).mean().shift(1)
    iod_w = iod3.reindex(pd.date_range(iod3.index.min(), w.index.max(), freq="D")).ffill().reindex(w.index)
    f["iod_3m"] = iod_w
    f["iod_max12m"] = iod_w.rolling(52, min_periods=26).max()               # strong IOD in the past year
    return w, f


# One variable per physical pathway, so odds ratios are interpretable (all VIF < 3). Dropped as near-duplicates:
# rain − evaporation 26 wk (r = 0.99 with rain 26 wk) and Lake Victoria (r = 0.77 with Albert; its water reaches the
# Sudd through Lake Albert). Both stay in the Q2 early-warning models, where collinearity does not matter.
Q1_DRIVERS = {
    "Lake Albert level anomaly": "albert_anom",
    "Indian Ocean Dipole, max past 12 months": "iod_max12m",
    "Lake-basin runoff, 52 weeks": "ro_lakes_a52",
    "Sobat / Ethiopian highland rain, 13 weeks": "tp_sobat_a13",
    "South Sudan rain, 26 weeks": "tp_ssd_a26",
    "South Sudan rain, 4 weeks": "tp_ssd_a4",
}
LOCAL = ["tp_ssd_a4", "tp_ssd_a13", "tp_ssd_a26", "ro_ssd_a13", "tp_sudd_a13", "tp_ghazal_a13", "wb_ssd_a26",
         "et_ssd_a4"]
UPSTREAM = ["albert_anom", "albert_d26", "victoria_anom", "kyoga_anom", "ro_lakes_a52", "tp_lakes_a52",
            "tp_sobat_a13", "ro_sobat_a13", "iod_3m", "iod_max12m"]
STATE = ["log_unusual", "log_recurring", "unusual_share"]
SEASON = ["woy_sin", "woy_cos"]


# --------------------------------------------------------------------------------------
# Q1 — odds ratios
# --------------------------------------------------------------------------------------
def q1(w, f):
    cols = list(Q1_DRIVERS.values())
    rows = []
    for label, (a, b) in {"2003–2019 (seasonal regime)": (2003, 2019), "2003–2025 (all years)": (2003, 2025)}.items():
        m = np.asarray((w.index.year >= a) & (w.index.year <= b) & (w.index.month >= 7))
        X = f.loc[m, cols]
        ok = X.notna().all(axis=1).values
        X, y, idx = X.values[ok], w["extreme"].values[m][ok], w.index[m][ok]
        mu, sd = X.mean(0), X.std(0)
        Z = (X - mu) / sd

        def fit(i):
            return LogisticRegression(C=1.0, max_iter=5000).fit(Z[i], y[i]).coef_[0]

        est = fit(np.arange(len(y)))
        blocks = R.year_blocks(idx)
        keys = list(blocks)
        boots = []
        for _ in range(R.B):
            i = np.concatenate([blocks[k] for k in R.RNG.choice(keys, len(keys))])
            if len(np.unique(y[i])) == 2:
                boots.append(fit(i))
        boots = np.array(boots)
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        for j, (name, col) in enumerate(Q1_DRIVERS.items()):
            rows.append(dict(period=label, driver=name, odds_ratio_per_sd=np.exp(est[j]),
                             ci_low=np.exp(lo[j]), ci_high=np.exp(hi[j]),
                             significant=bool(lo[j] > 0 or hi[j] < 0),
                             n_weeks=len(y), n_extreme=int(y.sum())))
    out = pd.DataFrame(rows)
    out.to_csv(T.OUT_T / "final_q1_odds_ratios.csv", index=False)
    return out


# --------------------------------------------------------------------------------------
# Q2 — early-warning skill
# --------------------------------------------------------------------------------------
Q2_MODELS = {
    "Local only": LOCAL + SEASON,
    "Upstream only": UPSTREAM + SEASON,
    "Local + upstream": LOCAL + UPSTREAM + SEASON,
    "Persistence": STATE + SEASON,
    "Persistence + upstream": STATE + UPSTREAM + SEASON,
}


def q2(w, f):
    dates = w.index
    m = (dates >= T.START) & (dates <= T.END)
    rows, diffs = [], []
    for h in LEADS:
        y = w["extreme"].shift(-h).values
        valid = m & ~np.isnan(y) & (np.arange(len(w)) >= T.LOOKBACK)
        woy = dates.isocalendar().week.clip(upper=52).values
        P = {}
        clim = np.full(len(w), np.nan)
        for mname, cols in Q2_MODELS.items():
            X = f[cols].fillna(0).values
            oof = np.full(len(w), np.nan)
            for _, tr, te in T.make_folds(dates, valid):
                sc = StandardScaler().fit(X[tr])
                mdl = LogisticRegression(C=0.1, max_iter=5000).fit(sc.transform(X[tr]), y[tr])
                oof[te] = mdl.predict_proba(sc.transform(X[te]))[:, 1]
                if mname == "Local only":   # climatology of the target week, from the same training folds
                    tab = pd.Series(y[tr]).groupby(np.roll(woy, -h)[tr]).mean()
                    clim[te] = pd.Series(np.roll(woy, -h)[te]).map(tab).fillna(y[tr].mean()).values
            P[mname] = oof
        ok = valid & ~np.isnan(clim)
        yy, idx = y[ok].astype(int), dates[ok]
        Pk = {k: v[ok] for k, v in P.items()}
        ck = clim[ok]

        def bss(p, i):
            return 1 - np.mean((p[i] - yy[i]) ** 2) / np.mean((ck[i] - yy[i]) ** 2)

        for mname, p in Pk.items():
            for metric, fn in [("AUC", lambda i, p=p: R.fast_auc(yy[i], p[i])), ("BSS", lambda i, p=p: bss(p, i))]:
                est, lo, hi = R.boot_ci(fn, idx)
                rows.append(dict(lead_weeks=h, model=mname, metric=metric, est=est, ci_low=lo, ci_high=hi))
        for dname, (a, b) in {"Upstream only − Local only": ("Upstream only", "Local only"),
                              "Local + upstream − Local only": ("Local + upstream", "Local only"),
                              "Persistence + upstream − Persistence": ("Persistence + upstream", "Persistence")}.items():
            est, lo, hi = R.boot_ci(lambda i: bss(Pk[a], i) - bss(Pk[b], i), idx)
            diffs.append(dict(lead_weeks=h, comparison=dname, bss_difference=est, ci_low=lo, ci_high=hi,
                              significant=bool(lo > 0 or hi < 0)))
        print(f"  lead {h} weeks done", flush=True)
    sk, df = pd.DataFrame(rows), pd.DataFrame(diffs)
    sk.to_csv(T.OUT_T / "final_q2_skill_by_lead.csv", index=False)
    df.to_csv(T.OUT_T / "final_q2_upstream_added_value.csv", index=False)
    return sk, df


def q2_forward(w, f):
    """Same models, trained on 2003-2018 only, evaluated on 2019-2025 (could the regime shift be anticipated?)."""
    dates = w.index
    m = (dates >= T.START) & (dates <= T.END)
    rows = []
    for h in LEADS:
        y = w["extreme"].shift(-h).values
        valid = m & ~np.isnan(y) & (np.arange(len(w)) >= T.LOOKBACK)
        tr = valid & ((dates + pd.Timedelta(weeks=h)) <= pd.Timestamp("2018-12-31"))
        te = valid & (dates >= pd.Timestamp("2019-01-01"))
        for mname, cols in Q2_MODELS.items():
            X = f[cols].fillna(0).values
            sc = StandardScaler().fit(X[tr])
            p = LogisticRegression(C=0.1, max_iter=5000).fit(sc.transform(X[tr]), y[tr]).predict_proba(
                sc.transform(X[te]))[:, 1]
            rows.append(dict(lead_weeks=h, model=mname, auc=R.fast_auc(y[te].astype(int), p),
                             hit_rate=float(np.mean(p[y[te] == 1] > 0.5))))
    out = pd.DataFrame(rows)
    out.to_csv(T.OUT_T / "final_q2_forward_test.csv", index=False)
    return out


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def fig_q1(od):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    names = list(Q1_DRIVERS)
    for ax, (period, s) in zip(axes, od.groupby("period", sort=False)):
        s = s.set_index("driver").loc[names]
        for j, (name, r) in enumerate(s.iterrows()):
            col = C["blue"] if r["significant"] and r["odds_ratio_per_sd"] > 1 else (
                C["red"] if r["significant"] else C["muted"])
            ax.errorbar(r["odds_ratio_per_sd"], j, xerr=[[r["odds_ratio_per_sd"] - r["ci_low"]],
                        [r["ci_high"] - r["odds_ratio_per_sd"]]], fmt="o", ms=7, color=col, elinewidth=2)
        ax.axvline(1, color=C["ink2"], lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_yticks(range(len(names)), names)
        ax.invert_yaxis()
        ax.set_xlabel("Odds ratio per +1 SD (log scale, 95 % CI)")
        ax.set_title(f"{period}\n{int(s['n_extreme'].iloc[0])} extreme of {int(s['n_weeks'].iloc[0])} flood-season weeks",
                     fontsize=11)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Q1 · Which factors distinguish an extreme flood season from a normal one?\n"
                 "blue = raises the odds, red = lowers them, grey = not significant",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(T.OUT_F / "15_q1_factor_odds_ratios.png", dpi=170)
    plt.close(fig)


def fig_q2(sk, df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    for mname, c, ls in [("Local only", C["orange"], "-"), ("Upstream only", C["violet"], "-"),
                         ("Local + upstream", C["blue"], "-"), ("Persistence", C["ink2"], ":")]:
        s = sk[(sk.model == mname) & (sk.metric == "AUC")].sort_values("lead_weeks")
        ax.fill_between(s.lead_weeks, s.ci_low, s.ci_high, color=c, alpha=0.12, lw=0)
        ax.plot(s.lead_weeks, s.est, color=c, ls=ls, marker="o", ms=5, label=mname)
    ax.axhline(0.5, color="#c3c2b7", lw=1)
    ax.set_xticks(LEADS)
    ax.set_xlabel("Warning lead time (weeks)")
    ax.set_ylabel("ROC-AUC (95 % CI band)")
    ax.set_title("How well can each source of information warn?")
    ax.legend(fontsize=9, loc="lower left")
    ax = axes[1]
    for j, (dname, c) in enumerate([("Upstream only − Local only", C["violet"]),
                                    ("Persistence + upstream − Persistence", C["ink"])]):
        s = df[df.comparison == dname].sort_values("lead_weeks")
        x = np.arange(len(LEADS)) + (j - 0.5) * 0.22
        ax.errorbar(x, s.bss_difference, yerr=[s.bss_difference - s.ci_low, s.ci_high - s.bss_difference],
                    fmt="o", ms=6, color=c, elinewidth=2, label=dname)
    ax.axhline(0, color=C["ink2"], lw=1, ls="--")
    ax.set_xticks(range(len(LEADS)), [f"{h} wk" for h in LEADS])
    ax.set_ylabel("Difference in Brier skill score (95 % CI)")
    ax.set_title("Does upstream information add skill? (CI above 0 = yes)")
    ax.legend(fontsize=9, loc="upper left")
    fig.suptitle("Q2 · Can upstream anomalies give early indication? (leave-one-year-out, 2003–2025)",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(T.OUT_F / "16_q2_upstream_early_warning.png", dpi=170)
    plt.close(fig)


def main():
    w, f = features()
    pd.set_option("display.width", 200)
    print("\n=== Q1: odds ratios (flood-season weeks) ===")
    od = q1(w, f)
    print(od.round(2).to_string(index=False))
    fig_q1(od)
    print("\n=== Q2: early-warning skill ===")
    sk, df = q2(w, f)
    print(sk.pivot_table(index=["metric", "model"], columns="lead_weeks", values="est").round(2).to_string())
    print(df.round(3).to_string(index=False))
    fig_q2(sk, df)
    print("\n=== Q2: forward test (train <=2018, test 2019-2025) ===")
    print(q2_forward(w, f).pivot_table(index="model", columns="lead_weeks", values=["auc", "hit_rate"]).round(2).to_string())


if __name__ == "__main__":
    main()
