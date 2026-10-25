"""
Research question 12.1 — Step 2: drivers of extreme/unusual flooding and early warning from upstream anomalies.

Inputs : extreme_flood_early_warning/data/daily_series.parquet   (from build_dataset.py)
Outputs: extreme_flood_early_warning/outputs/{tables,figures}

Pipeline
  1. Weekly hydro-climatic features (all strictly causal: only information available at week t).
  2. Target: an "extreme / unusual flood week" = unusual-flood extent in South Sudan above the 90th percentile
     of the 2003-2019 reference period (the "normal seasonal flooding" regime).
  3. Association analysis: lagged correlations and onset composites of each driver.
  4. Prediction: multi-horizon GRU (2-26 weeks ahead) with feature-group ablations, benchmarked against
     climatology, persistence, logistic regression and gradient boosting.
     Validation: leave-one-year-out with a 13-week buffer, plus a forward test (train <=2018, test 2019-2025)
     that asks whether the 2020 regime shift could have been anticipated.
  5. Permutation importance of feature groups for the GRU.

Run from the raw_data folder with a Python that has torch:  python extreme_flood_early_warning/train_models.py
"""
import json
import os
import warnings
from pathlib import Path

# torch and sklearn ship different OpenMP runtimes; multi-threaded use of both segfaults on macOS/conda
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=RuntimeWarning)
torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "daily_series.parquet"
OUT_T = HERE / "outputs" / "tables"
OUT_F = HERE / "outputs" / "figures"
OUT_T.mkdir(parents=True, exist_ok=True)
OUT_F.mkdir(parents=True, exist_ok=True)

PIXEL_KM2 = 0.0534          # MCDWD pixel (0.0020833 deg) at ~7 N
REF = (2003, 2019)          # reference "normal regime" period for anomalies and the event threshold
START, END = "2003-01-01", "2025-12-31"
HORIZONS = [2, 4, 8, 13, 26]  # weeks ahead
LOOKBACK = 52
SEEDS = [0, 1, 2]
BUFFER_W = 13

# Palette (validated reference palette, light mode)
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", violet="#4a3aa7",
         red="#e34948", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", surface="#fcfcfb")
plt.rcParams.update({
    "figure.facecolor": C["surface"], "axes.facecolor": C["surface"], "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": C["ink2"], "xtick.color": C["muted"], "ytick.color": C["muted"], "text.color": C["ink"],
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "lines.linewidth": 2, "legend.frameon": False,
})

# --------------------------------------------------------------------------------------
# 1. Weekly features
# --------------------------------------------------------------------------------------
FEATURE_GROUPS = {
    "Lake Albert level (upstream)": ["albert_anom", "albert_d13", "albert_d26"],
    "Equatorial-lakes rain & runoff (upstream)": ["tp_lakes_a4", "tp_lakes_a13", "tp_lakes_a26", "tp_lakes_a52",
                                                  "ro_lakes_a13", "ro_lakes_a52"],
    "Ethiopian highlands / Sobat (upstream)": ["tp_sobat_a4", "tp_sobat_a13", "ro_sobat_a13"],
    # DFO 1505 drains to Lake Turkana, not the Nile: a regional-wetness proxy, deliberately not "upstream"
    "SW Ethiopia discharge (regional)": ["q_ethiopia_anom"],
    "Local rain, runoff & ET (South Sudan)": ["tp_ssd_a1", "tp_ssd_a4", "tp_ssd_a13", "tp_ssd_a26", "ro_ssd_a4",
                                              "ro_ssd_a13", "tp_sudd_a13", "tp_ghazal_a13", "et_ssd_a4",
                                              "wb_ssd_a13", "wb_ssd_a26"],
    "White Nile discharge at Malakal (local)": ["q_malakal_anom"],
    "Current flood state": ["log_unusual", "log_recurring", "unusual_share"],
    "Seasonality": ["woy_sin", "woy_cos"],
}
UPSTREAM = [g for g in FEATURE_GROUPS if "(upstream)" in g]
VARIANTS = {
    "GRU – all features": list(FEATURE_GROUPS),
    "GRU – hydro-climate only (no flood state)": [g for g in FEATURE_GROUPS if g != "Current flood state"],
    "GRU – without upstream": [g for g in FEATURE_GROUPS if g not in UPSTREAM],
    "GRU – upstream + season only": UPSTREAM + ["Seasonality"],
}
ENSEMBLE = "Ensemble (GRU + logistic + boosting)"


def woy_anomaly(s, ref_mask):
    """Standardised anomaly relative to the week-of-year climatology of the reference period."""
    woy = s.index.isocalendar().week.clip(upper=52).values
    ref = pd.DataFrame({"v": s.values, "w": woy})[ref_mask]
    clim = ref.groupby("w")["v"].agg(["mean", "std"])
    # smooth the climatology over +-2 weeks (circular) to reduce sampling noise
    clim = pd.concat([clim.iloc[-2:], clim, clim.iloc[:2]]).rolling(5, center=True).mean().iloc[2:-2]
    mu, sd = clim["mean"].reindex(woy).values, clim["std"].reindex(woy).values
    return pd.Series((s.values - mu) / np.where(sd > 0, sd, np.nan), index=s.index)


def build_weekly():
    d = pd.read_parquet(DATA)
    d.index = pd.to_datetime(d.index)
    flood_cols = [c for c in d if "_px_" in c]
    flux_cols = [c for c in d if c.startswith(("tp_", "ro_", "et_"))]
    w = pd.concat([
        d[flood_cols].resample("W-SUN").max() * PIXEL_KM2,          # robust to cloud-gapped days
        d[flux_cols].resample("W-SUN").sum(),                        # weekly totals [mm]
        d[["q_malakal", "q_ethiopia"]].resample("W-SUN").mean(),
        d[["lake_albert"]].resample("W-SUN").median(),
    ], axis=1)
    w.columns = [c.replace("_px_", "_km2_") for c in w.columns]
    w["wb_ssd"] = w["tp_ssd"] - w["et_ssd"]
    ref = (w.index.year >= REF[0]) & (w.index.year <= REF[1])

    f = pd.DataFrame(index=w.index)
    # Upstream lake: last-observation-carried-forward (causal; satellite revisit is ~10-35 days)
    alb = w["lake_albert"].ffill(limit=26)
    f["albert_anom"] = alb - alb[ref].mean()
    f["albert_d13"] = alb - alb.shift(13)
    f["albert_d26"] = alb - alb.shift(26)
    # Fluxes: standardised weekly anomalies, then accumulated over k weeks (SPI-like, scaled by sqrt(k))
    for v in ["tp_lakes", "ro_lakes", "tp_sobat", "ro_sobat", "tp_ssd", "ro_ssd", "tp_sudd", "tp_ghazal",
              "et_ssd", "wb_ssd"]:
        a = woy_anomaly(w[v], ref)
        for k in [1, 4, 13, 26, 52]:
            f[f"{v}_a{k}"] = a.rolling(k, min_periods=k).sum() / np.sqrt(k)
    for q in ["q_malakal", "q_ethiopia"]:
        f[f"{q}_anom"] = woy_anomaly(w[q], ref)
    f["log_unusual"] = np.log1p(w["unusual_km2_ssd"])
    f["log_recurring"] = np.log1p(w["recurring_km2_ssd"])
    f["unusual_share"] = w["unusual_km2_ssd"] / (w["unusual_km2_ssd"] + w["recurring_km2_ssd"] + 1e-9)
    woy = w.index.isocalendar().week.values.astype(float)
    f["woy_sin"], f["woy_cos"] = np.sin(2 * np.pi * woy / 52.18), np.cos(2 * np.pi * woy / 52.18)

    thr = w.loc[ref, "unusual_km2_ssd"].quantile(0.90)
    w["extreme"] = (w["unusual_km2_ssd"] > thr).astype(int)
    return w, f, thr


# --------------------------------------------------------------------------------------
# 2. Association analysis
# --------------------------------------------------------------------------------------
DRIVERS = {
    "Lake Albert level anomaly": "albert_anom",
    "Lake Albert 26-wk rise": "albert_d26",
    "Equatorial-lakes rain, 52-wk": "tp_lakes_a52",
    "Equatorial-lakes runoff, 52-wk": "ro_lakes_a52",
    "Sobat/Ethiopia rain, 13-wk": "tp_sobat_a13",
    "SW Ethiopia discharge (regional)": "q_ethiopia_anom",
    "South Sudan rain, 4-wk": "tp_ssd_a4",
    "South Sudan rain, 26-wk": "tp_ssd_a26",
    "South Sudan runoff, 13-wk": "ro_ssd_a13",
    "South Sudan P–ET, 26-wk": "wb_ssd_a26",
    "Sudd rain, 13-wk": "tp_sudd_a13",
    "Malakal discharge": "q_malakal_anom",
}


def association_analysis(w, f):
    target = woy_anomaly(np.log1p(w["unusual_km2_ssd"]), np.ones(len(w), bool))  # de-seasonalised log extent
    lags = list(range(0, 53, 2))
    rows = []
    for period, (a, b) in {"2003–2025": (2003, 2025), "2003–2019 (pre-shift)": (2003, 2019)}.items():
        m = (w.index.year >= a) & (w.index.year <= b)
        for name, col in DRIVERS.items():
            for lag in lags:
                x = f[col].shift(lag)[m]
                ok = x.notna() & target[m].notna()
                r = spearmanr(x[ok], target[m][ok]).statistic if ok.sum() > 30 else np.nan
                rows.append(dict(period=period, driver=name, lag_weeks=lag, spearman_r=r))
    corr = pd.DataFrame(rows)
    corr.to_csv(OUT_T / "lagged_correlations.csv", index=False)

    # Onset composites: first extreme week after >= 8 non-extreme weeks
    ext = w["extreme"].values
    onsets = [i for i in range(8, len(ext)) if ext[i] == 1 and ext[i - 8:i].sum() == 0
              and START <= str(w.index[i].date()) <= END]
    comp = {}
    for name, col in DRIVERS.items():
        mat = np.array([f[col].values[i - 26:i + 5] for i in onsets if i >= 26 and i + 5 <= len(f)])
        comp[name] = np.nanmean(mat, axis=0)
    comp = pd.DataFrame(comp, index=range(-26, 5))
    comp.to_csv(OUT_T / "onset_composites.csv")

    # Standardised driver contrast: extreme vs normal weeks, same flood season (Jul-Dec)
    m = (w.index.month >= 7) & (w.index >= START) & (w.index <= END)
    cont = []
    for name, col in DRIVERS.items():
        x = f.loc[m, col]
        e, n = x[w.loc[m, "extreme"] == 1], x[w.loc[m, "extreme"] == 0]
        sd = np.sqrt((e.var() + n.var()) / 2)
        cont.append(dict(driver=name, mean_extreme=e.mean(), mean_normal=n.mean(), cohens_d=(e.mean() - n.mean()) / sd))
    cont = pd.DataFrame(cont).sort_values("cohens_d", ascending=False)
    cont.to_csv(OUT_T / "driver_contrast_extreme_vs_normal.csv", index=False)
    return corr, comp, onsets, cont


def indicator_skill(w, f):
    """Early-warning value of each raw driver, no model fitting: ROC-AUC of the driver value at week t
    for discriminating extreme vs. normal weeks at t+h. AUC < 0.5 means the driver is inversely related."""
    rows = []
    for period, (a, b) in {"2003–2025": (2003, 2025), "2003–2019 (pre-shift)": (2003, 2019)}.items():
        m = (w.index.year >= a) & (w.index.year <= b)
        for name, col in DRIVERS.items():
            for h in [0] + HORIZONS:
                yy = w["extreme"].shift(-h)[m]
                x = f[col][m]
                ok = yy.notna() & x.notna()
                auc = roc_auc_score(yy[ok], x[ok]) if yy[ok].nunique() == 2 else np.nan
                rows.append(dict(period=period, driver=name, horizon=h, auc=auc))
    ind = pd.DataFrame(rows)
    ind.to_csv(OUT_T / "indicator_skill_univariate_auc.csv", index=False)

    # Lead of upstream threshold crossings over the onset of the sustained extreme regime of the 2020s
    ext = w["extreme"]
    share_ahead = ext[::-1].rolling(26, min_periods=26).mean()[::-1]   # share of extreme weeks in [t, t+25]
    regime_start = share_ahead[(share_ahead >= 0.5) & (share_ahead.index >= "2019-01-01")].index[0]

    def first_sustained(s, thr, n=4):
        hit = (s > thr).astype(int).rolling(n).sum() == n
        hit = hit[hit.index >= "2019-01-01"]
        return hit[hit].index[0] - pd.Timedelta(weeks=n - 1) if hit.any() else None

    leads = []
    for label, col, thr in [("Lake Albert anomaly > +0.5 m", "albert_anom", 0.5),
                            ("Lake Albert anomaly > +1.0 m", "albert_anom", 1.0),
                            ("Equatorial-lakes 52-wk rain > +2σ", "tp_lakes_a52", 2.0),
                            ("Equatorial-lakes 52-wk runoff > +2σ", "ro_lakes_a52", 2.0)]:
        d = first_sustained(f[col], thr)
        # how often did the same signal fire in 2003-2018 (false-alarm check)?
        pre = f[col][(f.index >= START) & (f.index < "2019-01-01")]
        leads.append(dict(signal=label, first_crossing=None if d is None else str(d.date()),
                          sustained_regime_onset=str(regime_start.date()),
                          lead_weeks=None if d is None else int((regime_start - d).days // 7),
                          weeks_above_threshold_2003_2018=int((pre > thr).sum())))
    leads = pd.DataFrame(leads)
    leads.to_csv(OUT_T / "regime_shift_early_warning_leads.csv", index=False)
    return ind, leads, regime_start


def fig_indicator(ind):
    show = [("Lake Albert level anomaly", C["violet"]), ("Equatorial-lakes runoff, 52-wk", C["blue"]),
            ("Sobat/Ethiopia rain, 13-wk", C["yellow"]), ("South Sudan rain, 26-wk", C["aqua"]),
            ("South Sudan rain, 4-wk", C["orange"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, period in zip(axes, ind["period"].unique()):
        for name, c in show:
            s = ind[(ind.period == period) & (ind.driver == name)]
            ax.plot(s["horizon"], s["auc"], color=c, marker="o", ms=4.5, label=name)
        ax.axhline(0.5, color="#c3c2b7", lw=1)
        ax.set_xticks([0] + HORIZONS)
        ax.set_xlabel("Lead time (weeks)")
        ax.set_title(period)
    axes[0].set_ylabel("ROC-AUC of the raw driver value")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5)
    fig.suptitle("Early-warning value of single drivers (no model): does the anomaly at week t flag an extreme at t + lead?",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_F / "08_single_driver_early_warning_skill.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# 3. Models
# --------------------------------------------------------------------------------------
class GRUNet(nn.Module):
    def __init__(self, n_in, n_out, hidden=32):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, batch_first=True)
        self.drop = nn.Dropout(0.25)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(self.drop(h[-1]))


def make_targets(w):
    y = np.stack([w["extreme"].shift(-h).values for h in HORIZONS], axis=1)
    return y  # NaN where t+h is beyond the record


def sequences(X, idx):
    return np.stack([X[i - LOOKBACK + 1:i + 1] for i in idx])


def fit_gru(Xtr, ytr, seed, epochs=60):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = GRUNet(Xtr.shape[2], ytr.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-2)
    pos = np.nanmean(ytr, axis=0)
    pw = torch.tensor((1 - pos) / np.clip(pos, 0.05, None), dtype=torch.float32).clamp(max=10)
    Xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)
    mask = ~torch.isnan(yt)
    yt = torch.nan_to_num(yt)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
    n = len(Xt)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(n)
        for b in range(0, n, 64):
            j = perm[b:b + 64]
            xb = Xt[j] + 0.05 * torch.randn_like(Xt[j])  # input noise: regularises on a small sample
            loss = (lossf(net(xb), yt[j]) * mask[j]).sum() / mask[j].sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
    net.eval()
    return net


def predict_gru(nets, X):
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32)
        return np.mean([torch.sigmoid(n(Xt)).numpy() for n in nets], axis=0)


def cols_for(groups):
    return [c for g in groups for c in FEATURE_GROUPS[g]]


def make_folds(dates, valid):
    """Leave-one-year-out with a buffer, plus the forward 'regime shift' test."""
    years = sorted(set(dates[valid].year))
    folds = []
    for y in years:
        te = valid & (dates.year == y)
        lo, hi = pd.Timestamp(f"{y}-01-01") - pd.Timedelta(weeks=BUFFER_W), pd.Timestamp(f"{y}-12-31") + pd.Timedelta(weeks=BUFFER_W)
        tr = valid & ~((dates >= lo) & (dates <= hi))
        folds.append((f"LOYO {y}", tr, te))
    return folds


def run_models(w, f, train_mask, test_mask, y, tag, permute=False, fitted=None):
    """Fit every model on train_mask, predict test_mask. Returns {model: probs[n_test, H]} and permutation results.
    If `fitted` is a dict, the fitted scalers/models are stored in it (used for the final, all-data models)."""
    preds, perm_out = {}, {}
    idx_tr, idx_te = np.where(train_mask)[0], np.where(test_mask)[0]
    ytr = y[idx_tr]

    # Climatology: P(extreme at t+h | week of year of t+h) from training targets
    woy = w.index.isocalendar().week.clip(upper=52).values
    clim = np.zeros((len(idx_te), len(HORIZONS)))
    for k, h in enumerate(HORIZONS):
        wt = np.roll(woy, -h)
        tab = pd.Series(ytr[:, k]).groupby(wt[idx_tr]).mean()
        base = np.nanmean(ytr[:, k])
        clim[:, k] = pd.Series(wt[idx_te]).map(tab).fillna(base).values
    preds["Climatology"] = clim

    # Tabular models on the current-week feature vector
    def tab_model(cols, make, name=None):
        out = np.zeros((len(idx_te), len(HORIZONS)))
        Xa = f[cols].fillna(0).values
        sc = StandardScaler().fit(Xa[idx_tr])
        mdls = []
        for k in range(len(HORIZONS)):
            ok = ~np.isnan(ytr[:, k])
            if len(np.unique(ytr[ok, k])) < 2:
                out[:, k] = np.nanmean(ytr[:, k])
                mdls.append(None)
                continue
            mdl = make().fit(sc.transform(Xa[idx_tr][ok]), ytr[ok, k])
            out[:, k] = mdl.predict_proba(sc.transform(Xa[idx_te]))[:, 1]
            mdls.append(mdl)
        if fitted is not None and name:
            fitted[name] = dict(columns=cols, scaler=sc, models=mdls)
        return out

    preds["Persistence (logistic, flood state + season)"] = tab_model(
        cols_for(["Current flood state", "Seasonality"]), lambda: LogisticRegression(C=1.0, max_iter=2000))
    all_cols = cols_for(FEATURE_GROUPS)
    preds["Logistic regression – all features"] = tab_model(
        all_cols, lambda: LogisticRegression(C=0.1, max_iter=5000), "logistic")
    preds["Gradient boosting – all features"] = tab_model(
        all_cols, lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                         l2_regularization=1.0, random_state=0), "gbm")

    # GRU variants
    for vname, groups in VARIANTS.items():
        cols = cols_for(groups)
        Xa = f[cols].values.astype(float)
        sc = StandardScaler().fit(np.nan_to_num(Xa[idx_tr]))
        Xs = np.nan_to_num(sc.transform(np.nan_to_num(Xa)))
        nets = [fit_gru(sequences(Xs, idx_tr), ytr, s) for s in SEEDS]
        Xte = sequences(Xs, idx_te)
        preds[vname] = predict_gru(nets, Xte)
        if fitted is not None and vname == "GRU – all features":
            fitted["gru"] = dict(columns=cols, scaler=sc, state_dicts=[n.state_dict() for n in nets],
                                 n_in=len(cols), lookback=LOOKBACK, horizons=HORIZONS)

        if permute and vname == "GRU – all features":
            rng = np.random.default_rng(0)
            for g in FEATURE_GROUPS:
                ci = [cols.index(c) for c in FEATURE_GROUPS[g]]
                reps = []
                for _ in range(5):
                    Xp = Xte.copy()
                    p = rng.permutation(len(Xp))
                    Xp[:, :, ci] = Xte[p][:, :, ci]   # swap whole input histories of this group between samples
                    reps.append(predict_gru(nets, Xp))
                perm_out[g] = np.mean(reps, axis=0)

    preds[ENSEMBLE] = np.mean([preds["GRU – all features"], preds["Logistic regression – all features"],
                               preds["Gradient boosting – all features"]], axis=0)
    print(f"    {tag}: done")
    return preds, perm_out


def score(y, p):
    ok = ~np.isnan(y) & ~np.isnan(p)
    y, p = y[ok], p[ok]
    if len(np.unique(y)) < 2:
        return dict(auc=np.nan, ap=np.nan, brier=brier_score_loss(y, p), n=len(y), pos=y.mean())
    return dict(auc=roc_auc_score(y, p), ap=average_precision_score(y, p), brier=brier_score_loss(y, p),
                n=len(y), pos=y.mean())


# --------------------------------------------------------------------------------------
# 4. Figures
# --------------------------------------------------------------------------------------
def fig_timeline(w, f, thr):
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True, gridspec_kw=dict(height_ratios=[2, 1.2, 1.2]))
    m = (w.index >= START) & (w.index <= END)
    ww, ff = w[m], f[m]
    ax = axes[0]
    for i, (s, e) in enumerate(_runs(ww["extreme"].values)):
        ax.axvspan(ww.index[s], ww.index[e], color=C["red"], alpha=0.08, lw=0, label="Extreme week" if i == 0 else None)
    ax.plot(ww.index, ww["recurring_km2_ssd"], color=C["blue"], lw=1.2, label="Recurring (normal seasonal) flood")
    ax.plot(ww.index, ww["unusual_km2_ssd"], color=C["orange"], lw=1.2, label="Unusual flood")
    ax.axhline(thr, color=C["ink2"], lw=1, ls="--")
    ax.text(ww.index[5], thr * 1.25, f"Extreme threshold: P90 of 2003–2019 = {thr:,.0f} km²", color=C["ink2"], fontsize=8.5)
    ax.set_yscale("log")
    ax.set_ylabel("Detected flood extent, km² (log)")
    ax.set_title("Flood regime in South Sudan: normal seasonal vs. unusual flooding (weekly max of 3-day composites)")
    ax.legend(loc="upper left", ncol=3, fontsize=8.5)
    ax = axes[1]
    ax.plot(ff.index, ff["albert_anom"], color=C["violet"], lw=1.6)
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.set_ylabel("m vs. 2003–19 mean")
    ax.set_title("Upstream: Lake Albert water-level anomaly (satellite altimetry)")
    ax = axes[2]
    ax.fill_between(ff.index, 0, ff["tp_lakes_a52"], where=ff["tp_lakes_a52"] > 0, color=C["blue"], alpha=0.6, lw=0)
    ax.fill_between(ff.index, 0, ff["tp_lakes_a52"], where=ff["tp_lakes_a52"] <= 0, color=C["red"], alpha=0.45, lw=0)
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.set_ylabel("Standardised anomaly")
    ax.set_title("Upstream: Equatorial-lakes basin rainfall, 52-week accumulated anomaly (ERA5)")
    fig.tight_layout()
    fig.savefig(OUT_F / "01_flood_regime_and_upstream_timeline.png", dpi=170)
    plt.close(fig)


def _runs(a):
    runs, s = [], None
    for i, v in enumerate(a):
        if v and s is None:
            s = i
        if not v and s is not None:
            runs.append((s, i - 1)); s = None
    if s is not None:
        runs.append((s, len(a) - 1))
    return runs


def fig_lagcorr(corr):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", [C["blue"], "#f0efec", C["red"]])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for ax, period in zip(axes, corr["period"].unique()):
        piv = corr[corr.period == period].pivot(index="driver", columns="lag_weeks", values="spearman_r")
        piv = piv.loc[list(DRIVERS)]
        im = ax.imshow(piv.values, cmap=cmap, vmin=-0.8, vmax=0.8, aspect="auto")
        ax.set_xticks(range(0, piv.shape[1], 2), [str(c) for c in piv.columns[::2]])
        ax.set_yticks(range(len(piv)), piv.index)
        ax.set_xlabel("Driver leads flood anomaly by (weeks)")
        ax.set_title(period)
        ax.grid(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label("Spearman r with de-seasonalised unusual-flood extent")
    fig.suptitle("Which drivers lead unusual flooding, and by how long?", x=0.01, ha="left", fontweight="bold")
    fig.savefig(OUT_F / "02_lagged_driver_correlations.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_onsets(comp, n_onsets):
    show = ["Lake Albert level anomaly", "Equatorial-lakes rain, 52-wk", "South Sudan rain, 26-wk",
            "South Sudan rain, 4-wk", "Sobat/Ethiopia rain, 13-wk"]
    cols = [C["violet"], C["blue"], C["aqua"], C["orange"], C["yellow"]]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for name, c in zip(show, cols):
        ax.plot(comp.index, comp[name], color=c, label=name)
        ax.text(comp.index[-1] + 0.4, comp[name].iloc[-1], name, color=C["ink2"], fontsize=8, va="center")
    ax.axvline(0, color=C["ink2"], lw=1, ls="--")
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.set_xlim(-26, 11)
    ax.set_xlabel("Weeks relative to onset of an extreme-flood episode")
    ax.set_ylabel("Mean anomaly (m for Albert; σ for rainfall)")
    ax.set_title(f"Driver anomalies before the onset of extreme-flood episodes (n = {n_onsets} onsets, 2003–2025)")
    fig.tight_layout()
    fig.savefig(OUT_F / "03_onset_composites.png", dpi=170)
    plt.close(fig)


MODEL_STYLE = {
    "Climatology": (C["muted"], "--"),
    "Persistence (logistic, flood state + season)": (C["ink2"], ":"),
    "Logistic regression – all features": (C["aqua"], "-"),
    "Gradient boosting – all features": (C["yellow"], "-"),
    "GRU – all features": (C["blue"], "-"),
    ENSEMBLE: (C["red"], "-"),
    "GRU – hydro-climate only (no flood state)": (C["aqua"], "-"),
    "GRU – without upstream": (C["orange"], "-"),
    "GRU – upstream + season only": (C["violet"], "-"),
}
MAIN_MODELS = ["Climatology", "Persistence (logistic, flood state + season)", "Logistic regression – all features",
               "Gradient boosting – all features", "GRU – all features", ENSEMBLE]
ABLATIONS = ["Persistence (logistic, flood state + season)"] + list(VARIANTS)


def fig_skill(summary, title, fname, models):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, metric, lab in [(axes[0], "auc", "ROC-AUC"), (axes[1], "bss", "Brier skill score vs. climatology")]:
        for mname in models:
            c, ls = MODEL_STYLE[mname]
            s = summary[summary.model == mname].sort_values("horizon")
            label = mname + (" (BSS = 0 by definition)" if mname == "Climatology" else "")
            if metric == "bss" and mname == "Climatology":
                ax.plot([], [], color=c, ls=ls, marker="o", ms=4.5, lw=1.8, label=label)
                continue
            ax.plot(s["horizon"], s[metric], color=c, ls=ls, marker="o", ms=4.5, lw=1.8, label=label)
        ax.set_xticks(HORIZONS)
        ax.set_xlabel("Lead time (weeks)")
        ax.set_title(lab)
        if metric == "bss":
            ax.axhline(0, color="#c3c2b7", lw=1)
        else:
            ax.axhline(0.5, color="#c3c2b7", lw=1)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_F / fname, dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_importance(imp):
    piv = imp.pivot(index="group", columns="horizon", values="auc_drop")
    piv = piv.loc[piv.mean(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    hs = list(piv.columns)
    cols = [C["blue"], C["aqua"], C["yellow"], C["orange"], C["violet"]]
    bh = 0.8 / len(hs)
    for k, (h, c) in enumerate(zip(hs, cols)):
        ax.barh(np.arange(len(piv)) + (k - len(hs) / 2 + 0.5) * bh, piv[h], height=bh * 0.85, color=c, label=f"{h} wk")
    ax.set_yticks(range(len(piv)), piv.index)
    ax.axvline(0, color="#c3c2b7", lw=1)
    ax.set_xlabel("Drop in out-of-sample ROC-AUC when the group is permuted")
    ax.set_title("What the GRU relies on, by lead time (leave-one-year-out)")
    ax.legend(title="Lead", fontsize=8, title_fontsize=8, loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT_F / "05_gru_feature_group_importance.png", dpi=170)
    plt.close(fig)


def fig_forward(w, test_mask, y, preds, k_show=(2, 3)):
    dates = w.index[test_mask]
    fig, axes = plt.subplots(len(k_show), 1, figsize=(11, 3.2 * len(k_show)), sharex=True)
    for ax, k in zip(np.atleast_1d(axes), k_show):
        h = HORIZONS[k]
        tgt_dates = dates + pd.Timedelta(weeks=h)
        yy = y[test_mask][:, k]
        for i, (s, e) in enumerate(_runs(np.nan_to_num(yy).astype(int))):
            ax.axvspan(tgt_dates[s], tgt_dates[e], color=C["red"], alpha=0.08, lw=0,
                       label="Observed extreme week" if i == 0 else None)
        for mname, c in [("GRU – all features", C["blue"]), ("GRU – upstream + season only", C["violet"]),
                         ("Persistence (logistic, flood state + season)", C["ink2"])]:
            ax.plot(tgt_dates, preds[mname][:, k], color=c, lw=1.4, label=mname)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("P(extreme)")
        ax.set_title(f"Forward test — trained on 2003–2018 only — forecast issued {h} weeks ahead")
        ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_F / "06_forward_test_2019_2025.png", dpi=170)
    plt.close(fig)


# --------------------------------------------------------------------------------------
def main():
    w, f, thr = build_weekly()
    print(f"Extreme threshold (P90 of weekly unusual extent, {REF[0]}-{REF[1]}): {thr:,.0f} km²")
    m = (w.index >= START) & (w.index <= END)
    print("Extreme-week share by year:\n", w.loc[m, "extreme"].groupby(w.index[m].year).mean().round(2).to_string())

    fig_timeline(w, f, thr)
    corr, comp, onsets, cont = association_analysis(w, f)
    fig_lagcorr(corr)
    fig_onsets(comp, len(onsets))
    print("\nDriver contrast (Jul-Dec, extreme vs normal weeks):\n", cont.round(2).to_string(index=False))
    ind, leads, regime_start = indicator_skill(w, f)
    fig_indicator(ind)
    print("\nSingle-driver early-warning AUC:\n",
          ind.pivot_table(index=["period", "driver"], columns="horizon", values="auc").round(2).to_string())
    print("\nUpstream threshold crossings vs. onset of the sustained extreme regime:\n", leads.to_string(index=False))

    y = make_targets(w)
    pos = np.arange(len(w))
    valid = m & (pos >= LOOKBACK - 1)
    valid &= ~np.all(np.isnan(y), axis=1)
    dates = w.index

    # ---- Leave-one-year-out ----
    print("\nLeave-one-year-out CV ...")
    oof, oof_perm = {}, {}
    for name, tr, te in make_folds(dates, valid):
        p, pp = run_models(w, f, tr, te, y, name, permute=True)
        for k, v in p.items():
            oof.setdefault(k, np.full((len(w), len(HORIZONS)), np.nan))[te] = v
        for k, v in pp.items():
            oof_perm.setdefault(k, np.full((len(w), len(HORIZONS)), np.nan))[te] = v

    rows = []
    for mname, P in oof.items():
        for k, h in enumerate(HORIZONS):
            s = score(y[valid, k], P[valid, k])
            rows.append(dict(model=mname, horizon=h, **s))
    loyo = pd.DataFrame(rows)
    clim_b = loyo[loyo.model == "Climatology"].set_index("horizon")["brier"]
    loyo["bss"] = 1 - loyo["brier"] / loyo["horizon"].map(clim_b)
    loyo.to_csv(OUT_T / "skill_leave_one_year_out.csv", index=False)
    print(loyo.round(3).to_string(index=False))

    base = oof["GRU – all features"]
    imp = []
    for g, P in oof_perm.items():
        for k, h in enumerate(HORIZONS):
            imp.append(dict(group=g, horizon=h, auc_drop=score(y[valid, k], base[valid, k])["auc"]
                            - score(y[valid, k], P[valid, k])["auc"]))
    imp = pd.DataFrame(imp)
    imp.to_csv(OUT_T / "gru_permutation_importance.csv", index=False)

    oof_df = pd.concat({m_: pd.DataFrame(P[valid], index=dates[valid], columns=[f"p_{h}wk" for h in HORIZONS])
                        for m_, P in oof.items()}, names=["model", "week"])
    oof_df.join(pd.DataFrame(y[valid], index=dates[valid], columns=[f"y_{h}wk" for h in HORIZONS]), on="week") \
          .to_parquet(OUT_T / "oof_predictions_leave_one_year_out.parquet")

    fig_skill(loyo, "Out-of-sample skill by lead time — leave-one-year-out, 2003–2025",
              "04a_skill_by_lead_loyo_models.png", MAIN_MODELS)
    fig_skill(loyo, "Which information carries the skill? GRU feature-group ablations — leave-one-year-out",
              "04b_skill_by_lead_loyo_ablations.png", ["Climatology"] + ABLATIONS)
    fig_importance(imp)

    # ---- Forward test: could the 2020 shift have been anticipated? ----
    print("\nForward test (train <=2018, test 2019-2025) ...")
    tgt_end = dates + pd.Timedelta(weeks=max(HORIZONS))
    tr = valid & (tgt_end <= pd.Timestamp("2018-12-31"))
    te = valid & (dates >= pd.Timestamp("2019-01-01"))
    fp, _ = run_models(w, f, tr, te, y, "forward")
    rows = []
    for mname, P in fp.items():
        for k, h in enumerate(HORIZONS):
            rows.append(dict(model=mname, horizon=h, **score(y[te, k], P[:, k])))
    fwd = pd.DataFrame(rows)
    clim_b = fwd[fwd.model == "Climatology"].set_index("horizon")["brier"]
    fwd["bss"] = 1 - fwd["brier"] / fwd["horizon"].map(clim_b)
    fwd.to_csv(OUT_T / "skill_forward_test_2019_2025.csv", index=False)
    print(fwd.round(3).to_string(index=False))
    fig_skill(fwd, "Forward test — trained on 2003–2018, evaluated on the 2019–2025 regime shift",
              "07_skill_by_lead_forward.png", MAIN_MODELS + ["GRU – upstream + season only"])
    fig_forward(w, te, y, fp)

    # ---- Final models on all data + outlook from the latest observed week ----
    print("\nFitting final models on 2003-2025 ...")
    fitted = {}
    last = np.zeros(len(w), bool)
    last[np.where(m)[0][-1]] = True
    outlook, _ = run_models(w, f, valid, last, y, "final", fitted=fitted)
    issued = w.index[last][0]
    out = pd.DataFrame({mname: P[0] for mname, P in outlook.items()}, index=[f"{h} weeks" for h in HORIZONS]).T
    out.columns.name = f"P(extreme week) at lead, issued {issued.date()}"
    out.to_csv(OUT_T / "outlook_from_latest_week.csv")
    print(out.round(2).to_string())
    mdir = HERE / "outputs" / "models"
    mdir.mkdir(parents=True, exist_ok=True)
    torch.save(fitted.pop("gru"), mdir / "gru_all_features.pt")
    import joblib
    joblib.dump(dict(fitted, horizons=HORIZONS, threshold_km2=thr, feature_groups=FEATURE_GROUPS),
                mdir / "tabular_models.joblib")

    json.dump(dict(threshold_km2=float(thr), n_onsets=len(onsets), regime_onset=str(regime_start.date()),
                   outlook_issued=str(issued.date())),
              open(OUT_T / "summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
