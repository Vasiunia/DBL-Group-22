"""
Research question 12.1 — Step 3: a simple, physically constrained logistic regression.

Same target, horizons and validation as train_models.py, but only a handful of physical features:
current flood state, seasonality, Lake Albert anomaly, Lake Albert 26-week rise and Sobat rain.
The Lake Albert coefficients are forced to be >= 0 (more water upstream can only raise flood risk).
An unconstrained version with the same features is fitted as well, to isolate the effect of the constraint.

Question: does this model anticipate the 2022-25 extremes in the forward test (train <=2018), where the GRU failed?

Run from the raw_data folder:  python extreme_flood_early_warning/physical_model.py   (~1 min)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit

import train_models as T

FEATURES = ["log_unusual", "log_recurring", "unusual_share", "woy_sin", "woy_cos",
            "albert_anom", "albert_d26", "tp_sobat_a13"]
POSITIVE = ["albert_anom", "albert_d26"]
L2 = 1.0  # same strength as sklearn's C=1.0 used for the persistence baseline


class ConstrainedLogit:
    """L2-penalised logistic regression with optional lower bound 0 on selected coefficients."""

    def __init__(self, positive=()):
        self.positive = positive

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9   # standardising keeps coefficient signs
        Z = (X - self.mu) / self.sd
        n, p = Z.shape

        def loss(b):
            z = b[0] + Z @ b[1:]
            pr = expit(z)
            nll = np.sum(np.logaddexp(0, z) - y * z)
            g = pr - y
            return nll + 0.5 * L2 * b[1:] @ b[1:], np.r_[g.sum(), Z.T @ g + L2 * b[1:]]

        bounds = [(None, None)] + [(0, None) if c in self.positive else (None, None) for c in FEATURES]
        self.b = minimize(loss, np.zeros(p + 1), jac=True, method="L-BFGS-B", bounds=bounds).x
        return self

    def predict_proba(self, X):
        return expit(self.b[0] + ((X - self.mu) / self.sd) @ self.b[1:])


def fit_predict(f, y, tr, te, positive):
    X = f[FEATURES].fillna(0).values
    P = np.zeros((te.sum(), len(T.HORIZONS)))
    coefs = {}
    for k, h in enumerate(T.HORIZONS):
        ok = tr & ~np.isnan(y[:, k])
        mdl = ConstrainedLogit(positive).fit(X[ok], y[ok, k])
        P[:, k] = mdl.predict_proba(X[te])
        coefs[h] = dict(zip(FEATURES, mdl.b[1:]))
    return P, pd.DataFrame(coefs)


def skill(y, P, clim_brier):
    rows = []
    for k, h in enumerate(T.HORIZONS):
        s = T.score(y[:, k], P[:, k])
        rows.append(dict(horizon=h, auc=s["auc"], brier=s["brier"], bss=1 - s["brier"] / clim_brier[h]))
    return pd.DataFrame(rows)


def main():
    w, f, _ = T.build_weekly()
    y = T.make_targets(w)
    dates = w.index
    m = (dates >= T.START) & (dates <= T.END)
    valid = m & (np.arange(len(w)) >= T.LOOKBACK - 1) & ~np.all(np.isnan(y), axis=1)
    models = {"Physical logistic – Albert ≥ 0": POSITIVE, "Physical logistic – unconstrained": ()}

    # ---- Forward test, identical split to train_models.py ----
    tgt_end = dates + pd.Timedelta(weeks=max(T.HORIZONS))
    tr = valid & (tgt_end <= pd.Timestamp("2018-12-31"))
    te = valid & (dates >= pd.Timestamp("2019-01-01"))
    fwd_prev = pd.read_csv(T.OUT_T / "skill_forward_test_2019_2025.csv")
    clim_b = fwd_prev[fwd_prev.model == "Climatology"].set_index("horizon")["brier"]

    fwd, preds = [], {}
    for name, pos in models.items():
        P, coefs = fit_predict(f, y, tr, te, pos)
        preds[name] = P
        fwd.append(skill(y[te], P, clim_b).assign(model=name))
        coefs.to_csv(T.OUT_T / f"physical_logit_coefficients_forward_{'constrained' if pos else 'free'}.csv")
        print(f"\n{name} — standardised coefficients (forward fit, train <=2018):\n", coefs.round(2).to_string())
    fwd = pd.concat(fwd)
    keep = ["Persistence (logistic, flood state + season)", "GRU – all features", T.ENSEMBLE]
    comp = pd.concat([fwd_prev[fwd_prev.model.isin(keep)][["model", "horizon", "auc", "brier", "bss"]], fwd])
    comp.to_csv(T.OUT_T / "skill_forward_test_physical_vs_others.csv", index=False)
    print("\nForward test 2019-2025:\n", comp.pivot(index="model", columns="horizon", values="bss").round(3).to_string())
    print("\n", comp.pivot(index="model", columns="horizon", values="auc").round(3).to_string())

    # Skill on the persistent 2022-25 extremes only (the part the GRU missed)
    late = dates[te] >= pd.Timestamp("2022-01-01")
    rows = []
    for name, P in preds.items():
        for k, h in enumerate(T.HORIZONS):
            yy = y[te][late, k]
            ok = ~np.isnan(yy)
            rows.append(dict(model=name, horizon=h, hit_rate_2022_25=np.mean(P[late][ok, k][yy[ok] == 1] > 0.5)))
    print("\nShare of 2022-25 extreme weeks forecast with P > 0.5:\n",
          pd.DataFrame(rows).pivot(index="model", columns="horizon", values="hit_rate_2022_25").round(2).to_string())

    # ---- Leave-one-year-out, for comparison with the main table ----
    loyo_prev = pd.read_csv(T.OUT_T / "skill_leave_one_year_out.csv")
    clim_l = loyo_prev[loyo_prev.model == "Climatology"].set_index("horizon")["brier"]
    loyo = []
    for name, pos in models.items():
        oof = np.full((len(w), len(T.HORIZONS)), np.nan)
        for _, tr_, te_ in T.make_folds(dates, valid):
            oof[te_] = fit_predict(f, y, tr_, te_, pos)[0]
        loyo.append(skill(y[valid], oof[valid], clim_l).assign(model=name))
    loyo = pd.concat(loyo)
    loyo.to_csv(T.OUT_T / "skill_loyo_physical_logit.csv", index=False)
    print("\nLeave-one-year-out:\n", loyo.pivot(index="model", columns="horizon", values="bss").round(3).to_string())

    fig_forward(w, te, y, preds)


def fig_forward(w, te, y, preds):
    dates = w.index[te]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True)
    for ax, k in zip(axes, [2, 3]):
        h = T.HORIZONS[k]
        td = dates + pd.Timedelta(weeks=h)
        for i, (s, e) in enumerate(T._runs(np.nan_to_num(y[te][:, k]).astype(int))):
            ax.axvspan(td[s], td[e], color=T.C["red"], alpha=0.08, lw=0, label="Observed extreme week" if i == 0 else None)
        ax.plot(td, preds["Physical logistic – Albert ≥ 0"][:, k], color=T.C["violet"], lw=1.6,
                label="Physical logistic – Albert ≥ 0")
        ax.plot(td, preds["Physical logistic – unconstrained"][:, k], color=T.C["aqua"], lw=1.4,
                label="Physical logistic – unconstrained")
        ax.axhline(0.5, color="#c3c2b7", lw=1, ls="--")
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("P(extreme)")
        ax.set_title(f"Forward test — trained on 2003–2018 only — forecast issued {h} weeks ahead")
        ax.legend(loc="upper left", fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(T.OUT_F / "09_forward_test_physical_logit.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
