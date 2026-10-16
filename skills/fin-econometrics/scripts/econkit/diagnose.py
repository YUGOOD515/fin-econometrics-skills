"""Run baseline OLS and the standard battery of diagnostics, then emit
rule-based recommendations. Rules are documented in references/decision-rules.md."""
import os, math
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_white, acorr_breusch_godfrey, acorr_ljungbox, linear_reset
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller, kpss, coint
from .common import ALPHA, ensure_dir
from .data import prepare, LAG_RE
from .models import _pack_sm, nw_lags


def _lm_test(e2, Z):
    """N*R^2 LM test of e^2 on Z (Koenker/'iid' BP)."""
    Z = sm.add_constant(np.asarray(Z, float), has_constant="add")
    r = sm.OLS(e2, Z).fit()
    lm = len(e2) * r.rsquared
    df = Z.shape[1] - 1
    return {"stat": lm, "df": df, "p": float(1 - stats.chi2.cdf(lm, df))}


def _stationarity(s):
    s = pd.Series(s).dropna().astype(float)
    out = {}
    try:
        adf = adfuller(s, autolag="AIC")
        out["adf_stat"], out["adf_p"], out["adf_lags"] = adf[0], adf[1], adf[2]
    except Exception as e:
        out["adf_error"] = str(e)
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            k = kpss(s, regression="c", nlags="auto")
        out["kpss_stat"], out["kpss_p"] = k[0], k[1]
    except Exception as e:
        out["kpss_error"] = str(e)
    ur = out.get("adf_p", 0) > ALPHA
    st = out.get("kpss_p", 1) < ALPHA
    out["verdict"] = ("unit_root" if ur and st else "stationary" if (not ur and not st)
                      else "inconclusive")
    return out


def _plots(res, d, a, out_dir, ts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    p = ensure_dir(os.path.join(out_dir, "plots"))
    paths = []
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(res.fittedvalues, res.resid, s=12, color="#3E5C76", alpha=.7)
    ax[0].axhline(0, color="#B8913A", lw=1); ax[0].set_xlabel("fitted"); ax[0].set_ylabel("residual")
    ax[0].set_title("Residuals vs fitted")
    sm.qqplot(res.resid, line="s", ax=ax[1], markersize=3, color="#3E5C76"); ax[1].set_title("Normal Q-Q")
    if len(ax[1].get_lines()) > 1:
        ax[1].get_lines()[1].set_color("#B8913A")
    fig.tight_layout(); f = os.path.join(p, f"{a.label}_resid.png"); fig.savefig(f, dpi=120); plt.close(fig)
    paths.append(f)
    if ts:
        from statsmodels.graphics.tsaplots import plot_acf
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        ax[0].plot(d[a.time].values, res.resid.values, marker="o", ms=3, color="#3E5C76")
        ax[0].axhline(0, color="#B8913A", lw=1); ax[0].set_title("Residuals over time")
        plot_acf(res.resid, ax=ax[1], lags=min(12, len(d) // 2 - 1)); ax[1].set_title("Residual ACF")
        fig.tight_layout(); f = os.path.join(p, f"{a.label}_acf.png"); fig.savefig(f, dpi=120); plt.close(fig)
        paths.append(f)
    return paths


def run(a, df):
    ts = bool(a.time) and not a.entity
    panel = bool(a.time and a.entity)
    d, info = prepare(df, a.y, a.x, [a.cluster] if a.cluster else [], a.time, a.entity, a.query)
    X = sm.add_constant(d[a.x].astype(float), has_constant="add")
    y = d[a.y].astype(float)
    res = sm.OLS(y, X).fit()
    n, k = X.shape
    e = res.resid
    T = {}
    flags, rec = [], []

    # ---- functional form / normality / collinearity / influence
    try:
        r = linear_reset(res, power=4, use_f=True)
        T["reset"] = {"stat": float(r.fvalue), "p": float(r.pvalue), "note": "Ramsey RESET with yhat^2..yhat^4 (Stata ovtest)"}
    except Exception as ex:
        T["reset"] = {"error": str(ex)}
    jb = jarque_bera(e)
    T["jarque_bera"] = {"stat": float(jb[0]), "p": float(jb[1]), "skew": float(jb[2]), "kurtosis": float(jb[3])}
    if len(a.x) > 1:
        T["vif"] = {c: float(variance_inflation_factor(X.values, i)) for i, c in enumerate(X.columns) if c != "const"}
    infl = res.get_influence()
    cd = infl.cooks_distance[0]
    big = np.where(cd > 4 / n)[0]
    T["influence"] = {"cooks_threshold": 4 / n, "n_influential": int(len(big)),
                      "top": [{"row": int(d.index[i]), "cooks_d": float(cd[i])} for i in np.argsort(-cd)[:5]]}

    # ---- heteroskedasticity
    e2 = e.values ** 2
    T["bp_fitted"] = _lm_test(e2, res.fittedvalues.values.reshape(-1, 1))
    T["bp_fitted"]["note"] = "BP/Koenker on fitted values (Stata: estat hettest, iid)"
    T["bp_rhs"] = _lm_test(e2, d[a.x].values)
    T["bp_rhs"]["note"] = "BP/Koenker on all regressors (Stata: estat hettest, iid rhs)"
    try:
        w = het_white(e, X)
        T["white"] = {"stat": float(w[0]), "p": float(w[1]), "note": "White test with squares and cross products"}
    except Exception as ex:
        T["white"] = {"error": str(ex)}
    het = min(T["bp_rhs"]["p"], T.get("white", {}).get("p", 1)) < ALPHA

    # ---- autocorrelation (time series or panel)
    auto = False
    if ts:
        T["durbin_watson"] = float(durbin_watson(e))
        lag_list = sorted(set([1, min(4, max(1, n // 5))]))
        T["breusch_godfrey"] = {}
        for L in lag_list:
            bg = acorr_breusch_godfrey(res, nlags=L)
            T["breusch_godfrey"][f"lag{L}"] = {"lm": float(bg[0]), "p": float(bg[1])}
        qlag = max(1, min(int(n / 2) - 2, 40))
        lb = acorr_ljungbox(e, lags=[qlag], return_df=True)
        T["ljung_box"] = {"lags": qlag, "q": float(lb["lb_stat"].iloc[0]), "p": float(lb["lb_pvalue"].iloc[0])}
        T["acf_resid_lag1"] = float(pd.Series(e.values).autocorr(1))
        auto = min(v["p"] for v in T["breusch_godfrey"].values()) < ALPHA
        # stationarity of levels
        bases = []
        for v in [a.y, *a.x]:
            m = LAG_RE.match(v)
            b = m.group(3) if m else v
            if b not in bases:
                bases.append(b)
        src = df.sort_values(a.time)
        T["stationarity"] = {b: _stationarity(src[b]) for b in bases}
        nonst = [b for b, r in T["stationarity"].items() if r["verdict"] == "unit_root"]
        if a.y in nonst or (LAG_RE.match(a.y) and LAG_RE.match(a.y).group(3) in nonst):
            try:
                c = coint(y, d[a.x].astype(float), trend="c")
                T["engle_granger"] = {"stat": float(c[0]), "p": float(c[1]),
                                      "note": "H0: no cointegration among y and x"}
            except Exception as ex:
                T["engle_granger"] = {"error": str(ex)}

    if panel:  # Wooldridge (2002) test for AR(1) in panel errors + within-unit persistence
        try:
            import statsmodels.api as _sm
            P = d[[a.entity, a.time, a.y, *a.x]].dropna().sort_values([a.entity, a.time]).copy()
            g = P.groupby(a.entity)
            consec = (P[a.time] - g[a.time].shift(1)) == 1
            D_ = P[[a.y, *a.x]].astype(float) - g[[a.y, *a.x]].shift(1).astype(float)
            ok = consec & D_.notna().all(axis=1)
            fd = _sm.OLS(D_.loc[ok, a.y], D_.loc[ok, list(a.x)]).fit()
            P.loc[ok, "_e"] = fd.resid
            P["_e1"] = P.groupby(a.entity)["_e"].shift(1)
            c2 = ((P[a.time] - P.groupby(a.entity)[a.time].shift(1)) == 1) & P["_e"].notna() & P["_e1"].notna()
            w = _sm.OLS(P.loc[c2, "_e"], P.loc[c2, ["_e1"]]).fit(cov_type="cluster", cov_kwds={"groups": P.loc[c2, a.entity]})
            F = float(((w.params["_e1"] + 0.5) / w.bse["_e1"]) ** 2)
            G = int(P.loc[c2, a.entity].nunique())
            pw = float(stats.f.sf(F, 1, max(G - 1, 1)))
            yd = P[a.y].astype(float) - g[a.y].transform("mean")
            rho_y = float(pd.Series(yd.values).corr(pd.Series(yd.groupby(P[a.entity]).shift(1).values)))
            T["panel_serial_correlation"] = {"wooldridge_F": F, "df": [1, G - 1], "p": pw, "within_unit_ar1_y": rho_y,
                                             "note": "H0: no first-order autocorrelation in the idiosyncratic errors"}
            auto = auto or pw < ALPHA
        except Exception as ex:
            T["panel_serial_correlation"] = {"error": str(ex)[:160]}

    # ---- endogeneity (only when user/agent specified suspects + instruments)
    if a.endog and a.instruments:
        from .models import fit_iv
        out, _, _ = fit_iv(a, df, "2sls")
        T["endogeneity"] = {k: v for k, v in out["iv_diagnostics"].items()}

    # ---- rules -> recommendations
    def add(issue, severity, evidence, action, command, ref):
        flags.append(issue)
        rec.append({"issue": issue, "severity": severity, "evidence": evidence, "action": action,
                    "command": command, "ref": ref})

    xs = " ".join(a.x)
    base = f"--data {a.data} --y {a.y} --x {xs}" + (f" --time {a.time}" if a.time else "") + (f" --entity {a.entity}" if a.entity else "")
    yv = d[a.y]
    if set(yv.unique()) <= {0, 1}:
        add("binary_outcome", "high", "dependent variable is 0/1", "use logit/probit and report marginal effects; LPM only as robustness",
            f"fit --model logit {base}", "decision-rules.md#binary")
    if het:
        add("heteroskedasticity", "medium",
            f"BP(rhs) p={T['bp_rhs']['p']:.4g}, White p={T.get('white', {}).get('p', float('nan')):.4g}",
            "keep OLS coefficients, report robust (HC1) SE; if variance form is clear and efficiency matters, add FGLS/WLS as comparison",
            f"fit --model ols --cov robust {base}  |  fit --model fgls --het-vars <z> {base}", "decision-rules.md#heteroskedasticity")
    if ts and auto:
        L = nw_lags(n)
        add("autocorrelation", "high",
            f"BG p={min(v['p'] for v in T['breusch_godfrey'].values()):.4g}, DW={T['durbin_watson']:.3f}, rho1={T['acf_resid_lag1']:.3f}",
            f"first check dynamic misspecification (add lags of y/x); then report Newey-West HAC SE (lags={L}) and/or Prais-Winsten FGLS",
            f"fit --model ols --cov hac --lags {L} {base}  |  fit --model prais {base}", "decision-rules.md#autocorrelation")
    if ts and T.get("stationarity"):
        nonst = [b for b, r in T["stationarity"].items() if r["verdict"] == "unit_root"]
        if nonst:
            eg = T.get("engle_granger", {})
            if eg.get("p") is not None and eg["p"] < ALPHA:
                add("unit_root_cointegrated", "medium", f"unit-root: {nonst}; Engle-Granger p={eg['p']:.4g}",
                    "levels regression is a cointegrating relation (super-consistent) but use HAC SE; consider ECM for dynamics",
                    f"fit --model ols --cov hac {base}", "decision-rules.md#nonstationarity")
            elif n >= 50:
                add("spurious_regression_risk", "high", f"unit-root: {nonst}; no evidence of cointegration (T={n})",
                    "do not interpret levels regression; difference the I(1) series (D.x) or use growth rates",
                    "use D.var operators or transform --gen", "decision-rules.md#nonstationarity")
            else:
                add("possible_nonstationarity", "low", f"ADF/KPSS suggest unit root in {nonst}, but T={n} < 50 so tests have low power",
                    "do not over-react: keep levels model if theory says stationary, but add a differenced or detrended specification as a robustness check",
                    "fit ... --x D.var ...", "decision-rules.md#nonstationarity")
    if T["reset"].get("p") is not None and T["reset"]["p"] < ALPHA:
        add("functional_form", "medium", f"RESET p={T['reset']['p']:.4g}",
            "try logs, squares or interactions guided by theory; re-run diagnose", "transform --gen ...", "decision-rules.md#specification")
    if T.get("vif") and max(T["vif"].values()) > 10:
        worst = max(T["vif"], key=T["vif"].get)
        hi = [c for c, v in T["vif"].items() if v > 10]
        # VIF of x and x^2 / x*z terms is high by construction (harmless): flag only if a high-VIF
        # variable is not a transform of another regressor
        def _related(c):
            return any(o != c and len(o) >= 1 and (o in c or c in o) and abs(np.corrcoef(d[o].astype(float), d[c].astype(float))[0, 1]) > .8
                       for o in a.x if o in d and c in d)
        if hi and all(_related(c) for c in hi):
            T["vif_note"] = f"high VIF only among {hi}: polynomial/interaction terms built from the same variable; collinearity by construction, not a problem (test them jointly)"
        else:
            add("multicollinearity", "low", f"max VIF={T['vif'][worst]:.1f} ({worst})",
                "coefficients imprecise but unbiased; do not drop theory-relevant controls mechanically; report joint tests",
                "", "decision-rules.md#multicollinearity")
    if T["influence"]["n_influential"] > max(3, 0.05 * n):
        add("influential_obs", "low", f"{T['influence']['n_influential']} obs with Cook's D > 4/n",
            "inspect top rows; re-estimate without them as robustness (never silently delete)", "", "decision-rules.md#outliers")
    psc = T.get("panel_serial_correlation", {})
    if panel and psc.get("p") is not None and psc["p"] < ALPHA:
        add("panel_serial_correlation", "high",
            f"Wooldridge F={psc['wooldridge_F']:.2f}, p={psc['p']:.3g}; within-unit AR(1) of y={psc['within_unit_ar1_y']:.2f}",
            "errors are serially correlated within units: iid/robust SE overstate precision - cluster by entity (or a coarser level)",
            f"fit --model twfe --cov cluster --cluster {a.entity} {base}", "decision-rules.md#panel")
    if panel:
        add("panel_structure", "high", f"entity={a.entity}, time={a.time}",
            "estimate FE and RE, run Hausman; cluster SE by entity; consider two-way FE",
            f"fit --model fe {base}  |  hausman {base}", "decision-rules.md#panel")
    if "endogeneity" in T:
        en = T["endogeneity"]
        p_end = en.get("wooldridge_regression_endog", en.get("wu_hausman", {})).get("p")
        if p_end is not None and p_end < ALPHA:
            add("endogeneity", "high", f"endogeneity test p={p_end:.4g}", "prefer IV estimates; check weak-IV and over-id",
                f"fit --model 2sls --endog {' '.join(a.endog)} --instruments {' '.join(a.instruments)} {base}", "decision-rules.md#endogeneity")
        if en.get("weak_iv_flag"):
            add("weak_instruments", "high", "first-stage F < 10", "use LIML and weak-IV robust inference; seek stronger instruments",
                "fit --model liml ...", "decision-rules.md#endogeneity")
    elif not (a.endog and a.instruments):
        rec.append({"issue": "endogeneity_check", "severity": "info",
                    "evidence": "OLS diagnostics cannot detect endogeneity",
                    "action": "argue from theory: omitted variables (e.g. ability), simultaneity, measurement error. If a key regressor is plausibly endogenous and instruments exist, re-run diagnose with --endog/--instruments",
                    "command": f"diagnose {base} --endog <x> --instruments <z1 z2>", "ref": "decision-rules.md#endogeneity"})
    if not flags:
        rec.append({"issue": "none", "severity": "info", "evidence": "no test rejects at 5%",
                    "action": "OLS with robust SE is adequate", "command": f"fit --model ols --cov robust {base}", "ref": ""})

    ols = _pack_sm(res, "ols", a.label, "nonrobust", a.y, {"sample": info})
    plots = [] if a.no_plots else _plots(res, d, a, a.out_dir, ts)
    return {"ok": True, "label": a.label, "data_structure": "panel" if panel else "time_series" if ts else "cross_section",
            "baseline_ols": ols, "tests": T, "flags": flags, "recommendations": rec, "plots": plots}
