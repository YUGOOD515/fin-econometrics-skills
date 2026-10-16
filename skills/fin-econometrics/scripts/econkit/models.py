import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from .common import fail, ALPHA
from .data import prepare

COV_ALIASES = {"classic": "nonrobust", "nonrobust": "nonrobust", "ols": "nonrobust",
               "robust": "HC1", "hc1": "HC1", "hc0": "HC0", "hc2": "HC2", "hc3": "HC3",
               "hac": "HAC", "newey": "HAC", "cluster": "cluster"}


def nw_lags(T):
    """Newey-West (1994) rule of thumb: floor(4*(T/100)^(2/9))."""
    return max(1, int(math.floor(4 * (T / 100.0) ** (2.0 / 9.0))))


def _stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def _coef_rows(params, se, tvals, pvals, ci=None):
    rows = []
    for n in params.index:
        r = {"name": n, "coef": params[n], "se": se[n], "t": tvals[n], "p": pvals[n],
             "stars": _stars(pvals[n])}
        if ci is not None:
            r["ci_low"], r["ci_high"] = ci.loc[n].iloc[0], ci.loc[n].iloc[1]
        rows.append(r)
    return rows


def _design(d, x, const=True):
    X = d[x].astype(float)
    return sm.add_constant(X, has_constant="add") if const else X


def _sm_cov(cov, d, cluster=None, lags=None, T=None):
    ct = COV_ALIASES.get((cov or "nonrobust").lower())
    if ct is None:
        fail(f"unknown --cov {cov}", allowed=sorted(COV_ALIASES))
    kw = {}
    if ct == "HAC":
        kw = {"maxlags": int(lags or nw_lags(T or len(d))), "use_correction": True}
    if ct == "cluster":
        if not cluster:
            fail("--cov cluster needs --cluster <var>")
        kw = {"groups": pd.factorize(d[cluster])[0]}
    return ct, kw


def _pack_sm(res, model, label, cov_label, dep, extra=None):
    out = {"ok": True, "label": label, "model": model, "cov_type": cov_label, "dep": dep,
           "nobs": int(res.nobs),
           "coefs": _coef_rows(res.params, res.bse, res.tvalues, res.pvalues, res.conf_int()),
           "stats": {}}
    for k, attr in (("r2", "rsquared"), ("adj_r2", "rsquared_adj"), ("aic", "aic"), ("bic", "bic"),
                    ("loglik", "llf"), ("pseudo_r2", "prsquared")):
        if hasattr(res, attr):
            try:
                out["stats"][k] = float(getattr(res, attr))
            except Exception:
                pass
    try:
        out["stats"]["f"] = float(np.squeeze(res.fvalue)); out["stats"]["f_p"] = float(res.f_pvalue)
    except Exception:
        pass
    try:
        out["stats"]["rmse"] = float(np.sqrt(res.mse_resid))
    except Exception:
        pass
    if extra:
        out.update(extra)
    return out


# ---------------- cross-section / time-series linear models ----------------

def fit_ols(a, df, weights=None):
    d, info = prepare(df, a.y, a.x, [a.cluster, weights] if (a.cluster or weights) else [],
                      a.time, a.entity, a.query)
    X = _design(d, a.x, not a.noconst)
    ct, kw = _sm_cov(a.cov, d, a.cluster, a.lags, len(d))
    if weights:
        m = sm.WLS(d[a.y], X, weights=d[weights])
        name = "wls"
    else:
        m = sm.OLS(d[a.y], X)
        name = "ols"
    res = m.fit(cov_type=ct, cov_kwds=kw or None)
    extra = {"sample": info}
    if ct == "HAC":
        extra["hac_lags"] = kw["maxlags"]
    return _pack_sm(res, name, a.label, ct, a.y, extra), res, d


def fit_fgls_het(a, df):
    """Feasible GLS for heteroskedasticity: ln(e^2) = Z*g (+const) -> w = 1/exp(fitted)."""
    hv = a.het_vars or a.x
    d, info = prepare(df, a.y, a.x, hv, a.time, a.entity, a.query)
    X = _design(d, a.x, not a.noconst)
    e = sm.OLS(d[a.y], X).fit().resid
    Z = d[hv].astype(float)
    if not a.het_noconst:
        Z = sm.add_constant(Z, has_constant="add")
    aux = sm.OLS(np.log(e ** 2), Z).fit()
    w = 1.0 / np.exp(aux.fittedvalues)
    ct, kw = _sm_cov(a.cov, d, a.cluster, a.lags, len(d))
    res = sm.WLS(d[a.y], X, weights=w).fit(cov_type=ct, cov_kwds=kw or None)
    return _pack_sm(res, "fgls_het", a.label, ct, a.y,
                    {"sample": info, "variance_function": {"z": hv, "const": not a.het_noconst,
                                                           "aux_r2": aux.rsquared}}), res, d


def fit_ar1(a, df, method="pw"):
    """Prais-Winsten (pw) or Cochrane-Orcutt (co) iterated FGLS for AR(1) errors."""
    if not a.time:
        fail("AR(1) FGLS needs --time")
    d, info = prepare(df, a.y, a.x, [], a.time, a.entity, a.query)
    y = d[a.y].astype(float).values
    X = _design(d, a.x, not a.noconst)
    names = list(X.columns); Xv = X.values
    b = np.linalg.lstsq(Xv, y, rcond=None)[0]
    rho = 0.0
    for _ in range(200):
        e = y - Xv @ b
        rho_new = float(np.sum(e[1:] * e[:-1]) / np.sum(e[:-1] ** 2))
        ys, Xs = y[1:] - rho_new * y[:-1], Xv[1:] - rho_new * Xv[:-1]
        if method == "pw":
            s = math.sqrt(max(1e-12, 1 - rho_new ** 2))
            ys = np.r_[s * y[0], ys]; Xs = np.vstack([s * Xv[0], Xs])
        b_new = np.linalg.lstsq(Xs, ys, rcond=None)[0]
        conv = abs(rho_new - rho) < 1e-8
        rho, b = rho_new, b_new
        if conv:
            break
    res = sm.OLS(ys, pd.DataFrame(Xs, columns=names)).fit()
    eps = ys - Xs @ b
    dw = float(np.sum(np.diff(eps) ** 2) / np.sum(eps ** 2))
    return _pack_sm(res, "prais_winsten" if method == "pw" else "cochrane_orcutt", a.label,
                    "nonrobust", a.y, {"sample": info, "rho": rho, "dw_transformed": dw}), res, d


# ---------------- instrumental variables ----------------

def fit_iv(a, df, estimator="2sls"):
    from linearmodels.iv import IV2SLS, IVLIML, IVGMM
    if not a.endog or not a.instruments:
        fail("IV needs --endog and --instruments")
    exog = [v for v in a.x if v not in a.endog]
    d, info = prepare(df, a.y, exog, [*a.endog, *a.instruments, *([a.cluster] if a.cluster else [])],
                      a.time, a.entity, a.query)
    X = _design(d, exog, not a.noconst)
    cov = (a.cov or "robust").lower()
    lm_cov = {"nonrobust": "unadjusted", "classic": "unadjusted", "robust": "robust", "hc1": "robust",
              "hac": "kernel", "cluster": "clustered"}.get(cov, "robust")
    kw = {}
    if lm_cov == "kernel":
        kw = {"bandwidth": a.lags or nw_lags(len(d))}
    if lm_cov == "clustered":
        kw = {"clusters": pd.factorize(d[a.cluster])[0]}
    cls = {"2sls": IV2SLS, "liml": IVLIML, "gmm": IVGMM}[estimator]
    m = cls(d[a.y], X, d[a.endog], d[a.instruments])
    res = m.fit(cov_type=lm_cov, **kw) if estimator != "gmm" else m.fit(cov_type=lm_cov if lm_cov != "unadjusted" else "robust", **kw)
    ci = res.conf_int()
    out = {"ok": True, "label": a.label, "model": f"iv_{estimator}", "cov_type": lm_cov, "dep": a.y,
           "nobs": int(res.nobs), "endog": a.endog, "instruments": a.instruments,
           "coefs": _coef_rows(res.params, res.std_errors, res.tstats, res.pvalues, ci),
           "stats": {"r2": float(res.rsquared)}, "sample": info, "iv_diagnostics": {}}
    dg = out["iv_diagnostics"]
    # first stage (computed with 2SLS regardless of estimator)
    try:
        fs = IV2SLS(d[a.y], X, d[a.endog], d[a.instruments]).fit(cov_type=lm_cov, **kw).first_stage.diagnostics
        dg["first_stage"] = {e: {"partial_r2": float(fs.loc[e, "partial.rsquared"]),
                                 "shea_r2": float(fs.loc[e, "shea.rsquared"]),
                                 "f_stat": float(fs.loc[e, "f.stat"]), "f_p": float(fs.loc[e, "f.pval"])}
                             for e in fs.index}
        minf = min(v["f_stat"] for v in dg["first_stage"].values())
        dg["weak_iv_flag"] = bool(minf < 10)
        dg["weak_iv_rule"] = "first-stage F < 10 => weak instruments (Staiger-Stock); prefer LIML / report AR-robust inference"
    except Exception as e:
        dg["first_stage_error"] = str(e)
    # Stata-style (ivreg2) weak-ID battery: Sanderson-Windmeijer F per endog, Cragg-Donald, Kleibergen-Paap, AR
    if lm_cov != "kernel":
        try:
            from . import hdfe
            vce = {"unadjusted": "iid", "robust": "robust", "clustered": "cluster"}[lm_cov]
            r = hdfe.estimate(d, a.y, exog, a.endog, a.instruments, fe=None, vce=vce,
                              cluster=[[a.cluster]] if vce == "cluster" else None, convention="ivreg2",
                              constant=not a.noconst, small=True, estimator="liml" if estimator == "liml" else "2sls")
            iv = r.get("iv", {})
            keep = {k: iv[k] for k in ("cragg_donald_F", "kp_rk_LM", "kp_rk_LM_p", "kp_rk_Wald_F", "AR_F", "AR_p",
                                        "endogeneity_test") if k in iv}
            keep["first_stage"] = [{k: f[k] for k in ("endog", "partial_F", "SW_F", "partial_R2")}
                                   for f in iv.get("first_stage", [])]
            lmfs = dg.get("first_stage", {})
            dg["first_stage"] = {f["endog"]: {"partial_F": f["partial_F"], "SW_F": f["SW_F"], "partial_r2": f["partial_R2"],
                                              "shea_r2": lmfs.get(f["endog"], {}).get("shea_r2")} for f in keep.pop("first_stage")}
            dg["first_stage_note"] = "partial_F / SW_F follow ivreg2 (robust/cluster-consistent, small-sample scaled)"
            dg.update(keep)
            if "weak_iv_flag" in iv:
                sw_min = min([v["SW_F"] for v in dg["first_stage"].values()] or [np.inf])
                dg["weak_iv_flag"] = bool(iv["weak_iv_flag"] or sw_min < 10)
                dg["weak_iv_rule"] = ("weak-ID: Kleibergen-Paap rk Wald F (robust/cluster) or Cragg-Donald F (iid) < 10, "
                                      "or any Sanderson-Windmeijer F < 10 with several endogenous regressors; "
                                      "compare with Stock-Yogo critical values; then prefer LIML and report AR-robust inference")
        except Exception as e:
            dg["stata_style_error"] = str(e)[:200]
    n_over = len(a.instruments) - len(a.endog)
    dg["overid_df"] = n_over
    if n_over > 0:
        for name, attr in (("sargan", "sargan"), ("wooldridge_score", "wooldridge_overid")):
            try:
                t = getattr(res, attr) if estimator != "gmm" else res.j_stat
                dg[name] = {"stat": float(t.stat), "p": float(t.pval)}
            except Exception:
                pass
        if estimator == "gmm":
            dg["hansen_j"] = {"stat": float(res.j_stat.stat), "p": float(res.j_stat.pval)}
    if estimator == "2sls":
        for name, fn in (("durbin", "durbin"), ("wu_hausman", "wu_hausman")):
            try:
                t = getattr(res, fn)()
                dg[name] = {"stat": float(t.stat), "p": float(t.pval)}
            except Exception:
                pass
        try:
            t = res.wooldridge_regression
            dg["wooldridge_regression_endog"] = {"stat": float(t.stat), "p": float(t.pval),
                                                 "note": "heteroskedasticity-robust endogeneity test"}
        except Exception:
            pass
    return out, res, d


# ---------------- panel data ----------------

def _panel_frame(a, df):
    if not (a.entity and a.time):
        fail("panel models need --entity and --time")
    d, info = prepare(df, a.y, a.x, [], a.time, a.entity, a.query)
    d = d.set_index([a.entity, a.time])
    return d, info


def fit_panel(a, df, kind="fe"):
    from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS
    d, info = _panel_frame(a, df)
    X = d[a.x].astype(float)
    cov = (a.cov or "cluster").lower()
    kw = {}
    if cov == "cluster":
        lm_cov, kw = "clustered", {"cluster_entity": True}
    elif cov in ("robust", "hc1"):
        lm_cov = "robust"
    elif cov == "driscoll-kraay" or cov == "hac":
        lm_cov = "kernel"
    else:
        lm_cov = "unadjusted"
    if kind in ("fe", "twfe"):
        m = PanelOLS(d[a.y], X, entity_effects=True, time_effects=(kind == "twfe"), drop_absorbed=True)
    elif kind == "re":
        m = RandomEffects(d[a.y], sm.add_constant(X))
    else:
        m = PooledOLS(d[a.y], sm.add_constant(X))
    res = m.fit(cov_type=lm_cov, **kw)
    out = {"ok": True, "label": a.label, "model": f"panel_{kind}", "cov_type": lm_cov, "dep": a.y,
           "nobs": int(res.nobs), "entities": int(res.entity_info["total"]),
           "coefs": _coef_rows(res.params, res.std_errors, res.tstats, res.pvalues, res.conf_int()),
           "stats": {"r2_within": float(getattr(res, "rsquared_within", np.nan)),
                     "r2_between": float(getattr(res, "rsquared_between", np.nan)),
                     "r2_overall": float(getattr(res, "rsquared_overall", np.nan))},
           "sample": info}
    if kind == "fe":
        try:
            out["poolability_f"] = {"stat": float(res.f_pooled.stat), "p": float(res.f_pooled.pval),
                                    "note": "H0: all entity effects = 0 (pooled OLS adequate)"}
        except Exception:
            pass
    return out, res, d


def hausman_fe_re(a, df):
    from linearmodels.panel import PanelOLS, RandomEffects
    d, info = _panel_frame(a, df)
    X = d[a.x].astype(float)
    fe = PanelOLS(d[a.y], X, entity_effects=True).fit()
    re = RandomEffects(d[a.y], sm.add_constant(X)).fit()
    common = [c for c in fe.params.index if c in re.params.index]
    b = fe.params[common] - re.params[common]
    V = fe.cov.loc[common, common] - re.cov.loc[common, common]
    stat = float(b.values @ np.linalg.pinv(V.values) @ b.values)
    p = float(1 - stats.chi2.cdf(stat, len(common)))
    return {"ok": True, "test": "hausman_fe_vs_re", "stat": stat, "df": len(common), "p": p,
            "decision": "use FE (RE inconsistent)" if p < ALPHA else "RE not rejected (RE efficient); FE still consistent",
            "note": "classic Hausman uses non-robust covariances; with clustering prefer Mundlak/auxiliary test"}


# ---------------- binary choice ----------------

def fit_binary(a, df, kind="logit"):
    d, info = prepare(df, a.y, a.x, [a.cluster] if a.cluster else [], a.time, a.entity, a.query)
    X = _design(d, a.x, not a.noconst)
    M = sm.Logit if kind == "logit" else sm.Probit
    ct, kw = _sm_cov(a.cov or "robust", d, a.cluster)
    res = M(d[a.y], X).fit(disp=0, cov_type=ct, cov_kwds=kw or None)
    me = res.get_margeff(at="overall").summary_frame()
    out = _pack_sm(res, kind, a.label, ct, a.y, {"sample": info})
    out["marginal_effects"] = [{"name": i, "dydx": r.iloc[0], "se": r.iloc[1], "p": r.iloc[3]} for i, r in me.iterrows()]
    pred = (res.predict(X) > 0.5).astype(int)
    out["stats"]["pct_correct"] = float((pred == d[a.y]).mean())
    return out, res, d


def fit(a, df):
    m = a.model.lower()
    if m in ("ols", "reg"):
        return fit_ols(a, df)
    if m == "wls":
        if not a.weights:
            fail("--model wls needs --weights <column>")
        return fit_ols(a, df, weights=a.weights)
    if m in ("fgls", "fgls_het"):
        return fit_fgls_het(a, df)
    if m in ("prais", "pw"):
        return fit_ar1(a, df, "pw")
    if m in ("corc", "co"):
        return fit_ar1(a, df, "co")
    if m in ("iv", "2sls", "iv2sls"):
        return fit_iv(a, df, "2sls")
    if m == "liml":
        return fit_iv(a, df, "liml")
    if m in ("gmm", "ivgmm"):
        return fit_iv(a, df, "gmm")
    if m in ("fe", "re", "pooled", "twfe"):
        return fit_panel(a, df, m)
    if m in ("logit", "probit"):
        return fit_binary(a, df, m)
    fail(f"unknown --model {a.model}",
         allowed=["ols", "wls", "fgls", "prais", "corc", "2sls", "liml", "gmm", "fe", "twfe", "re", "pooled", "logit", "probit"])
