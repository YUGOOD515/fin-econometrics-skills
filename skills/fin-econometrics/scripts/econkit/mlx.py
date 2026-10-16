"""Extra Stata estimators for the do-file engine: newey, qreg, tobit, ologit/oprobit, mlogit,
clogit/xtlogit-fe, and psmatch2 (approximate). Inputs are clean numpy arrays; outputs follow the
stata_estimates.json schema (coefs dict of {coef, se, t, p, ci_low, ci_high})."""
import numpy as np
import pandas as pd
from scipy import stats, optimize


def _rows(names, b, se, df_r=None):
    b, se = np.asarray(b, float), np.asarray(se, float)
    t = b / se
    if df_r:
        p = 2 * stats.t.sf(np.abs(t), df_r); c = stats.t.ppf(.975, df_r)
    else:
        p = 2 * stats.norm.sf(np.abs(t)); c = 1.959963984540054
    return {n: {"coef": b[i], "se": se[i], "t": t[i], "p": p[i], "ci_low": b[i] - c * se[i], "ci_high": b[i] + c * se[i]}
            for i, n in enumerate(names)}


def _sandwich(Hinv, S, vce, cl=None):
    N = S.shape[0]
    if vce == "robust":
        return Hinv @ (S.T @ S) @ Hinv * N / (N - 1)
    if vce == "cluster":
        codes = pd.factorize(cl)[0]; G = codes.max() + 1
        Sg = np.zeros((G, S.shape[1])); np.add.at(Sg, codes, S)
        return Hinv @ (Sg.T @ Sg) @ Hinv * G / (G - 1)
    return Hinv


def num_hess(f, x, eps=1e-5):
    k = len(x); H = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k); ej = np.zeros(k); ei[i] = eps; ej[j] = eps
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * eps * eps)
    return H


def num_score_obs(fobs, x, eps=1e-6):
    cols = []
    for i in range(len(x)):
        e = np.zeros(len(x)); e[i] = eps
        cols.append((fobs(x + e) - fobs(x - e)) / (2 * eps))
    return np.column_stack(cols)


def newey(y, X, names, order, lags, groups=None):
    """Stata newey: OLS + Bartlett HAC, scaled N/(N-K); t with N-K df. Lags within panels if groups given."""
    g = np.zeros(len(y)) if groups is None else pd.factorize(groups)[0]
    o = np.lexsort((order, g)); y, X, g, tt = y[o], X[o], g[o], np.asarray(order, float)[o]
    N, K = X.shape
    XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ y; e = y - X @ b
    U = X * e[:, None]; S = U.T @ U
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        ok = (g[l:] == g[:-l]) & (np.abs(tt[l:] - tt[:-l] - l) < 1e-9 if groups is not None else True)
        G = (U[l:] * np.asarray(ok, float).reshape(-1, 1)).T @ U[:-l]; S += w * (G + G.T)
    V = XtXi @ S @ XtXi * N / (N - K)
    return {"estimator": "newey", "N": N, "coefs": _rows(names, b, np.sqrt(np.diag(V)), N - K), "names": names,
            "vce": f"newey(lag {lags})", "df_r": N - K, "vcov": V.tolist(), "_b": b}


def qreg(y, X, names, q=0.5, vce="iid"):
    import statsmodels.api as sm
    r = sm.QuantReg(y, X).fit(q=q, vcov="robust" if vce == "robust" else "iid", max_iter=5000)
    N, K = X.shape
    return {"estimator": "qreg", "quantile": q, "N": N, "coefs": _rows(names, r.params, r.bse, N - K), "names": names,
            "vce": vce, "df_r": N - K, "vcov": np.asarray(r.cov_params()).tolist(), "_b": np.asarray(r.params),
            "note": "coefficients match Stata qreg; SEs use statsmodels' kernel density estimate and may differ in the 2nd-3rd digit"}


def tobit(y, X, names, ll=None, ul=None, vce="iid", cl=None):
    """Censored-normal MLE (Stata tobit) by Newton-Raphson with analytic scores; reports sigma."""
    N, K = X.shape
    left = (y <= ll) if ll is not None else np.zeros(N, bool)
    right = (y >= ul) if ul is not None else np.zeros(N, bool)
    mid = ~(left | right)

    def lobs(th):
        b, s = th[:K], np.exp(th[K]); xb = X @ b; out = np.zeros(N)
        out[mid] = stats.norm.logpdf((y[mid] - xb[mid]) / s) - np.log(s)
        if left.any():
            out[left] = stats.norm.logcdf((ll - xb[left]) / s)
        if right.any():
            out[right] = stats.norm.logsf((ul - xb[right]) / s)
        return out

    def sobs(th):  # N x (K+1) scores wrt (b, ln sigma)
        b, s = th[:K], np.exp(th[K]); xb = X @ b
        gb = np.zeros(N); gs = np.zeros(N)
        z = (y[mid] - xb[mid]) / s; gb[mid] = z / s; gs[mid] = z * z - 1
        if left.any():
            c = (ll - xb[left]) / s; lam = np.exp(stats.norm.logpdf(c) - stats.norm.logcdf(c))
            gb[left] = -lam / s; gs[left] = -lam * c
        if right.any():
            c = (ul - xb[right]) / s; mu = np.exp(stats.norm.logpdf(c) - stats.norm.logsf(c))
            gb[right] = mu / s; gs[right] = mu * c
        return np.column_stack([X * gb[:, None], gs])

    def hess(th):
        H = np.zeros((K + 1, K + 1))
        for i in range(K + 1):
            h = 1e-6 * max(1.0, abs(th[i])); e = np.zeros(K + 1); e[i] = h
            H[:, i] = (sobs(th + e).sum(0) - sobs(th - e).sum(0)) / (2 * h)
        return (H + H.T) / 2
    b0 = np.linalg.lstsq(X, y, rcond=None)[0]
    th = np.r_[b0, np.log(np.std(y - X @ b0) + 1e-9)]
    f0 = lobs(th).sum()
    for _ in range(200):
        g = sobs(th).sum(0); H = hess(th)
        step = np.linalg.solve(H, -g)
        t = 1.0
        while t > 1e-8 and lobs(th + t * step).sum() < f0 - 1e-12:
            t /= 2
        th = th + t * step; f1 = lobs(th).sum()
        if abs(f1 - f0) < 1e-11 * (1 + abs(f0)) and np.max(np.abs(t * step)) < 1e-9 * (1 + np.max(np.abs(th))):
            f0 = f1; break
        f0 = f1
    # covariance for (b, sigma): delta method from (b, ln sigma)
    sig = np.exp(th[K]); Jm = np.eye(K + 1); Jm[K, K] = sig
    Hinv = np.linalg.inv(-hess(th))
    V0 = Hinv if vce == "iid" else _sandwich(Hinv, sobs(th), vce, cl)
    V = Jm @ V0 @ Jm.T
    nm = names + ["sigma"]
    return {"estimator": "tobit", "N": N, "N_lc": int(left.sum()), "N_rc": int(right.sum()),
            "coefs": _rows(nm, np.r_[th[:K], sig], np.sqrt(np.diag(V)), N - K), "names": nm, "vce": vce, "df_r": N - K,
            "ll": float(f0), "vcov": V.tolist(), "_b": th[:K]}


def ordered(y, X, names, distr="logit", vce="iid", cl=None):
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    m = OrderedModel(y, X, distr=distr)
    r = m.fit(method="bfgs", disp=0, maxiter=2000)
    r = m.fit(start_params=r.params, method="newton", disp=0, maxiter=200) if not r.mle_retvals.get("converged", True) else r
    p = np.asarray(r.params); K = X.shape[1]; J = len(p) - K
    cut = np.r_[p[K], p[K] + np.cumsum(np.exp(p[K + 1:]))]
    Jac = np.zeros((K + J, K + J)); Jac[:K, :K] = np.eye(K)
    for j in range(J):
        Jac[K + j, K] = 1
        for i in range(1, j + 1):
            Jac[K + j, K + i] = np.exp(p[K + i])
    Hinv = np.linalg.inv(-m.hessian(p))
    V0 = Hinv if vce == "iid" else _sandwich(Hinv, m.score_obs(p), vce, cl)
    V = Jac @ V0 @ Jac.T
    nm = names + [f"cut{j + 1}" for j in range(J)]
    return {"estimator": "o" + distr, "N": len(y), "coefs": _rows(nm, np.r_[p[:K], cut], np.sqrt(np.diag(V))),
            "names": nm, "vce": vce, "pseudo_r2": float(r.prsquared), "ll": float(r.llf), "vcov": V.tolist(), "_b": p[:K]}


def mlogit(y, X, names, base=None, vce="iid", cl=None):
    import statsmodels.api as sm
    vals, cnt = np.unique(y, return_counts=True)
    base = vals[np.argmax(cnt)] if base is None else base  # Stata default: most frequent outcome
    order = [base] + [v for v in vals if v != base]
    yy = np.array([order.index(v) for v in y])
    m = sm.MNLogit(yy, X); r = m.fit(disp=0, method="newton", maxiter=200)
    P = np.asarray(r.params)  # K x (J-1)
    flat = P.T.ravel()
    Hinv = np.linalg.inv(-m.hessian(flat))
    S = m.score_obs(flat)
    V = Hinv if vce == "iid" else _sandwich(Hinv, S, vce, cl)
    se = np.sqrt(np.diag(V))
    nm = [f"{_lab(order[j + 1])}:{n}" for j in range(P.shape[1]) for n in names]
    return {"estimator": "mlogit", "base_outcome": _lab(base), "N": len(y), "coefs": _rows(nm, flat, se), "names": nm,
            "vce": vce, "pseudo_r2": float(r.prsquared), "ll": float(r.llf), "vcov": V.tolist()}


def _lab(v):
    f = float(v)
    return str(int(f)) if f.is_integer() else str(v)


def clogit(y, X, names, groups, vce="iid"):
    from statsmodels.discrete.conditional_models import ConditionalLogit
    g = pd.Series(groups); yy = pd.Series(y)
    var = yy.groupby(g.values).transform(lambda s: s.nunique() > 1).values.astype(bool)
    y, X, groups = y[var], X[var], np.asarray(groups)[var]
    keep = [j for j in range(X.shape[1]) if np.ptp(X[:, j]) > 0 and not np.allclose(pd.Series(X[:, j]).groupby(groups).transform("std").fillna(0), 0)]
    X = X[:, keep]; names = [names[j] for j in keep]
    m = ConditionalLogit(y, X, groups=groups)
    r = m.fit(disp=0, method="bfgs", maxiter=1000)
    V = np.asarray(r.cov_params())
    if vce in ("robust", "cluster"):
        V = np.asarray(m.fit(disp=0, method="bfgs", maxiter=1000, cov_type="cluster", cov_kwds={"groups": groups}).cov_params())
    return {"estimator": "clogit", "N": int(len(y)), "N_g": int(len(np.unique(groups))), "coefs": _rows(names, r.params, np.sqrt(np.diag(V))),
            "names": names, "vce": vce, "ll": float(r.llf), "vcov": V.tolist(), "_b": np.asarray(r.params),
            "note": "groups with no outcome variation dropped (as Stata)"}


def psmatch2(treat, X, names, outcome=None, model="probit", k=1, caliper=None, common=False, replace=True, ties=False):
    """Nearest-neighbour propensity-score matching (psmatch2-like). Returns (estimate, generated columns)."""
    import statsmodels.api as sm
    Xc = np.column_stack([X, np.ones(len(X))])
    m = (sm.Probit if model == "probit" else sm.Logit)(treat, Xc).fit(disp=0, method="newton", maxiter=200)
    ps = m.predict(Xc)
    N = len(treat); T = treat == 1; C = ~T
    support = np.ones(N, bool)
    if common:
        lo, hi = ps[C].min(), ps[C].max()
        support[T] = (ps[T] >= lo) & (ps[T] <= hi)
    weight = np.full(N, np.nan); n1 = np.full(N, np.nan); pdif = np.full(N, np.nan)
    cidx = np.where(C)[0]; cps = ps[cidx]; used = np.zeros(len(cidx), bool)
    for i in np.where(T & support)[0]:
        d = np.abs(cps - ps[i])
        if not replace:
            d = np.where(used, np.inf, d)
        if not np.isfinite(d).any():
            continue
        if ties:
            kth = np.sort(d)[min(k, len(d)) - 1]; sel = np.where(d <= kth + 1e-15)[0]
        else:
            sel = np.argsort(d, kind="stable")[:k]
        if caliper is not None:
            sel = sel[d[sel] <= caliper]
        if len(sel) == 0:
            continue
        weight[i] = 1.0; n1[i] = cidx[sel[0]]; pdif[i] = d[sel[0]]
        for s in sel:
            weight[cidx[s]] = (0 if np.isnan(weight[cidx[s]]) else weight[cidx[s]]) + 1.0 / len(sel)
            used[s] = True
    out = {"estimator": "psmatch2", "N": int(np.sum(~np.isnan(weight))), "N_treated_matched": int(np.sum(T & ~np.isnan(weight))),
           "N_controls_used": int(np.sum(C & ~np.isnan(weight))), "pscore_model": model,
           "coefs": _rows(names + ["_cons"], m.params, m.bse), "names": names + ["_cons"], "vce": "iid",
           "pseudo_r2": float(m.prsquared),
           "note": "psmatch2 reimplementation: matches depend on sort order/ties; Stata breaks ties by the current (often random) sort order, so matched-sample N can differ slightly"}
    if outcome is not None:
        yt = outcome[T & ~np.isnan(weight)]
        wc = weight[C & ~np.isnan(weight)]; yc = outcome[C & ~np.isnan(weight)]
        att = yt.mean() - (wc * yc).sum() / wc.sum()
        out["ATT"] = float(att); out["unmatched_diff"] = float(outcome[T].mean() - outcome[C].mean())
    cols = {"_pscore": ps, "_treated": T.astype(float), "_support": support.astype(float), "_weight": weight,
            "_n1": n1, "_pdif": pdif}
    return out, cols
