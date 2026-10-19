"""Linear models with high-dimensional fixed effects, IV, and Stata-compatible
small-sample conventions (reg / areg / xtreg fe / reghdfe / ivreg2 / ivreghdfe).

Everything works on numpy arrays after listwise deletion; FE are swept out by
alternating projections (pyfixest's compiled demean when available).
"""
import numpy as np
import pandas as pd
from .stexpr import _isstr
from scipy import stats
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

try:
    from pyfixest.core.demean import demean as _pf_demean
except Exception:  # pragma: no cover
    _pf_demean = None


# ----------------------------------------------------------------------------- utilities
# Stock-Yogo (2005) critical values, 10% maximal IV size, keyed by (n_endog, n_excluded_instruments)
SY10 = {(1, 1): 16.38, (1, 2): 19.93, (1, 3): 22.30, (1, 4): 24.58, (1, 5): 26.87, (1, 6): 29.18, (1, 7): 31.50,
        (1, 8): 33.84, (1, 9): 36.19, (1, 10): 38.54, (2, 2): 7.03, (2, 3): 13.43, (2, 4): 16.87, (2, 5): 19.45,
        (2, 6): 21.68, (2, 7): 23.72, (2, 8): 25.64, (2, 9): 27.51, (2, 10): 29.32}

def group_codes(df, cols):
    """Integer codes for the interaction of one or more columns (a#b). Missing -> -1."""
    if isinstance(cols, str):
        cols = [cols]
    sub = df[list(cols)]
    miss = sub.isna().any(axis=1).values
    codes = sub.groupby(list(cols), sort=True, dropna=False).ngroup().values.astype(np.int64)
    codes[miss] = -1
    return codes


def _recode(c):
    _, inv = np.unique(c, return_inverse=True)
    return inv.astype(np.int64)


def singleton_mask(fe_codes, fw=None):
    """Iteratively flag observations that are alone in some FE group (reghdfe default).
    With frequency weights a row counts as `fw` observations (reghdfe behaviour)."""
    n = len(fe_codes[0]) if fe_codes else 0
    keep = np.ones(n, bool)
    if not fe_codes:
        return keep
    changed = True
    while changed:
        changed = False
        for c in fe_codes:
            cc = c[keep]
            cnt = np.bincount(_recode(cc), weights=None if fw is None else fw[keep])
            single = cnt[_recode(cc)] == 1
            if single.any():
                idx = np.where(keep)[0][single]
                keep[idx] = False
                changed = True
    return keep


def demean(M, fe_codes, w=None, tol=1e-10, maxiter=100000):
    M = np.ascontiguousarray(np.asarray(M, dtype=np.float64))
    if M.ndim == 1:
        M = M[:, None]
    if not fe_codes:
        return M
    w = np.ones(M.shape[0]) if w is None else np.asarray(w, dtype=np.float64)
    if _pf_demean is not None:
        fl = np.ascontiguousarray(np.column_stack([_recode(c) for c in fe_codes]).astype(np.uint64))
        out, ok = _pf_demean(M, fl, w, tol, maxiter)
        return np.asarray(out)
    out = M.copy()  # fallback: alternating projections
    codes = [_recode(c) for c in fe_codes]
    for _ in range(maxiter):
        prev = out.copy()
        for c in codes:
            sw = np.bincount(c, weights=w)
            for j in range(out.shape[1]):
                out[:, j] -= (np.bincount(c, weights=w * out[:, j]) / sw)[c]
        if np.max(np.abs(out - prev)) < tol:
            break
    return out


def nested_in(fe, cl):
    """True if every FE level maps to a single cluster level."""
    d = pd.DataFrame({"f": fe, "c": cl})
    return bool((d.groupby("f")["c"].nunique() <= 1).all())


def fe_dof(fe_codes, fe_names, cluster_codes=None):
    """reghdfe-style absorbed dof. FE nested within a cluster variable contribute 0; among the
    remaining FE the first is fully counted, the second loses its connected components with the
    first, later ones lose 1 (conservative, as reghdfe). Returns (rows, df_a_vcov, df_a_full, df_a_iv)."""
    def _table(nested_flags):
        rows, tot, first = [], 0, None
        for j, (c, name) in enumerate(zip(fe_codes, fe_names)):
            L = len(np.unique(c))
            if nested_flags[j]:
                red = L
            elif any((not nested_flags[i]) and nested_in(fe_codes[i], c) for i in range(j)):
                red = L  # an earlier absorbed FE is finer than (or equal to) this one: fully redundant
            elif any((not nested_flags[i]) and nested_in(c, fe_codes[i]) for i in range(j)):
                # this FE is finer than an earlier one (e.g. id#industry after id): earlier dummies are spanned
                red = max(len(np.unique(fe_codes[i])) for i in range(j)
                          if (not nested_flags[i]) and nested_in(c, fe_codes[i]))
            elif first is None:
                red, first = 0, j
            elif first is not None and all(nested_flags[i] or i == first for i in range(j)):
                a, b = _recode(fe_codes[first]), _recode(c)
                na, nb = a.max() + 1, b.max() + 1
                g = coo_matrix((np.ones(len(a)), (a, b + na)), shape=(na + nb, na + nb))
                red = connected_components(g, directed=False)[0]
            else:
                red = 1
            rows.append({"fe": name, "categories": L, "redundant": int(red), "coefs": int(L - red),
                         "nested_in_cluster": bool(nested_flags[j])})
            tot += L - red
        return rows, tot
    nested = [any(nested_in(c, cl) for cl in (cluster_codes or [])) for c in fe_codes]
    rows, df_a = _table(nested)
    _, df_a_full = _table([False] * len(fe_codes))
    # ivreghdfe (calibrated on two papers): when an FE is nested in the cluster var the constant is not counted
    df_a_iv = max(df_a - 1, 0) if any(nested) else df_a
    if all(nested):
        df_a += 1  # constant no longer absorbed (reghdfe)
    return rows, df_a, df_a_full, df_a_iv


# ----------------------------------------------------------------------------- vcov
def _meat(S, clusters=None):
    """S: N x k scores. clusters: list of code arrays (multiway = inclusion-exclusion)."""
    if not clusters:
        return S.T @ S, None
    if len(clusters) == 1:
        c = _recode(clusters[0])
        G = c.max() + 1
        Sg = np.zeros((G, S.shape[1]))
        np.add.at(Sg, c, S)
        return Sg.T @ Sg, [G]
    raise ValueError("use _meat_multi")


def _cluster_sum(S, c):
    c = _recode(c)
    Sg = np.zeros((c.max() + 1, S.shape[1]))
    np.add.at(Sg, c, S)
    return Sg.T @ Sg, c.max() + 1


def cgm_meat(S, clusters):
    """Multiway (Cameron-Gelbach-Miller) cluster meat sum_r (-1)^(r+1) sum_{|c|=r} S_c'S_c, PSD-projected."""
    from itertools import combinations
    k = len(clusters)
    if k == 1:
        return _cluster_sum(S, clusters[0])[0]
    M = np.zeros((S.shape[1], S.shape[1]))
    for r in range(1, k + 1):
        for comb in combinations(range(k), r):
            cc = clusters[comb[0]] if r == 1 else group_codes(pd.DataFrame({i: clusters[i] for i in comb}), list(comb))
            M += (1 if r % 2 == 1 else -1) * _cluster_sum(S, cc)[0]
    ev, U = np.linalg.eigh((M + M.T) / 2)
    if (ev < 0).any():
        M = U @ np.diag(np.clip(ev, 0, None)) @ U.T
    return M


def sandwich(bread, S, kind, clusters=None, scale_fn=None):
    """kind: iid handled outside; 'robust' or 'cluster'. scale_fn(G) -> multiplier."""
    if kind == "robust":
        return bread @ (S.T @ S) @ bread * scale_fn(None), None
    # cluster (one- or multi-way, Cameron-Gelbach-Miller inclusion-exclusion)
    from itertools import combinations
    k = len(clusters)
    Gs = [len(np.unique(c)) for c in clusters]
    Gmin = min(Gs)
    V = np.zeros_like(bread)
    for r in range(1, k + 1):
        for comb in combinations(range(k), r):
            if r == 1:
                cc = clusters[comb[0]]
            else:
                cc = group_codes(pd.DataFrame({i: clusters[i] for i in comb}), list(comb))
            M, G = _cluster_sum(S, cc)
            sign = 1 if r % 2 == 1 else -1
            V += sign * bread @ M @ bread * (scale_fn(Gmin) if k > 1 else scale_fn(G))
    if k > 1:  # ensure PSD
        ev, U = np.linalg.eigh((V + V.T) / 2)
        if (ev < 0).any():
            V = U @ np.diag(np.clip(ev, 0, None)) @ U.T
    return V, Gs


# ----------------------------------------------------------------------------- core estimator
CONVENTIONS = {
    # small: t/F with df; dof_fe: how absorbed FE enter the dof ("nested"= reghdfe, "full"= areg, "xtreg")
    "reg": dict(small=True, dof_fe="full"),
    "areg": dict(small=True, dof_fe="full"),
    "reghdfe": dict(small=True, dof_fe="nested"),
    "xtreg": dict(small=True, dof_fe="xtreg"),
    "ivreg2": dict(small=False, dof_fe="nested"),
    "ivreghdfe": dict(small=False, dof_fe="nested"),
    "ivregress": dict(small=False, dof_fe="full"),
    "xtivreg": dict(small=True, dof_fe="xtreg"),
}


def _inv(A):
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A)


def _drop_collinear(X, names, tol=1e-9):
    """Sequential removal of collinear columns, in order (Stata 'omitted because of collinearity').
    Works on the scaled Gram matrix like Stata's _rmcoll: column j is dropped when its squared residual on
    the kept earlier columns is < tol x its squared norm. O(K^2 N + K^3) instead of a QR loop."""
    K = X.shape[1]
    if K == 0:
        return X, names, []
    nrm = np.sqrt((X * X).sum(0))
    ok = nrm > 1e-12
    Xs = np.where(ok, X / np.where(ok, nrm, 1), 0.0)
    G = Xs.T @ Xs
    L = np.zeros((K, K))
    keep, dropped = [], []
    for j in range(K):
        if not ok[j]:
            dropped.append(names[j]); continue
        lj = L[j, keep] if keep else np.zeros(0)
        d = G[j, j] - lj @ lj
        if d < tol:
            dropped.append(names[j]); continue
        L[j, j] = np.sqrt(d)
        if j + 1 < K:
            L[j + 1:, j] = (G[j + 1:, j] - (L[j + 1:, keep] @ lj if keep else 0.0)) / L[j, j]
        keep.append(j)
    if len(keep) < K and len(keep) <= 400:  # confirm borderline cases exactly on the kept set
        Q, R = np.linalg.qr(Xs[:, keep])
        dg = np.abs(np.diag(R))
        if dg.size and dg.min() < np.sqrt(tol):
            bad = [keep[i] for i in range(len(keep)) if dg[i] < np.sqrt(tol)]
            dropped += [names[b] for b in bad]; keep = [k for k in keep if k not in bad]
    return X[:, keep], [names[j] for j in keep], dropped


def estimate(df, y, exog, endog=None, instruments=None, fe=None, vce="iid", cluster=None,
             weights=None, wtype="aweight", convention="reghdfe", drop_singletons=True,
             constant=True, small=None, estimator="2sls", fe_names=None, label=None):
    """General linear (IV) estimator with absorbed FE.

    df: DataFrame already containing all needed (generated) columns.
    fe: list of column-lists (each item = an interaction, e.g. ["industry","year"]).
    cluster: list of column-lists (multiway allowed).
    vce: iid | robust | cluster.
    """
    endog = list(endog or [])
    instruments = list(instruments or [])
    exog = list(exog)
    fe = [[f] if isinstance(f, str) else list(f) for f in (fe or [])]
    cluster = [[c] if isinstance(c, str) else list(c) for c in (cluster or [])]
    if cluster and vce != "cluster":
        vce = "cluster"
    conv = CONVENTIONS.get(convention, CONVENTIONS["reghdfe"])
    small = conv["small"] if small is None else small
    is_iv = bool(endog)

    need = [y] + exog + endog + instruments + sum(fe, []) + sum(cluster, []) + ([weights] if weights else [])
    need = list(dict.fromkeys(need))
    d = df[need].copy()
    for c in [y] + exog + endog + instruments + ([weights] if weights else []):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    ok = ~d.isna().any(axis=1)
    for c in sum(fe, []) + sum(cluster, []):
        if _isstr(d[c]):
            ok &= d[c].astype(str).str.len() > 0
    if weights:
        ok &= d[weights] > 0
    d = d[ok]
    n_listwise = len(d)
    fe_codes = [group_codes(d, f) for f in fe]
    n_singletons = 0
    if fe_codes and drop_singletons:
        fwv = d[weights].values.astype(float) if (weights and wtype == "fweight") else None
        keep = singleton_mask(fe_codes, fwv)
        n_singletons = int((~keep).sum())
        d = d[keep]
        fe_codes = [c[keep] for c in fe_codes]
    N = len(d)
    if N == 0:
        raise ValueError("no observations")
    cl_codes = [group_codes(d, c) for c in cluster]
    w = d[weights].values.astype(float) if weights else None
    if w is not None and wtype in ("aweight", "pweight"):
        w = w * N / w.sum()
    Nd = float(w.sum()) if (w is not None and wtype == "fweight") else N  # dof sample size
    if wtype == "pweight" and weights and vce == "iid":
        vce = "robust"

    yv = d[y].values.astype(float)
    Xex = d[exog].values.astype(float) if exog else np.zeros((N, 0))
    Xen = d[endog].values.astype(float) if endog else np.zeros((N, 0))
    Zx = d[instruments].values.astype(float) if instruments else np.zeros((N, 0))
    has_fe = bool(fe_codes)
    if has_fe:
        allm = demean(np.column_stack([yv, Xex, Xen, Zx]), fe_codes, w)
        yt = allm[:, 0]
        Xex_t = allm[:, 1:1 + Xex.shape[1]]
        Xen_t = allm[:, 1 + Xex.shape[1]:1 + Xex.shape[1] + Xen.shape[1]]
        Zx_t = allm[:, 1 + Xex.shape[1] + Xen.shape[1]:]
        constant_col = False
    else:
        yt, Xex_t, Xen_t, Zx_t = yv, Xex, Xen, Zx
        constant_col = constant
    ex_names = list(exog)
    if constant_col:
        Xex_t = np.column_stack([Xex_t, np.ones(N)])
        ex_names = ex_names + ["_cons"]
    # sqrt-weights transform
    sw = np.sqrt(w) if w is not None else np.ones(N)
    yt_w, Xex_w, Xen_w, Zx_w = yt * sw, Xex_t * sw[:, None], Xen_t * sw[:, None], Zx_t * sw[:, None]

    # collinearity
    if constant_col and Xex_w.shape[1] > 1:  # check _cons first so that a dummy (not the constant) is dropped
        order = [len(ex_names) - 1] + list(range(len(ex_names) - 1))
        _, kept, dropped = _drop_collinear(Xex_w[:, order], [ex_names[i] for i in order])
        ex_names_k = [n for n in ex_names if n in kept]
        Xex_w = Xex_w[:, [ex_names.index(n) for n in ex_names_k]]
    else:
        Xex_w, ex_names_k, dropped = _drop_collinear(Xex_w, ex_names)
    keep_idx = [ex_names.index(n) for n in ex_names_k]
    Xex_t = Xex_t[:, keep_idx]
    if is_iv and Zx_w.shape[1]:
        nex = Xex_w.shape[1]
        _, zk, zdrop = _drop_collinear(np.column_stack([Xex_w, Zx_w]), ex_names_k + ["__z:" + z for z in instruments])
        keepz = [i for i, z in enumerate(instruments) if "__z:" + z in zk]
        dz = [z for z in instruments if "__z:" + z in zdrop]
        if dz:
            dropped = dropped + [f"instrument {z}" for z in dz]
            instruments = [instruments[i] for i in keepz]
            Zx_w = Zx_w[:, keepz]
    if is_iv and Xen_w.shape[1]:
        _, ek_, edrop = _drop_collinear(np.column_stack([Xex_w, Xen_w]), ex_names_k + ["__e:" + v for v in endog])
        de = [v for v in endog if "__e:" + v in edrop]
        if de:
            keepe = [i for i, v in enumerate(endog) if v not in de]
            dropped = dropped + [f"endogenous {v}" for v in de]
            endog = [endog[i] for i in keepe]
            Xen_w = Xen_w[:, keepe]
            Xen = Xen[:, keepe]
            is_iv = bool(endog)
    X = np.column_stack([Xen_w, Xex_w])
    names = list(endog) + ex_names_k
    if is_iv:
        Z = np.column_stack([Zx_w, Xex_w])
        L, L2, K2 = Z.shape[1], Zx_w.shape[1], len(endog)
        if L2 < K2:
            gone = [d[len("instrument "):] for d in dropped if str(d).startswith("instrument ")]
            raise ValueError(f"underidentified: {L2} excluded instrument(s) left for {K2} endogenous regressor(s)"
                             + (f"; dropped as collinear with controls/absorbed FE: {', '.join(gone)}" if gone else "")
                             + ". A time-invariant instrument is absorbed by unit fixed effects: identification must come "
                               "from cross-sectional variation (no unit FE) or from interacting it with a time-varying series")
        ZZi = _inv(Z.T @ Z)
        PzX = Z @ (ZZi @ (Z.T @ X))
        if estimator == "liml":
            Yy = np.column_stack([yt_w, Xen_w])
            def _resid(A, B):
                return A - B @ np.linalg.lstsq(B, A, rcond=None)[0] if B.shape[1] else A
            Wm = _resid(Yy, Xex_w); Zm = _resid(Yy, Z)
            kappa = float(np.min(np.real(np.linalg.eigvals(_inv(Zm.T @ Zm) @ (Wm.T @ Wm)))))
            Xr = X - PzX
            A = X.T @ X - kappa * (Xr.T @ X)
            bread = _inv(A)
            b = bread @ (X.T @ yt_w - kappa * (Xr.T @ yt_w))
            Xs = (1 - kappa) * X + kappa * PzX  # score regressor
        else:
            kappa = 1.0
            bread = _inv(PzX.T @ X)
            b = bread @ (PzX.T @ yt_w)
            Xs = PzX
    else:
        bread = _inv(X.T @ X)
        b = bread @ (X.T @ yt_w)
        Xs = X
        L = L2 = K2 = 0
    e = yt_w - X @ b
    rss = float(e @ e)
    K = X.shape[1]

    # ---- absorbed dof
    fe_names = fe_names or ["#".join(f) for f in fe]
    if has_fe:
        dof_rows, df_a_nested, df_a_full, df_a_iv = fe_dof(fe_codes, fe_names, cl_codes if vce == "cluster" else None)
    else:
        dof_rows, df_a_nested, df_a_full = [], (1 if constant else 0), (1 if constant else 0)
        df_a_iv = 0
        K_nocons = K - (1 if constant_col else 0)
    if has_fe:
        K_nocons = K
    if conv["dof_fe"] == "nested":
        df_a = df_a_nested if has_fe else (1 if constant else 0)
    elif conv["dof_fe"] == "xtreg":
        # xtreg fe: absorbed group dof counted for iid/robust... cluster(id) treats them as nested (+1 for _cons)
        df_a = (1 + (df_a_nested - 0)) if (vce == "cluster" and has_fe) else df_a_full
    else:
        df_a = df_a_full if has_fe else (1 if constant else 0)
    df_r = Nd - K_nocons - df_a
    df_r_full = Nd - K_nocons - (df_a_full if has_fe else (1 if constant else 0))

    # ---- vcov
    G_list = None
    if vce == "iid":
        s2 = rss / (df_r if (small or convention == "xtivreg") else Nd)  # xtivreg fe: dof-corrected sigma, z stats
        if is_iv and estimator == "liml":
            V = s2 * bread
        else:
            V = s2 * bread
    else:
        S = Xs * e[:, None]
        if vce == "robust":
            sc = (Nd / df_r) if small else 1.0
            V, _ = sandwich(bread, S, "robust", scale_fn=lambda G: sc)
        else:
            if small:
                V, G_list = sandwich(bread, S, "cluster", cl_codes,
                                     scale_fn=lambda G: (G / (G - 1)) * ((Nd - 1) / (Nd - K_nocons - df_a)))
            else:  # ivreg2/ivreghdfe cluster: (N-1)/(N-K) * G/(G-1) even without `small` (calibrated)
                V, G_list = sandwich(bread, S, "cluster", cl_codes,
                                     scale_fn=lambda G: (G / (G - 1)) * ((Nd - 1) / (Nd - K_nocons - df_a_iv)))
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    with np.errstate(all="ignore"):
        tv = b / se
    if small:
        dft = (min(G_list) - 1) if G_list else df_r
        pv = 2 * stats.t.sf(np.abs(tv), dft)
        crit = stats.t.ppf(0.975, dft)
    else:
        dft = None
        pv = 2 * stats.norm.sf(np.abs(tv))
        crit = stats.norm.ppf(0.975)

    # ---- fit statistics (unweighted-scale R2 like Stata with aweights uses weighted)
    wv = w if w is not None else np.ones(N)
    ybar = np.sum(wv * yv) / np.sum(wv)
    tss = float(np.sum(wv * (yv - ybar) ** 2))
    tss_w = float(np.sum((yt_w) ** 2)) if has_fe else tss
    r2 = 1 - rss / tss if tss > 0 else np.nan
    r2_within = (1 - rss / tss_w) if has_fe and tss_w > 0 else None
    adj_r2 = 1 - (1 - r2) * (Nd - 1) / df_r_full if df_r_full > 0 else np.nan
    rmse = np.sqrt(rss / (df_r_full if small else Nd))

    coefs = {}
    for i, nm in enumerate(names):
        coefs[nm] = {"coef": b[i], "se": se[i], "t": tv[i], "p": pv[i],
                     "ci_low": b[i] - crit * se[i], "ci_high": b[i] + crit * se[i]}
    # intercept reported by reghdfe/areg/xtreg: mean(y) - mean(X) b (weighted)
    if has_fe and constant:
        Xraw = np.column_stack([Xen, Xex[:, [exog.index(n) for n in ex_names_k if n in exog]]]) if K else np.zeros((N, 0))
        xbar = (wv @ Xraw) / wv.sum() if K else np.zeros(0)
        coefs["_cons"] = {"coef": float(ybar - xbar @ b), "se": None, "t": None, "p": None}

    # model F test (all slope coefs, excluding _cons)
    slope = [i for i, nm in enumerate(names) if nm != "_cons"]
    F = Fp = None
    if slope:
        Vs = V[np.ix_(slope, slope)]
        bs = b[slope]
        try:
            W = float(bs @ np.linalg.solve(Vs, bs))
            q = len(slope)
            if vce == "cluster":
                q = min(q, min(G_list) - 1)
            F = W / len(slope)
            Fp = float(stats.f.sf(F, len(slope), dft if dft else 1e9)) if small else float(stats.chi2.sf(W, len(slope)))
        except np.linalg.LinAlgError:
            pass

    out = {
        "label": label, "convention": convention, "estimator": (estimator if is_iv else "ols"),
        "dep": y, "N": (int(Nd) if Nd != N else N), "N_rows": N, "N_listwise": n_listwise, "singletons_dropped": n_singletons,
        "coefs": coefs, "names": names, "vcov": V.tolist(), "vce": vce,
        "cluster": ["#".join(c) for c in cluster] or None, "n_clusters": G_list,
        "fe": fe_names, "absorbed_dof": dof_rows, "df_a": df_a, "df_r": df_r, "small": small,
        "r2": r2, "adj_r2": adj_r2, "r2_within": r2_within, "rmse": rmse, "rss": rss,
        "F": F, "F_p": Fp, "omitted_collinear": dropped,
    }
    if is_iv:
        out["iv"] = iv_diagnostics(yt_w, Xen_w, Zx_w, Xex_w, e, b[:K2], vce, cl_codes, N, small,
                                   df_a=df_a_iv, endog=endog, instruments=instruments, kappa=kappa)
        out["endog"], out["instruments"] = endog, instruments
    out["_index"] = d.index
    out["_resid"] = e / sw
    out["_b"] = b
    return out


# ----------------------------------------------------------------------------- IV diagnostics
def _partial(A, W):
    if W.shape[1] == 0:
        return A
    return A - W @ np.linalg.lstsq(W, A, rcond=None)[0]


def _wald(b, V):
    try:
        return float(b @ np.linalg.solve(V, b))
    except np.linalg.LinAlgError:
        return float(b @ np.linalg.pinv(V) @ b)


def _ols_vcov(Xm, e, vce, cl, N, adj=1.0):
    bread = _inv(Xm.T @ Xm)
    if vce == "iid":
        return (e @ e / N) * bread
    S = Xm * e[:, None]
    if vce == "robust":
        return bread @ (S.T @ S) @ bread
    V, _ = sandwich(bread, S, "cluster", cl, scale_fn=lambda G: 1.0)
    return V


def _msqrt(A):
    ev, U = np.linalg.eigh((A + A.T) / 2)
    return U @ np.diag(np.sqrt(np.clip(ev, 0, None))) @ U.T


def kp_rk(X, Z, vce, cl, N, wald=True):
    """Kleibergen-Paap (2006) rk statistic for H0: rank(Pi)=K-1 (X,Z already partialled).
    Follows ranktest's construction. Returns chi2 stat with df L-K+1."""
    K, L = X.shape[1], Z.shape[1]
    zz, xx = Z.T @ Z, X.T @ X
    pihat = np.linalg.solve(zz, Z.T @ X)               # L x K
    zhalf = _msqrt(zz / N)
    xhalf = _msqrt(xx / N)
    that = zhalf @ pihat @ np.linalg.inv(xhalf)          # L x K
    U, s, Vt = np.linalg.svd(that)
    Vv = Vt.T
    q = K - 1
    u12, u22 = U[:q, q:], U[q:, q:]
    v12, v22 = Vv[:q, q:], Vv[q:, q:]
    aq = np.vstack([u12, u22]) @ np.linalg.inv(u22) @ _msqrt(u22 @ u22.T)
    bq = np.vstack([v12, v22]) @ np.linalg.inv(v22).T @ _msqrt(v22 @ v22.T)
    kron = np.kron(bq.T, aq.T)
    lab = kron @ that.reshape(-1, order="F")
    # covariance of vec(pihat)*sqrt(N)
    E = X - Z @ pihat if wald else X
    zzi = np.linalg.inv(zz / N)
    if vce == "iid":
        if wald:
            shat = np.kron(E.T @ E / N, zz / N)
        else:
            shat = np.kron(xx / N, zz / N)
    else:
        Sc = np.einsum("ik,il->ikl", E, Z).reshape(N, K * L)  # row i: vec(z_i e_i') ordered (k major) -> fix order
        # want vec(Z'E) column-major over (L x K): index = k*L + l
        Sc = np.column_stack([Z * E[:, [k]] for k in range(K)])
        if vce == "robust":
            shat = Sc.T @ Sc / N
        else:
            shat = cgm_meat(Sc, cl) / N
    Ik = np.eye(K)
    vpi = np.kron(Ik, zzi) @ shat @ np.kron(Ik, zzi)       # avar of sqrt(N) vec(pihat)
    T = np.kron(np.linalg.inv(xhalf).T, zhalf)
    vth = T @ vpi @ T.T
    vlab = kron @ vth @ kron.T
    rk = float(N * lab @ np.linalg.solve(vlab, lab))
    return rk, (L - K + 1)


def _ar_set(yp, xp, Zp, vce, cl, N, L, L2, G, b_hat, level=0.95):
    """AR confidence set for one endogenous regressor: {b0 : AR_F(b0) < F crit}. Returns a list of [lo, hi]
    intervals (None = unbounded) plus the grid used, so a disjoint or unbounded set is visible."""
    crit = stats.f.ppf(level, L2, max(N - L, 1))
    ZtZi = np.linalg.pinv(Zp.T @ Zp)
    scale = (N - L) / N if vce != "cluster" else (N - L) / (N - 1)
    cadj = (G / (G - 1)) if vce == "cluster" else 1

    def arF(b0):
        u = yp - b0 * xp
        g = ZtZi @ (Zp.T @ u); r = u - Zp @ g
        return _wald(g, _ols_vcov(Zp, r, vce, cl, N) * cadj) / L2 * scale
    # grid: centred on the IV estimate, width from the reduced-form / first-stage scale
    pi = ZtZi @ (Zp.T @ xp); gam = ZtZi @ (Zp.T @ yp)
    span = max(abs(b_hat), 1e-6) * 50 + 10 * float(np.linalg.norm(gam) / max(np.linalg.norm(pi), 1e-12))
    grid = np.unique(np.r_[np.linspace(b_hat - span, b_hat + span, 1601), b_hat])
    acc = np.array([arF(b) < crit for b in grid])
    if not acc.any():
        return {"intervals": [], "note": "empty at grid resolution (model rejects every value: misspecification?)"}
    iv, i = [], 0
    while i < len(grid):
        if acc[i]:
            j = i
            while j + 1 < len(grid) and acc[j + 1]:
                j += 1
            lo = None if i == 0 else float(_bisect(arF, grid[i - 1], grid[i], crit))
            hi = None if j == len(grid) - 1 else float(_bisect(arF, grid[j], grid[j + 1], crit))
            iv.append([lo, hi]); i = j + 1
        else:
            i += 1
    out = {"intervals": iv, "level": level}
    if any(a is None or b is None for a, b in iv):
        out["note"] = "unbounded within the search grid -> instruments too weak for informative inference"
    return out


def _bisect(f, a, b, crit, it=40):
    fa = f(a) - crit
    for _ in range(it):
        m = (a + b) / 2; fm = f(m) - crit
        if (fm < 0) == (fa < 0):
            a, fa = m, fm
        else:
            b = m
    return (a + b) / 2


def iv_diagnostics(y, Xen, Zx, W, e, b_en, vce, cl, N, small, df_a, endog, instruments, kappa=1.0):
    """First-stage and weak/under-identification diagnostics (ivreg2 conventions, large-sample)."""
    K2, L2 = Xen.shape[1], Zx.shape[1]
    L = L2 + W.shape[1] + df_a  # total instruments incl. exogenous & absorbed (ivreg2: L incl. partialled)
    Xp, Zp, yp = _partial(Xen, W), _partial(Zx, W), _partial(y[:, None], W)[:, 0]
    G = min(len(np.unique(c)) for c in cl) if (vce == "cluster" and cl) else None  # multiway: G = min
    out = {"n_endog": K2, "n_excluded_instruments": L2, "overid_df": L2 - K2}
    # first stages
    fs = []
    for j, nm in enumerate(endog):
        xj = Xp[:, j]
        pi = np.linalg.lstsq(Zp, xj, rcond=None)[0]
        v = xj - Zp @ pi
        Vpi = _ols_vcov(Zp, v, vce, cl, N)
        if vce == "cluster":
            Vpi = Vpi * G / (G - 1)
        Fj = _wald(pi, Vpi) / L2 * ((N - L) / N if vce != "cluster" else (N - L) / N * (G - 1) / G)
        # Fj in ivreg2 small-sample-like scaling for first-stage F
        r2p = 1 - (v @ v) / (xj @ xj)
        # Sanderson-Windmeijer conditional F
        if K2 > 1:
            others = [k for k in range(K2) if k != j]
            Xo = Xp[:, others]
            PzXo = Zp @ np.linalg.lstsq(Zp, Xo, rcond=None)[0]
            d_ = np.linalg.lstsq(PzXo, xj, rcond=None)[0]
            ej = xj - Xo @ d_
            g_ = np.linalg.lstsq(Zp, ej, rcond=None)[0]
            u_ = ej - Zp @ g_
            Vg = _ols_vcov(Zp, u_, vce, cl, N) * ((G / (G - 1)) if vce == "cluster" else 1)
            sw = _wald(g_, Vg) / (L2 - K2 + 1) * ((N - L) / N if vce != "cluster" else (N - L) / N * (G - 1) / G)
        else:
            sw = Fj
        # reported first-stage SEs follow ivreg2's first-stage regressions (small-sample: N/(N-L) or (N-1)/(N-L))
        adj = ((N - 1) / (N - L)) if vce == "cluster" else (N / (N - L))
        fs.append({"endog": nm, "coefs": dict(zip(instruments, pi)),
                   "se": dict(zip(instruments, np.sqrt(np.diag(Vpi) * adj))),
                   "partial_F": Fj, "partial_F_p": float(stats.f.sf(Fj, L2, N - L)), "partial_R2": r2p,
                   "SW_F": sw})
    out["first_stage"] = fs
    # Cragg-Donald (iid) Wald F
    PZ = Zp @ np.linalg.lstsq(Zp, Xp, rcond=None)[0]
    MZ = Xp - PZ
    try:
        A = np.linalg.solve(_msqrt(MZ.T @ MZ), np.eye(K2))
        mineig = float(np.min(np.linalg.eigvalsh(A.T @ (PZ.T @ PZ) @ A)))
        cd = mineig * (N - L) / L2
    except Exception:
        cd = None
    out["cragg_donald_F"] = cd
    # Kleibergen-Paap
    try:
        rkW, df = kp_rk(Xp, Zp, vce, cl, N, wald=True)
        rkLM, _ = kp_rk(Xp, Zp, vce, cl, N, wald=False)
        if vce == "cluster":
            rkW_adj = rkW * (G - 1) / G  # (ranktest uses no G/(G-1)); keep raw
        kpF = rkW / L2 * ((N - L) / (N - 1) * (G - 1) / G if vce == "cluster" else (N - L) / N)
        out.update({"kp_rk_LM": rkLM, "kp_rk_LM_p": float(stats.chi2.sf(rkLM, df)), "kp_rk_Wald": rkW,
                    "kp_rk_Wald_F": kpF, "underid_df": df})
    except Exception as ex:
        out["kp_error"] = str(ex)
    # Anderson-Rubin (H0: all endog coefs = 0)
    g_ = np.linalg.lstsq(Zp, yp, rcond=None)[0]
    u_ = yp - Zp @ g_
    Vg = _ols_vcov(Zp, u_, vce, cl, N) * ((G / (G - 1)) if vce == "cluster" else 1)
    arW = _wald(g_, Vg)
    out["AR_chi2"] = arW
    out["AR_F"] = arW / L2 * ((N - L) / N if vce != "cluster" else (N - L) / (N - 1))
    out["AR_p"] = float(stats.chi2.sf(arW, L2))
    out["AR_F_p"] = float(stats.f.sf(out["AR_F"], L2, max(N - L, 1)))
    if K2 == 1:  # weak-IV-robust 95% confidence set by inverting the AR test over a grid
        try:
            out["AR_CI95"] = _ar_set(yp, Xp[:, 0], Zp, vce, cl, N, L, L2, G, float(b_en[0]))
        except Exception as ex:
            out["AR_CI95_error"] = str(ex)[:120]
    # overidentification
    if L2 > K2:
        Zfull = Zp
        ep = _partial(e[:, None], W)[:, 0]
        if vce == "iid":
            Pe = Zfull @ np.linalg.lstsq(Zfull, ep, rcond=None)[0]
            J = float(N * (Pe @ Pe) / (ep @ ep)); name = "Sargan"
        else:
            Sc = Zfull * ep[:, None]
            if vce == "cluster":
                Sm = cgm_meat(Sc, cl)
            else:
                Sm = Sc.T @ Sc
            Wm = _inv(Sm)
            # efficient two-step GMM using 2SLS residual-based weight matrix
            ZX, Zy = Zfull.T @ Xp, Zfull.T @ yp
            bg = np.linalg.solve(ZX.T @ Wm @ ZX, ZX.T @ Wm @ Zy)
            eg = yp - Xp @ bg
            gbar = Zfull.T @ eg
            J = float(gbar @ Wm @ gbar); name = "Hansen J"
        out["overid"] = {"test": name, "stat": J, "df": L2 - K2, "p": float(stats.chi2.sf(J, L2 - K2))}
    else:
        out["overid"] = {"test": "none", "note": "exactly identified: exclusion restriction cannot be tested"}
    # regression-based (control function) endogeneity test, robust to vce
    V_ = Xp - PZ
    R = np.column_stack([Xp, V_])
    br = np.linalg.lstsq(R, yp, rcond=None)[0]
    er = yp - R @ br
    Vr = _ols_vcov(R, er, vce, cl, N) * ((G / (G - 1)) if vce == "cluster" else 1)
    idx = list(range(K2, 2 * K2))
    ew = _wald(br[idx], Vr[np.ix_(idx, idx)])
    out["endogeneity_test"] = {"stat": ew, "df": K2, "p": float(stats.chi2.sf(ew, K2)),
                               "method": "control-function Wald (H0: regressors exogenous)"}
    # weak-IV flags
    minF = min([f["SW_F"] for f in fs]) if fs else None
    eff = out.get("kp_rk_Wald_F") if vce != "iid" else cd
    out["weak_iv_flag"] = bool(eff is not None and eff < 10)
    sy = SY10.get((K2, L2))
    if sy is not None and eff is not None:
        out["stock_yogo_10pct_size"] = sy
        out["below_stock_yogo_10pct"] = bool(eff < sy)
    out["note"] = ("weak-ID statistic: KP rk Wald F (non-iid) / Cragg-Donald (iid); compare to Stock-Yogo 10% "
                   "critical values (~16.38 for 1 endog/1 IV) or rule-of-thumb 10; per-endog use SW_F")
    return out
