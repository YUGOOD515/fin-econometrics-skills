"""Heterogeneity-robust difference-in-differences.

* Callaway & Sant'Anna (2021) ATT(g,t) for panel data (csdid / R did): never-treated or not-yet-treated
  controls, `varying` (default) or `long2` base periods, estimators dripw (Sant'Anna-Zhao doubly robust,
  default), reg (outcome regression), stdipw (normalized IPW); analytic influence-function SEs clustered by
  unit (or a coarser cluster), aggregations simple / group / calendar / event incl. the weight-estimation
  influence terms of R did::aggte.
* Sun & Abraham (2021) interaction-weighted event study (eventstudyinteract).
* de Chaisemartin & D'Haultfoeuille DID_M (did_multiplegt, 2020) and DID_l event-study effects /
  placebos (did_multiplegt_dyn, 2024).
"""
import numpy as np
import pandas as pd
from scipy import stats


# ============================================================================ Callaway & Sant'Anna
def _logit(X, D, w=None, it=100):
    n, k = X.shape
    b = np.zeros(k)
    w = np.ones(n) if w is None else w
    for _ in range(it):
        eta = np.clip(X @ b, -35, 35)
        p = 1 / (1 + np.exp(-eta))
        g = X.T @ (w * (D - p))
        H = (X * (w * p * (1 - p))[:, None]).T @ X
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        b += step
        if np.max(np.abs(step)) < 1e-10:
            break
    eta = np.clip(X @ b, -35, 35)
    p = 1 / (1 + np.exp(-eta))
    H = (X * (w * p * (1 - p))[:, None]).T @ X
    return p, np.linalg.pinv(H)  # pinv(H) = vcov of MLE (n * Hessian_ps in DRDID notation / n)


def _ols(X, y, w):
    XtX = (X * w[:, None]).T @ X
    beta = np.linalg.lstsq(XtX, (X * w[:, None]).T @ y, rcond=None)[0]
    return beta, XtX


def att_2x2(dy, D, X, method="dripw"):
    """Panel 2x2 ATT and its influence function (DRDID: drdid_panel / reg_did_panel / std_ipw_did_panel).
    dy: outcome change, D: treated indicator, X: covariates incl. intercept (n,k)."""
    n = len(dy)
    D = D.astype(float)
    if X.shape[1] == 1 or method in ("reg", "dripw", "drimp", "ipw", "stdipw"):
        pass
    w_treat = D
    if X.shape[1] == 1:  # no covariates: all estimators reduce to the simple 2x2 DID
        mt, mc = dy[D == 1].mean(), dy[D == 0].mean()
        att = mt - mc
        p1 = D.mean()
        inf = D * (dy - mt) / p1 - (1 - D) * (dy - mc) / (1 - p1)
        return att, inf
    if method in ("reg", "dripw", "drimp"):
        wc = 1 - D
        beta, XtX = _ols(X, dy, wc)
        out = X @ beta
        XpX_inv = np.linalg.pinv(XtX / n)
        lin_ols = (wc * (dy - out))[:, None] * X @ XpX_inv
    if method in ("dripw", "ipw", "stdipw", "drimp"):
        ps, Vps = _logit(X, D)
        ps = np.minimum(ps, 1 - 1e-6)
        lin_ps = ((D - ps)[:, None] * X) @ (Vps * n)
    if method == "reg":
        att_t = w_treat * dy; att_c = w_treat * out
        et = att_t.mean() / w_treat.mean(); ec = att_c.mean() / w_treat.mean()
        inf_t = (att_t - w_treat * et) / w_treat.mean()
        M1 = (w_treat[:, None] * X).mean(0)
        inf_c = (att_c - w_treat * ec + lin_ols @ M1) / w_treat.mean()
        return et - ec, inf_t - inf_c
    w_cont = ps * (1 - D) / (1 - ps)
    if method in ("ipw", "stdipw"):
        att_t = w_treat * dy; att_c = w_cont * dy
        et = att_t.mean() / w_treat.mean(); ec = att_c.mean() / w_cont.mean()
        inf_t = (att_t - w_treat * et) / w_treat.mean()
        M2 = ((w_cont * (dy - ec))[:, None] * X).mean(0)
        inf_c = (att_c - w_cont * ec + lin_ps @ M2) / w_cont.mean()
        return et - ec, inf_t - inf_c
    # dripw: doubly robust with IPW-normalized weights (drdid_panel)
    r = dy - out
    att_t = w_treat * r; att_c = w_cont * r
    et = att_t.mean() / w_treat.mean(); ec = att_c.mean() / w_cont.mean()
    M1 = (w_treat[:, None] * X).mean(0)
    inf_t = (att_t - w_treat * et - lin_ols @ M1) / w_treat.mean()
    M2 = ((w_cont * (r - ec))[:, None] * X).mean(0)
    M3 = (w_cont[:, None] * X).mean(0)
    inf_c = (att_c - w_cont * ec + lin_ps @ M2 - lin_ols @ M3) / w_cont.mean()
    return et - ec, inf_t - inf_c


def cs_attgt(Y, G, times, Xb=None, control="never", base="varying", method="dripw", cluster=None, alpha=0.05):
    """Y: (n,T) balanced outcomes; G: (n,) first-treatment period (0/NaN = never); times: (T,) period values;
    Xb: (n,T,k) covariates (values at the base period are used) or None; cluster: (n,) codes or None.
    Returns dict with att table, influence matrix (n, J) and helpers for aggregation."""
    n, T = Y.shape
    G = np.where(np.isnan(G), 0, G).astype(float)
    glist = sorted(g for g in np.unique(G) if g > 0)
    rows, IF = [], []
    tpos = {t: i for i, t in enumerate(times)}
    for g in glist:
        if g not in tpos:
            gi = int(np.searchsorted(times, g))
        else:
            gi = tpos[g]
        if gi == 0:
            continue
        for ti in range(1, T):
            t = times[ti]
            if ti >= gi:
                bi = gi - 1
            else:
                bi = ti - 1 if base == "varying" else gi - 1
            if bi == ti:
                continue
            tb = times[bi]
            treated = G == g
            if control == "never":
                ctrl = G == 0
            else:
                ctrl = (G == 0) | ((G > max(t, tb)) & (G != g))
            sel = treated | ctrl
            if treated.sum() == 0 or ctrl.sum() == 0:
                continue
            dy = Y[sel, ti] - Y[sel, bi]
            D = treated[sel].astype(float)
            if Xb is not None:
                X = np.column_stack([np.ones(sel.sum()), Xb[sel, bi, :]])
            else:
                X = np.ones((sel.sum(), 1))
            att, inf = att_2x2(dy, D, X, method)
            full = np.zeros(n)
            full[sel] = inf * (n / sel.sum())
            rows.append({"g": g, "t": t, "base": tb, "att": att, "post": t >= g, "n_treat": int(treated.sum()),
                         "n_ctrl": int(ctrl.sum())})
            IF.append(full)
    IF = np.column_stack(IF) if IF else np.zeros((n, 0))
    out = {"rows": rows, "IF": IF, "G": G, "n": n, "glist": glist, "cluster": cluster, "times": times}
    se = _se_from_if(IF, cluster, n)
    crit = stats.norm.ppf(1 - alpha / 2)
    for r, s in zip(rows, se):
        r["se"] = s
        r["z"] = r["att"] / s if s > 0 else np.nan
        r["p"] = float(2 * stats.norm.sf(abs(r["z"]))) if s > 0 else None
        r["ci"] = (r["att"] - crit * s, r["att"] + crit * s)
    out["V"] = _vcov_from_if(IF, cluster, n)
    return out


def _vcov_from_if(IF, cluster, n):
    if IF.shape[1] == 0:
        return np.zeros((0, 0))
    if cluster is None:
        return IF.T @ IF / n ** 2
    c = pd.factorize(cluster)[0]
    Gc = c.max() + 1
    S = np.zeros((Gc, IF.shape[1]))
    np.add.at(S, c, IF)
    S -= S.mean(0)
    return S.T @ S / n ** 2 * Gc / (Gc - 1)


def _se_from_if(IF, cluster, n):
    V = _vcov_from_if(IF, cluster, n)
    return np.sqrt(np.clip(np.diag(V), 0, None)) if V.size else np.zeros(0)


def _agg(res, keep, weights, use_wif):
    """Aggregate ATT(g,t) entries `keep` with weights (normalized pg) -> (att, IF)."""
    rows, IF, G, n = res["rows"], res["IF"], res["G"], res["n"]
    att = np.array([rows[k]["att"] for k in keep])
    w = np.asarray(weights, float)
    est = float(w @ att)
    inf = IF[:, keep] @ w
    if use_wif:
        pg = np.array([np.mean(G == rows[k]["g"]) for k in keep])
        ind = np.column_stack([(G == rows[k]["g"]).astype(float) for k in keep])
        if1 = (ind - pg) / pg.sum()
        if2 = np.outer((ind - pg).sum(1), pg / pg.sum() ** 2)
        inf = inf + (if1 - if2) @ att
    return est, inf


def cs_aggregate(res, kind="simple", emin=None, emax=None, balance=None):
    rows, G, n = res["rows"], res["G"], res["n"]
    pgs = {g: np.mean(G == g) for g in res["glist"]}
    out = []

    def add(name, keep, w, wif):
        if not keep:
            return None
        e, inf = _agg(res, keep, w, wif)
        out.append({"name": name, "att": e, "inf": inf})
        return out[-1]

    if kind == "simple":
        keep = [k for k, r in enumerate(rows) if r["post"]]
        pg = np.array([pgs[rows[k]["g"]] for k in keep])
        add("ATT", keep, pg / pg.sum(), True)
    elif kind == "group":
        ests = []
        for g in res["glist"]:
            keep = [k for k, r in enumerate(rows) if r["g"] == g and r["post"]]
            if keep:
                ests.append(add(f"G{int(g)}", keep, np.ones(len(keep)) / len(keep), False))
        if ests:
            gl = [int(e["name"][1:]) for e in ests]
            pg = np.array([pgs[g] for g in gl])
            att = np.array([e["att"] for e in ests])
            w = pg / pg.sum()
            inf = np.column_stack([e["inf"] for e in ests]) @ w
            ind = np.column_stack([(G == g).astype(float) for g in gl])
            if1 = (ind - pg) / pg.sum()
            if2 = np.outer((ind - pg).sum(1), pg / pg.sum() ** 2)
            inf = inf + (if1 - if2) @ att
            out.insert(0, {"name": "GAverage", "att": float(w @ att), "inf": inf})
    elif kind == "calendar":
        ests = []
        for t in res["times"]:
            keep = [k for k, r in enumerate(rows) if r["t"] == t and r["post"]]
            if keep:
                pg = np.array([pgs[rows[k]["g"]] for k in keep])
                ests.append(add(f"T{int(t) if float(t).is_integer() else t}", keep, pg / pg.sum(), True))
        if ests:
            inf = np.column_stack([e["inf"] for e in ests]).mean(1)
            out.insert(0, {"name": "CAverage", "att": float(np.mean([e["att"] for e in ests])), "inf": inf})
    elif kind == "event":
        es = sorted(set(round(r["t"] - r["g"], 8) for r in rows))
        if emin is not None:
            es = [e for e in es if e >= emin]
        if emax is not None:
            es = [e for e in es if e <= emax]
        pre, post = [], []
        for e in es:
            keep = [k for k, r in enumerate(rows) if abs((r["t"] - r["g"]) - e) < 1e-8]
            if balance is not None:
                keep = [k for k in keep if any(abs(rr["t"] - rr["g"] - balance) < 1e-8 and rr["g"] == rows[k]["g"]
                                               for rr in rows)]
            if not keep:
                continue
            pg = np.array([pgs[rows[k]["g"]] for k in keep])
            ei = int(round(e))
            r_ = add(f"Tm{-ei}" if ei < 0 else f"Tp{ei}", keep, pg / pg.sum(), True)
            (pre if e < 0 else post).append(r_)
        summ = []
        if pre:
            summ.append({"name": "Pre_avg", "att": float(np.mean([e["att"] for e in pre])),
                         "inf": np.column_stack([e["inf"] for e in pre]).mean(1)})
        if post:
            summ.append({"name": "Post_avg", "att": float(np.mean([e["att"] for e in post])),
                         "inf": np.column_stack([e["inf"] for e in post]).mean(1)})
        out[:0] = summ
    if not out:
        return []
    IF = np.column_stack([o["inf"] for o in out])
    V = _vcov_from_if(IF, res["cluster"], n)
    for o, s in zip(out, np.sqrt(np.clip(np.diag(V), 0, None))):
        o["se"] = float(s)
        del o["inf"]
    return out, V


# ============================================================================ Sun & Abraham
# (implemented in stata_ext via the reghdfe engine: cohort x relative-time interactions + IW aggregation)


# ============================================================================ de Chaisemartin & D'Haultfoeuille
def dcdh_prepare(Yc, Dc, Nc):
    """Group x period cell arrays (n,T): mean outcome, mean treatment, cell size (0 = missing)."""
    return Yc, Dc, Nc


def did_m(Y, D, N):
    """DID_M (de Chaisemartin & D'Haultfoeuille 2020): instantaneous effect of switching, joiners and leavers.
    Y, D, N: (n,T) cell means / sizes (N=0 -> missing). Returns dict with DID_M, DID_plus, DID_minus and the
    per-period components. Only cells with D in {0,1} (binary) or general D (switch defined by any change)."""
    n, T = Y.shape
    num_p = num_m = 0.0
    Np = Nm = 0.0
    comp = []
    for t in range(1, T):
        ok = (N[:, t] > 0) & (N[:, t - 1] > 0)
        dy = Y[:, t] - Y[:, t - 1]
        d0, d1 = D[:, t - 1], D[:, t]
        for dval in np.unique(d0[ok]):
            base = ok & (d0 == dval)
            stay = base & (d1 == dval)
            if stay.sum() == 0:
                continue
            ctrl = np.sum(N[stay, t] * dy[stay]) / np.sum(N[stay, t])
            up = base & (d1 > dval)
            dn = base & (d1 < dval)
            if up.any():
                eff = np.sum(N[up, t] * dy[up]) / np.sum(N[up, t]) - ctrl
                num_p += np.sum(N[up, t]) * eff; Np += np.sum(N[up, t])
                comp.append({"t": t, "type": "joiners", "N": float(np.sum(N[up, t])), "did": float(eff)})
            if dn.any():
                eff = ctrl - np.sum(N[dn, t] * dy[dn]) / np.sum(N[dn, t])
                num_m += np.sum(N[dn, t]) * eff; Nm += np.sum(N[dn, t])
                comp.append({"t": t, "type": "leavers", "N": float(np.sum(N[dn, t])), "did": float(eff)})
    tot = Np + Nm
    return {"DID_M": (num_p + num_m) / tot if tot else np.nan, "DID_plus": num_p / Np if Np else np.nan,
            "DID_minus": num_m / Nm if Nm else np.nan, "N_switch": tot, "components": comp}


def did_dyn(Y, D, N, effects=1, placebo=0, normalized=False, cluster=None):
    """DID_l (de Chaisemartin & D'Haultfoeuille 2024, did_multiplegt_dyn) for l = 1..effects and placebos.

    Y, D, N: (n,T) group x period cell means and sizes (N=0 -> missing; group-level data -> N=1).
    D_g1 = treatment at the group's first period; F_g = first period with D != D_g1; S_g = +1 if the first change
    is upward (switcher in) else -1. Effect l for switcher g: Y_{F_g-1+l} - Y_{F_g-1} minus the N-weighted mean
    change of groups with the same D_g1 that have not switched by F_g-1+l. Placebo l: same with
    Y_{F_g-1-l} - Y_{F_g-1}, for the switchers and controls used by effect l.
    Variance: the cohort-demeaned influence function of the companion paper (switchers demeaned within
    (D_g1, F_g, D_{g,F_g}), controls within (D_g1, t); DoF factors sqrt(n/(n-1)); singleton cohorts fall back
    to the union of switchers and controls at (D_g1, t)); clustered by group or `cluster` (n,) codes."""
    n, T = Y.shape
    obs = N > 0
    first = np.argmax(obs, 1)
    D1 = np.round(D[np.arange(n), first], 10)
    F = np.full(n, T + 10)
    for g in range(n):
        idx = np.where(obs[g] & (np.abs(D[g] - D1[g]) > 1e-12))[0]
        if idx.size:
            F[g] = idx[0]
    sw = F <= T - 1
    S = np.zeros(n)
    DF = np.full(n, np.nan)
    for g in np.where(sw)[0]:
        S[g] = 1.0 if D[g, F[g]] > D1[g] else -1.0
        DF[g] = D[g, F[g]]
    out = []
    for kind, ells in (("effect", range(1, effects + 1)), ("placebo", range(1, placebo + 1))):
        for l in ells:
            C = np.zeros((n, T))     # coefficient on the long difference dY[g, t] (t = F-1+l indexing)
            DY = np.full((n, T), np.nan)
            role = np.zeros((n, T), int)  # 1 switcher cell, 2 control cell
            Nl = 0.0; dsum = 0.0
            for t in range(T):
                if t - l < 0:
                    continue
                if kind == "effect":
                    a, b = t, t - l
                else:
                    a, b = t - 2 * l, t - l
                if a < 0:
                    continue
                ok = obs[:, a] & obs[:, b]
                dy = np.where(ok, Y[:, a] - Y[:, b], np.nan)
                for d in np.unique(D1):
                    swi = sw & (F - 1 + l == t) & (D1 == d) & ok
                    if kind == "placebo":
                        swi &= (F - 1 - l >= 0)
                    if not swi.any():
                        continue
                    ctrl = ((~sw) | (F > t)) & (D1 == d) & ok
                    if not ctrl.any():
                        continue
                    ws = N[swi, t] if kind == "effect" else N[swi, t - l + l]
                    wsum = float(np.sum(S[swi] * N[swi, t]))
                    C[swi, t] += S[swi] * N[swi, t]
                    C[ctrl, t] -= wsum * N[ctrl, t] / np.sum(N[ctrl, t])
                    DY[swi, t] = dy[swi]; DY[ctrl, t] = dy[ctrl]
                    role[swi, t] = 1; role[ctrl, t] = 2
                    Nl += float(np.sum(N[swi, t]))
                    if normalized:
                        for g in np.where(swi)[0]:
                            dsum += N[g, t] * abs(np.sum(D[g, F[g]:t + 1] - D1[g]))
            if Nl == 0:
                continue
            est = float(np.nansum(C * np.nan_to_num(DY))) / Nl
            # ---- variance: demeaned contributions
            E = np.zeros((n, T)); DOF = np.ones((n, T))
            for t in range(T):
                for d in np.unique(D1):
                    s_idx = np.where((role[:, t] == 1) & (D1 == d))[0]
                    c_idx = np.where((role[:, t] == 2) & (D1 == d))[0]
                    union = np.concatenate([s_idx, c_idx])

                    def mean_of(ix):
                        return float(np.sum(N[ix, t] * DY[ix, t]) / np.sum(N[ix, t]))
                    # switchers: cohorts (D1, F, D_F) -- F is fixed by t, so cohort = D_F
                    for dfv in np.unique(DF[s_idx]) if s_idx.size else []:
                        ix = s_idx[DF[s_idx] == dfv]
                        if ix.size >= 2:
                            E[ix, t] = mean_of(ix); DOF[ix, t] = np.sqrt(ix.size / (ix.size - 1))
                        elif union.size >= 2:
                            E[ix, t] = mean_of(union); DOF[ix, t] = np.sqrt(union.size / (union.size - 1))
                    if c_idx.size >= 2:
                        E[c_idx, t] = mean_of(c_idx); DOF[c_idx, t] = np.sqrt(c_idx.size / (c_idx.size - 1))
                    elif c_idx.size == 1 and union.size >= 2:
                        E[c_idx, t] = mean_of(union); DOF[c_idx, t] = np.sqrt(union.size / (union.size - 1))
            U = np.nansum(C * DOF * (np.nan_to_num(DY) - E) * (role > 0), 1) / Nl
            if cluster is not None:
                cc = pd.factorize(cluster)[0]
                Uc = np.zeros(cc.max() + 1); np.add.at(Uc, cc, U)
                se = float(np.sqrt(np.sum(Uc ** 2)))
            else:
                se = float(np.sqrt(np.sum(U ** 2)))
            denom = (dsum / Nl) if (normalized and dsum > 0) else 1.0
            out.append({"kind": kind, "l": l, "est": est / denom, "se": se / denom, "N_switchers": Nl,
                        "n_switcher_groups": int(np.sum(np.any(role == 1, 1))),
                        "n_obs": int(np.sum(role > 0)), "U": U / denom})
    return out
