"""v0.3 commands for ek stata: dynamic panel GMM (xtabond2 / xtabond / xtdpdsys), heterogeneity-robust DID
(csdid, eventstudyinteract, did_multiplegt / did_multiplegt_dyn), wild cluster bootstrap (boottest) and
sensitivity analysis (sensemakr). Each command reads the Session state and stores a result dict like the
built-in estimators (coefs / names / vcov / N) plus `extra_lines` for the markdown report."""
import re

import numpy as np
import pandas as pd
from scipy import stats

COMMANDS = {"xtabond2", "xtabond", "xtdpdsys", "csdid", "eventstudyinteract", "did_multiplegt", "did_multiplegt_dyn",
            "boottest", "sensemakr", "did_imputation_ek"}
POST = {"boottest"}  # post-estimation: attach to e(last) instead of creating a new estimate


class ExtError(Exception):
    pass


# ---------------------------------------------------------------------------- option parsing
def parse_opts_multi(s):
    """Like stata.parse_options but keeps repeated options: [(name, value_or_True), ...]."""
    out, i, s = [], 0, (s or "").strip()
    while i < len(s):
        m = re.compile(r"\s*([A-Za-z_][\w]*)").match(s, i)
        if not m:
            i += 1; continue
        name = m.group(1).lower(); i = m.end()
        while i < len(s) and s[i] == " " and s[i:].lstrip().startswith("("):
            i += 1
        if i < len(s) and s[i] == "(":
            depth, j = 0, i
            while j < len(s):
                if s[j] == "(":
                    depth += 1
                elif s[j] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            out.append((name, s[i + 1:j].strip())); i = j + 1
        else:
            out.append((name, True))
    return out


def _abbr(name, full, minlen):
    return full.startswith(name) and len(name) >= minlen


def _get(opts, full, minlen=None, default=None):
    """Last occurrence of option `full` (abbreviation allowed down to minlen chars)."""
    minlen = minlen or len(full)
    v = default
    for k, val in opts:
        if _abbr(k, full, minlen):
            v = val
    return v


def _all(opts, full, minlen=None):
    minlen = minlen or len(full)
    return [val for k, val in opts if _abbr(k, full, minlen)]


def _split_sub(v):
    """'vars, subopts' -> (vars, [(name, val)])"""
    from .stata import top_split
    parts = top_split(v, ",")
    return parts[0].strip(), parse_opts_multi(",".join(parts[1:])) if len(parts) > 1 else []


def _lagpair(v, default=(1, None)):
    if v in (None, True):
        return default
    toks = re.split(r"[\s,]+", str(v).strip())
    vals = [None if t in (".", "") else int(t) for t in toks if t != ""]
    if len(vals) == 1:
        vals.append(None)
    a, b = vals[0], vals[1]
    if a is not None and b is not None and a > b:
        a, b = b, a
    return (a if a is not None else 1), b


def _fmt_p(p):
    return "" if p is None or (isinstance(p, float) and np.isnan(p)) else f"{p:.4f}"


# ---------------------------------------------------------------------------- panel helpers
class Rect:
    """Map the Session data frame onto a rectangular (unit x time) grid."""

    def __init__(self, S, need_time=True):
        if not S.panel or not S.panel[0] or (need_time and not S.panel[1]):
            raise ExtError("panel not set: run `xtset id time` first")
        df = S.need_df()
        pid, tv = S.panel
        self.pid, self.tv = pid, tv
        idv = df[pid]
        self.ids, self.uid = pd.factorize(idv, sort=True)
        tt = pd.to_numeric(df[tv], errors="coerce").astype(float)
        delta = float(getattr(S, "panel_delta", 1) or 1)
        ok = (self.ids >= 0) & tt.notna().values
        t0 = tt[ok].min()
        self.t = np.where(ok, np.round((tt.fillna(t0).values - t0) / delta), -1).astype(int)
        self.ok = ok
        self.t0, self.delta = t0, delta
        self.n, self.T = int(self.ids.max()) + 1, int(self.t[ok].max()) + 1
        key = pd.Series(self.ids[ok] * (self.T + 1) + self.t[ok])
        if key.duplicated().any():
            raise ExtError("repeated time values within panel")

    def arr(self, values, mask=None):
        v = np.asarray(pd.to_numeric(pd.Series(values), errors="coerce"), float)
        A = np.full((self.n, self.T), np.nan)
        sel = self.ok if mask is None else (self.ok & np.asarray(mask, bool))
        A[self.ids[sel], self.t[sel]] = v[sel]
        return A

    def times(self):
        return self.t0 + self.delta * np.arange(self.T)


def _store(S, res, ln, c, line, eststo_name):
    res.update({"cmd": c, "line": ln, "command": line, "id": len(S.estimates) + 1})
    S.last = res
    S.estimates.append(res)
    if eststo_name:
        S.eststo_n += 1
        nm = f"est{S.eststo_n}" if eststo_name == "__auto" else eststo_name
        S.stored[nm] = res; res.setdefault("stored_as", []).append(nm)
    S.log(ln, line, "estimate", None, id=res["id"])


def dispatch(S, ln, c, rest, line, eststo_name):
    fn = {"xtabond2": c_xtabond2, "xtabond": c_xtabond, "xtdpdsys": c_xtdpdsys, "csdid": c_csdid,
          "eventstudyinteract": c_eventstudyinteract, "did_multiplegt": c_did_multiplegt,
          "did_multiplegt_dyn": c_did_multiplegt, "boottest": c_boottest, "sensemakr": c_sensemakr}[c]
    try:
        out = fn(S, ln, c, rest, line)
    except ExtError as ex:
        from .stata import StataError
        raise StataError(str(ex))
    if out is None:
        return
    if c in POST:
        return
    _store(S, out, ln, c, line, eststo_name)


# ============================================================================ dynamic panel GMM
def _dpd_common(S, p, E):
    from .stata import split_command  # noqa
    R = Rect(S)
    msk = S.mask(p["cond"], p["inrange"]).values
    return R, msk


def _dpd_finish(S, R, y_name, E, msk, xnames, specs, system, fod, twostep, robust, small, constant, h, artests,
                diffsargan, label, cons_instr_hidden=False, extra_warn=()):
    from . import dpd
    y = R.arr(E[y_name].values, msk)
    X = np.stack([R.arr(E[n].values, msk) for n in xnames], 2) if xnames else np.zeros((R.n, R.T, 0))
    r = dpd.run(y, X, xnames, specs, system=system, fod=fod, twostep=twostep, robust=robust, small=small,
                constant=constant, h=h, artests=artests, diffsargan=diffsargan)
    r["dep"] = y_name
    r["estimator"] = label
    r["vce"] = ("WC-robust (Windmeijer)" if twostep else "robust (cluster " + R.pid + ")") if robust else \
        ("two-step, NOT robust" if twostep else "one-step, homoskedastic")
    r["cluster"] = [R.pid] if robust else []
    r["n_clusters"] = [r["n_groups"]] if robust else []
    if cons_instr_hidden:
        pass
    L = r["n_instruments"]
    lines = [f"{label}: groups={r['n_groups']} (obs/group min {r['obs_per_group']['min']}, avg "
             f"{r['obs_per_group']['avg']:.2f}, max {r['obs_per_group']['max']}); instruments={L}; "
             + (f"F({r['wald']['df'][0]},{r['wald']['df'][1]})={r['wald']['F']:.2f}" if small else
                f"Wald chi2({r['wald']['df']})={r['wald']['chi2']:.2f}")]
    ar = "; ".join(f"AR({a['order']}) z={a['z']:.3f} p={_fmt_p(a['p'])}" for a in r["ar_tests"])
    lines.append(f"- Arellano-Bond test in first differences: {ar}")
    sg, hs = r["sargan"], r["hansen"]
    if sg["df"] > 0:
        lines.append(f"- Sargan chi2({sg['df']})={sg['chi2']:.3f} p={_fmt_p(sg['p'])} (not robust); "
                     f"Hansen chi2({hs['df']})={hs['chi2']:.3f} p={_fmt_p(hs['p'])} (robust; weakened by many instruments)")
    else:
        lines.append("- exactly identified: no overidentification test")
    for d in r.get("diff_hansen", [])[:12]:
        lines.append(f"- diff-in-Hansen [{d['group']}]: excluding J({d['df_excl']})={d['hansen_excluding']:.3f} "
                     f"p={_fmt_p(d['p_excl'])}; difference chi2({d['df_diff']})={d['diff']:.3f} p={_fmt_p(d['p_diff'])}")
    r["extra_lines"] = lines
    for w in list(r.get("dpd_warnings", [])) + list(extra_warn):
        S.warnings.append(f"{label}: {w}")
    r["coefs"] = r["coefs"]
    return r


def c_xtabond2(S, ln, c, rest, line):
    from .stata import split_command
    from .dpd import Spec
    p = split_command(rest)
    opts = parse_opts_multi(p["optstr"])
    E = pd.DataFrame(index=S.need_df().index)
    toks = S.tokens(p["body"])
    y_name = S.term_columns(toks[0], E)[0]
    xnames = S.design(toks[1:], E)
    R, msk = _dpd_common(S, p, E)
    system = _get(opts, "noleveleq", 3) is None
    fod = _get(opts, "orthogonal", 4) is not None
    twostep = _get(opts, "twostep", 3) is not None
    robust = _get(opts, "robust", 1) is not None or _get(opts, "vce") not in (None, True) and \
        str(_get(opts, "vce")).lower().startswith("r")
    small = _get(opts, "small") is not None
    constant = system and _get(opts, "noconstant", 6) is None and _get(opts, "nocons", 6) is None
    h = int(_get(opts, "h", 1) or 3) if _get(opts, "h", 1) not in (None, True) else 3
    art = _get(opts, "artests", 2)
    artests = int(art) if art not in (None, True) else 2
    warn = []
    if _get(opts, "cluster", 2) not in (None, True):
        warn.append("cluster() other than the panel variable is not implemented; clustering on the panel id")
    if _get(opts, "arlevels", 4) is not None:
        warn.append("arlevels ignored: AR tests are on first-differenced residuals")
    if p["weight"]:
        warn.append("weights ignored by the dynamic panel GMM engine")
    specs = []
    for kind, full in (("gmm", "gmmstyle"), ("iv", "ivstyle")):
        for v in _all(opts, full, 2):
            vars_, sub = _split_sub(v)
            names = S.design(S.tokens(vars_), E)
            arr = np.stack([R.arr(E[n].values) for n in names], 2)
            eq = _get(sub, "equation", 2, "both")
            eq = {"d": "diff", "l": "level", "b": "both"}.get(str(eq)[:1].lower(), "both")
            if kind == "gmm":
                lag = _lagpair(_get(sub, "laglimits", 3), (1, None))
                if _get(sub, "split") is not None:
                    warn.append("gmm(..., split) treated as eq(both) without splitting the diff-in-Hansen group")
                if _get(sub, "orthogonal", 4) is not None:
                    warn.append("gmm(..., orthogonal) suboption ignored")
                specs.append(Spec("gmm", arr, names, lag=lag, collapse=_get(sub, "collapse", 3) is not None, eq=eq,
                                  passthru=_get(sub, "passthru", 4) is not None))
            else:
                specs.append(Spec("iv", arr, names, eq=eq, passthru=_get(sub, "passthru", 4) is not None,
                                  mz=_get(sub, "mz") is not None))
    if not any(s.kind == "gmm" for s in specs):
        warn.append("no gmmstyle() instruments: this is a plain (transformed) IV regression")
    lab = ("system GMM" if system else "difference GMM") + (" (FOD)" if fod else "") + (", two-step" if twostep else ", one-step")
    r = _dpd_finish(S, R, y_name, E, msk, xnames, specs, system, fod, twostep, robust, small, constant, h, artests,
                    _get(opts, "nodiffsargan", 6) is None, "xtabond2 " + lab, extra_warn=warn)
    return r


def _abond_parse(S, p, E, opts, R, system):
    """Shared xtabond / xtdpdsys mapping onto xtabond2 instrument specs."""
    from .dpd import Spec
    toks = S.tokens(p["body"])
    y_name = S.term_columns(toks[0], E)[0]
    exog = S.design(toks[1:], E)
    lags = int(_get(opts, "lags", 3) or 1) if _get(opts, "lags", 3) not in (None, True) else 1
    mld = _get(opts, "maxldep", 4)
    mld = int(mld) if mld not in (None, True) else None
    mlg = _get(opts, "maxlags", 4)
    mlg = int(mlg) if mlg not in (None, True) else None
    ylag = []
    for k in range(1, lags + 1):
        ylag += S.term_columns((f"L{k}." if k > 1 else "L.") + toks[0], E)
    xnames = list(ylag)
    specs = [Spec("gmm", R.arr(E[ylag[0]].values)[:, :, None], [ylag[0]],
                  lag=(1, mld), eq="both" if system else "diff")]
    lab_inst = [f"L(2/{'.' if mld is None else 1 + mld}).{toks[0]}"]

    def lagstruct(v, default_max):
        vars_, sub = _split_sub(v)
        ls = _get(sub, "lagstruct", 3)
        a, b = (0, None)
        if ls not in (None, True):
            parts = [x.strip() for x in str(ls).split(",")]
            a = int(parts[0]) if parts[0] not in (".", "") else 0
            b = None if len(parts) < 2 or parts[1] in (".", "") else int(parts[1])
        if b is None:
            b = default_max
        return S.tokens(vars_), a, b

    for kind in ("pre", "endogenous"):
        for v in _all(opts, kind, 3):
            vs, a, b = lagstruct(v, mlg)
            for var in vs:
                cols = []
                for k in range(0, a + 1):
                    cols += S.term_columns((f"L{k}." if k > 1 else ("L." if k == 1 else "")) + var, E)
                xnames += cols
                base = cols[-1]
                first = 1 if kind == "pre" else 2
                specs.append(Spec("gmm", R.arr(E[base].values)[:, :, None], [base],
                                  lag=(first, None if b is None else first + b - 1), eq="both" if system else "diff"))
                lab_inst.append(f"L({first}/{'.' if b is None else first + b - 1}).{base}")
    xnames += [x for x in exog if x not in xnames]
    if exog:
        specs.append(Spec("iv", np.stack([R.arr(E[n].values) for n in exog], 2), exog, eq="diff"))
    extra = _get(opts, "inst", 4)
    if extra not in (None, True):
        nm = S.design(S.tokens(extra), E)
        specs.append(Spec("iv", np.stack([R.arr(E[n].values) for n in nm], 2), nm, eq="diff"))
    return y_name, xnames, specs, lab_inst


def c_xtabond(S, ln, c, rest, line, system=False):
    from .stata import split_command
    p = split_command(rest)
    opts = parse_opts_multi(p["optstr"])
    E = pd.DataFrame(index=S.need_df().index)
    R, msk = _dpd_common(S, p, E)
    y_name, xnames, specs, lab_inst = _abond_parse(S, p, E, opts, R, system)
    twostep = _get(opts, "twostep", 3) is not None
    vce = _get(opts, "vce")
    robust = vce not in (None, True) and str(vce).lower().startswith("r")
    nocons = _get(opts, "noconstant", 5) is not None
    art = _get(opts, "artests", 2)
    artests = int(art) if art not in (None, True) else 2
    if system:
        constant, h = (not nocons), 3
    else:
        constant, h = ("level" if not nocons else False), 2
    lab = ("xtdpdsys (Arellano-Bover/Blundell-Bond system GMM)" if system else "xtabond (Arellano-Bond difference GMM)") + \
        (", two-step" if twostep else ", one-step")
    r = _dpd_finish(S, R, y_name, E, msk, xnames, specs, system, False, twostep, robust, False, constant, h, artests,
                    False, lab)
    r["extra_lines"].insert(1, "- GMM-type instruments: " + " ".join(lab_inst))
    if not system and constant:
        r["extra_lines"].append("- _cons is identified only by the level-equation moment (does not affect slopes); "
                                "matches Stata to ~1e-4 in the two-step case")
    if not twostep and not robust:
        r["extra_lines"].append("- default vce(gmm): one-step homoskedastic SEs; use vce(robust) unless errors are iid")
    return r


def c_xtdpdsys(S, ln, c, rest, line):
    return c_xtabond(S, ln, c, rest, line, system=True)


# ============================================================================ DID
def _coef_block(names, b, V, dist="z", df=None):
    se = np.sqrt(np.clip(np.diag(V), 0, None)) if len(b) else np.zeros(0)
    crit = stats.norm.ppf(0.975) if df is None else stats.t.ppf(0.975, df)
    co = {}
    for i, nm in enumerate(names):
        z = b[i] / se[i] if se[i] > 0 else np.nan
        p = (2 * stats.norm.sf(abs(z)) if df is None else 2 * stats.t.sf(abs(z), df)) if se[i] > 0 else None
        co[nm] = {"coef": float(b[i]), "se": float(se[i]), "t": float(z), "p": None if p is None else float(p),
                  "ci_low": float(b[i] - crit * se[i]), "ci_high": float(b[i] + crit * se[i])}
    return co


def _wide(S, R, cols, msk):
    return {c: R.arr(S.need_df()[c].values if c in S.need_df() else c, msk) for c in cols}


def c_csdid(S, ln, c, rest, line):
    from .stata import split_command
    from . import did
    p = split_command(rest)
    opts = parse_opts_multi(p["optstr"])
    df = S.need_df()
    ivar = _get(opts, "ivar", 1); tvar = _get(opts, "time", 1); gvar = _get(opts, "gvar", 1)
    if ivar in (None, True):
        raise ExtError("csdid without ivar() (repeated cross-sections) is not implemented; panel data only")
    if tvar in (None, True) or gvar in (None, True):
        raise ExtError("csdid needs time() and gvar()")
    ivar, tvar, gvar = S.resolve_name(ivar), S.resolve_name(tvar), S.resolve_name(gvar)
    old = S.panel
    S.panel = (ivar, tvar)
    try:
        R = Rect(S)
    finally:
        S.panel = old
    E = pd.DataFrame(index=df.index)
    toks = S.tokens(p["body"])
    y = S.term_columns(toks[0], E)[0]
    xs = S.design(toks[1:], E)
    msk = S.mask(p["cond"], p["inrange"]).values & E[[y] + xs].notna().all(1).values & df[gvar].notna().values
    Y = R.arr(E[y].values, msk)
    Gw = R.arr(df[gvar].values, msk)
    G = np.nanmax(np.where(np.isnan(Gw), -np.inf, Gw), 1)
    G = np.where(np.isfinite(G), G, np.nan)
    warn = []
    keep = ~np.isnan(Y).any(1) & ~np.isnan(G)
    if (~keep).any():
        warn.append(f"panel not balanced on the estimation sample: {int((~keep).sum())} of {R.n} units dropped "
                    "(as R did / csdid with balanced panels)")
    times = R.times()
    G = np.where(G > times.max(), 0, G)
    early = (G > 0) & (G <= times.min())
    if early.any():
        warn.append(f"{int(early.sum())} units treated in the first period dropped")
        keep &= ~early
    Xb = None
    if xs:
        Xb = np.stack([R.arr(E[x].values, msk) for x in xs], 2)[keep]
    ctrl = "notyet" if _get(opts, "notyet", 6) is not None else "never"
    base = "long2" if (_get(opts, "long2", 5) is not None or _get(opts, "long", 4) is not None) else "varying"
    meth = str(_get(opts, "method", 4, "dripw")).lower()
    meth = {"dripw": "dripw", "drimp": "dripw", "reg": "reg", "ipw": "stdipw", "stdipw": "stdipw"}.get(meth, None)
    if meth is None:
        raise ExtError("method() must be dripw, drimp, reg, ipw or stdipw")
    if str(_get(opts, "method", 4, "")).lower() == "drimp":
        warn.append("method(drimp) approximated by dripw (same estimand; improved-DR weights not implemented)")
    if str(_get(opts, "method", 4, "")).lower() == "ipw":
        warn.append("method(ipw): normalized (Hajek) IPW used, as stdipw")
    if p["weight"]:
        warn.append("weights ignored by csdid engine")
    if _get(opts, "wboot", 2) is not None:
        warn.append("wboot: multiplier bootstrap not implemented; analytic influence-function SEs reported")
    cl = None
    clv = _get(opts, "cluster", 2)
    if clv not in (None, True):
        cl = R.arr(pd.factorize(df[S.resolve_name(clv)])[0].astype(float), msk)
        cl = np.nanmax(cl, 1)[keep]
    Gk = G[keep]
    res = did.cs_attgt(Y[keep], Gk, times, Xb, control=ctrl, base=base, method=meth, cluster=cl)
    tl = lambda v: str(int(v)) if float(v).is_integer() else str(v)
    names = [f"ATT({tl(r['g'])},{tl(r['t'])})" for r in res["rows"]]
    b = np.array([r["att"] for r in res["rows"]])
    aggs = {}
    for k in ("simple", "group", "calendar", "event"):
        o = did.cs_aggregate(res, k)
        if o:
            aggs[k] = o
    agg = _get(opts, "agg", 3)
    lines = [f"csdid (Callaway-Sant'Anna): method={meth}, control={'not-yet-treated' if ctrl == 'notyet' else 'never-treated'}, "
             f"base={'long2' if base == 'long2' else 'varying (short gaps pre-treatment)'}; units={int(keep.sum())}, "
             f"cohorts={', '.join(tl(g) for g in res['glist'])}; SE: analytic influence function, clustered by "
             + (S.resolve_name(clv) if cl is not None else ivar)]
    for k, (o, V) in aggs.items():
        lines.append(f"- agg({k}): " + "; ".join(f"{x['name']} {x['att']:.4f} ({x['se']:.4f})" for x in o))
    if base == "varying":
        lines.append("- pre-treatment ATT(g,t) use short gaps (t-1 vs t): read them as placebo changes, not a cumulative path; "
                     "use long2 for event-study plots comparable to TWFE")
    if agg not in (None, True) and str(agg).lower()[:2] in ("si", "gr", "ca", "ev"):
        k = {"si": "simple", "gr": "group", "ca": "calendar", "ev": "event"}[str(agg).lower()[:2]]
        o, V = aggs[k]
        names = [x["name"] for x in o]; b = np.array([x["att"] for x in o]); Vm = V
    else:
        Vm = res["V"]
    out = {"coefs": _coef_block(names, b, Vm), "names": names, "vcov": np.asarray(Vm).tolist(),
           "N": int(keep.sum() * len(times)), "dep": y, "vce": f"influence function (cluster {ivar if cl is None else clv})",
           "n_units": int(keep.sum()), "extra_lines": lines,
           "aggregations": {k: [{kk: vv for kk, vv in x.items()} for x in o] for k, (o, V) in aggs.items()},
           "attgt": [{k: (v if not isinstance(v, (np.floating, np.integer)) else float(v)) for k, v in r.items()}
                     for r in res["rows"]]}
    for w in warn:
        S.warnings.append(f"L{ln} csdid: {w}")
    return out


def c_eventstudyinteract(S, ln, c, rest, line):
    """Sun & Abraham (2021) interaction-weighted estimator, following eventstudyinteract.ado (v0.1)."""
    from .stata import split_command
    from .hdfe import estimate
    p = split_command(rest)
    opts = parse_opts_multi(p["optstr"])
    df = S.need_df()
    E = pd.DataFrame(index=df.index)
    toks = S.tokens(p["body"])
    y = S.term_columns(toks[0], E)[0]
    rel = S.design(toks[1:], E)
    coh = _get(opts, "cohort", 3); ctl = _get(opts, "control_cohort", 9)
    ab = _get(opts, "absorb", 1)
    if coh in (None, True) or ctl in (None, True) or ab in (None, True):
        raise ExtError("eventstudyinteract needs cohort(), control_cohort() and absorb()")
    coh, ctl = S.resolve_name(coh), S.resolve_name(ctl)
    covs = _get(opts, "covariates", 3)
    covn = S.design(S.tokens(covs), E) if covs not in (None, True) else []
    fe = []
    for t in str(ab).split():
        cols = [S.resolve_name(re.sub(r"^i\.", "", x)) for x in t.split("#")]
        for col in cols:
            E[col] = df[col].values
        fe.append(cols)
    E["__coh"] = pd.to_numeric(df[coh], errors="coerce").values
    E["__ctl"] = pd.to_numeric(df[ctl], errors="coerce").values
    vopts = {k: v for k, v in [(k, v) for k, v in opts if k in ("vce", "cluster", "cl", "robust", "r")]}
    vce, clv = S.vce_spec(vopts, "eventstudyinteract")
    clcols = []
    for t in clv:
        cols = [S.resolve_name(x) for x in t.split("#")]
        for col in cols:
            E[col] = df[col].values
        clcols.append(cols)
    need = [y] + rel + covn + ["__coh", "__ctl"] + [x for f in fe for x in f]
    msk = S.mask(p["cond"], p["inrange"]).values & E[need].notna().all(1).values
    Ed = E[msk].copy()
    ctl1 = Ed["__ctl"].values == 1
    for r_ in rel:
        Ed.loc[ctl1, r_] = 0.0
    cohorts = sorted(pd.unique(Ed.loc[~ctl1, "__coh"]))
    X = Ed.loc[~ctl1, rel].values.astype(float)
    Nn = X.shape[0]
    ff_w, resid = [], []
    for e in cohorts:
        ci = (Ed.loc[~ctl1, "__coh"].values == e).astype(float)
        bb = np.linalg.lstsq(X, ci, rcond=None)[0]
        bb[np.all(X == 0, 0)] = 0.0
        ff_w.append(bb); resid.append(ci - X @ bb)
    ff_w = np.array(ff_w)  # (ncoh, nrel)
    Sxx = X.T @ X / Nn
    Sxxi = np.linalg.pinv(Sxx)
    Zs = np.concatenate([np.asarray(resid)[k][:, None] * X for k in range(len(cohorts))], 1)  # cohort-major
    Smat = Zs.T @ Zs / Nn
    K = np.kron(np.eye(len(cohorts)), Sxxi)
    Sigma_ff = K @ Smat @ K / Nn
    inames = []
    tl = lambda v: str(int(v)) if float(v).is_integer() else str(v)
    for r_ in rel:
        for e in cohorts:
            nm = f"{r_}_x_{tl(e)}"
            Ed[nm] = (Ed["__coh"].values == e) * Ed[r_].values
            inames.append(nm)
    res = estimate(Ed, y, inames + covn, fe=fe, vce=vce, cluster=clcols, convention="reghdfe", drop_singletons=True)
    res.pop("_index", None); res.pop("_b", None); res.pop("_resid", None)
    nr, nc = len(rel), len(cohorts)
    bvec = np.array([res["coefs"].get(nm, {}).get("coef", 0.0) or 0.0 for nm in inames])
    Vfull = np.array(res["vcov"]) if res.get("vcov") is not None else None
    idx = {nm: i for i, nm in enumerate(res["names"])}
    VV = np.zeros((nr * nc, nr * nc))
    for a, na in enumerate(inames):
        for b_, nb in enumerate(inames):
            if na in idx and nb in idx and Vfull is not None:
                VV[a, b_] = Vfull[idx[na], idx[nb]]
    delta = bvec.reshape(nr, nc).T  # (ncoh, nrel)
    b_iw = (ff_w * delta).sum(0)
    wlong = np.zeros((nr, nr * nc))
    for i in range(nr):
        wlong[i, i * nc:(i + 1) * nc] = ff_w[:, i]
    V_iw = wlong @ VV @ wlong.T
    sidx = np.arange(nc) * nr
    for i in range(nr):
        for j in range(i + 1):
            Vs = Sigma_ff[np.ix_(sidx + i, sidx + j)]
            V_iw[i, j] += delta[:, i] @ Vs @ delta[:, j]
    V_iw = np.tril(V_iw) + np.tril(V_iw, -1).T
    dfr = res.get("df_r")
    co = _coef_block(rel, b_iw, V_iw, df=dfr if (res.get("small") or vce == "cluster") and dfr else None)
    lines = [f"eventstudyinteract (Sun-Abraham IW): cohorts {', '.join(tl(e) for e in cohorts)}; control cohort = {ctl}==1; "
             f"interacted regression N={res.get('N')}, FE: {' '.join('#'.join(f) for f in fe)}; "
             "SE = interacted-regression VCE + cohort-share estimation term (as eventstudyinteract.ado)",
             "- cohort shares by relative time: " + "; ".join(
                 f"{r_}: " + ",".join(f"{tl(e)}={ff_w[k, i]:.3f}" for k, e in enumerate(cohorts) if ff_w[k, i] > 0)
                 for i, r_ in enumerate(rel))]
    if res.get("omitted_collinear"):
        om = [o for o in res["omitted_collinear"] if "_x_" in o]
        if om:
            lines.append(f"- {len(om)} cohort x relative-time cells not identified (set to 0, as Stata): " + ", ".join(om[:8]))
    out = {"coefs": co, "names": list(rel), "vcov": V_iw.tolist(), "N": res.get("N"), "dep": y,
           "vce": res.get("vce"), "cluster": res.get("cluster"), "n_clusters": res.get("n_clusters"),
           "fe": res.get("fe"), "extra_lines": lines,
           "b_interact": {nm: float(v) for nm, v in zip(inames, bvec)}, "ff_w": ff_w.tolist()}
    return out


def c_did_multiplegt(S, ln, c, rest, line):
    from .stata import split_command
    from . import did
    p = split_command(rest)
    opts = parse_opts_multi(p["optstr"])
    df = S.need_df()
    toks = S.tokens(p["body"])
    if len(toks) < 4:
        raise ExtError(f"{c} syntax: {c} Y G T D, ...")
    yv, gv, tv, dv = [S.resolve_name(t) for t in toks[:4]]
    msk = S.mask(p["cond"], p["inrange"]).values
    d = df.loc[msk, [yv, gv, tv, dv]].dropna()
    warn = []
    if _get(opts, "controls", 4) not in (None, True):
        warn.append("controls() not implemented: estimates are without covariates")
    if _get(opts, "trends_lin", 8) is not None or _get(opts, "trends_nonparam", 8) not in (None, True):
        warn.append("trends options not implemented")
    if p["weight"] or _get(opts, "weight", 6) not in (None, True):
        warn.append("weights ignored (cells weighted by their number of observations)")
    cell = d.groupby([gv, tv]).agg(y=(yv, "mean"), D=(dv, "mean"), N=(yv, "size")).reset_index()
    if cell.N.max() == 1:
        pass
    gid, gu = pd.factorize(cell[gv], sort=True)
    tu = np.array(sorted(cell[tv].unique()))
    tix = np.searchsorted(tu, cell[tv].values)
    n, T = len(gu), len(tu)
    Y = np.full((n, T), np.nan); D = np.full((n, T), np.nan); Nc = np.zeros((n, T))
    Y[gid, tix] = cell.y.values; D[gid, tix] = cell.D.values; Nc[gid, tix] = cell.N.values
    if np.nanmax(np.abs(D - np.round(D))) > 0 and c == "did_multiplegt":
        warn.append("treatment varies within (group, period) cells: cell means used")
    clv = _get(opts, "cluster", 2)
    clus = None
    if clv not in (None, True):
        cv = S.resolve_name(clv)
        cmap = d.groupby(gv)[cv].first()
        clus = pd.factorize(cmap.reindex(gu).values)[0]
    dyn = c == "did_multiplegt_dyn" or _get(opts, "robust_dynamic", 8) is not None or \
        _get(opts, "dynamic", 3) not in (None, True) or _get(opts, "effects", 3) not in (None, True)
    tl = lambda v: str(int(v)) if float(v).is_integer() else str(v)
    if dyn:
        ne = _get(opts, "effects", 3) if c == "did_multiplegt_dyn" else _get(opts, "dynamic", 3)
        ne = int(ne) + (0 if c == "did_multiplegt_dyn" else 1) if ne not in (None, True) else 1
        npl = _get(opts, "placebo", 2)
        npl = int(npl) if npl not in (None, True) else 0
        res = did.did_dyn(Y, D, Nc, effects=ne, placebo=npl, normalized=_get(opts, "normalized", 4) is not None,
                          cluster=clus)
        names, b, se = [], [], []
        for r in res:
            nm = (f"Effect_{r['l']}" if r["kind"] == "effect" else f"Placebo_{r['l']}")
            names.append(nm); b.append(r["est"]); se.append(r["se"])
        effs = [r for r in res if r["kind"] == "effect"]
        if effs:
            Ntot = sum(r["N_switchers"] for r in effs)
            U = sum(r["U"] * r["N_switchers"] for r in effs) / Ntot
            avg = sum(r["est"] * r["N_switchers"] for r in effs) / Ntot
            names.append("Av_eff"); b.append(avg)
            if clus is not None:
                Uc = np.zeros(clus.max() + 1); np.add.at(Uc, clus, U); se.append(float(np.sqrt(np.sum(Uc ** 2))))
            else:
                se.append(float(np.sqrt(np.sum(U ** 2))))
        V = np.diag(np.array(se) ** 2)
        lines = [f"{c} (de Chaisemartin-D'Haultfoeuille DID_l, did_multiplegt_dyn estimator): groups={n}, periods={T}; "
                 "controls = groups with the same baseline treatment not yet switched; analytic cohort-demeaned SEs"
                 + (f" clustered by {clv}" if clus is not None else " (clustered by group)")]
        for r in res:
            lines.append(f"- {r['kind']} {r['l']}: {r['est']:.4f} ({r['se']:.4f}); switchers N={r['N_switchers']:.0f}, cells={r['n_obs']}")
        lines.append("- Av_eff = switcher-weighted mean of the effects (Stata's Av_tot_eff divides by the average number "
                     "of treatment-periods; compare effects, not this line)")
        if c == "did_multiplegt":
            lines.append("- did_multiplegt with robust_dynamic/dynamic(): mapped to the did_multiplegt_dyn estimator; "
                         "dynamic(k) -> effects 1..k+1 (old command's effect 0..k)")
        if npl:
            lines.append("- joint placebo test: " + _joint_placebo([r for r in res if r["kind"] == "placebo"], clus))
        co = {}
        for nm, bi, si in zip(names, b, se):
            z = bi / si if si > 0 else np.nan
            co[nm] = {"coef": float(bi), "se": float(si), "t": float(z), "p": float(2 * stats.norm.sf(abs(z))) if si > 0 else None,
                      "ci_low": float(bi - 1.96 * si), "ci_high": float(bi + 1.96 * si)}
        out = {"coefs": co, "names": names, "vcov": V.tolist(), "N": int(np.sum(Nc > 0)), "dep": yv,
               "vce": "dCDH analytic", "extra_lines": lines}
    else:
        r = did.did_m(Y, D, Nc)
        breps = _get(opts, "breps", 2)
        breps = int(breps) if breps not in (None, True) else 0
        se = None
        if breps > 0:
            rng = np.random.default_rng(int(_get(opts, "seed", 4) or 1) if _get(opts, "seed", 4) not in (None, True) else 1)
            units = np.arange(n) if clus is None else clus
            uu = np.unique(units)
            bs = []
            for _ in range(breps):
                pick = rng.choice(uu, len(uu), replace=True)
                ix = np.concatenate([np.where(units == u)[0] for u in pick])
                bs.append(did.did_m(Y[ix], D[ix], Nc[ix])["DID_M"])
            se = float(np.nanstd(bs, ddof=1))
        co = {"DID_M": {"coef": float(r["DID_M"]), "se": se, "t": (r["DID_M"] / se) if se else None,
                        "p": float(2 * stats.norm.sf(abs(r["DID_M"] / se))) if se else None}}
        lines = [f"did_multiplegt (DID_M, de Chaisemartin-D'Haultfoeuille 2020): instantaneous effect of switching; "
                 f"joiners DID_+ = {r['DID_plus']:.4f}, leavers DID_- = {r['DID_minus']:.4f}, switching cells N={r['N_switch']:.0f}; "
                 + (f"SE: cluster bootstrap with {breps} reps" if breps else "no SE: add breps(#) (bootstrap), or use did_multiplegt_dyn")]
        out = {"coefs": co, "names": ["DID_M"], "vcov": [[se ** 2 if se else 0.0]], "N": int(np.sum(Nc > 0)), "dep": yv,
               "vce": "bootstrap" if breps else "none", "extra_lines": lines}
    for w in warn:
        S.warnings.append(f"L{ln} {c}: {w}")
    return out


def _joint_placebo(pls, clus):
    if not pls:
        return "n/a"
    U = np.column_stack([r["U"] for r in pls])
    if clus is not None:
        Uc = np.zeros((clus.max() + 1, U.shape[1])); np.add.at(Uc, clus, U); U = Uc
    V = U.T @ U
    b = np.array([r["est"] for r in pls])
    try:
        W = float(b @ np.linalg.solve(V, b))
    except np.linalg.LinAlgError:
        W = float(b @ np.linalg.pinv(V) @ b)
    return f"chi2({len(b)})={W:.3f}, p={stats.chi2.sf(W, len(b)):.4f}"


# ============================================================================ boottest
def _parse_hyps(S, body, names):
    """boottest hypotheses -> list of (label, R (q,K), r (q,)) ; {..} {..} = separate tests, else one joint test."""
    body = body.strip()
    groups = re.findall(r"\{([^}]*)\}", body) if body.startswith("{") else [body]
    out = []
    for g in groups:
        g = g.strip()
        if not g:
            cands = [n for n in names if n != "_cons"]
            g = cands[0] if cands else ""
        cons = re.findall(r"\(([^()]*(?:\([^()]*\)[^()]*)*)\)", g) if g.startswith("(") else None
        if not cons:
            if "=" in g:
                cons = [g]
            else:
                cons = [t for t in S.tokens(g)]
        Rs, rs = [], []
        for cst in cons:
            lhs, _, rhs = cst.partition("=")
            rhs = rhs.strip() or "0"
            a, c0, _ = S._lin(f"({lhs})-({rhs})", names)
            Rs.append(a); rs.append(-c0)
        out.append((g, np.array(Rs), np.array(rs)))
    return out


def c_boottest(S, ln, c, rest, line):
    from .stata import split_command
    from .wild import WildBoot
    L = S.last
    if L is None or "_boot" not in L:
        raise ExtError("boottest: run it right after reg / areg / reghdfe / xtreg,fe (OLS-type estimates; IV/MLE not supported)")
    p = split_command(rest)
    opts = parse_opts_multi(p["optstr"])
    spec = dict(L["_boot"])
    names_all = list(spec["x"])
    clv = _get(opts, "cluster", 2)
    if clv not in (None, True):
        spec["cluster"] = [[S.resolve_name(x) for x in t.split("#")] for t in str(clv).split()]
        spec["vce"] = "cluster"
        for cc in spec["cluster"]:
            for col in cc:
                if col not in spec["E"]:
                    spec["E"] = spec["E"].assign(**{col: S.need_df().loc[spec["E"].index, col].values})
    bcl = _get(opts, "bootcluster", 5)
    bcols = None
    if bcl not in (None, True):
        bcols = [S.resolve_name(x) for x in str(bcl).split()]
        for col in bcols:
            if col not in spec["E"]:
                spec["E"] = spec["E"].assign(**{col: S.need_df().loc[spec["E"].index, col].values})
    wb = WildBoot(spec, boot_cluster=bcols)
    reps = int(_get(opts, "reps", 3) or 999) if _get(opts, "reps", 3) not in (None, True) else 999
    seed = _get(opts, "seed", 4)
    seed = int(seed) if seed not in (None, True) else 12345
    weight = str(_get(opts, "weighttype", 6, "rademacher")) if _get(opts, "weighttype", 6) not in (None, True) else \
        (str(_get(opts, "weight", 6)) if _get(opts, "weight", 6) not in (None, True) else "rademacher")
    null = _get(opts, "nonull", 5) is None
    ptype = str(_get(opts, "ptype", 2, "symmetric")).lower()
    lv = _get(opts, "level", 3)
    level = float(lv) / 100 if lv not in (None, True) else 0.95
    noci = _get(opts, "noci", 4) is not None
    # coefficient names of the bootstrap design (main regressors only)
    names = [names_all[i] for i in range(len(names_all))]
    hyps = _parse_hyps(S, p["body"], names)
    lines = L.setdefault("extra_lines", [])
    tests = L.setdefault("tests", [])
    for lab, Rm, rv in hyps:
        Rfull = np.zeros((Rm.shape[0], wb.K))
        for qi in range(Rm.shape[0]):
            for j, nm in enumerate(names):
                if Rm[qi, j] != 0:
                    if j not in wb.col_map:
                        raise ExtError(f"boottest: {nm} was omitted from the model")
                    Rfull[qi, wb.col_map[j]] = Rm[qi, j]
        o = wb.test(Rfull, rv, reps=reps, weight=weight, seed=seed, null=null, ptype=ptype, level=level,
                    ci=(not noci and Rm.shape[0] == 1))
        kind = "WCR" if null else "WCU"
        if wb.hetero:
            kind = "wild (heteroskedastic) " + ("restricted" if null else "unrestricted")
        st = (f"t={o['stat']:.4f}" if o["q"] == 1 else f"F({o['q']})={o['stat']:.4f}")
        ci = ""
        if o.get("ci"):
            lo, hi = o["ci"]
            ci = f"; {int(round(level * 100))}% CI [{'-inf' if lo is None else f'{lo:.4f}'}, {'+inf' if hi is None else f'{hi:.4f}'}]"
        lines.append(f"- boottest `{lab}`: {kind}, {weight} weights, {o['reps']} {'(full enumeration) ' if o['enumerated'] else ''}"
                     f"draws, bootstrap clusters G={o['G_boot']}: {st}, p={o['p']:.4f}{ci}")
        tests.append({"test": "boottest", "expr": lab, "t" if o["q"] == 1 else "F": o["stat"], "p": o["p"],
                      "ci": o.get("ci"), "reps": o["reps"], "G_boot": o["G_boot"], "weight": weight, "kind": kind})
        S.log(ln, line, "display", f"boottest {lab}: p={o['p']:.4f}{ci}")
    for nt in wb.notes:
        S.warnings.append(f"L{ln} boottest: {nt}")
    if wb.Gb < 12 and weight.startswith("r"):
        S.warnings.append(f"L{ln} boottest: only {wb.Gb} bootstrap clusters - Rademacher has at most 2^{wb.Gb} distinct draws; "
                          "consider weight(webb)")
    return None


def c_sensemakr(S, ln, c, rest, line):
    """sensemakr depvar covars, treat(d) [benchmark(vars) gbenchmark(vars) gname() kd() ky() q() alpha() r2dz() r2yz() noreduce]"""
    from .stata import split_command
    from . import sens
    p = split_command(rest)
    opts = parse_opts_multi(p["optstr"])
    df = S.need_df()
    E = pd.DataFrame(index=df.index)
    toks = S.tokens(p["body"])
    y = S.term_columns(toks[0], E)[0]
    xs = S.design(toks[1:], E)
    tr = _get(opts, "treat", 2)
    if tr in (None, True):
        raise ExtError("sensemakr needs treat(varname)")
    d = S.term_columns(str(tr), E)[0]
    xs = [x for x in xs if x != d]
    msk = S.mask(p["cond"], p["inrange"]).values & E[[y, d] + xs].notna().all(1).values
    Ed = E[msk].astype(float)
    from .hdfe import _drop_collinear
    Xc = np.column_stack([Ed[xs].values, np.ones(len(Ed))]) if xs else np.ones((len(Ed), 1))
    _, kept, dropped = _drop_collinear(Xc, xs + ["_cons"])
    xs = [x for x in kept if x != "_cons"]
    yv = Ed[y].values; dv = Ed[d].values
    X = np.column_stack([dv, Ed[xs].values, np.ones(len(Ed))])
    b, se, dfr = sens.ols(yv, X)
    est, sd = float(b[0]), float(se[0]); t = est / sd
    q = float(_get(opts, "q", 1) or 1) if _get(opts, "q", 1) not in (None, True) else 1.0
    al = _get(opts, "alpha", 2)
    alpha = float(al) if al not in (None, True) else 0.05
    reduce = _get(opts, "noreduce", 5) is None
    r2yd = sens.partial_r2(t, dfr)
    rvq = sens.rv(t, dfr, q)
    rva = sens.rv(t, dfr, q, alpha)
    lines = [f"sensemakr (Cinelli-Hazlett): treatment {d}; estimate {est:.4f} (se {sd:.4f}, t {t:.3f}, df {dfr}); "
             f"partial R2 of treatment with outcome R2_YD|X = {100 * r2yd:.2f}%; robustness value RV_q={q:g} = "
             f"{100 * rvq:.2f}%; RV_q,alpha={alpha:g} = {100 * rva:.2f}%",
             f"- reading: confounders explaining {100 * rvq:.2f}% of the residual variance of both {d} and {y} would "
             f"{'reduce the estimate to zero' if q == 1 else f'reduce the estimate by {100 * q:.0f}%'}; "
             f"{100 * rva:.2f}% would make it statistically insignificant at alpha={alpha:g}"]
    kdv = _get(opts, "kd", 2); kyv = _get(opts, "ky", 2)
    kd = [float(v) for v in str(kdv).replace(",", " ").split()] if kdv not in (None, True) else [1.0, 2.0, 3.0]
    ky = [float(v) for v in str(kyv).replace(",", " ").split()] if kyv not in (None, True) else list(kd)
    if len(ky) == 1 and len(kd) > 1:
        ky = ky * len(kd)
    bench = []
    bv = _get(opts, "benchmark", 5)
    if bv not in (None, True):
        for v in S.design(S.tokens(str(bv)), E):
            if v in xs:
                bench.append((v, [v]))
    gb = _get(opts, "gbenchmark", 6)
    if gb not in (None, True):
        gv = [v for v in S.design(S.tokens(str(gb)), E) if v in xs]
        gname = _get(opts, "gname", 5)
        bench.append((str(gname) if gname not in (None, True) else "group", gv))
    bounds = []
    Xd = np.column_stack([Ed[xs].values, np.ones(len(Ed))])
    for nm, cols in bench:
        jy = [1 + xs.index(cv) for cv in cols]
        jd = [xs.index(cv) for cv in cols]
        if len(cols) == 1:
            bj, sj, _ = sens.ols(yv, X)
            r2y = sens.partial_r2(bj[jy[0]] / sj[jy[0]], dfr)
            bdd, sdd, dfd = sens.ols(dv, Xd)
            r2d = sens.partial_r2(bdd[jd[0]] / sdd[jd[0]], dfd)
        else:
            r2y = sens.group_partial_r2(yv, X, jy)
            r2d = sens.group_partial_r2(dv, Xd, jd)
        for kdi, kyi in zip(kd, ky):
            bb = sens.bounds(r2d, r2y, kdi, kyi)
            if bb is None:
                lines.append(f"- bound {kdi:g}x {nm}: implied confounder impossible (R2 >= 1)")
                continue
            r2dz, r2yz = bb
            adj, adj_se, at, lo, hi = sens.adjusted(est, sd, dfr, r2dz, r2yz, reduce=reduce, alpha=alpha)
            bounds.append({"bound": f"{kdi:g}x {nm}", "kd": kdi, "ky": kyi, "r2dz_x": r2dz, "r2yz_dx": r2yz,
                           "adjusted_estimate": adj, "adjusted_se": adj_se, "adjusted_t": at, "ci": [lo, hi]})
            lines.append(f"- bound {kdi:g}x {nm}: R2_DZ|X={100 * r2dz:.2f}%, R2_YZ|DX={100 * r2yz:.2f}% -> adjusted estimate "
                         f"{adj:.4f} (se {adj_se:.4f}, t {at:.3f}, {100 * (1 - alpha):.0f}% CI [{lo:.4f}, {hi:.4f}])")
    for nmv, vals in (("r2dz", _get(opts, "r2dz", 4)), ("r2yz", _get(opts, "r2yz", 4))):
        pass
    r2dzv, r2yzv = _get(opts, "r2dz", 4), _get(opts, "r2yz", 4)
    if r2dzv not in (None, True) and r2yzv not in (None, True):
        for a_, b_ in zip(str(r2dzv).split(), str(r2yzv).split()):
            adj, adj_se, at, lo, hi = sens.adjusted(est, sd, dfr, float(a_), float(b_), reduce=reduce, alpha=alpha)
            lines.append(f"- manual confounder R2_DZ|X={a_}, R2_YZ|DX={b_}: adjusted {adj:.4f} (se {adj_se:.4f}, CI [{lo:.4f}, {hi:.4f}])")
    co = {d: {"coef": est, "se": sd, "t": t, "p": float(2 * stats.t.sf(abs(t), dfr)),
              "ci_low": est - stats.t.ppf(0.975, dfr) * sd, "ci_high": est + stats.t.ppf(0.975, dfr) * sd}}
    return {"coefs": co, "names": [d], "vcov": [[sd ** 2]], "N": int(len(Ed)), "dep": y, "vce": "iid (sensemakr uses OLS SEs)",
            "extra_lines": lines, "sensitivity": {"r2yd_x": r2yd, "rv_q": rvq, "rv_qa": rva, "q": q, "alpha": alpha,
                                                  "dof": int(dfr), "bounds": bounds},
            "omitted_collinear": [x for x in dropped if x != "_cons"]}
