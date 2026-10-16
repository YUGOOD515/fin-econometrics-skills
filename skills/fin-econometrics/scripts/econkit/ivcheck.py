"""Instrument-construction diagnostics for panel IVs (`ek ivcheck`).

Checks that the replication guide (references/replication.md, 4c) asks for, mechanically:
 time invariance / within-unit share, the level at which the instrument varies, variation left after the
 fixed effects, first-stage and reduced-form partial correlations, shift-share rank-1 structure, and
 closeness to a leave-one-out (peer) mean of the endogenous variable.
"""
import numpy as np
import pandas as pd
from .hdfe import demean, group_codes
from .common import _clean


def _codes(df, cols):
    return group_codes(df, cols) if cols else None


def _resid(df, v, fe_codes):
    x = df[v].to_numpy(float)[:, None]
    return demean(x, fe_codes)[:, 0] if fe_codes else x[:, 0] - x.mean()


def _corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else None


def run(df, z_list, x=None, y=None, entity=None, time=None, absorb=None, levels=None, group=None):
    cols = [c for c in set(z_list + [x, y, entity, time] + (absorb or []) + (levels or []) + ([group] if group else []))
            if c]
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"unknown variables: {miss}")
    d = df[cols].dropna().reset_index(drop=True)
    absorb = absorb if absorb is not None else [c for c in (entity, time) if c]
    fe = [np.asarray(_codes(d, [a])) for a in absorb]
    res = {"N": int(len(d)), "absorb": absorb, "instruments": {}}
    xr = _resid(d, x, fe) if x else None
    yr = _resid(d, y, fe) if y else None
    for z in z_list:
        s = d[z].astype(float); out = {}
        out["sd"] = float(s.std())
        if entity:
            g = d[entity]
            within = s - s.groupby(g).transform("mean")
            out["within_entity_share"] = float(within.var() / s.var()) if s.var() > 0 else None
            out["time_invariant"] = bool(np.nanmax(np.abs(within)) < 1e-9 * max(1.0, abs(s).max()))
            if out["time_invariant"]:
                out["warning_time_invariant"] = ("constant within each unit: lags are identical (one instrument), and unit FE "
                                                 "absorb it entirely - identification only from cross-section")
        if time:  # level of variation: smallest candidate group within which z is constant at each t
            lv, const = [], {}
            for c in (levels or []):
                k = d.groupby([c, time])[z].nunique()
                const[c] = float((k <= 1).mean())
                if const[c] >= 0.95:  # constant in (almost) every cell
                    lv.append(c)
            if const:
                out["share_cells_constant"] = const
            if lv:
                ncell = {c: int(d.groupby([c, time]).ngroups) for c in lv}
                coarsest = min(ncell, key=ncell.get)
                out["varies_only_at"] = f"{coarsest} x {time}"
                out["n_cells"] = ncell[coarsest]
                if coarsest != entity:
                    out["cluster_advice"] = (f"z varies only across {coarsest}-{time} cells ({ncell[coarsest]}): also report "
                                             f"SE clustered by {coarsest} (G={d[coarsest].nunique()})")
            if not lv and levels:
                out["varies_only_at"] = f"unit level (not constant within any of {levels} x {time})"
        zr = _resid(d, z, fe)
        out["sd_after_fe"] = float(zr.std())
        out["share_var_left_after_fe"] = float(zr.var() / s.var()) if s.var() > 0 else None
        if out["share_var_left_after_fe"] is not None and 0 < out["share_var_left_after_fe"] < 0.01:
            out["warning_little_variation"] = (f"only {100 * out['share_var_left_after_fe']:.2f}% of the variance survives the "
                                               "fixed effects: identification rests on a sliver of variation")
        if x:
            out["partial_corr_with_x"] = _corr(zr, xr)
        if y:
            out["partial_corr_with_y_reduced_form"] = _corr(zr, yr)
        if entity and time and not out.get("time_invariant"):
            P = d.pivot_table(index=entity, columns=time, values=z, aggfunc="mean").dropna()
            if P.shape[0] >= 3 and P.shape[1] >= 3:
                sv = np.linalg.svd(P.to_numpy(float), compute_uv=False)
                r1 = float(sv[0] ** 2 / (sv ** 2).sum())
                out["rank1_share"] = r1
                if r1 > 0.999:
                    out["warning_shift_share"] = ("z ~ share_i x shock_t (rank-1): the variation comes from one common time "
                                                  "series scaled by fixed shares; over-id tests among such instruments have no power")
        if x and time and group:
            gx = d.groupby([group, time])[x]
            sm, n = gx.transform("sum"), gx.transform("count")
            loo = (sm - d[x]) / (n - 1).replace(0, np.nan)
            ok = loo.notna()
            if ok.sum() > 10:
                out["corr_with_leave_one_out_mean_x"] = _corr(s[ok].to_numpy(), loo[ok].to_numpy())
                if abs(out["corr_with_leave_one_out_mean_x"] or 0) > 0.95:
                    out["warning_loo"] = (f"z ~ leave-one-out mean of {x} within {group}-{time}: common shocks / sorting "
                                          "affect own y (exclusion); with FE only peer-mean changes identify")
        res["instruments"][z] = out
    if len(z_list) > 1:
        Zr = np.column_stack([_resid(d, z, fe) for z in z_list])
        sv = np.linalg.svd(Zr - Zr.mean(0), compute_uv=False)
        res["instrument_matrix_rank_after_fe"] = int((sv > sv[0] * 1e-8).sum()) if sv[0] > 0 else 0
        if res["instrument_matrix_rank_after_fe"] < len(z_list):
            res["warning_rank"] = (f"only {res['instrument_matrix_rank_after_fe']} of {len(z_list)} instruments are linearly "
                                   "independent after the fixed effects")
    return _clean(res)
