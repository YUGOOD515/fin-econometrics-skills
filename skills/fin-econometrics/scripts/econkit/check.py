"""Compare replicated estimates with numbers reported in a paper.

targets.json:
{"tolerance": {"rel": 0.02},          # optional
 "targets": [
   {"label": "Table 1 (1)", "match": {"id": 1} | {"stored": "m1"} | {"line": 13} | {"dep": "y", "contains": "absorb(id"},
    "coefs": {"AI": {"coef": "0.0392***", "t": "4.6000"}, "Size": {"coef": "0.2556***", "se": "0.0238"}},
    "N": 10563, "stats": {"kp_rk_Wald_F": "25.9028", "r2": "0.80"}}]}
Stats keys: N r2 adj_r2 r2_within pseudo_r2 kp_rk_LM (= Anderson LM under iid) anderson_LM kp_rk_Wald_F cd_F
AR_F hansen_J hansen_p sargan first_stage_F SW_F (first endogenous regressor); any other key = top-level field.
Strings keep the paper's rounding: '0.0392***' -> value 0.0392, 4 decimals, stars '***'.
Per-coefficient fields: coef, se, t (z for large-sample estimators). r2 is the overall R2 Stata prints for
reg/areg/reghdfe (reghdfe "R-squared"); use r2_within for xtreg fe "within" / reghdfe "Within R-sq.".
Optional top-level "estimates": path or list of paths selects which run(s) to check; a target may carry its own
"estimates" path. match keys: id, stored, line, dep, cmd, contains, nth (n-th hit of the other keys).
`ek check --list` prints the index (id / line / stored / cmd / N) of an estimates file.
Verdict: replicated=完全复现 / largely replicated=基本复现 (>=80% exact|close) / partially=部分复现 / not=未能复现.
"""
import json, re, math
from .common import _clean

STAT_KEYS = {"kp_rk_LM": ("iv", "kp_rk_LM"), "kp_rk_Wald_F": ("iv", "kp_rk_Wald_F"), "cd_F": ("iv", "cragg_donald_F"),
             "cragg_donald_F": ("iv", "cragg_donald_F"), "AR_F": ("iv", "AR_F"), "hansen_J": ("iv", "overid", "stat"),
             "anderson_LM": ("iv", "kp_rk_LM"), "underid_LM": ("iv", "kp_rk_LM"), "first_stage_F": ("iv", "first_stage", 0, "partial_F"),
             "SW_F": ("iv", "first_stage", 0, "SW_F"), "hansen_p": ("iv", "overid", "p"), "sargan": ("iv", "overid", "stat"),
             "r2": ("r2",), "adj_r2": ("adj_r2",), "r2_a": ("adj_r2",), "r2_within": ("r2_within",), "N": ("N",),
             "pseudo_r2": ("pseudo_r2",)}


def parse_num(v):
    """-> (value, decimals or None, stars or None)"""
    if v is None:
        return None, None, None
    if isinstance(v, (int, float)):
        s = repr(float(v))
        dec = len(s.split(".")[1]) if "." in s and not s.endswith(".0") else 0
        return float(v), dec, None
    s = str(v).strip().replace("（", "(").replace("）", ")").replace("−", "-").replace("—", "-")
    stars = re.search(r"\*+", s)
    num = re.search(r"-?\d+(?:\.(\d+))?(?:[eE][-+]?\d+)?", s)
    if not num:
        return None, None, stars.group(0) if stars else None
    dec = len(num.group(1)) if num.group(1) else 0
    return float(num.group(0)), dec, (stars.group(0) if stars else "")


def stars_of(p, levels=(0.1, 0.05, 0.01)):
    if p is None:
        return None
    return "***" if p < levels[2] else "**" if p < levels[1] else "*" if p < levels[0] else ""


def compare(mine, paper, dec, rel):
    if mine is None or paper is None:
        return "missing", None
    diff = abs(mine - paper)
    if dec is not None and diff <= 0.5 * 10 ** (-dec) + 1e-12:
        return "exact", diff
    if diff <= rel * max(abs(paper), 1e-12) or (dec is not None and diff <= 10 ** (-dec) + 1e-12):
        return "close", diff
    return "differs", diff


def find(est, m):
    for r in est:
        if "id" in m and r.get("id") != m["id"]:
            continue
        if "stored" in m and m["stored"] not in (r.get("stored_as") or []):
            continue
        if "line" in m and r.get("line") != m["line"]:
            continue
        if "dep" in m and r.get("dep") != m["dep"]:
            continue
        if "contains" in m and m["contains"] not in (r.get("command") or ""):
            continue
        if "cmd" in m and r.get("cmd") != m["cmd"]:
            continue
        if m.get("nth", 1) > 1:  # e.g. {"contains": "ivreg2", "nth": 3}
            m = dict(m, nth=m["nth"] - 1); continue
        return r
    return None


def listing(est):
    """Short index of the estimates (to write `match` keys)."""
    return [f"id={r.get('id')} line={r.get('line')} stored={','.join(r.get('stored_as') or []) or '-'} N={r.get('N')} | "
            f"{(r.get('command') or '')[:110]}" for r in est]


def load_estimates(paths):
    """One path or a list; ids/lines stay per file, so put the file first in `match` via 'file' if needed."""
    out = []
    for p in ([paths] if isinstance(paths, str) else paths):
        for r in json.load(open(p, encoding="utf-8")):
            r = dict(r); r["_file"] = p; out.append(r)
    return out


def _get(r, path):
    cur = r
    for k in path:
        if isinstance(k, int) and isinstance(cur, list) and len(cur) > k:
            cur = cur[k]; continue
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


VERDICT_CN = {"replicated": "完全复现", "largely replicated": "基本复现", "partially replicated": "部分复现",
              "not replicated": "未能复现"}


def n_hint(r, n_paper):
    """Most likely reason for an N mismatch, from the estimate's bookkeeping (shows the arithmetic)."""
    n, sd = int(r.get("N") or 0), int(r.get("singletons_dropped") or 0)
    nl = int(r.get("N_listwise") or 0)
    if sd and n + sd == n_paper:
        return (f"N_listwise({nl or n + sd}) - singletons({sd}) = {n} = your N; paper N = {n_paper} -> the paper kept "
                "singletons: add keepsingletons (or use areg/xtreg)")
    if sd and n < n_paper:
        return f"{sd} singletons dropped (N_listwise={nl}); try keepsingletons, then check sample filters"
    if nl and nl != n and nl == n_paper:
        return f"N_listwise({nl}) = paper N; your estimator dropped {nl - n} more rows (singletons / collinear / FE)"
    if abs(n - n_paper) <= 5:
        return f"off by {n - n_paper}: a missing value in one variable, a boundary year, or one singleton"
    return "check sample filters (years, industries, if-conditions), missing-value handling and lag-induced drops"


def run(estimates_path, targets_path, levels=(0.1, 0.05, 0.01)):
    spec = json.load(open(targets_path, encoding="utf-8"))
    est = load_estimates(estimates_path)
    import os
    base = os.path.dirname(os.path.abspath(targets_path))
    extra = {}
    rel = float(spec.get("tolerance", {}).get("rel", 0.02))
    levels = tuple(spec.get("star_levels", levels))
    rows, summary = [], {"exact": 0, "close": 0, "differs": 0, "missing": 0}
    sign_sig = {"match": 0, "mismatch": 0}
    se_conv = []
    for t in spec["targets"]:
        pool = est
        if t.get("estimates"):  # per-target override
            fp = t["estimates"] if os.path.isabs(t["estimates"]) or os.path.exists(t["estimates"]) else os.path.join(base, t["estimates"])
            if fp not in extra:
                extra[fp] = load_estimates(fp)
            pool = extra[fp]
        r = find(pool, t.get("match", {}))
        lab = t.get("label") or json.dumps(t.get("match"))
        if r is None:
            rows.append({"target": lab, "item": "(estimate)", "status": "missing", "note": "no estimate matched"})
            summary["missing"] += 1
            continue
        for var, spec_v in (t.get("coefs") or {}).items():
            c = r["coefs"].get(var)
            if c is None:
                rows.append({"target": lab, "item": var, "status": "missing", "note": "coefficient not in estimate"})
                summary["missing"] += 1
                continue
            if not isinstance(spec_v, dict):
                spec_v = {"coef": spec_v}
            pv, pdec, pst = parse_num(spec_v.get("coef"))
            st, d = compare(c["coef"], pv, pdec, rel)
            summary[st] += 1
            row = {"target": lab, "item": var, "paper": spec_v.get("coef"), "mine": c["coef"], "status": st}
            my_st = stars_of(c.get("p"), levels)
            for k in ("se", "t"):
                if spec_v.get(k) is not None:
                    v, dd, _ = parse_num(spec_v[k])
                    if k == "t" and c.get("t") is None:
                        continue
                    s2, _ = compare(c.get(k), v, dd, rel)
                    row[f"{k}_paper"], row[f"{k}_mine"], row[f"{k}_status"] = spec_v[k], c.get(k), s2
                    summary[s2] += 1
            if row["status"] in ("exact", "close") and any(row.get(f"{k}_status") == "differs" for k in ("se", "t")):
                row["note"] = "coefficient matches but se/t differ -> SE convention (cluster level, robust vs iid, small, dof)"
                se_conv.append(f"{lab}: {var}")
            if pst is not None and pv is not None and (c.get("p") is not None or var != "_cons"):
                same = (math.copysign(1, c["coef"]) == math.copysign(1, pv)) and (my_st == pst)
                row["stars_paper"], row["stars_mine"] = pst, my_st
                sign_sig["match" if same else "mismatch"] += 1
            rows.append(row)
        for k, v in ({"N": t["N"]} if "N" in t else {}).items():
            st, _ = compare(float(r.get("N") or 0), float(v), 0, 0)
            row = {"target": lab, "item": "N", "paper": v, "mine": r.get("N"), "status": st}
            if st != "exact":
                row["hint"] = n_hint(r, int(float(v)))
            rows.append(row); summary[st] += 1
        for k, v in (t.get("stats") or {}).items():
            path = STAT_KEYS.get(k, (k,))
            mine = _get(r, path)
            pv, pdec, _ = parse_num(v)
            st, _ = compare(mine, pv, pdec, rel)
            rows.append({"target": lab, "item": k, "paper": v, "mine": mine, "status": st}); summary[st] += 1
    n = sum(summary.values())
    ok = summary["exact"] + summary["close"]
    verdict = ("replicated" if n and summary["differs"] == 0 and summary["missing"] == 0 else
               "largely replicated" if n and ok / n >= 0.8 else "partially replicated" if ok else "not replicated")
    lines = ["# Replication check", "", f"verdict: **{verdict}** — {ok}/{n} numbers exact/close; "
             f"sign+significance agree for {sign_sig['match']}/{sign_sig['match'] + sign_sig['mismatch']} coefficients", "",
             "| target | item | paper | mine | status | se/t (paper→mine) | stars paper/mine |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        f = lambda x: f"{x:.4g}" if isinstance(x, float) else ("" if x is None else str(x))
        st = "; ".join(f"{k}: {f(r.get(k + '_paper'))}→{f(r.get(k + '_mine'))} ({r.get(k + '_status')})" for k in ("se", "t") if k + "_status" in r)
        lines.append(f"| {r['target']} | {r['item']} | {f(r.get('paper'))} | {f(r.get('mine'))} | {r['status']} | {st} | "
                     f"{r.get('stars_paper', '')}/{r.get('stars_mine', '') if 'stars_mine' in r else ''} |")
    if se_conv:
        lines += ["", "**SE convention differs** (coefficients match, se/t do not): " + "; ".join(se_conv[:20]),
                  "Try one change at a time with `ek stata --tag`: cluster level, robust vs iid, `small`, areg vs reghdfe dof."]
    if any(r.get("hint") for r in rows):
        lines += ["", "N hints:"] + [f"- {r['target']}: {r['hint']}" for r in rows if r.get("hint")]
    lines += ["", "Note: the verdict is about numbers only. State separately whether the SE convention matches the "
              "table notes and whether the conclusions hold."]
    return _clean({"ok": True, "verdict": verdict, "verdict_cn": VERDICT_CN[verdict], "se_convention_mismatch": se_conv, "counts": summary, "sign_significance": sign_sig, "rows": rows}), "\n".join(lines)
