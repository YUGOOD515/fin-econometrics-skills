#!/usr/bin/env python3
"""Grade one run directory: python grade.py <case_dir> <run_dir>
Checks the analysis.json contract + numeric targets. Qualitative rubric is printed for a judge."""
import json, os, sys

def _collect_estimates(run_dir):
    """All estimates produced in the run: ek stata (list) and ek fit (dict) json files."""
    out = []
    for root, _, files in os.walk(run_dir):
        for f in files:
            if not f.endswith(".json"):
                continue
            p = os.path.join(root, f)
            try:
                o = json.load(open(p, encoding="utf-8"))
            except Exception:
                continue
            if isinstance(o, list) and o and isinstance(o[0], dict) and "coefs" in o[0]:
                for e in o:
                    out.append({"N": e.get("N"), "coefs": {k: v for k, v in e["coefs"].items()}, "src": p})
            elif isinstance(o, dict) and isinstance(o.get("coefs"), list):
                out.append({"N": o.get("nobs"), "coefs": {r["name"]: r for r in o["coefs"]}, "src": p})
    return out


def _match(est, t):
    tol = 0.5 * 10 ** (-t.get("dec", 3)) + 1e-9
    for e in est:
        c = e["coefs"].get(t["var"])
        if not c or c.get("coef") is None or abs(c["coef"] - t["coef"]) > tol:
            continue
        if t.get("N") and e.get("N") and int(e["N"]) != int(t["N"]):
            continue
        ok_t = "t" not in t or (c.get("t") is not None and abs(c["t"] - t["t"]) <= 0.011 + 0.005 * abs(t["t"]))
        ok_se = "se" not in t or (c.get("se") is not None and abs(c["se"] - t["se"]) <= tol)
        return True, ok_t and ok_se, e["src"]
    return False, False, None


def grade_replication(E, run_dir, res, chk):
    ap = None
    for root, _, files in os.walk(run_dir):
        if "analysis.json" in files:
            ap = os.path.join(root, "analysis.json"); break
    chk("analysis_json_exists", ap is not None, 2)
    A = {}
    if ap:
        try:
            A = json.load(open(ap, encoding="utf-8")); chk("analysis_json_valid", True)
        except Exception as e:
            chk("analysis_json_valid", False, 1, str(e))
    R = A.get("replication") or {}
    chk("mode_replication", A.get("mode") == "replication", 1, str(A.get("mode")))
    chk("verdict_present_ok", any(v in json.dumps(R, ensure_ascii=False) for v in E["verdict_ok"]), 2, str(R.get("verdict")))
    chk("conclusions_supported_field", bool(R.get("conclusions_supported")), 1, str(R.get("conclusions_supported")))
    est = _collect_estimates(run_dir)
    for t in E["targets"]:
        found, se_ok, src = _match(est, t)
        chk(f"coef_{t['label']}", found, t.get("w", 1), src or "not found")
        if "t" in t or "se" in t:
            chk(f"inference_{t['label']}", se_ok, 1, "t/se match" if se_ok else "")
    rep = None
    for root, _, files in os.walk(run_dir):
        for f in files:
            if f in ("replication_report.md", "replication_report.html", "report.html"):
                rep = os.path.join(root, f); break
    chk("report_exists", rep is not None, 2, str(rep))
    text = (open(rep, encoding="utf-8").read() if rep else "") + json.dumps(A, ensure_ascii=False)
    chk("report_chinese", sum('\u4e00' <= ch <= '\u9fff' for ch in text) > 300, 1)
    for k in E["critique_keywords"]:
        chk(f"critique_{k['name']}", any(w.lower() in text.lower() for w in k["any"]), k.get("w", 1))
    chk("targets_json_and_check", any(os.path.exists(os.path.join(r, "check_report.md")) for r, _, _ in os.walk(run_dir)), 1)
    chk("history_logged", any("history.sh" in fs for _, _, fs in os.walk(run_dir)), 1)


def main(case_dir, run_dir):
    E = json.load(open(os.path.join(case_dir, "expected.json")))
    if E.get("type") == "replication":
        res = {"case": E["case"], "run": run_dir, "checks": {}, "score": 0, "max": 0}
        def chk(name, ok, w=1, detail=""):
            res["checks"][name] = {"ok": bool(ok), "weight": w, "detail": detail}
            res["max"] += w; res["score"] += w if ok else 0
        grade_replication(E, run_dir, res, chk)
        res["pct"] = round(100 * res["score"] / res["max"], 1)
        res["rubric_for_judge"] = E["rubric"]
        print(json.dumps(res, ensure_ascii=False, indent=2)); return
    res = {"case": E["case"], "run": run_dir, "checks": {}, "score": 0, "max": 0}
    def chk(name, ok, w=1, detail=""):
        res["checks"][name] = {"ok": bool(ok), "weight": w, "detail": detail}
        res["max"] += w; res["score"] += w if ok else 0
    ap = None
    for root, _, files in os.walk(run_dir):
        if "analysis.json" in files:
            ap = os.path.join(root, "analysis.json"); break
    chk("analysis_json_exists", ap is not None, 2)
    if not ap:
        print(json.dumps(res, ensure_ascii=False, indent=2)); return
    base = os.path.dirname(ap)
    try:
        A = json.load(open(ap)); chk("analysis_json_valid", True)
    except Exception as e:
        chk("analysis_json_valid", False, 1, str(e)); print(json.dumps(res, ensure_ascii=False, indent=2)); return
    issues = [i.lower() for i in A.get("issues_detected", [])]
    for m in E["must_detect"]:
        chk(f"detect_{m}", any(m in i for i in issues), 3)
    for m in E.get("bonus_detect", []):
        chk(f"bonus_{m}", any(m in i for i in issues), 1)
    fm = A.get("final_model", {})
    fp = fm.get("fit_file") and os.path.join(base, fm["fit_file"])
    F = None
    if fp and os.path.exists(fp):
        F = json.load(open(fp))
    chk("final_fit_file_exists", F is not None, 2, str(fp))
    if F:
        chk("final_model_type", F["model"] in E["final_model_ok"], 2, F["model"])
        chk("final_cov_type", F["cov_type"] in E["final_cov_ok"] or F["model"] in ("prais_winsten", "cochrane_orcutt"), 2, F["cov_type"])
        r = next((c for c in F["coefs"] if c["name"] == E["key_var"]), None)
        lo, hi = E["key_range"]
        chk("key_coef_in_range", r is not None and lo <= r["coef"] <= hi, 3, str(r and r["coef"]))
    chk("has_comparison_fits", len(A.get("comparison_fits", [])) >= 2, 1)
    chk("endogeneity_assessment_written", len(A.get("endogeneity_assessment", "")) > 40, 1)
    chk("caveats_present", len(A.get("caveats", [])) >= 1, 1)
    txt = json.dumps(A, ensure_ascii=False)
    chk("conclusion_keywords", all(k in txt for k in E["conclusion_keywords"]), 1)
    rp = A.get("story", {}).get("report")
    rpath = rp and os.path.join(base, rp)
    ok = rpath and os.path.exists(rpath)
    chk("report_exists", ok, 2, str(rpath))
    if ok:
        h = open(rpath, encoding="utf-8").read()
        chk("report_has_charts", h.count("Plotly.newPlot") >= 1 and ('"data": [' in h or "FIGS=[{" in h), 1)
    chk("run_sh_exists", any(os.path.exists(os.path.join(d, x)) for d in (base, run_dir) for x in ("run.sh", "ek_out/history.sh")), 1)
    res["pct"] = round(100 * res["score"] / res["max"], 1)
    res["rubric_for_judge"] = E["rubric"]
    print(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
