#!/usr/bin/env python3
"""Grade one run directory: python grade.py <case_dir> <run_dir>
Checks the analysis.json contract + numeric targets. Qualitative rubric is printed for a judge."""
import json, os, sys

def main(case_dir, run_dir):
    E = json.load(open(os.path.join(case_dir, "expected.json")))
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
    chk("run_sh_exists", any(os.path.exists(os.path.join(d, "run.sh")) for d in (base, run_dir)), 1)
    res["pct"] = round(100 * res["score"] / res["max"], 1)
    res["rubric_for_judge"] = E["rubric"]
    print(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
