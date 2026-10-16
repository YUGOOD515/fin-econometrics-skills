#!/usr/bin/env python3
"""econkit CLI. Run `python scripts/ek.py <command> -h` for options.
All commands print JSON; artifacts go to --out-dir (default ./ek_out)."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
from econkit.common import emit, fail
from econkit import data as D


def model_args(p, need_model=False):
    p.add_argument("--data", required=True)
    p.add_argument("--sheet")
    p.add_argument("--y", required=True)
    p.add_argument("--x", nargs="+", required=True, help="regressors; L.x / L2.x / D.x need --time")
    if need_model:
        p.add_argument("--model", required=True,
                       help="ols|wls|fgls|prais|corc|2sls|liml|gmm|fe|twfe|re|pooled|logit|probit")
    p.add_argument("--cov", help="nonrobust|robust(HC1)|hc3|hac|cluster  (defaults: ols=nonrobust, iv=robust, panel=cluster)")
    p.add_argument("--lags", type=int, help="HAC lags (default Newey-West rule floor(4(T/100)^(2/9)))")
    p.add_argument("--cluster")
    p.add_argument("--weights", help="analytic weight column for wls")
    p.add_argument("--freq", help="frequency-weight column (Stata fweight): rows are expanded")
    p.add_argument("--het-vars", nargs="+", help="variables in the variance function for fgls (default: x)")
    p.add_argument("--het-noconst", action="store_true")
    p.add_argument("--endog", nargs="+")
    p.add_argument("--instruments", nargs="+")
    p.add_argument("--time")
    p.add_argument("--entity")
    p.add_argument("--query", help="pandas query to subset rows, e.g. \"year>=2000\"")
    p.add_argument("--noconst", action="store_true")
    p.add_argument("--label", default=None)
    p.add_argument("--out-dir", default="ek_out")
    p.add_argument("--no-plots", action="store_true")


def load_model_data(a):
    df = D.load(a.data, a.sheet)
    if getattr(a, "freq", None):
        df = df.loc[df.index.repeat(df[a.freq].astype(int))].reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser(prog="ek", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("profile", help="describe a dataset and guess its structure")
    p.add_argument("--data", required=True); p.add_argument("--sheet")

    p = sub.add_parser("transform", help="create variables and save a new dataset")
    p.add_argument("--data", required=True); p.add_argument("--sheet")
    p.add_argument("--gen", nargs="+", default=[], help="'lny = log(y)' ...")
    p.add_argument("--query"); p.add_argument("--out", required=True)

    p = sub.add_parser("diagnose", help="baseline OLS + full diagnostic battery + recommendations")
    model_args(p)
    p = sub.add_parser("fit", help="estimate one model")
    model_args(p, need_model=True)
    p = sub.add_parser("hausman", help="Hausman FE vs RE")
    model_args(p)

    p = sub.add_parser("compare", help="side-by-side table of saved fits")
    p.add_argument("--fits", nargs="+", required=True); p.add_argument("--keep", nargs="+")
    p.add_argument("--out")

    p = sub.add_parser("report", help="render story.json to single-file HTML")
    p.add_argument("--story", required=True); p.add_argument("--out", required=True)
    p.add_argument("--offline", action="store_true", help="inline plotly.js (~4.6MB) instead of CDN")

    p = sub.add_parser("fetch", help="download public data")
    p.add_argument("source", choices=["fred", "worldbank", "url", "table"])
    p.add_argument("--series", nargs="+", help="FRED ids or WB indicator codes")
    p.add_argument("--countries", default="all", help="WB iso codes joined by ';' e.g. CHN;USA")
    p.add_argument("--start"); p.add_argument("--end")
    p.add_argument("--url"); p.add_argument("--match"); p.add_argument("--index", type=int, default=0)
    p.add_argument("--out", required=True)

    p = sub.add_parser("stata", help="run a Stata do-file / Stata commands (replication engine)")
    p.add_argument("--do", help="path to .do file")
    p.add_argument("--data", help="dataset to load first (.dta/.csv/.xlsx)")
    p.add_argument("--cmd", dest="stata_cmd", action="append", help="Stata command(s); repeatable; newlines allowed")
    p.add_argument("--data-dir", action="append", default=[], help="extra folders to search for used/merged files")
    p.add_argument("--est-filter", help="regex: only run estimation lines matching it (data steps always run)")
    p.add_argument("--keep-dummies", action="store_true", help="show factor-dummy coefficients in results")
    p.add_argument("--quiet", action="store_true", help="omit per-estimate brief lines from stdout")
    p.add_argument("--out-dir", default=None,
                   help="default ek_out/stata for --do runs, ek_out/stata/cmd for --cmd-only runs (never overwrites the do-file run)")
    p.add_argument("--tag", help="write to ek_out/stata/<tag>/ (keep several runs side by side)")

    p = sub.add_parser("check", help="compare estimates with the paper's reported numbers")
    p.add_argument("--estimates", default=None,
                   help="stata_estimates.json; default: targets.json 'estimates' key, else ek_out/stata/stata_estimates.json")
    p.add_argument("--targets", required=True, help="targets.json (see econkit/check.py docstring)")
    p.add_argument("--out", default=None, help="markdown report path (default next to estimates)")

    a = ap.parse_args()
    try:  # replayable command log
        import shlex
        od = getattr(a, "out_dir", None) or "ek_out"
        od = "ek_out" if od.startswith("ek_out") else od
        os.makedirs(od, exist_ok=True)
        with open(os.path.join(od, "history.sh"), "a", encoding="utf-8") as f:
            f.write("python " + shlex.join([os.path.abspath(sys.argv[0])] + sys.argv[1:]) + "\n")
    except Exception:
        pass
    if getattr(a, "label", "x") is None:
        a.label = (getattr(a, "model", None) or a.cmd) + ("_" + a.cov if getattr(a, "cov", None) else "")

    if a.cmd == "profile":
        emit({"ok": True, "data": a.data, **D.profile(D.load(a.data, a.sheet))})
    elif a.cmd == "transform":
        df = D.load(a.data, a.sheet)
        df, made = D.transform(df, a.gen)
        if a.query:
            df = df.query(a.query)
        D.save(df, a.out)
        emit({"ok": True, "out": a.out, "created": made, "shape": list(df.shape)})
    elif a.cmd == "diagnose":
        from econkit import diagnose
        r = diagnose.run(a, load_model_data(a))
        fp = os.path.join(a.out_dir, "fits", f"{a.label}.json")
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        json.dump(__import__("econkit.common", fromlist=["_clean"])._clean(r["baseline_ols"]), open(fp, "w"), ensure_ascii=False, indent=1)
        r["baseline_fit_saved"] = fp
        emit(r, os.path.join(a.out_dir, f"diagnose_{a.label}.json"))
    elif a.cmd == "fit":
        from econkit import models
        out, _, _ = models.fit(a, load_model_data(a))
        out["saved"] = os.path.join(a.out_dir, "fits", f"{a.label}.json")
        emit(out, out["saved"])
    elif a.cmd == "hausman":
        from econkit import models
        emit(models.hausman_fe_re(a, load_model_data(a)))
    elif a.cmd == "compare":
        F = [json.load(open(f)) for f in a.fits]
        names = a.keep or list(dict.fromkeys(r["name"] for f in F for r in f["coefs"]))
        hdr = "| var | " + " | ".join(f"({i+1}) {f['label']}" for i, f in enumerate(F)) + " |"
        lines = [hdr, "|" + "---|" * (len(F) + 1)]
        for n in names:
            cells = []
            for f in F:
                r = next((r for r in f["coefs"] if r["name"] == n), None)
                cells.append(f"{r['coef']:.4f}{r['stars']} ({r['se']:.4f})" if r else "")
            lines.append(f"| {n} | " + " | ".join(cells) + " |")
        lines.append("| N | " + " | ".join(str(f["nobs"]) for f in F) + " |")
        lines.append("| SE | " + " | ".join(f["cov_type"] for f in F) + " |")
        md = "\n".join(lines)
        if a.out:
            open(a.out, "w").write(md)
        emit({"ok": True, "markdown": md})
    elif a.cmd == "stata":
        from econkit import stata
        if not (a.do or a.stata_cmd):
            fail("give --do file.do and/or --cmd '...'")
        if not a.out_dir:
            a.out_dir = os.path.join("ek_out", "stata", a.tag) if a.tag else ("ek_out/stata" if a.do else "ek_out/stata/cmd")
        res, _ = stata.run(do=a.do, cmds=a.stata_cmd, data=a.data, data_dirs=a.data_dir, out_dir=a.out_dir,
                           est_filter=a.est_filter, keep_dummies=a.keep_dummies)
        if a.quiet:
            res.pop("estimates", None)
        emit(res)
    elif a.cmd == "check":
        from econkit import check
        if not a.estimates:
            tj = json.load(open(a.targets, encoding="utf-8"))
            e = tj.get("estimates") if isinstance(tj, dict) else None
            if e and not os.path.isabs(e) and not os.path.exists(e):
                e = os.path.join(os.path.dirname(os.path.abspath(a.targets)), e)
            a.estimates = e or "ek_out/stata/stata_estimates.json"
        if not os.path.exists(a.estimates):
            import glob as _g
            cands = sorted(_g.glob("ek_out/stata/**/stata_estimates.json", recursive=True), key=os.path.getmtime)
            fail(f"estimates file not found: {a.estimates}", candidates=cands)
        res, md = check.run(a.estimates, a.targets)
        res["estimates_file"] = os.path.abspath(a.estimates)
        ri = os.path.join(os.path.dirname(os.path.abspath(a.estimates)), "run_info.json")
        if os.path.exists(ri):
            res["estimates_run"] = json.load(open(ri, encoding="utf-8"))
        md = md.replace("# Replication check\n", f"# Replication check\n\nestimates: `{a.estimates}`\n", 1)
        out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.estimates)), "check_report.md")
        open(out, "w", encoding="utf-8").write(md)
        jp = os.path.splitext(out)[0] + ".json"
        json.dump(res, open(jp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        res["report_md"], res["report_json"] = out, jp
        bad = [r for r in res["rows"] if r["status"] in ("differs", "missing")]
        res["rows"] = bad[:40]  # only problems on stdout; full table in report_md
        res["rows_note"] = "only differing/missing rows shown; see report_md for all"
        emit(res)
    elif a.cmd == "report":
        from econkit import report
        emit(report.render(a.story, a.out, a.offline))
    elif a.cmd == "fetch":
        from econkit import fetch
        if a.source == "fred":
            df = fetch.fred(a.series, a.start, a.end)
        elif a.source == "worldbank":
            df = fetch.worldbank(a.series, a.countries, a.start, a.end)
        elif a.source == "url":
            df = fetch.url(a.url)
        else:
            df = fetch.html_table(a.url, a.match, a.index)
        D.save(df, a.out)
        emit({"ok": True, "out": a.out, **D.profile(df)})


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # always answer in JSON so the agent can self-correct
        import traceback
        emit({"ok": False, "error": f"{type(e).__name__}: {e}",
              "trace_tail": traceback.format_exc().strip().splitlines()[-3:]})
        sys.exit(1)
