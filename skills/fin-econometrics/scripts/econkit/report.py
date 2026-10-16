"""Render a single-file HTML story report from a story.json spec (see references/reporting.md)."""
import base64, html, json, os, re
import numpy as np
import pandas as pd
from .common import fail
from .data import load, transform

TOK = {"bg": "#FAFAFA", "surface": "#FFFFFF", "ink": "#15191D", "ink2": "#4E5761", "line": "#E5E7EB",
       "blue": "#3E5C76", "green": "#1F3D33", "gold": "#B8913A"}
PALETTE = [TOK["blue"], TOK["gold"], TOK["green"], "#8A9BA8", "#A65E46", "#6B7F5E"]
HERE = os.path.dirname(os.path.abspath(__file__))
CSS = os.path.join(HERE, "..", "..", "assets", "report.css")


def md(text):
    """Tiny markdown: paragraphs, '- ' lists, **bold**, *em*, `code`, [t](url)."""
    if not text:
        return ""
    def inline(s):
        s = html.escape(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
        s = re.sub(r"\[(.+?)\]\((https?://[^)]+)\)", r'<a href="\2">\1</a>', s)
        return s
    out, lst = [], []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.strip().splitlines()
        if all(l.strip().startswith(("- ", "* ")) for l in lines):
            out.append("<ul>" + "".join(f"<li>{inline(l.strip()[2:])}</li>" for l in lines) + "</ul>")
        else:
            out.append(f"<p>{inline(' '.join(l.strip() for l in lines))}</p>")
    return "\n".join(out)


ZH = {"obs": "观测值", "fit": "线性拟合（斜率 {s}）", "ci": "估计值及 95% 置信区间"}
EN = {"obs": "obs", "fit": "linear fit (slope {s})", "ci": "estimate with 95% CI"}
TXT = ZH


def _layout(title=None, ytitle=None, xtitle=None, y2title=None):
    lay = {"template": "plotly_white", "height": 400, "margin": {"l": 56, "r": 64 if y2title is not None else 24,
                                                                 "t": 48 if title else 16, "b": 96 if xtitle else 64},
           "font": {"family": '-apple-system, "PingFang SC", "Noto Sans SC", "Helvetica Neue", sans-serif',
                    "color": TOK["ink"], "size": 13},
           "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)", "colorway": PALETTE,
           "legend": {"orientation": "h", "x": 0, "y": 1.02, "yanchor": "bottom"},
           "xaxis": {"gridcolor": TOK["line"], "title": {"text": xtitle or ""}},
           "yaxis": {"gridcolor": TOK["line"], "title": {"text": ytitle or ""}}}
    if title:
        lay["title"] = {"text": title, "x": 0, "font": {"size": 15}}
        lay["margin"]["t"] = 80  # room for title + legend row
    if y2title is not None:
        lay["yaxis2"] = {"overlaying": "y", "side": "right", "showgrid": False, "title": {"text": y2title}}
    return lay


def _data(spec, base):
    p = spec.get("data")
    if not p:
        fail("chart needs 'data'")
    df = load(os.path.join(base, p) if not os.path.isabs(p) else p)
    if spec.get("gen"):
        df, _ = transform(df, spec["gen"])
    if spec.get("query"):
        df = df.query(spec["query"])
    return df


def _col(df, c):
    if c not in df:
        fail(f"chart column not found: {c}", available=list(df.columns))
    s = df[c]
    return s.astype(str).tolist() if s.dtype == object else s.tolist()


def r_ok(rows):
    return all(r.get("ci_low") is not None for r in rows)


def chart(spec, base):
    t = spec["type"]
    tr = []
    lay = _layout(spec.get("title"), spec.get("ytitle"), spec.get("xtitle"),
                  spec.get("y2title") if spec.get("y2") else None)
    if t == "image":
        p = spec["path"]; p = p if os.path.isabs(p) else os.path.join(base, p)
        b64 = base64.b64encode(open(p, "rb").read()).decode()
        return f'<img alt="{html.escape(spec.get("title", ""))}" src="data:image/png;base64,{b64}"/>'
    if t in ("line", "bar", "area"):
        df = _data(spec, base)
        if spec.get("sort", True) and t != "bar":
            df = df.sort_values(spec["x"])
        for i, c in enumerate(spec["y"] if isinstance(spec["y"], list) else [spec["y"]]):
            tr.append({"type": "bar" if t == "bar" else "scatter", "mode": "lines+markers" if t == "line" else None,
                       "fill": "tozeroy" if t == "area" else None, "x": _col(df, spec["x"]), "y": _col(df, c),
                       "name": spec.get("labels", {}).get(c, c), "marker": {"size": 5}})
        for c in spec.get("y2", []):
            tr.append({"type": "scatter", "mode": "lines+markers", "x": _col(df, spec["x"]), "y": _col(df, c),
                       "name": spec.get("labels", {}).get(c, c), "yaxis": "y2", "line": {"dash": "dot"}})
    elif t == "scatter":
        df = _data(spec, base)
        x, y = df[spec["x"]].astype(float), df[spec["y"]].astype(float)
        tr.append({"type": "scatter", "mode": "markers", "x": x.tolist(), "y": y.tolist(), "name": spec.get("name", TXT["obs"]),
                   "text": _col(df, spec["hover"]) if spec.get("hover") else None,
                   "marker": {"size": 7, "opacity": .7, "color": TOK["blue"]}})
        if spec.get("fit", True):
            b = np.polyfit(x, y, 1); xs = np.linspace(x.min(), x.max(), 50)
            tr.append({"type": "scatter", "mode": "lines", "x": xs.tolist(), "y": (b[0] * xs + b[1]).tolist(),
                       "name": spec.get("fit_name", TXT["fit"].format(s=f"{b[0]:.3g}")), "line": {"color": TOK["gold"], "width": 2}})
    elif t == "hist":
        df = _data(spec, base)
        tr.append({"type": "histogram", "x": _col(df, spec["x"]), "marker": {"color": TOK["blue"]}, "nbinsx": spec.get("bins", 30)})
    elif t == "coef":
        vars_ = spec["vars"]
        for i, fp in enumerate(spec["fits"]):
            f = load_fit(fp, base)
            rows = [r for r in f["coefs"] if r["name"] in vars_]
            if not r_ok(rows):
                continue
            tr.append({"type": "scatter", "mode": "markers", "name": f.get("label") or f["model"],
                       "y": [spec.get("labels", {}).get(r["name"], r["name"]) for r in rows], "x": [r["coef"] for r in rows],
                       "error_x": {"type": "data", "symmetric": False,
                                   "array": [r["ci_high"] - r["coef"] for r in rows],
                                   "arrayminus": [r["coef"] - r["ci_low"] for r in rows]},
                       "marker": {"size": 10}})
        lay["shapes"] = [{"type": "line", "x0": 0, "x1": 0, "yref": "paper", "y0": 0, "y1": 1,
                          "line": {"color": TOK["ink2"], "dash": "dot", "width": 1}}]
        lay["xaxis"]["title"] = {"text": spec.get("xtitle", TXT["ci"])}
        lay["height"] = max(240, 90 + 60 * len(vars_))
    else:
        fail(f"unknown chart type {t}", allowed=["line", "bar", "area", "scatter", "hist", "coef", "image"])
    tr = [{k: v for k, v in d.items() if v is not None} for d in tr]
    return {"data": tr, "layout": lay}


def load_fit(ref, base):
    """ek fit json, or 'path/stata_estimates.json#KEY' (KEY = stored name, id or line) from `ek stata`."""
    path, _, key = ref.partition("#")
    path = path if os.path.isabs(path) else os.path.join(base, path)
    obj = json.load(open(path, encoding="utf-8"))
    if not isinstance(obj, list):
        return obj
    if not key:
        fail("stata_estimates.json needs '#<stored name|id>' e.g. ek_out/stata/stata_estimates.json#m1")
    est = next((e for e in obj if key in (e.get("stored_as") or []) or str(e.get("id")) == key), None)
    if est is None:
        fail(f"estimate {key} not found in {path}", available=[e.get("stored_as") or e.get("id") for e in obj][:60])
    def st(p):
        return "" if p is None else "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""
    rows = [{"name": n, **v, "stars": st(v.get("p"))} for n, v in est["coefs"].items()]
    vce = est.get("vce") or ""
    if est.get("cluster"):
        vce = f"cluster({','.join(sum(est['cluster'], []) if isinstance(est['cluster'][0], list) else est['cluster'])})"
    fe = est.get("fe") or []
    return {"label": (est.get("stored_as") or [None])[0] or f"#{est.get('id')}", "model": est.get("cmd") or est.get("estimator"),
            "cov_type": vce, "nobs": est.get("N"), "coefs": rows,
            "fe": ", ".join("#".join(f) if isinstance(f, list) else str(f) for f in fe),
            "stats": {"r2": est.get("r2"), "r2_within": est.get("r2_within"), "pseudo_r2": est.get("pseudo_r2")}}


def model_table(fits, keep, base, labels=None):
    F = [load_fit(p, base) for p in fits]
    names = keep or list(dict.fromkeys(r["name"] for f in F for r in f["coefs"]))
    head = "<tr><th></th>" + "".join(f"<th>({i+1})<br><span>{html.escape(f.get('label') or f['model'])}</span></th>" for i, f in enumerate(F)) + "</tr>"
    body = []
    for n in names:
        cells = []
        for f in F:
            r = next((r for r in f["coefs"] if r["name"] == n), None)
            cells.append(f"<td>{r['coef']:.4f}{r['stars']}<br><span>({r['se']:.4f})</span></td>" if r else "<td></td>")
        body.append(f"<tr><th>{html.escape((labels or {}).get(n, n))}</th>{''.join(cells)}</tr>")
    meta = []
    labs = (("model", "估计方法"), ("fe", "固定效应"), ("cov_type", "标准误"), ("nobs", "观测值")) if TXT is ZH else \
        (("model", "Estimator"), ("fe", "FE"), ("cov_type", "SE"), ("nobs", "N"))
    for key, lab in labs:
        if key == "fe" and not any(f.get("fe") for f in F):
            continue
        meta.append(f"<tr class='meta'><th>{lab}</th>" + "".join(f"<td>{f.get(key, '')}</td>" for f in F) + "</tr>")
    r2 = "<tr class='meta'><th>R²</th>" + "".join(
        f"<td>{(f['stats'].get('r2') or f['stats'].get('r2_within') or f['stats'].get('pseudo_r2') or 0):.3f}</td>" for f in F) + "</tr>"
    return (f"<table class='reg'><thead>{head}</thead><tbody>{''.join(body)}{''.join(meta)}{r2}</tbody></table>"
            + ("<p class='note'>括号内为标准误。* p&lt;0.1，** p&lt;0.05，*** p&lt;0.01。</p>" if TXT is ZH else
               "<p class='note'>Standard errors in parentheses. * p&lt;0.1, ** p&lt;0.05, *** p&lt;0.01.</p>"))


def check_table(path, base, labels=None):
    """Paper-vs-replication table from `ek check` (check_report.json)."""
    R = json.load(open(path if os.path.isabs(path) else os.path.join(base, path), encoding="utf-8"))
    zh = TXT is ZH
    stt = {"exact": "一致", "close": "接近", "differs": "不同", "missing": "缺失"} if zh else {}
    f = lambda x: f"{x:.4f}" if isinstance(x, float) else ("" if x is None else str(x))
    hd = ("目标", "项目", "论文", "复现", "状态") if zh else ("target", "item", "paper", "replication", "status")
    rows = "".join(f"<tr class='st-{r['status']}'><td>{html.escape(str(r['target']))}</td><td>{html.escape((labels or {}).get(r['item'], r['item']))}</td>"
                   f"<td>{html.escape(f(r.get('paper')))}</td><td>{html.escape(f(r.get('mine')))}{html.escape(r.get('stars_mine') or '')}</td>"
                   f"<td>{stt.get(r['status'], r['status'])}</td></tr>" for r in R["rows"])
    c, ss = R["counts"], R.get("sign_significance", {})
    n = sum(c.values())
    cap = (f"核对结论：<b>{R['verdict']}</b>；{c['exact'] + c['close']}/{n} 个数值一致或接近；"
           f"系数符号与显著性一致 {ss.get('match', 0)}/{ss.get('match', 0) + ss.get('mismatch', 0)}") if zh else \
          (f"Verdict: <b>{R['verdict']}</b>; {c['exact'] + c['close']}/{n} numbers exact/close; sign+significance agree "
           f"{ss.get('match', 0)}/{ss.get('match', 0) + ss.get('mismatch', 0)}")
    return (f"<p class='note'>{cap}</p><table class='reg'><thead><tr>{''.join(f'<th>{x}</th>' for x in hd)}</tr></thead>"
            f"<tbody>{rows}</tbody></table>")


def diag_table(path, base):
    D = json.load(open(path if os.path.isabs(path) else os.path.join(base, path)))
    rows = [f"<tr><td>{html.escape(r['issue'])}</td><td>{html.escape(r['evidence'])}</td><td>{html.escape(r['action'])}</td></tr>"
            for r in D["recommendations"] if r["severity"] != "info" or r["issue"] == "none"]
    return ("<table class='diag'><thead><tr><th>Issue</th><th>Evidence</th><th>Treatment</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def render(story_path, out_path, offline=False):
    S = json.load(open(story_path, encoding="utf-8"))
    global TXT
    TXT = ZH if str(S.get("lang", "zh-CN")).lower().startswith("zh") else EN
    base = os.path.dirname(os.path.abspath(story_path))
    css = open(CSS, encoding="utf-8").read()
    figs, parts = [], []
    def put_chart(spec):
        c = chart(spec, base)
        if isinstance(c, str):
            return f"<figure>{c}<figcaption>{html.escape(spec.get('caption', ''))}</figcaption></figure>"
        figs.append(c); i = len(figs) - 1
        return f"<figure><div class='chart' id='fig{i}'></div><figcaption>{html.escape(spec.get('caption', ''))}</figcaption></figure>"
    hero = S.get("hero") or {}
    if hero:
        parts.append(f"<section class='hero'><div class='kpi'>{html.escape(str(hero.get('number', '')))}</div>"
                     f"<div class='kpi-cap'>{md(hero.get('caption', ''))}</div></section>")
    if S.get("summary"):
        parts.append(f"<section class='lede'>{md(S['summary'])}</section>")
    for sec in S.get("sections", []):
        h = f"<h2>{html.escape(sec.get('heading', ''))}</h2>" if sec.get("heading") else ""
        ch = "".join(put_chart(c) for c in (sec.get("charts") or ([sec["chart"]] if sec.get("chart") else [])))
        tb = model_table(sec["table"]["fits"], sec["table"].get("keep"), base, sec["table"].get("labels")) if sec.get("table") else ""
        ck = check_table(sec["check"], base, sec.get("labels")) if sec.get("check") else ""
        parts.append(f"<section>{h}{md(sec.get('body', ''))}{ch}{tb}{ck}</section>")
    appendix = []
    if S.get("models"):
        m = S["models"]
        appendix.append(f"<h3>{html.escape(m.get('heading', '模型对比' if TXT is ZH else 'Model comparison'))}</h3>" + md(m.get("caption", "")) +
                        model_table(m["fits"], m.get("keep"), base, m.get("labels")))
    if S.get("diagnostics"):
        appendix.append(("<h3>诊断与处理</h3>" if TXT is ZH else "<h3>Diagnostics &amp; treatment</h3>") + diag_table(S["diagnostics"], base))
    for key, title in ((("method_notes", "方法说明"), ("caveats", "局限与注意")) if TXT is ZH else (("method_notes", "Method notes"), ("caveats", "Caveats"))):
        if S.get(key):
            appendix.append(f"<h3>{html.escape(S.get(key + '_title', title))}</h3><ul>" +
                            "".join(f"<li>{html.escape(x)}</li>" for x in S[key]) + "</ul>")
    if S.get("sources"):
        appendix.append("<h3>Data sources</h3><ul>" + "".join(
            f"<li><a href='{html.escape(s.get('url', '#'))}'>{html.escape(s['name'])}</a></li>" for s in S["sources"]) + "</ul>")
    if appendix:
        parts.append(f"<section class='appendix'><h2>{html.escape(S.get('appendix_title', '技术附录' if TXT is ZH else 'Technical appendix'))}</h2>{''.join(appendix)}</section>")
    if offline:
        import plotly
        js = open(os.path.join(os.path.dirname(plotly.__file__), "package_data", "plotly.min.js"), encoding="utf-8").read()
        pl = f"<script>{js}</script>"
    else:
        pl = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
    figjs = json.dumps(figs, ensure_ascii=False)
    lang = S.get("lang", "zh-CN")
    doc = f"""<!doctype html><html lang="{lang}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(S.get('title', 'Report'))}</title><style>{css}</style>{pl}</head><body><main>
<header><p class="eyebrow">{html.escape(S.get('eyebrow', '数据故事' if TXT is ZH else 'Data story'))}</p><h1>{html.escape(S.get('title', ''))}</h1>
<p class="sub">{html.escape(S.get('subtitle', ''))}</p></header>
{''.join(parts)}
<footer>{html.escape(S.get('footer', 'Generated with fin-econometrics skill · econkit'))}</footer></main>
<script>const FIGS={figjs};const cfg={{displayModeBar:false,responsive:true}};
FIGS.forEach((f,i)=>Plotly.newPlot('fig'+i,f.data,f.layout,cfg));</script></body></html>"""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    open(out_path, "w", encoding="utf-8").write(doc)
    return {"ok": True, "report": os.path.abspath(out_path), "charts": len(figs), "bytes": len(doc), "offline": offline}
