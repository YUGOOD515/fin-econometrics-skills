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


def _layout(title=None, ytitle=None, xtitle=None, y2title=None):
    lay = {"template": "plotly_white", "height": 380, "margin": {"l": 56, "r": 24, "t": 48 if title else 16, "b": 48},
           "font": {"family": '-apple-system, "PingFang SC", "Noto Sans SC", "Helvetica Neue", sans-serif',
                    "color": TOK["ink"], "size": 13},
           "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)", "colorway": PALETTE,
           "legend": {"orientation": "h", "y": -0.18},
           "xaxis": {"gridcolor": TOK["line"], "title": {"text": xtitle or ""}},
           "yaxis": {"gridcolor": TOK["line"], "title": {"text": ytitle or ""}}}
    if title:
        lay["title"] = {"text": title, "x": 0, "font": {"size": 15}}
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
        tr.append({"type": "scatter", "mode": "markers", "x": x.tolist(), "y": y.tolist(), "name": spec.get("name", "obs"),
                   "text": _col(df, spec["hover"]) if spec.get("hover") else None,
                   "marker": {"size": 7, "opacity": .7, "color": TOK["blue"]}})
        if spec.get("fit", True):
            b = np.polyfit(x, y, 1); xs = np.linspace(x.min(), x.max(), 50)
            tr.append({"type": "scatter", "mode": "lines", "x": xs.tolist(), "y": (b[0] * xs + b[1]).tolist(),
                       "name": f"linear fit (slope {b[0]:.3g})", "line": {"color": TOK["gold"], "width": 2}})
    elif t == "hist":
        df = _data(spec, base)
        tr.append({"type": "histogram", "x": _col(df, spec["x"]), "marker": {"color": TOK["blue"]}, "nbinsx": spec.get("bins", 30)})
    elif t == "coef":
        vars_ = spec["vars"]
        for i, fp in enumerate(spec["fits"]):
            f = json.load(open(fp if os.path.isabs(fp) else os.path.join(base, fp)))
            rows = [r for r in f["coefs"] if r["name"] in vars_]
            tr.append({"type": "scatter", "mode": "markers", "name": f.get("label") or f["model"],
                       "y": [r["name"] for r in rows], "x": [r["coef"] for r in rows],
                       "error_x": {"type": "data", "symmetric": False,
                                   "array": [r["ci_high"] - r["coef"] for r in rows],
                                   "arrayminus": [r["coef"] - r["ci_low"] for r in rows]},
                       "marker": {"size": 10}})
        lay["shapes"] = [{"type": "line", "x0": 0, "x1": 0, "yref": "paper", "y0": 0, "y1": 1,
                          "line": {"color": TOK["ink2"], "dash": "dot", "width": 1}}]
        lay["xaxis"]["title"] = {"text": spec.get("xtitle", "estimate with 95% CI")}
        lay["height"] = max(240, 90 + 60 * len(vars_))
    else:
        fail(f"unknown chart type {t}", allowed=["line", "bar", "area", "scatter", "hist", "coef", "image"])
    tr = [{k: v for k, v in d.items() if v is not None} for d in tr]
    return {"data": tr, "layout": lay}


def model_table(fits, keep, base, labels=None):
    F = [json.load(open(p if os.path.isabs(p) else os.path.join(base, p))) for p in fits]
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
    for key, lab in (("model", "Estimator"), ("cov_type", "SE"), ("nobs", "N")):
        meta.append(f"<tr class='meta'><th>{lab}</th>" + "".join(f"<td>{f.get(key, '')}</td>" for f in F) + "</tr>")
    r2 = "<tr class='meta'><th>R²</th>" + "".join(
        f"<td>{(f['stats'].get('r2') or f['stats'].get('r2_within') or f['stats'].get('pseudo_r2') or 0):.3f}</td>" for f in F) + "</tr>"
    return (f"<table class='reg'><thead>{head}</thead><tbody>{''.join(body)}{''.join(meta)}{r2}</tbody></table>"
            "<p class='note'>Standard errors in parentheses. * p&lt;0.1, ** p&lt;0.05, *** p&lt;0.01.</p>")


def diag_table(path, base):
    D = json.load(open(path if os.path.isabs(path) else os.path.join(base, path)))
    rows = [f"<tr><td>{html.escape(r['issue'])}</td><td>{html.escape(r['evidence'])}</td><td>{html.escape(r['action'])}</td></tr>"
            for r in D["recommendations"] if r["severity"] != "info" or r["issue"] == "none"]
    return ("<table class='diag'><thead><tr><th>Issue</th><th>Evidence</th><th>Treatment</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def render(story_path, out_path, offline=False):
    S = json.load(open(story_path, encoding="utf-8"))
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
        parts.append(f"<section>{h}{md(sec.get('body', ''))}{ch}</section>")
    appendix = []
    if S.get("models"):
        m = S["models"]
        appendix.append(f"<h3>{html.escape(m.get('heading', 'Model comparison'))}</h3>" + md(m.get("caption", "")) +
                        model_table(m["fits"], m.get("keep"), base, m.get("labels")))
    if S.get("diagnostics"):
        appendix.append("<h3>Diagnostics &amp; treatment</h3>" + diag_table(S["diagnostics"], base))
    for key, title in (("method_notes", "Method notes"), ("caveats", "Caveats")):
        if S.get(key):
            appendix.append(f"<h3>{html.escape(S.get(key + '_title', title))}</h3><ul>" +
                            "".join(f"<li>{html.escape(x)}</li>" for x in S[key]) + "</ul>")
    if S.get("sources"):
        appendix.append("<h3>Data sources</h3><ul>" + "".join(
            f"<li><a href='{html.escape(s.get('url', '#'))}'>{html.escape(s['name'])}</a></li>" for s in S["sources"]) + "</ul>")
    if appendix:
        parts.append(f"<section class='appendix'><h2>{html.escape(S.get('appendix_title', 'Technical appendix'))}</h2>{''.join(appendix)}</section>")
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
<header><p class="eyebrow">{html.escape(S.get('eyebrow', 'Data story'))}</p><h1>{html.escape(S.get('title', ''))}</h1>
<p class="sub">{html.escape(S.get('subtitle', ''))}</p></header>
{''.join(parts)}
<footer>{html.escape(S.get('footer', 'Generated with fin-econometrics skill · econkit'))}</footer></main>
<script>const FIGS={figjs};const cfg={{displayModeBar:false,responsive:true}};
FIGS.forEach((f,i)=>Plotly.newPlot('fig'+i,f.data,f.layout,cfg));</script></body></html>"""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    open(out_path, "w", encoding="utf-8").write(doc)
    return {"ok": True, "report": os.path.abspath(out_path), "charts": len(figs), "bytes": len(doc), "offline": offline}
