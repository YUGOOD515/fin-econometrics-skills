# Reporting and data storytelling

The audience is a general reader: smart, busy, and not an econometrician. The technical
appendix is for experts. Write the story in the user's language (Chinese by default for
Chinese users).

## Narrative arc (fill story.json in this order)
1. **Hook / hero**: one number that matters, with a one-line plain caption.
   For example: "0.88: when output rises 1%, cost rises only 0.88%".
2. **Summary (lede)**: the answer in 2–3 sentences, including how sure we are.
3. **Section: the question**: why anyone should care (a person, a price, a policy).
4. **Section: what the data show**: a descriptive chart (scatter or line) before any model.
5. **Section: the naive answer and why it misleads**: baseline vs corrected (coef chart).
   This is where heteroskedasticity, autocorrelation, or endogeneity become a story, e.g.
   "smart people both study longer and earn more, so the simple comparison overstates the
   effect of schooling".
6. **Section: so what**: practical meaning with concrete magnitudes (yuan, %, years).
7. **Caveats**: what this does *not* show.
8. The appendix is generated automatically from `models` and `diagnostics`.

Plain-language rules: name variables in words, not codes; one idea per paragraph; turn
coefficients into everyday units; say "about" and give ranges (the 95% CI); never write
"significant" without saying what it means.

## story.json schema
```json
{
  "title": "...", "subtitle": "...", "eyebrow": "Data story", "lang": "zh-CN",
  "hero": {"number": "6.1%", "caption": "多读一年书，工资大约高 **6%**"},
  "summary": "markdown text",
  "sections": [
    {"heading": "...", "body": "markdown (paragraphs, - lists, **bold**)",
     "charts": [
       {"type": "scatter", "data": "data.csv", "x": "s", "y": "lnw", "fit": true, "title": "...", "caption": "...",
        "xtitle": "...", "ytitle": "..."},
       {"type": "line", "data": "ts.csv", "x": "time", "y": ["consumption"], "y2": ["temp"],
        "labels": {"consumption": "消费", "temp": "气温"}},
       {"type": "bar", "data": "...", "x": "...", "y": "..."},
       {"type": "hist", "data": "...", "x": "..."},
       {"type": "coef", "fits": ["ek_out/fits/ols.json", "ek_out/fits/iv.json"], "vars": ["s"]},
       {"type": "image", "path": "ek_out/plots/base_resid.png"}
     ]}
  ],
  "models": {"fits": ["ek_out/fits/..."], "keep": ["s", "iq"], "labels": {"s": "受教育年限"}, "caption": "..."},
  "diagnostics": "ek_out/diagnose_base.json",
  "method_notes": ["..."], "caveats": ["..."],
  "sources": [{"name": "...", "url": "..."}],
  "appendix_title": "技术附录", "footer": "..."
}
```
Charts accept `"gen": ["lnx = log(x)"]` and `"query": "year>=2000"`. Paths are relative to
story.json. Render the report with `$EK report --story story.json --out report.html`. Add
`--offline` when the reader may have no internet connection.

## Chart choice
- Relationship between two variables → scatter with a fit line. Trend → line (use `y2` for a
  second unit). Distribution → hist. Model comparison → coef (dot with 95% CI).
- Use at most 4 charts in the story. Every chart gets a caption stating its takeaway, not its
  contents.

## QA before you hand off
- Open report.html in a browser (headless screenshot if available) and check that the charts
  render.
- The numbers in the text match `analysis.json` and the fit files.
- If you could not visually verify the report, say so.
