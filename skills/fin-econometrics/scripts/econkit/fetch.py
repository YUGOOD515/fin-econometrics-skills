"""One-shot data acquisition. For scheduled/persistent crawling see references/data-acquisition.md."""
import io, json, os, urllib.request
import pandas as pd
from .common import fail

UA = {"User-Agent": "Mozilla/5.0 (econkit; research use)"}


def _get(url, timeout=20, tries=2):
    import time
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            last = e
            time.sleep(2 * (i + 1))
    fail(f"download failed after {tries} tries: {type(last).__name__}: {last}", url=url,
         hint="source may block this network; try another mirror/source or ask the user for the file")


def fred(series, start=None, end=None):
    """FRED public CSV endpoint (no API key)."""
    frames = []
    for s in series:
        q = f"&cosd={start}" if start else ""
        q += f"&coed={end}" if end else ""
        raw = _get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}{q}")
        df = pd.read_csv(io.BytesIO(raw))
        df.columns = ["date", s]
        df["date"] = pd.to_datetime(df["date"])
        df[s] = pd.to_numeric(df[s], errors="coerce")
        frames.append(df.set_index("date"))
    out = pd.concat(frames, axis=1).reset_index()
    if start:
        out = out[out["date"] >= pd.Timestamp(start)]
    if end:
        out = out[out["date"] <= pd.Timestamp(end)]
    return out


def worldbank(indicators, countries="all", start=None, end=None):
    """World Bank WDI API v2 -> long panel (country, iso3, year, indicators...)."""
    frames = []
    date = f"&date={start}:{end}" if start and end else ""
    for ind in indicators:
        url = (f"https://api.worldbank.org/v2/country/{countries}/indicator/{ind}"
               f"?format=json&per_page=20000{date}")
        js = json.loads(_get(url))
        if not isinstance(js, list) or len(js) < 2 or js[1] is None:
            fail(f"World Bank returned no data for {ind}", response=js[0] if js else None)
        rows = [{"country": r["country"]["value"], "iso3": r["countryiso3code"], "year": int(r["date"]),
                 ind: r["value"]} for r in js[1]]
        frames.append(pd.DataFrame(rows).set_index(["country", "iso3", "year"]))
    return pd.concat(frames, axis=1).reset_index().sort_values(["iso3", "year"])


def url(u):
    raw = _get(u)
    low = u.lower().split("?")[0]
    if low.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw))
    if low.endswith(".json"):
        return pd.read_json(io.BytesIO(raw))
    if low.endswith(".dta"):
        return pd.read_stata(io.BytesIO(raw))
    return pd.read_csv(io.BytesIO(raw))


def html_table(u, match=None, index=0):
    raw = _get(u).decode("utf-8", errors="ignore")
    tables = pd.read_html(io.StringIO(raw), match=match or ".+")
    if not tables:
        fail("no HTML tables found")
    return tables[index]
