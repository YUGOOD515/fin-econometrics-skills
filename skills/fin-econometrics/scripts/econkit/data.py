import os, re
import numpy as np
import pandas as pd
from .common import fail

LAG_RE = re.compile(r"^(L|D|F)(\d*)\.(.+)$")


def load(path, sheet=None):
    if not os.path.exists(path):
        fail(f"file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".tsv", ".txt"):
        df = pd.read_csv(path, sep="\t")
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, sheet_name=sheet or 0)
    elif ext == ".dta":
        df = pd.read_stata(path, convert_categoricals=False)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext == ".json":
        df = pd.read_json(path)
    else:
        fail(f"unsupported file type: {ext}")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def save(df, path):
    ext = os.path.splitext(path)[1].lower()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if ext == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _sorted(df, time=None, entity=None):
    keys = [k for k in (entity, time) if k]
    return df.sort_values(keys).reset_index(drop=True) if keys else df


def resolve(df, names, time=None, entity=None):
    """Return DataFrame with requested columns. Supports Stata-like operators
    L.x / L2.x (lag), D.x (first difference), F.x (lead). Requires --time."""
    out = {}
    for n in names:
        m = LAG_RE.match(n)
        if n in df.columns:
            out[n] = df[n]
        elif m:
            op, k, base = m.group(1), int(m.group(2) or 1), m.group(3)
            if base not in df.columns:
                fail(f"unknown variable in {n}: {base}", available=list(df.columns))
            if not time:
                fail(f"operator {n} requires --time")
            g = df.groupby(entity)[base] if entity else df[base]
            if op == "L":
                out[n] = g.shift(k)
            elif op == "F":
                out[n] = g.shift(-k)
            else:
                s = df[base]
                out[n] = s - (g.shift(k))
        else:
            fail(f"unknown variable: {n}", available=list(df.columns))
    return pd.DataFrame(out, index=df.index)


def prepare(df, y, x, extra=(), time=None, entity=None, query=None):
    """Sort, filter, resolve and listwise-delete. Returns (frame, info)."""
    if query:
        df = df.query(query)
    df = _sorted(df, time, entity)
    cols = list(dict.fromkeys([c for c in [y, *x, *extra] if c]))
    data = resolve(df, cols, time, entity)
    for k in (time, entity):
        if k and k not in data:
            data[k] = df[k]
    n0 = len(data)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    return data, {"n_input": n0, "n_used": len(data), "n_dropped_missing": n0 - len(data)}


def profile(df, max_cols=60):
    cols = []
    for c in list(df.columns)[:max_cols]:
        s = df[c]
        info = {"name": c, "dtype": str(s.dtype), "missing": int(s.isna().sum()),
                "unique": int(s.nunique(dropna=True))}
        if pd.api.types.is_numeric_dtype(s):
            d = s.describe()
            info.update({k: float(d[k]) for k in ("mean", "std", "min", "50%", "max") if k in d})
            info["binary"] = bool(set(s.dropna().unique()) <= {0, 1})
            if s.dropna().gt(0).all() and info["unique"] > 10:
                sk = float(s.skew())
                info["skew"] = sk
                if sk > 2:
                    info["hint"] = "strong right skew; consider log()"
        else:
            info["top"] = s.astype(str).value_counts().head(5).to_dict()
        cols.append(info)
    # structure guess
    lc = {c.lower(): c for c in df.columns}
    time_c = [lc[k] for k in ("year", "date", "time", "t", "month", "quarter", "period") if k in lc]
    id_c = [c for c in df.columns if c.lower() in ("id", "code", "firm", "country", "state", "province", "city", "stkcd", "entity", "region")]
    structure = "cross_section"
    if time_c and id_c:
        structure = "panel"
    elif time_c:
        structure = "time_series"
    return {"shape": list(df.shape), "columns": cols, "structure_guess": structure,
            "time_candidates": time_c, "entity_candidates": id_c}


SAFE = {"np": np, "log": np.log, "exp": np.exp, "sqrt": np.sqrt, "abs": np.abs}


def transform(df, gens):
    """gens: list of 'new = expression' using column names and log/exp/sqrt/abs/np."""
    created = []
    for g in gens:
        if "=" not in g:
            fail(f"--gen needs 'name = expr': {g}")
        name, expr = [p.strip() for p in g.split("=", 1)]
        env = dict(SAFE)
        env.update({c: df[c] for c in df.columns if c.isidentifier()})
        try:
            df[name] = eval(expr, {"__builtins__": {}}, env)
        except Exception as e:
            fail(f"cannot evaluate '{g}': {e}")
        created.append(name)
    return df, created
