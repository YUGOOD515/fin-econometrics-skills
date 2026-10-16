"""A pragmatic Stata do-file interpreter for replication work.

Supports the data-management and estimation commands that dominate applied
economics replication packages (gen/replace/egen/keep/drop/merge/winsor2/tab,
reg/areg/reghdfe/xtreg/ivreg2/ivreghdfe/ivregress/xtivreg/probit/logit/
poisson/nbreg/ppmlhdfe/sgmediation2/ttest ...), with macros, loops, prefixes,
factor variables and time-series operators. Unsupported commands are logged
as warnings instead of aborting the run.
"""
import os, re, json, glob, math, fnmatch, time
import numpy as np
import pandas as pd
from .stexpr import _isstr
from scipy import stats

from . import stexpr
from .hdfe import estimate, group_codes
from .common import _clean

OUTPUT_ONLY = {"esttab", "estout", "estadd", "outreg2", "outreg", "logout", "display", "di", "dis", "set", "cd",
               "log", "graph", "twoway", "tw", "scatter", "line", "histogram", "hist", "coefplot", "binscatter",
               "kdensity", "label", "lab", "format", "order", "compress", "notes", "note", "version", "clear",
               "cls", "more", "pause", "describe", "des", "codebook", "list", "l", "browse", "br", "edit",
               "putdocx", "putexcel", "matrix", "mat", "mata", "timer", "erase", "shell", "translate", "which",
               "help", "ssc", "net", "ado", "adopath", "sysdir", "macro", "return", "ereturn", "marksample", "esttab,", "graph2tex", "xtline", "xtdes", "xtdescribe",
               "inspect", "sum2docx", "corr2docx", "reg2docx", "t2docx", "tabout", "outsheet", "export", "lookfor", "ds", "misstable", "mvencode", "cap", "noi", "estimates", "est",
               "eststo", "estpost", "asdoc", "local", "global", "scalar", "drop", "keep"}
EXTRA_EST = {"newey", "qreg", "tobit", "ologit", "oprobit", "mlogit", "clogit", "xtlogit", "psmatch2"}
UNSUPPORTED = {"xtabond", "xtdpdsys", "xtdpd", "xtabond2", "sensemakr", "bdiff", "teffects",
               "didregress", "xtdidregress", "csdid", "did_multiplegt", "eventdd", "rdrobust", "synth",
               "heckman", "sqreg", "xtqreg", "spxtregress", "xsmle", "gmm", "sem", "gsem", "boottest",
               "bootstrap", "permute", "simulate", "ritest", "xtgls", "xtpcse", "arima", "var", "vec",
               "garch", "xtunitroot", "xtcointtest", "xttobit",
               "xtprobit", "ivprobit", "biprobit", "medeff", "khb", "estat"}


class StataError(Exception):
    pass


def read_text(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "gb18030", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def strip_comments(text):
    """Remove /* */, // and * comments; join /// continuations. Returns [(lineno, line)]."""
    out, buf, buf_ln = [], "", None
    in_block = 0
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for ln, raw in enumerate(lines, 1):
        s, res, i, q = raw, "", 0, False
        while i < len(s):
            if in_block:
                j = s.find("*/", i); k = s.find("/*", i)
                if k != -1 and (j == -1 or k < j):
                    in_block += 1; i = k + 2; continue
                if j == -1:
                    i = len(s); break
                in_block -= 1; i = j + 2; continue
            ch = s[i]
            if ch == '"':
                q = not q; res += ch; i += 1; continue
            if not q and s.startswith("/*", i):
                in_block += 1; i += 2; continue
            if not q and s.startswith("///", i) and i > 0 and s[i - 1] in " \t":
                res += " \x00CONT"; break
            if not q and s.startswith("//", i) and (i == 0 or s[i - 1] in " \t"):
                break
            res += ch; i += 1
        cont = res.endswith("\x00CONT")
        if cont:
            res = res[: -len(" \x00CONT")]
        if buf_ln is None:
            if res.strip().startswith("*"):
                res = ""
            buf_ln = ln
        buf += " " + res if buf else res
        if not cont:
            if buf.strip():
                out.append((buf_ln, buf.strip()))
            buf, buf_ln = "", None
    if buf.strip():
        out.append((buf_ln, buf.strip()))
    return out


def top_split(s, sep):
    """Split on separator character at depth 0 (outside (), [], quotes)."""
    parts, depth, cur, q = [], 0, "", False
    for ch in s:
        if ch == '"':
            q = not q
        if not q:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            elif ch == sep and depth == 0:
                parts.append(cur); cur = ""; continue
        cur += ch
    parts.append(cur)
    return parts


def find_top(s, pattern):
    """Find regex pattern at depth 0; returns match or None."""
    depth, q = 0, False
    for i, ch in enumerate(s):
        if ch == '"':
            q = not q
        if q:
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif depth == 0:
            m = re.compile(pattern).match(s, i)
            if m and (i == 0 or not (s[i - 1].isalnum() or s[i - 1] in "_.")):
                return m
    return None


def parse_options(s):
    opts, i, s = {}, 0, s.strip()
    while i < len(s):
        m = re.compile(r"\s*([A-Za-z_][\w]*)").match(s, i)
        if not m:
            i += 1; continue
        name = m.group(1).lower(); i = m.end()
        while i < len(s) and s[i] == " " and s[i:].lstrip().startswith("("):
            i += 1
        if i < len(s) and s[i] == "(":
            depth, j = 0, i
            while j < len(s):
                if s[j] == "(":
                    depth += 1
                elif s[j] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            opts[name] = s[i + 1:j].strip(); i = j + 1
        else:
            opts[name] = True
    return opts


def getopt(opts, *names):
    """names: full names with optional minimal abbreviations given as 'cl:cluster'."""
    for spec in names:
        if ":" in spec:
            mn, full = spec.split(":")
        else:
            mn = full = spec
        for k, v in opts.items():
            if full.startswith(k) and k.startswith(mn):
                return v
    return None


def split_command(rest):
    """-> dict(body, cond, inrange, weight, opts)"""
    parts = top_split(rest, ",")
    main = parts[0]
    optstr = ",".join(parts[1:]) if len(parts) > 1 else ""
    weight = None
    m = re.search(r"\[\s*(aw|aweight|pw|pweight|fw|fweight|iw|iweight|weight)\s*=\s*([^\]]+)\]", main)
    if m:
        weight = (m.group(1), m.group(2).strip())
        main = main[:m.start()] + main[m.end():]
    cond = inr = None
    mi = find_top(main, r"in\s+(\S+/\S+|\d+)")
    if mi:
        inr = mi.group(1); main = main[:mi.start()] + main[mi.end():]
    mf = find_top(main, r"if\s+")
    if mf:
        cond = main[mf.end():].strip(); main = main[:mf.start()]
    return {"body": main.strip(), "cond": cond, "inrange": inr, "weight": weight, "opts": parse_options(optstr),
            "optstr": optstr}


def stars(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


DUMMY_RE = re.compile(r"^\d+[bo]?\.|#\d+[bo]?\.|^[\w]*\d+$")


class Session:
    def __init__(self, base_dirs=None, out_dir="ek_out/stata", est_filter=None, max_rows_print=60):
        self.base_dirs = [d for d in (base_dirs or []) if d]
        self.out_dir = out_dir
        self.df = None
        self.globals = {}
        self.locals = {}
        self.scalars = {}
        self.panel = None  # (id, time)
        self.preserved = []
        self.saved = {}
        self.saved_panels = {}
        self.last = None
        self.stored = {}
        self.estimates = []
        self.events = []
        self.warnings = []
        self.errors = []
        self.est_filter = re.compile(est_filter) if est_filter else None
        self.vce_variants = []
        self.eststo_n = 0
        self.stop = False
        self.descriptives = []

    # ------------------------------------------------------------------ logging
    def log(self, ln, cmd, kind, msg=None, **kw):
        ev = {"line": ln, "cmd": cmd[:300], "kind": kind}
        if msg:
            ev["msg"] = msg
        ev.update(kw)
        self.events.append(ev)
        if kind == "warning":
            self.warnings.append(f"L{ln}: {msg} [{cmd[:80]}]")
        if kind == "error" and "underidentified" in str(msg):  # a modelling fact, not an engine failure
            self.warnings.append(f"L{ln}: estimate skipped - {msg} [{cmd[:80]}]")
        elif kind == "error":
            self.errors.append(f"L{ln}: {msg} [{cmd[:120]}]")

    # ------------------------------------------------------------------ files
    def find_file(self, name, exts=(".dta", ".csv", ".xlsx", ".xls")):
        name = name.strip().strip('"').strip("'")
        cands = [name] + ([name + e for e in exts] if not os.path.splitext(name)[1] else [])
        base = os.path.basename(name)
        for c in cands:
            key = os.path.basename(c)
            if key in self.saved:
                return ("mem", key)
        for d in [""] + self.base_dirs:
            for c in cands:
                p = os.path.join(d, c) if d else c
                if os.path.isfile(p):
                    return ("file", p)
        for d in self.base_dirs:  # recursive basename search
            for c in cands:
                hits = glob.glob(os.path.join(d, "**", os.path.basename(c)), recursive=True)
                if hits:
                    return ("file", hits[0])
        if os.path.splitext(name)[1] in ("", ".dta"):  # replication packages often rename data files
            for d in self.base_dirs[:1]:
                dtas = glob.glob(os.path.join(d, "*.dta"))
                if len(dtas) == 1:
                    self.warnings.append(f"file '{name}' not found; using the only .dta in {d}: {os.path.basename(dtas[0])}")
                    return ("file", dtas[0])
        raise StataError(f"file not found: {name} (searched {self.base_dirs}); pass --data-dir or --data")

    def load(self, name):
        kind, p = self.find_file(name)
        if kind == "mem":
            return self.saved[p].copy()
        ext = os.path.splitext(p)[1].lower()
        if ext == ".dta":
            try:
                df = pd.read_stata(p, convert_categoricals=False)
            except Exception:
                import pyreadstat
                df, _ = pyreadstat.read_dta(p)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(p)
        else:
            df = pd.read_csv(p)
        for c in df.columns:
            if str(df[c].dtype).startswith(("int", "uint", "Int", "float32", "bool")):
                df[c] = df[c].astype(float)
            elif _isstr(df[c]):
                df[c] = df[c].fillna("").astype(str)
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------ macros
    def expand(self, s):
        for _ in range(20):
            new = re.sub(r"`([^`']*)'", lambda m: self._local(m.group(1)), s)
            new = re.sub(r"\$\{([A-Za-z_]\w*)\}|\$([A-Za-z_]\w*)", lambda m: self.globals.get(m.group(1) or m.group(2), ""), new)
            if new == s:
                break
            s = new
        return s

    def _local(self, name):
        name = name.strip()
        if name.startswith("="):
            return str(self.scalar_eval(name[1:]))
        if name.startswith(":"):
            m = re.match(r":\s*word\s+count\s+(.*)", name)
            if m:
                return str(len(m.group(1).split()))
            m = re.match(r":\s*word\s+(\d+)\s+of\s+(.*)", name)
            if m:
                w = m.group(2).split(); k = int(m.group(1))
                return w[k - 1] if k <= len(w) else ""
            return ""
        return self.locals.get(name, "")

    def scalar_eval(self, expr):
        expr = self.sub_results(expr)
        d1 = pd.DataFrame(index=[0])
        def res(n):
            if n in self.scalars:
                return pd.Series([self.scalars[n]], index=d1.index)
            if self.df is not None and n in self.df.columns:
                return pd.Series([self.df[n].iloc[0]], index=d1.index)
            raise StataError(f"unknown name in scalar expression: {n}")
        v = stexpr.evaluate(expr, d1, res).iloc[0]
        try:
            v = float(v)
            return int(v) if v.is_integer() else v
        except Exception:
            return v

    def sub_results(self, expr):
        L = self.last or {}
        co = L.get("coefs", {})
        def b(m):
            v = co.get(m.group(2), {}).get("coef" if m.group(1) == "_b" else "se")
            return repr(float(v)) if v is not None else "."
        expr = re.sub(r"(_b|_se)\[\s*([^\]]+?)\s*\]", b, expr)
        def e(m):
            k = m.group(1)
            mp = {"N": L.get("N"), "r2": L.get("r2"), "r2_a": L.get("adj_r2"), "r2_within": L.get("r2_within"),
                  "rmse": L.get("rmse"), "F": L.get("F"), "df_r": L.get("df_r"), "N_clust": (L.get("n_clusters") or [None])[0]}
            if k == "sample":
                return "__esample"
            v = mp.get(k)
            return repr(float(v)) if v is not None else "."
        expr = re.sub(r"\be\((\w+)\)", e, expr)
        def r_(m):
            v = self.scalars.get("r(" + m.group(1) + ")")
            return repr(float(v)) if v is not None else "."
        expr = re.sub(r"\br\((\w+)\)", r_, expr)
        return expr

    # ------------------------------------------------------------------ variables
    def need_df(self):
        if self.df is None:
            dtas = glob.glob(os.path.join(self.base_dirs[0], "*.dta")) if self.base_dirs else []
            if len(dtas) == 1:  # do-files that assume the data were opened by hand
                self.warnings.append(f"no 'use' before first data command; auto-loaded {os.path.basename(dtas[0])}")
                self.df = self.load(dtas[0])
            else:
                raise StataError("no data in memory (use ... first, or pass --data)")
        return self.df

    def varnames(self):
        return [c for c in self.need_df().columns if not c.startswith("__")]

    def resolve_name(self, n):
        cols = self.varnames()
        if n in cols:
            return n
        hits = [c for c in cols if c.startswith(n)]
        if len(hits) == 1:
            return hits[0]
        lo = [c for c in cols if c.lower() == n.lower()]
        if len(lo) == 1 and False:
            return lo[0]
        raise StataError(f"variable {n} not found" + (f" (ambiguous: {hits[:5]})" if hits else ""))

    def expand_varlist(self, toks):
        cols = self.varnames()
        out = []
        for t in toks:
            if t in ("", "_all", "*") and t != "":
                out += cols; continue
            if any(ch in t for ch in "*?"):
                hits = [c for c in cols if fnmatch.fnmatchcase(c, t)]
                if not hits:
                    raise StataError(f"no variables match {t}")
                out += hits; continue
            if "-" in t and not t.startswith("-"):
                a, b = t.split("-", 1)
                a, b = self.resolve_name(a), self.resolve_name(b)
                i, j = cols.index(a), cols.index(b)
                out += cols[i:j + 1]; continue
            out.append(self.resolve_name(t))
        return list(dict.fromkeys(out))

    def tsop(self, ops, name, df=None):
        """Apply time-series operator string like 'L2', 'D', 'F', 'LD' to variable name."""
        df = self.need_df() if df is None else df
        s = self.series(name, df)
        for op in re.findall(r"([LDFS])(\d*)", ops.upper()):
            k = int(op[1]) if op[1] else 1
            if op[0] == "L":
                s = self.shift(s, k, df)
            elif op[0] == "F":
                s = self.shift(s, -k, df)
            elif op[0] == "D":
                for _ in range(k):
                    s = s - self.shift(s, 1, df)
            elif op[0] == "S":
                s = s - self.shift(s, k, df)
        return s

    def shift(self, s, k, df):
        if not self.panel or not self.panel[1]:
            guess = self.guess_panel(df)
            if guess:
                self.panel = guess
                self.warnings.append(f"time-series operator used without xtset; inferred panel ({guess[0]}, {guess[1]}) "
                                     "from variable names (the original .dta was probably saved after xtset)")
            else:
                self.warnings.append("time-series operator used without xtset/tsset; using row order")
                return s.shift(k)
        pid, t = self.panel
        tv = df[t].astype(float)
        ids = df[pid] if pid else pd.Series(0, index=df.index)
        key = pd.MultiIndex.from_arrays([ids.values, tv.values])
        src = pd.Series(s.values, index=key)
        src = src[~src.index.duplicated()]
        tgt = pd.MultiIndex.from_arrays([ids.values, (tv - k * (self.panel_delta or 1)).values])
        return pd.Series(src.reindex(tgt).values, index=df.index)

    panel_delta = 1

    def guess_panel(self, df):
        ids = ["id", "ID", "Id", "code", "Code", "stkcd", "Stkcd", "firm", "firmid", "gvkey", "permno", "country",
               "iso3", "countrycode", "province", "city", "Symbol", "symbol", "ticker", "id_firm"]
        ts = ["year", "Year", "t", "time", "date", "month", "quarter", "yr"]
        for t in ts:
            if t not in df.columns:
                continue
            for i in ids:
                if i in df.columns and not df.duplicated([i, t]).any():
                    return (i, t)
        return None

    def series(self, name, df=None):
        df = self.need_df() if df is None else df
        m = re.match(r"^([LDFSlfds]\d*(?:[LDFSldfs]\d*)*)\.(.+)$", name)
        if m and m.group(1)[0] in "LDFSldfs":
            return self.tsop(m.group(1), m.group(2), df)
        if name == "__esample":
            return df["__esample"].astype(float) if "__esample" in df else pd.Series(0.0, index=df.index)
        if name in self.scalars and name not in df.columns:
            return pd.Series(self.scalars[name], index=df.index)
        n = self.resolve_name(name) if name not in df.columns else name
        return df[n]

    def eval(self, expr, df=None, groups=None):
        df = self.need_df() if df is None else df
        expr = self.sub_results(expr)
        return stexpr.evaluate(expr, df, lambda n: self.series(n, df), groups)

    def mask(self, cond, inrange=None, df=None):
        df = self.need_df() if df is None else df
        m = pd.Series(True, index=df.index)
        if cond:
            v = self.eval(cond, df)
            m &= stexpr._truth(v) if v.dtype != bool else v
        if inrange:
            a, _, b = inrange.partition("/")
            n = len(df)
            a = 1 if a in ("f", "F") else int(a); b = n if b in ("l", "L", "") else int(b)
            if a < 0: a = n + a + 1
            if b < 0: b = n + b + 1
            pos = np.arange(1, n + 1)
            m &= pd.Series((pos >= a) & (pos <= b), index=df.index)
        return m

    # ------------------------------------------------------------------ design (factor vars)
    def tokens(self, body):
        """Split a varlist preserving parentheses groups."""
        toks, cur, depth = [], "", 0
        for ch in body:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if ch.isspace() and depth == 0:
                if cur:
                    toks.append(cur); cur = ""
                continue
            cur += ch
        if cur:
            toks.append(cur)
        # glue "a - b" ranges and "x = y"
        out = []
        for t in toks:
            if out and (t == "-" or out[-1].endswith("-")) and not t.startswith("("):
                out[-1] += t
            else:
                out.append(t)
        return out

    def term_columns(self, term, E):
        """Expand one varlist term into columns of E (DataFrame). Returns list of names."""
        df = self.need_df()
        if "##" in term:
            parts = term.split("##")
            names = []
            from itertools import combinations
            for r in range(1, len(parts) + 1):
                for comb in combinations(parts, r):
                    names += self.term_columns("#".join(comb), E)
            return names
        if "#" in term:
            parts = term.split("#")
            pieces = []
            for p in parts:
                if p.startswith("c."):
                    pieces.append([(p, self.series(p[2:]).astype(float))])
                else:
                    pieces.append(self.factor_dummies(p if re.match(r"^i[b(\d]?", p) and "." in p else "i." + p))
            combos = [("", None)]
            for pc in pieces:
                combos = [((a + "#" if a else "") + n, (s if v is None else v * s)) for a, v in combos for n, s in pc]
            for n, s in combos:
                E[n] = s.values
            return [n for n, _ in combos]
        if re.match(r"^i[^.]*\.", term) or re.match(r"^[ib]\d+\.", term):
            pcs = self.factor_dummies(term)
            for n, s in pcs:
                E[n] = s.values
            return [n for n, _ in pcs]
        m = re.match(r"^([LFDSlfds])\(([\d/ ]+)\)\.(.+)$", term)
        if m:
            op, rng, var = m.groups()
            if "/" in rng:
                a, b = map(int, rng.split("/")); ks = range(a, b + 1)
            else:
                ks = [int(x) for x in rng.split()]
            names = []
            for k in ks:
                nm = (f"{op.upper()}{k}." if k != 1 else f"{op.upper()}.") + var if k else var
                names += self.term_columns(nm, E)
            return names
        if term.startswith("c."):
            term = term[2:]
        mt = re.match(r"^([LDFSlfds]\d*(?:[LDFSldfs]\d*)*)\.(.+)$", term)
        if mt:
            nm = mt.group(1).upper() + "." + self.resolve_name(mt.group(2)) if not any(c in mt.group(2) for c in "*?") else None
            if nm is None:
                names = []
                for v in self.expand_varlist([mt.group(2)]):
                    names += self.term_columns(mt.group(1) + "." + v, E)
                return names
            E[nm] = self.series(nm).astype(float).values
            return [nm]
        names = self.expand_varlist([term])
        for n in names:
            E[n] = pd.to_numeric(df[n], errors="coerce").values if _isstr(df[n]) else df[n].values
        return names

    def factor_dummies(self, term):
        m = re.match(r"^(i|ib\(?(\w+)\)?|b(\d+)|i(\d+))\.(.+)$", term)
        if not m:
            raise StataError(f"cannot parse factor variable {term}")
        base = m.group(2) or m.group(3)
        var = self.resolve_name(m.group(5))
        s = self.series(var)
        lv = sorted(pd.unique(s.dropna())) if (not _isstr(s)) else sorted(set(s) - {""})
        if base in (None,):
            b = lv[0] if lv else None
        elif base == "last":
            b = lv[-1]
        elif base == "freq":
            b = s.value_counts().idxmax()
        elif base in ("n", "none"):
            b = None
        else:
            b = float(base) if (not _isstr(s)) else base
        out = []
        for v in lv:
            if b is not None and v == b:
                continue
            lab = (str(int(v)) if isinstance(v, float) and float(v).is_integer() else str(v))
            d = (s == v).astype(float)
            d[s.isna()] = np.nan
            out.append((f"{lab}.{var}", d))
        return out

    def design(self, toks, E):
        names = []
        for t in toks:
            names += self.term_columns(t, E)
        return list(dict.fromkeys(names))

    # ------------------------------------------------------------------ main loop
    def run_file(self, path):
        self.base_dirs = [os.path.dirname(os.path.abspath(path))] + self.base_dirs
        self.run_lines(strip_comments(read_text(path)))

    def run_text(self, text):
        self.run_lines(strip_comments(text))

    def run_lines(self, lines):
        i = 0
        while i < len(lines) and not self.stop:
            ln, line = lines[i]
            head = self.expand(line)
            m = re.match(r"^(foreach|forvalues|forval|while)\b(.*)\{\s*$", head.strip())
            if m:
                depth, j, body = 1, i + 1, []
                while j < len(lines):
                    l2 = lines[j][1].strip()
                    opens = l2.endswith("{")
                    if l2 == "}" or (l2.startswith("}") and not opens):
                        depth -= 1
                        if depth == 0:
                            break
                    elif opens:
                        depth += 1
                    body.append(lines[j]); j += 1
                self.loop(ln, m.group(1), m.group(2).strip(), body)
                i = j + 1
                continue
            m = re.match(r"^if\s+(.*)\{\s*$", head.strip())
            if m:  # program-control if block
                depth, j, body = 1, i + 1, []
                while j < len(lines):
                    l2 = lines[j][1].strip()
                    if l2.startswith("}"):
                        depth -= 1
                        if depth == 0:
                            break
                    elif l2.endswith("{"):
                        depth += 1
                    body.append(lines[j]); j += 1
                try:
                    cond = self.scalar_eval(m.group(1))
                except Exception:
                    cond = 0
                if cond:
                    self.run_lines(body)
                i = j + 1
                if i < len(lines) and re.match(r"^else\s*\{", lines[i][1].strip()):
                    depth, j, body = 1, i + 1, []
                    while j < len(lines):
                        l2 = lines[j][1].strip()
                        if l2.startswith("}"):
                            depth -= 1
                            if depth == 0:
                                break
                        body.append(lines[j]); j += 1
                    if not cond:
                        self.run_lines(body)
                    i = j + 1
                continue
            if head.strip() in ("}", "{"):
                i += 1; continue
            self.exec_line(ln, line)
            i += 1

    def loop(self, ln, kind, spec, body):
        if kind == "foreach":
            m = re.match(r"(\w+)\s+(in|of)\s+(\w+)?\s*(.*)$", spec)
            if not m:
                raise StataError(f"bad foreach: {spec}")
            var, how, lt, rest = m.groups()
            if how == "in":
                items = (lt + " " + rest).split() if lt else rest.split()
                items = [x.strip('"') for x in items]
            elif lt in ("varlist", "newlist"):
                items = self.expand_varlist(rest.split()) if lt == "varlist" else rest.split()
            elif lt == "numlist":
                items = self.numlist(rest)
            elif lt in ("local", "global"):
                items = (self.locals if lt == "local" else self.globals).get(rest.strip(), "").split()
            else:
                items = rest.split()
        elif kind in ("forvalues", "forval"):
            m = re.match(r"(\w+)\s*=\s*(.*)$", spec)
            var = m.group(1); items = self.numlist(m.group(2))
        else:
            self.log(ln, "while " + spec, "warning", "while loops not supported; skipped")
            return
        for it in items:
            self.locals[var] = str(it)
            self.run_lines(body)

    def numlist(self, s):
        s = s.strip(); out = []
        for part in s.split():
            m = re.match(r"^(-?[\d.]+)\((-?[\d.]+)\)(-?[\d.]+)$", part)
            if m:
                a, st, b = map(float, m.groups()); x = a
                while (st > 0 and x <= b + 1e-9) or (st < 0 and x >= b - 1e-9):
                    out.append(int(x) if float(x).is_integer() else x); x += st
                continue
            m = re.match(r"^(-?\d+)/(-?\d+)$", part)
            if m:
                a, b = map(int, m.groups()); out += list(range(a, b + 1)); continue
            out.append(int(float(part)) if float(part).is_integer() else float(part))
        return out

    def exec_line(self, ln, line):
        raw = line
        try:
            line = self.expand(line).strip()
        except Exception as ex:
            self.log(ln, raw, "error", f"macro expansion failed: {ex}"); return
        capture = False; by = None; eststo_name = None
        while True:  # prefixes
            m = re.match(r"^(quietly|quietl|quiet|quie|qui|qu|noisily|noisil|noisi|nois|noi|n)\s*:?\s+(.*)$", line)
            if m and not re.match(r"^n\s*=", line):
                line = m.group(2).strip(); continue
            m = re.match(r"^(capture|captur|captu|capt|cap)\s*:?\s+(.*)$", line)
            if m:
                capture = True; line = m.group(2).strip(); continue
            m = re.match(r"^(eststo|estimates\s+store)\s*([\w]*)\s*:\s*(.*)$", line)
            if m and m.group(1) == "eststo":
                eststo_name = m.group(2) or "__auto"; line = m.group(3).strip(); continue
            m = re.match(r"^(bysort|bys|by)\s+([^:]+):\s*(.*)$", line)
            if m:
                by = (m.group(1), m.group(2).strip()); line = m.group(3).strip(); continue
            m = re.match(r"^xi\s*:\s*(.*)$", line)
            if m:
                line = m.group(1).strip(); continue
            m = re.match(r"^(logout|asdoc)\b[^:]*:\s*(.*)$", line)
            if m:
                line = m.group(2).strip(); continue
            m = re.match(r"^asdoc\s+(.*)$", line)
            if m:
                parts = split_command(m.group(1))
                keep = {k: v for k, v in parts["opts"].items() if k not in ("save", "replace", "append", "title", "dec", "fs", "font", "label", "nest", "rep", "stat", "add", "cnames", "abb", "tzok", "fhr", "fhc", "setstars")}
                line = re.split(r",", m.group(1))[0] + (", " + " ".join(f"{k}({v})" if v is not True else k for k, v in keep.items()) if keep else "")
                continue
            break
        m = re.match(r"^([A-Za-z_]\w*)\s*(.*)$", line)
        if not m:
            self.log(ln, raw, "warning", "unparsed line"); return
        cmd, rest = m.group(1), m.group(2)
        try:
            self.dispatch(ln, cmd, rest, line, by, eststo_name)
        except Exception as ex:
            if capture:
                self.log(ln, line, "captured", str(ex)[:200])
            else:
                self.log(ln, line, "error", f"{type(ex).__name__}: {str(ex)[:300]}")

    # ------------------------------------------------------------------ dispatcher
    def dispatch(self, ln, cmd, rest, line, by, eststo_name):
        c = cmd.lower()
        if c in ("exit",):
            self.stop = True; return
        if c in ("global",):
            m = re.match(r"^(\w+)\s*(.*)$", rest)
            name, val = m.group(1), m.group(2).strip()
            if val.startswith("="):
                val = str(self.scalar_eval(val[1:]))
            self.globals[name] = self._unquote(val); return
        if c in ("local", "loc"):
            m = re.match(r"^(\w+)\s*(.*)$", rest)
            name, val = m.group(1), m.group(2).strip()
            if val.startswith("="):
                val = str(self.scalar_eval(val[1:]))
            elif val.startswith(":"):
                val = self._local(val)
            self.locals[name] = self._unquote(val); return
        if c == "scalar" or (c == "sca"):
            m = re.match(r"^(?:define\s+)?(\w+)\s*=\s*(.*)$", rest)
            if m:
                self.scalars[m.group(1)] = self.scalar_eval(m.group(2))
            return
        if c in ("display", "di", "dis"):
            try:
                txt = rest
                val = None
                m = re.match(r'^\s*"([^"]*)"\s*(.*)$', rest)
                if m:
                    txt, expr = m.group(1), m.group(2)
                    val = self.scalar_eval(expr) if expr.strip() else None
                else:
                    txt, val = "", self.scalar_eval(rest)
                self.log(ln, line, "display", f"{txt} {'' if val is None else val}".strip())
            except Exception:
                pass
            return
        if c in ("use",):
            return self.c_use(ln, rest, line)
        if c == "log" and rest.strip().startswith("using"):
            self.log(ln, line, "warning", "log using ignored: results are written to stata_results.md / stata_estimates.json")
            return
        if c in ("import", "insheet"):
            body = split_command(rest)["body"] if "," in rest else rest
            body = re.sub(r"^(delimited|excel)\s+", "", body.strip())
            body = re.sub(r"^using\s+", "", body.strip()).strip().strip('"')
            if not body:
                raise StataError("import: could not read the file name")
            self.df = self.load(body); self.panel = None
            self.log(ln, line, "data", f"loaded {body}: N={len(self.df)}")
            return
        if c in ("tempfile", "tempvar", "tempname"):
            for nm in rest.split():
                self.locals[nm] = f"__{c[4:]}_{nm}"
            return
        if c in ("sysuse", "webuse"):
            raise StataError("sysuse/webuse need internet/Stata; skipped")
        if c in ("xtset", "tsset"):
            toks = split_command(rest)["body"].split()
            opts = split_command(rest)["opts"]
            if c == "xtset":
                self.panel = (self.resolve_name(toks[0]), self.resolve_name(toks[1]) if len(toks) > 1 else None)
            else:
                self.panel = (self.resolve_name(toks[1]) if len(toks) > 1 else None, self.resolve_name(toks[0])) if len(toks) > 1 else (None, self.resolve_name(toks[0]))
                if len(toks) > 1:
                    self.panel = (self.resolve_name(toks[0]), self.resolve_name(toks[1]))
            d = getopt(opts, "delta")
            self.panel_delta = float(d) if d and d is not True else 1
            if self.panel[1]:
                self.df = self.df.sort_values([x for x in self.panel if x]).reset_index(drop=True)
            return
        if c in ("gen", "generate", "g", "ge", "gene", "egen", "replace", "rep", "repl", "replac"):
            return self.c_gen(ln, c, rest, line, by)
        if c in ("drop", "keep"):
            return self.c_dropkeep(ln, c, rest, line, by)
        if c in ("rename", "ren"):
            p = split_command(rest)["body"].split()
            if p and p[0].startswith("(") :
                raise StataError("group rename not supported")
            self.df = self.df.rename(columns={self.resolve_name(p[0]): p[1]}); return
        if c in ("sort", "gsort"):
            return self.c_sort(rest, c)
        if c in ("preserve",):
            self.preserved.append((self.df.copy(), self.panel)); return
        if c in ("restore",):
            if self.preserved:
                self.df, self.panel = self.preserved.pop()
            return
        if c in ("save", "sa", "sav"):
            m = re.match(r'^\s*"?([^",]+)"?', rest)
            if not m or not m.group(1).strip():
                raise StataError("save: no file name (an undefined macro? declare it with `tempfile name` first)")
            nm = os.path.basename(m.group(1).strip())
            nm = nm if nm.endswith(".dta") else nm + ".dta"
            self.saved[nm] = self.df.copy(); self.saved_panels[nm] = self.panel
            os.makedirs(os.path.join(self.out_dir, "saved"), exist_ok=True)
            pth = os.path.join(self.out_dir, "saved", nm[:-4] + ".csv")
            self.df[[c_ for c_ in self.df.columns if not c_.startswith("__")]].to_csv(pth, index=False)
            self.log(ln, line, "data", f"saved (in memory as {nm}; on disk as {pth})")
            return
        if c in ("tab", "tabulate", "ta", "tab1"):
            return self.c_tab(ln, rest, line)
        if c in ("winsor2", "winsor", "winsorizej"):
            return self.c_winsor(ln, rest, line)
        if c in ("encode",):
            p = split_command(rest); v = self.resolve_name(p["body"])
            g = getopt(p["opts"], "gen:generate")
            lv = sorted(set(self.df[v].astype(str)) - {""})
            mp = {x: i + 1 for i, x in enumerate(lv)}
            self.df[g] = self.df[v].astype(str).map(mp).astype(float); return
        if c in ("destring",):
            p = split_command(rest)
            vs = self.expand_varlist(p["body"].split()) if p["body"] else self.varnames()
            gen = getopt(p["opts"], "gen:generate")
            gl = gen.split() if gen else None
            for k, v in enumerate(vs):
                if _isstr(self.df[v]):
                    conv = pd.to_numeric(self.df[v].str.replace(",", "") if getopt(p["opts"], "ignore") else self.df[v], errors="coerce")
                    if gl:
                        self.df[gl[k]] = conv
                    else:
                        self.df[v] = conv
            return
        if c in ("tostring",):
            p = split_command(rest)
            for v in self.expand_varlist(p["body"].split()):
                self.df[v] = self.df[v].map(lambda x: "" if pd.isna(x) else (str(int(x)) if float(x).is_integer() else str(x)))
            return
        if c in ("duplicates",):
            p = split_command(rest); toks = p["body"].split()
            if toks and toks[0] == "drop":
                vs = self.expand_varlist(toks[1:]) if len(toks) > 1 else None
                n0 = len(self.df); self.df = self.df.drop_duplicates(subset=vs).reset_index(drop=True)
                self.log(ln, line, "data", f"duplicates dropped: {n0 - len(self.df)}")
            return
        if c in ("merge",):
            return self.c_merge(ln, rest, line)
        if c in ("append",):
            m = re.match(r"^using\s+(.*)$", rest.split(",")[0].strip())
            for f in m.group(1).split():
                self.df = pd.concat([self.df, self.load(f)], ignore_index=True)
            return
        if c in ("summarize", "sum", "su", "summ", "summa", "tabstat"):
            return self.c_sum(ln, c, rest, line, by)
        if c in ("ttest",):
            return self.c_ttest(ln, rest, line)
        if c in ("pwcorr", "corr", "correlate", "cor"):
            return self.c_corr(ln, rest, line, pairwise=(c == "pwcorr"))
        if c in ("xtsum",):
            return self.c_xtsum(ln, rest, line)
        if c in ("count",):
            p = split_command(rest); n = int(self.mask(p["cond"], p["inrange"]).sum())
            self.scalars["r(N)"] = n; self.log(ln, line, "display", f"count = {n}"); return
        if c in ("est", "estimates", "eststo", "esto"):
            return self.c_est(ln, c, rest, line)
        if c in ("predict",):
            return self.c_predict(ln, rest, line)
        if c in ("test", "testparm", "lincom", "nlcom"):
            return self.c_test(ln, c, rest, line)
        if c in ("margins",):
            self.log(ln, line, "info", "margins: average marginal effects are reported with probit/logit estimates"); return
        if c in ("export", "outsheet"):
            m = re.search(r'(?:delimited\s+)?(?:using\s+)?"?([^",]+\.csv)"?', rest)
            if m:
                os.makedirs(os.path.join(self.out_dir, "saved"), exist_ok=True)
                pth = os.path.join(self.out_dir, "saved", os.path.basename(m.group(1).strip()))
                self.df[[c_ for c_ in self.df.columns if not c_.startswith("__")]].to_csv(pth, index=False)
                self.log(ln, line, "data", f"exported {pth}")
            return
        if c in ("estat",):
            self.log(ln, line, "info", "estat: IV diagnostics are already reported with each IV estimate"); return
        if c in ("reg", "regress", "areg", "reghdfe", "xtreg", "ivreghdfe", "ivreg2", "ivregress", "xtivreg",
                 "xtivreg2", "probit", "logit", "poisson", "nbreg", "ppmlhdfe", "sgmediation2", "sgmediation", "dprobit",
                 "ivreg") or c in EXTRA_EST:
            if self.est_filter and not self.est_filter.search(line):
                self.last = None
                self.log(ln, line, "skipped", "filtered out by --est-filter"); return
            out = self.c_estimate(ln, c, rest, line, eststo_name)
            if self.vce_variants and self.last is not None and c not in ("sgmediation2", "sgmediation", "psmatch2"):
                self._vce_scan(ln, c, rest, line)
            return out
        if c in UNSUPPORTED:
            self.log(ln, line, "warning", f"command '{c}' not supported by ek stata: run it in Stata or implement manually (see references/replication.md)")
            return
        if c in OUTPUT_ONLY:
            return
        self.log(ln, line, "warning", f"unknown command '{c}' ignored")

    VCE_OPT = re.compile(r"(?<![\w(])(vce\s*\([^)]*\)|cl(?:u(?:s(?:t(?:e(?:r)?)?)?)?)?\s*\([^)]*\)|robust|r|small)(?![\w(])")

    def _vce_scan(self, ln, c, rest, line):
        """Re-estimate the same line under each SE convention in --vce-scan; keep the original as e(last)."""
        orig = self.last
        parts = top_split(rest, ",")
        main, optstr = parts[0], ",".join(parts[1:]) if len(parts) > 1 else ""
        base_opts = self.VCE_OPT.sub(" ", optstr)
        rows = []
        for v in self.vce_variants:
            toks = v.split()
            small = "small" in toks; toks = [t for t in toks if t != "small"]
            kind = toks[0].lower() if toks else "iid"
            add = {"iid": "", "ols": "", "unadjusted": "", "robust": "vce(robust)", "r": "vce(robust)"}.get(kind)
            if add is None:  # cluster a [b]
                add = f"vce(cluster {' '.join(toks[1:])})" if kind.startswith("cl") else f"vce({' '.join(toks)})"
            new_rest = f"{main}, {base_opts} {add} {'small' if small else ''}".strip()
            n_before = len(self.estimates)
            try:
                self.c_estimate(ln, c, new_rest, f"{c} {new_rest}", None)
            except Exception as ex:
                rows.append({"variant": v, "error": str(ex)[:120]}); continue
            if len(self.estimates) > n_before:
                r = self.estimates[-1]; r["variant_of"] = orig["id"]; r["variant"] = v
                key = [n for n in r.get("names", []) if n != "_cons" and not self.is_dummy(n)][:2]
                rows.append({"variant": v, "id": r["id"], **{n: {k: r["coefs"][n].get(k) for k in ("coef", "se", "t")} for n in key},
                             **({"kp_rk_Wald_F": r["iv"].get("kp_rk_Wald_F"), "cd_F": r["iv"].get("cragg_donald_F")} if r.get("iv") else {})})
        orig.setdefault("vce_scan", rows)
        self.last = orig
        self.log(ln, line, "info", "vce scan", vce_scan=rows)

    @staticmethod
    def _unquote(v):
        v = v.strip()
        if v.startswith('`"') and v.endswith("\"'"):
            return v[2:-2]
        if len(v) >= 2 and v[0] == '"' and v[-1] == '"' and v.count('"') == 2:
            return v[1:-1]
        return v

    # ------------------------------------------------------------------ data commands
    def c_use(self, ln, rest, line):
        p = split_command(rest)
        body = p["body"]
        m = re.match(r"^(.*?)\s+using\s+(.*)$", body)
        if m:
            fname = m.group(2); cols = m.group(1).split()
        else:
            fname, cols = body, None
        fname = fname.strip().strip('"')
        df = self.load(fname)
        try:
            kind, key = self.find_file(fname)
        except Exception:
            kind, key = None, None
        if cols:
            df = df[[c for c in df.columns if any(fnmatch.fnmatchcase(c, t) for t in cols)]]
        self.df = df
        if p["cond"]:
            self.df = self.df[self.mask(p["cond"])].reset_index(drop=True)
        self.panel = self.saved_panels.get(key) if kind == "mem" else None  # xtset is stored with saved data
        self.log(ln, line, "data", f"loaded {fname}: N={len(df)}, vars={df.shape[1]}")

    def by_groups(self, by):
        if not by:
            return None
        kind, spec = by
        m = re.match(r"^([^()]*)(?:\(([^)]*)\))?\s*$", spec)
        gvars = self.expand_varlist(m.group(1).split())
        svars = self.expand_varlist(m.group(2).split()) if m.group(2) else []
        if kind in ("bysort", "bys") or svars:
            self.df = self.df.sort_values(gvars + svars, kind="mergesort").reset_index(drop=True)
        return pd.Series(group_codes(self.df, gvars), index=self.df.index), gvars

    def c_gen(self, ln, c, rest, line, by):
        grp = self.by_groups(by)
        groups, gvars = (grp if grp else (None, None))
        p = split_command(rest)
        body = p["body"]
        m = re.match(r"^(?:(byte|int|long|float|double|str\d*|strL)\s+)?([A-Za-z_]\w*)\s*=\s*(.*)$", body, re.S)
        if not m:
            raise StataError(f"cannot parse {c}: {body}")
        name, expr = m.group(2), m.group(3).strip()
        df = self.df
        cond = self.mask(p["cond"], p["inrange"]) if (p["cond"] or p["inrange"]) else None
        if c == "egen":
            byopt = getopt(p["opts"], "by")
            if byopt:
                gvars = self.expand_varlist(byopt.split()); groups = pd.Series(group_codes(df, gvars), index=df.index)
            val = self.egen(expr, groups, cond, p["opts"])
        else:
            val = self.eval(expr, df, groups)
        if not isinstance(val, pd.Series):
            val = pd.Series(val, index=df.index)
        if val.dtype == bool:
            val = val.astype(float)
        if c.startswith("rep"):
            name = self.resolve_name(name)
            if cond is not None:
                df.loc[cond, name] = val[cond]
            else:
                df[name] = val
            return
        if name in df.columns:
            raise StataError(f"variable {name} already defined")
        if cond is not None:
            val = val.where(cond, np.nan if (not _isstr(val)) else "")
        if (not _isstr(val)):
            val = val.astype(float)
        df[name] = val

    def egen(self, expr, groups, cond, opts):
        m = re.match(r"^(\w+)\((.*)\)\s*$", expr.strip(), re.S)
        if not m:
            raise StataError(f"bad egen expression {expr}")
        fn, arg = m.group(1).lower(), m.group(2)
        df = self.df
        idx = df.index
        if fn in ("group", "tag"):
            vs = self.expand_varlist(arg.split())
            sub = df[vs]
            ok = ~(sub.isna().any(axis=1) | (sub.astype(str) == "").any(axis=1))
            if fn == "group":
                g = pd.Series(np.nan, index=idx)
                g[ok] = sub[ok].groupby(vs, sort=True).ngroup().values + 1.0
                return g
            t = (~sub.duplicated()) & ok
            return t.astype(float)
        if fn.startswith("row"):
            vs = self.expand_varlist(arg.split())
            M = df[vs].astype(float)
            return {"rowmean": M.mean(axis=1), "rowtotal": M.sum(axis=1), "rowmax": M.max(axis=1), "rowmin": M.min(axis=1),
                    "rownonmiss": M.notna().sum(axis=1).astype(float), "rowmiss": M.isna().sum(axis=1).astype(float),
                    "rowsd": M.std(axis=1)}[fn]
        x = self.eval(arg, df).astype(float)
        if cond is not None:
            x = x.where(cond)
        g = groups if groups is not None else pd.Series(0, index=idx)
        gb = x.groupby(g.values)
        if fn == "mean":
            r = gb.transform("mean")
        elif fn == "median":
            r = gb.transform(lambda s: stata_pctile(s, 50))
        elif fn in ("pctile",):
            pp = float(getopt(opts, "p") or 50)
            r = gb.transform(lambda s: stata_pctile(s, pp))
        elif fn in ("sd",):
            r = gb.transform("std")
        elif fn in ("sum", "total"):
            r = gb.transform("sum")
        elif fn == "max":
            r = gb.transform("max")
        elif fn == "min":
            r = gb.transform("min")
        elif fn == "count":
            r = gb.transform("count").astype(float)
        elif fn in ("std",):
            r = (x - gb.transform("mean")) / gb.transform("std")
        elif fn == "rank":
            r = gb.rank()
        elif fn == "iqr":
            r = gb.transform(lambda s: stata_pctile(s, 75) - stata_pctile(s, 25))
        elif fn == "mode":
            r = gb.transform(lambda s: s.mode().min() if len(s.dropna()) else np.nan)
        else:
            raise StataError(f"egen function {fn}() not supported")
        r = pd.Series(np.asarray(r, dtype=float), index=idx)
        if cond is not None:
            r = r.where(cond)
        return r

    def c_dropkeep(self, ln, c, rest, line, by):
        grp = self.by_groups(by)
        groups = grp[0] if grp else None
        p = split_command(rest)
        n0 = len(self.df)
        if p["cond"] or p["inrange"]:
            if groups is not None:
                v = self.eval(p["cond"], self.df, groups)
                msk = stexpr._truth(v)
            else:
                msk = self.mask(p["cond"], p["inrange"])
            self.df = (self.df[~msk] if c == "drop" else self.df[msk]).reset_index(drop=True)
            self.log(ln, line, "data", f"{c}: {n0} -> {len(self.df)} obs")
        else:
            vs = self.expand_varlist(self.tokens(p["body"]))
            if c == "drop":
                self.df = self.df.drop(columns=vs)
            else:
                self.df = self.df[vs + (["__esample"] if "__esample" in self.df else [])]

    def c_sort(self, rest, c):
        toks = split_command(rest)["body"].split()
        cols, asc = [], []
        for t in toks:
            a = not t.startswith("-")
            t = t.lstrip("+-")
            for v in self.expand_varlist([t]):
                cols.append(v); asc.append(a)
        self.df = self.df.sort_values(cols, ascending=asc, kind="mergesort").reset_index(drop=True)

    def c_tab(self, ln, rest, line):
        p = split_command(rest)
        vs = self.expand_varlist(p["body"].split())
        gen = getopt(p["opts"], "gen:generate")
        df = self.df
        if gen:
            v = vs[0]
            s = df[v]
            msk = self.mask(p["cond"], p["inrange"])
            lv = sorted(pd.unique(s[msk & s.notna()])) if (not _isstr(s)) else sorted(set(s[msk]) - {""})
            for i, x in enumerate(lv, 1):
                d = (s == x).astype(float)
                d[s.isna() if (not _isstr(s)) else (s == "")] = np.nan
                df[f"{gen}{i}"] = d
            self.log(ln, line, "data", f"generated {gen}1-{gen}{len(lv)}")
        else:
            vc = df.loc[self.mask(p["cond"], p["inrange"]), vs[0]].value_counts().sort_index()
            self.descriptives.append({"line": ln, "cmd": line, "table": {str(k): int(v) for k, v in list(vc.items())[:50]}})

    def c_winsor(self, ln, rest, line):
        p = split_command(rest)
        vs = self.expand_varlist(p["body"].split())
        cuts = getopt(p["opts"], "cuts", "cut") or "1 99"
        lo, hi = map(float, str(cuts).split())
        replace = getopt(p["opts"], "replace") is True
        suffix = getopt(p["opts"], "suffix") or "_w"
        trim = getopt(p["opts"], "trim") is True
        byv = getopt(p["opts"], "by")
        df = self.df
        g = pd.Series(group_codes(df, self.expand_varlist(byv.split())), index=df.index) if byv else pd.Series(0, index=df.index)
        for v in vs:
            x = df[v].astype(float)
            out = x.copy()
            for code, idx in x.groupby(g.values).groups.items():
                s = x.loc[idx]
                a, b = stata_pctile(s, lo), stata_pctile(s, hi)
                if trim:
                    out.loc[idx] = s.where((s >= a) & (s <= b))
                else:
                    out.loc[idx] = s.clip(a, b)
            if replace:
                df[v] = out
            else:
                df[v + suffix] = out

    def c_merge(self, ln, rest, line):
        p = split_command(rest)
        m = re.match(r"^(1:1|m:1|1:m|m:m)\s+(.*?)\s+using\s+(.*)$", p["body"])
        if not m:
            raise StataError("merge syntax not understood")
        kind, keys, fname = m.groups()
        keys = keys.split()
        other = self.load(fname.strip().strip('"'))
        kv = getopt(p["opts"], "keepus:keepusing")
        if kv:
            other = other[keys + [c for c in other.columns if any(fnmatch.fnmatchcase(c, t) for t in kv.split()) and c not in keys]]
        dup = [c for c in other.columns if c in self.df.columns and c not in keys]
        other = other.drop(columns=dup)  # master wins (Stata default without update)
        md = self.df.merge(other, on=keys, how="outer", indicator=True)
        md["_merge"] = md["_merge"].map({"left_only": 1.0, "right_only": 2.0, "both": 3.0}).astype(float)
        keep = getopt(p["opts"], "keep")
        if keep:
            codes = set()
            for k in keep.split():
                codes |= {"master": {1.0}, "mas": {1.0}, "1": {1.0}, "using": {2.0}, "us": {2.0}, "2": {2.0},
                          "match": {3.0}, "mat": {3.0}, "3": {3.0}}.get(k, set())
            md = md[md["_merge"].isin(codes)]
        if getopt(p["opts"], "nogen:nogenerate"):
            md = md.drop(columns="_merge")
        self.log(ln, line, "data", f"merge: N={len(md)}")
        self.df = md.reset_index(drop=True)

    def c_sum(self, ln, c, rest, line, by):
        p = split_command(rest)
        vs = self.expand_varlist(p["body"].split()) if p["body"] else self.varnames()
        msk = self.mask(p["cond"], p["inrange"])
        d = self.df.loc[msk, vs].apply(pd.to_numeric, errors="coerce")
        tab = pd.DataFrame({"N": d.count(), "mean": d.mean(), "sd": d.std(), "min": d.min(),
                            "p50": d.median(), "max": d.max()})
        if len(vs) == 1:
            s = d[vs[0]]
            self.scalars.update({"r(N)": int(s.count()), "r(mean)": s.mean(), "r(sd)": s.std(), "r(min)": s.min(),
                                 "r(max)": s.max(), "r(sum)": s.sum(), "r(p50)": stata_pctile(s, 50)})
        self.descriptives.append({"line": ln, "cmd": line, "summary": tab.round(4).reset_index().rename(columns={"index": "var"}).to_dict("records")})

    def c_ttest(self, ln, rest, line):
        p = split_command(rest)
        msk = self.mask(p["cond"], p["inrange"])
        byv = getopt(p["opts"], "by")
        m = re.match(r"^(\S+)\s*(==\s*(\S+))?$", p["body"])
        x = self.series(m.group(1))[msk].astype(float)
        if byv:
            g = self.series(byv.strip())[msk]
            lv = sorted(pd.unique(g.dropna()))
            a, b = x[g == lv[0]].dropna(), x[g == lv[1]].dropna()
            uneq = getopt(p["opts"], "uneq:unequal", "welch") is not None
            t, pv = stats.ttest_ind(a, b, equal_var=not uneq)
            res = {"var": m.group(1), "by": byv, "groups": [float(lv[0]), float(lv[1])], "n": [len(a), len(b)],
                   "mean": [a.mean(), b.mean()], "diff": a.mean() - b.mean(), "t": t, "p_two_sided": pv}
        else:
            mu = float(m.group(3)) if m.group(3) else 0.0
            t, pv = stats.ttest_1samp(x.dropna(), mu)
            res = {"var": m.group(1), "mu0": mu, "n": int(x.count()), "mean": x.mean(), "t": t, "p_two_sided": pv}
        self.descriptives.append({"line": ln, "cmd": line, "ttest": _clean(res)})

    def c_corr(self, ln, rest, line, pairwise=False):
        p = split_command(rest)
        vs = self.expand_varlist(p["body"].split()) if p["body"] else self.varnames()
        d = self.df.loc[self.mask(p["cond"], p["inrange"]), vs].apply(pd.to_numeric, errors="coerce")
        if not pairwise:
            d = d.dropna()
        r = d.corr()
        out = {"line": ln, "cmd": line, "corr": r.round(4).to_dict(), "N": int(len(d)) if not pairwise else None}
        if pairwise:  # pwcorr: pairwise N and two-sided p-values (t-test of r)
            n = d.notna().astype(int).T @ d.notna().astype(int)
            tt = r * np.sqrt((n - 2) / (1 - r.clip(-0.999999999, 0.999999999) ** 2))
            pv = pd.DataFrame(2 * stats.t.sf(np.abs(tt), n - 2), index=r.index, columns=r.columns)
            out.update({"pairwise_N": n.to_dict(), "p": pv.round(4).to_dict()})
        self.descriptives.append(out)

    def c_xtsum(self, ln, rest, line):
        """Stata xtsum: overall / between / within mean, sd, min, max and N, n, T-bar."""
        if not self.panel or not self.panel[0]:
            raise StataError("xtsum needs xtset panelvar")
        p = split_command(rest); pid = self.panel[0]
        vs = self.expand_varlist(p["body"].split()) if p["body"] else [v for v in self.varnames() if v not in self.panel]
        msk = self.mask(p["cond"], p["inrange"]); rows = []
        for v in vs:
            x = pd.to_numeric(self.df.loc[msk, v], errors="coerce"); g = self.df.loc[msk, pid]
            ok = x.notna(); x, g = x[ok], g[ok]
            if not len(x):
                continue
            gm = x.groupby(g).transform("mean"); bm = x.groupby(g).mean()
            w = x - gm + x.mean()
            rows.append({"var": v, "mean": x.mean(), "overall_sd": x.std(), "overall_min": x.min(), "overall_max": x.max(),
                         "between_sd": bm.std(), "between_min": bm.min(), "between_max": bm.max(),
                         "within_sd": w.std(), "within_min": w.min(), "within_max": w.max(),
                         "N": int(len(x)), "n": int(bm.size), "T_bar": len(x) / bm.size})
        self.descriptives.append({"line": ln, "cmd": line, "xtsum": _clean(rows)})

    def c_est(self, ln, c, rest, line):
        toks = rest.split()
        if c in ("eststo", "esto"):
            if toks and toks[0] == "clear":
                return
            name = toks[0] if toks and not toks[0].startswith(",") else None
            if self.last:
                self.eststo_n += 1
                name = name or f"est{self.eststo_n}"
                self.stored[name] = self.last; self.last.setdefault("stored_as", []).append(name)
            return
        if not toks:
            return
        sub = toks[0]
        if sub.startswith("sto") and self.last:
            for n in toks[1:2]:
                self.stored[n] = self.last
                if n not in self.last.setdefault("stored_as", []):
                    self.last["stored_as"].append(n)
        elif sub.startswith("res") and len(toks) > 1 and toks[1] in self.stored:
            self.last = self.stored[toks[1]]

    def c_predict(self, ln, rest, line):
        p = split_command(rest)
        toks = p["body"].split()
        name = toks[-1]
        L = self.last
        if not L or "_design" not in L:
            raise StataError("predict: no estimation results")
        kind = "pr" if (getopt(p["opts"], "pr") or (L["cmd"] in ("probit", "logit") and not getopt(p["opts"], "xb"))) else "xb"
        if getopt(p["opts"], "r:residuals", "res:residuals"):
            kind = "resid"
        E = pd.DataFrame(index=self.df.index)
        names = self.design(L["_design"], E)
        co = L["coefs"]
        xb = pd.Series(co.get("_cons", {}).get("coef") or 0.0, index=self.df.index, dtype=float)
        for n in names:
            if n in co and co[n].get("coef") is not None:
                xb = xb + E[n].astype(float) * co[n]["coef"]
        if kind == "pr":
            val = pd.Series(stats.norm.cdf(xb) if L["cmd"] == "probit" else 1 / (1 + np.exp(-xb)), index=xb.index)
        elif kind == "resid":
            if L.get("fe"):
                raise StataError("predict residuals after absorbed FE not supported")
            val = self.series(L["dep"]).astype(float) - xb
        elif L["cmd"] in ("poisson", "nbreg") and not getopt(p["opts"], "xb"):
            val = np.exp(xb)
        else:
            val = xb
        self.df[name] = val.astype(float)

    # ------------------------------------------------------------------ post-estimation tests
    def _lin(self, expr, names):
        """Linear form a'b + c from an expression in coefficient names / _b[] -> (a, c)."""
        e = re.sub(r"_b\[\s*([^\]]+?)\s*\]", r"\1", expr)
        def f(vec):
            vals = dict(zip(names, vec))
            d1 = pd.DataFrame(index=[0])
            def res(n):
                if n in vals:
                    return pd.Series([vals[n]], index=d1.index)
                raise StataError(f"coefficient {n} not in last estimates")
            return float(stexpr.evaluate(self._protect(e, names), d1, lambda n: res(self._unprotect(n))).iloc[0])
        k = len(names)
        c = f(np.zeros(k))
        a = np.array([f(np.eye(k)[i]) - c for i in range(k)])
        return a, c, f

    _PROT = {}

    def _protect(self, e, names):
        # coefficient names like L.x or 2010.year#c.z are not valid identifiers: map to placeholders
        for i, n in enumerate(sorted(names, key=len, reverse=True)):
            if not re.match(r"^[A-Za-z_]\w*$", n) and n in e:
                ph = f"__c{i}"
                self._PROT[ph] = n
                e = e.replace(n, ph)
        return e

    def _unprotect(self, n):
        return self._PROT.get(n, n)

    def c_test(self, ln, c, rest, line):
        L = self.last
        if not L or "vcov" not in L:
            raise StataError(f"{c}: no estimation results with a covariance matrix")
        names = [n for n in L["names"]]
        b = np.array([L["coefs"][n]["coef"] for n in names])
        V = np.array(L["vcov"])
        dfr = (min(L["n_clusters"]) - 1) if L.get("n_clusters") else L.get("df_r")
        small = L.get("small", False) and dfr
        body = split_command(rest)["body"]
        out = {"test": c, "expr": body}
        if c in ("lincom", "nlcom"):
            expr = body.split(":", 1)[-1].strip() if c == "nlcom" else body
            a, c0, f = self._lin(expr, names)
            if c == "lincom":
                est = float(a @ b + c0); se = float(np.sqrt(a @ V @ a))
            else:
                est = f(b); g = np.zeros(len(b))
                for i in range(len(b)):
                    h = 1e-6 * max(1.0, abs(b[i])); bp = b.copy(); bp[i] += h; bm = b.copy(); bm[i] -= h
                    g[i] = (f(bp) - f(bm)) / (2 * h)
                se = float(np.sqrt(g @ V @ g))
            z = est / se if se > 0 else np.nan
            p = 2 * (stats.t.sf(abs(z), dfr) if (small and c == "lincom") else stats.norm.sf(abs(z)))
            out.update({"estimate": est, "se": se, "t": z, "p": p, "dist": "t" if (small and c == "lincom") else "normal"})
        else:
            if c == "testparm":
                terms = self.expand_varlist(body.split()) if body else []
                eqs = [f"{t} = 0" for t in names if t in terms or any(fnmatch.fnmatchcase(t, x) for x in body.split())]
            else:
                parts = re.findall(r"\(([^()]*)\)", body) if body.strip().startswith("(") else [body]
                eqs = []
                for pt in parts:
                    if "=" in pt:
                        lhs, rhs = pt.split("=", 1)
                        eqs.append(f"({lhs}) - ({rhs})" if rhs.strip() else lhs)
                    else:
                        eqs += [f"{t}" for t in pt.split()]
            R, r = [], []
            for e in eqs:
                e = e.replace(" = 0", "")
                a, c0, _ = self._lin(e, names)
                R.append(a); r.append(-c0)
            R, r = np.array(R), np.array(r)
            d = R @ b - r
            W = float(d @ np.linalg.pinv(R @ V @ R.T) @ d)
            q = np.linalg.matrix_rank(R)
            if small:
                F = W / q; p = float(stats.f.sf(F, q, dfr))
                out.update({"F": F, "df": [int(q), int(dfr)], "p": p})
            else:
                out.update({"chi2": W, "df": int(q), "p": float(stats.chi2.sf(W, q))})
            out["constraints"] = eqs
        L.setdefault("tests", []).append(_clean(out))
        self.descriptives.append({"line": ln, "cmd": line, "post_estimation": _clean(out), "est_id": L.get("id")})

    # ------------------------------------------------------------------ estimation
    def parse_iv_body(self, body):
        """y exog (endog = inst) exog -> y, exog_toks, endog_toks, inst_toks"""
        m = re.search(r"\(([^()]*(?:\([^()]*\)[^()]*)*)=([^()]*(?:\([^()]*\)[^()]*)*)\)", body)
        if not m:
            return None
        pre, post = body[:m.start()], body[m.end():]
        toks = self.tokens(pre) + self.tokens(post)
        return toks[0], toks[1:], self.tokens(m.group(1)), self.tokens(m.group(2))

    def vce_spec(self, opts, cmd):
        vce = getopt(opts, "vce")
        cl = getopt(opts, "cl:cluster")
        if vce and vce is not True:
            v = vce.split()
            k = v[0].lower()
            if k.startswith("cl"):
                return "cluster", v[1:]
            if k.startswith("r") or k.startswith("hc") or k == "hetero":
                return "robust", []
            if k in ("unadjusted", "ols", "un", "conventional", "oim", "opg"):
                return "iid", []
            if k.startswith("boot") or k.startswith("jack"):
                self.warnings.append(f"{cmd}: vce({vce}) approximated by robust")
                return "robust", []
        if cl and cl is not True:
            return "cluster", cl.split()
        if getopt(opts, "r:robust") is not None or getopt(opts, "rob:robust") is not None:
            return "robust", []
        return "iid", []

    def c_estimate(self, ln, c, rest, line, eststo_name):
        p = split_command(rest)
        opts = p["opts"]
        df = self.need_df()
        msk = self.mask(p["cond"], p["inrange"])
        E = pd.DataFrame(index=df.index)
        weights, wtype = None, "aweight"
        if p["weight"]:
            wt, wexpr = p["weight"]
            wtype = {"aw": "aweight", "pw": "pweight", "fw": "fweight", "iw": "aweight", "weight": "aweight"}.get(wt, wt)
            E["__w"] = self.eval(wexpr).astype(float).values; weights = "__w"
        vce, clv = self.vce_spec(opts, c)
        res = None
        if c in ("sgmediation2", "sgmediation"):
            res = self.est_mediation(p, opts, msk, vce, clv)
        elif c in EXTRA_EST:
            res = self.est_extra(c, p, opts, msk, vce, clv, ln)
        else:
            iv = self.parse_iv_body(p["body"]) if c in ("ivreghdfe", "ivreg2", "ivregress", "xtivreg", "xtivreg2", "ivreg") else None
            if iv:
                y, ex_t, en_t, in_t = iv
                if c == "ivregress":
                    est_kind = y.lower(); y = ex_t[0]; ex_t = ex_t[1:]
                else:
                    est_kind = "liml" if getopt(opts, "liml") else "2sls"
                    if getopt(opts, "gmm2s") or getopt(opts, "gmm"):
                        self.warnings.append(f"L{ln}: GMM estimator approximated by 2SLS (coefficients identical only if exactly identified)")
            else:
                toks = self.tokens(p["body"])
                y, ex_t, en_t, in_t, est_kind = toks[0], toks[1:], [], [], "ols"
            ycol = self.term_columns(y, E)[0]
            exn = self.design(ex_t, E)
            enn = self.design(en_t, E)
            inn = self.design(in_t, E)
            fe, conv, drop_single, constant = [], "reg", False, getopt(opts, "nocons:noconstant") is None
            fe_names = None
            if c == "reghdfe" or c == "ivreghdfe":
                ab = getopt(opts, "a:absorb")
                conv = "reghdfe" if c == "reghdfe" else "ivreghdfe"
                drop_single = getopt(opts, "keepsin:keepsingletons") is None
                if ab and ab is not True:
                    for t in ab.split():
                        t = re.sub(r"^\w+=", "", t)
                        if re.search(r"c\.|##", t):
                            raise StataError(f"heterogeneous slopes in absorb({t}) not supported")
                        cols = [self.resolve_name(re.sub(r"^i\.", "", x)) for x in t.split("#")]
                        for col in cols:
                            E[col] = df[col].values
                        fe.append(cols)
                if c == "ivreghdfe" and not fe:
                    conv = "ivreg2"
            elif c == "areg":
                ab = getopt(opts, "a:absorb")
                col = self.resolve_name(ab); E[col] = df[col].values; fe = [[col]]; conv = "areg"
            elif c in ("xtreg", "xtivreg", "xtivreg2"):
                if not self.panel or not self.panel[0]:
                    raise StataError(f"{c} requires xtset panelvar")
                pid = self.panel[0]
                if getopt(opts, "re") is not None and c == "xtreg":
                    res = self.est_xtre(ln, line, E, ycol, exn, pid, vce, clv, msk)
                elif getopt(opts, "fe") is not None or c == "xtivreg2":
                    E[pid] = df[pid].values; fe = [[pid]]; conv = "xtreg" if c == "xtreg" else "xtivreg"
                    if vce == "robust":
                        vce, clv = "cluster", [pid]
                else:
                    if c == "xtreg":
                        res = self.est_xtre(ln, line, E, ycol, exn, pid, vce, clv, msk)
                    else:
                        raise StataError(f"{c}: only fe is supported (re/be/fd not implemented)")
            elif c in ("ivreg2", "ivregress", "ivreg"):
                conv = "ivreg2" if c == "ivreg2" else "ivregress"
            if c in ("probit", "logit", "dprobit", "poisson", "nbreg"):
                res = self.est_mle(c, E, ycol, exn, vce, clv, msk, constant, weights)
            elif c == "ppmlhdfe":
                res = self.est_ppml(E, ycol, exn, opts, vce, clv, msk)
            if res is None:
                clcols = []
                for t in clv:
                    cols = [self.resolve_name(x) for x in t.split("#")]
                    for col in cols:
                        E[col] = df[col].values
                    clcols.append(cols)
                small = None
                if getopt(opts, "small") is not None:
                    small = True
                if c == "xtivreg":
                    small = getopt(opts, "small") is not None
                Ed = E[msk.values]
                res = estimate(Ed, ycol, exn, endog=enn, instruments=inn, fe=fe, vce=vce, cluster=clcols,
                               weights=weights, wtype=wtype, convention=conv, drop_singletons=drop_single,
                               constant=constant, small=small, estimator=("liml" if est_kind == "liml" else "2sls"))
                ridx = res.pop("_index")
                rname = getopt(opts, "res:residuals")
                if rname and rname is not True:
                    rr = pd.Series(np.nan, index=df.index); rr[ridx] = res.get("_resid")
                    df[rname] = rr
                sample = pd.Series(False, index=df.index); sample[ridx] = True
                res.pop("_b", None)
                df["__esample"] = sample.astype(float)
            res["_design"] = ex_t + en_t
        res.update({"cmd": c, "line": ln, "command": line, "id": len(self.estimates) + 1})
        if c in ("probit", "logit", "poisson", "nbreg"):
            res["_design"] = self.tokens(p["body"])[1:]
        self.last = res
        self.estimates.append(res)
        if eststo_name:
            self.eststo_n += 1
            nm = f"est{self.eststo_n}" if eststo_name == "__auto" else eststo_name
            self.stored[nm] = res; res.setdefault("stored_as", []).append(nm)
        self.log(ln, line, "estimate", None, id=res["id"])

    def est_mle(self, c, E, ycol, exn, vce, clv, msk, constant, weights):
        import statsmodels.api as sm
        df = self.df
        cols = [ycol] + exn
        clcol = None
        if vce == "cluster":
            clcol = self.resolve_name(clv[0]); E["__cl"] = df[clcol].values
        d = E[msk.values][cols + (["__cl"] if clcol else []) + ([weights] if weights else [])].apply(pd.to_numeric, errors="coerce").dropna()
        y = d[ycol].values
        X = d[exn].values.astype(float)
        from .hdfe import _drop_collinear
        X, names, dropped = _drop_collinear(X, exn)
        if constant:
            X = np.column_stack([X, np.ones(len(d))]); names = names + ["_cons"]
        model = {"probit": sm.Probit, "dprobit": sm.Probit, "logit": sm.Logit, "poisson": sm.Poisson,
                 "nbreg": sm.NegativeBinomial}[c](y, X)
        r = model.fit(disp=0, maxiter=200, method="newton" if c != "nbreg" else "bfgs")
        if c == "nbreg":
            r = model.fit(start_params=r.params, disp=0, maxiter=200, method="newton")
        params = np.asarray(r.params)
        H = model.hessian(params)
        Hinv = np.linalg.inv(-H)
        N = len(d)
        if vce == "iid":
            V = Hinv
        else:
            S = model.score_obs(params)
            if vce == "robust":
                V = Hinv @ (S.T @ S) @ Hinv * N / (N - 1)
            else:
                from .hdfe import _recode
                cc = _recode(group_codes(d, ["__cl"]))
                G = cc.max() + 1
                Sg = np.zeros((G, S.shape[1])); np.add.at(Sg, cc, S)
                V = Hinv @ (Sg.T @ Sg) @ Hinv * G / (G - 1)
        se = np.sqrt(np.diag(V))
        allnames = names + (["lnalpha"] if c == "nbreg" else [])
        if c == "nbreg":  # statsmodels alpha -> report lnalpha like Stata
            a = params[-1]; params = params.copy(); se = se.copy()
            se[-1] = se[-1] / a; params[-1] = np.log(a)
        z = params / se
        pv = 2 * stats.norm.sf(np.abs(z))
        coefs = {n: {"coef": params[i], "se": se[i], "t": z[i], "p": pv[i],
                     "ci_low": params[i] - 1.96 * se[i], "ci_high": params[i] + 1.96 * se[i]} for i, n in enumerate(allnames)}
        sample = pd.Series(False, index=df.index); sample[d.index] = True
        df["__esample"] = sample.astype(float)
        out = {"estimator": c, "dep": ycol, "N": N, "coefs": coefs, "names": allnames, "vce": vce,
               "cluster": [clcol] if clcol else None, "pseudo_r2": getattr(r, "prsquared", None), "ll": r.llf,
               "omitted_collinear": dropped, "vcov": V.tolist()}
        if c in ("probit", "logit"):
            try:
                mfx = r.get_margeff(at="overall")
                out["margins_dydx"] = {n: {"dydx": float(mfx.margeff[i]), "se": float(mfx.margeff_se[i])}
                                       for i, n in enumerate(names) if n != "_cons" and i < len(mfx.margeff)}
            except Exception:
                pass
        return out

    def est_extra(self, c, p, opts, msk, vce, clv, ln):
        from . import mlx
        df = self.df
        E = pd.DataFrame(index=df.index)
        toks = self.tokens(p["body"])
        ycol = self.term_columns(toks[0], E)[0]
        exn = self.design(toks[1:], E)
        E[ycol] = df[ycol].values if ycol in df else E[ycol]
        extra = {}
        if vce == "cluster":
            extra["__cl"] = self.resolve_name(clv[0])
        if c == "newey":
            if not self.panel or not self.panel[1]:
                raise StataError("newey requires tsset timevar")
            extra["__t"] = self.panel[1]
            if self.panel[0]:
                extra["__g"] = self.panel[0]
        if c == "clogit":
            extra["__g"] = self.resolve_name(getopt(opts, "gr:group", "strata"))
        if c == "xtlogit":
            if getopt(opts, "fe") is None:
                raise StataError("xtlogit: only fe (conditional logit) is supported; re needs quadrature (run in Stata)")
            extra["__g"] = self.panel[0]
        if c == "psmatch2":
            o = getopt(opts, "out:outcome")
            if o and o is not True:
                extra["__y"] = self.resolve_name(o.split()[0])
        for k, v in extra.items():
            E[k] = df[v].values
        cols = [ycol] + exn + list(extra)
        d = E[msk.values][cols]
        num = d.apply(pd.to_numeric, errors="coerce") if c not in ("clogit", "xtlogit") else d
        d = num.dropna()
        y = d[ycol].values.astype(float)
        from .hdfe import _drop_collinear
        X, names, dropped = _drop_collinear(d[exn].values.astype(float), exn)
        cons = getopt(opts, "nocons:noconstant") is None and c not in ("ologit", "oprobit", "clogit", "xtlogit", "psmatch2")
        if cons:
            X = np.column_stack([X, np.ones(len(d))]); names = names + ["_cons"]
        cl = d["__cl"].values if "__cl" in d else None
        if c == "newey":
            lg = getopt(opts, "lag")
            if not lg or lg is True:
                raise StataError("newey needs lag(#)")
            r = mlx.newey(y, X, names, d["__t"].values.astype(float), int(lg), d["__g"].values if "__g" in d else None)
        elif c == "qreg":
            q = getopt(opts, "q:quantile")
            q = float(q) if q and q is not True else 0.5
            r = mlx.qreg(y, X, names, q / 100 if q > 1 else q, vce)
            self.warnings.append(f"L{ln}: qreg SEs approximate Stata's (coefficients exact)")
        elif c == "tobit":
            lo, hi = getopt(opts, "ll"), getopt(opts, "ul")
            lo = (y.min() if lo is True else float(lo)) if lo else None
            hi = (y.max() if hi is True else float(hi)) if hi else None
            r = mlx.tobit(y, X, names, lo, hi, vce, cl)
        elif c in ("ologit", "oprobit"):
            r = mlx.ordered(y, X, names, "logit" if c == "ologit" else "probit", vce, cl)
        elif c == "mlogit":
            b = getopt(opts, "b:baseoutcome")
            r = mlx.mlogit(y, X, names, float(b) if b and b is not True else None, vce, cl)
        elif c in ("clogit", "xtlogit"):
            r = mlx.clogit(y, X, names, d["__g"].values, vce)
        elif c == "psmatch2":
            nn = getopt(opts, "n:neighbor")
            cal = getopt(opts, "cal:caliper")
            if getopt(opts, "kernel") is not None or getopt(opts, "radius") is not None or getopt(opts, "mahal:mahalanobis") is not None:
                raise StataError("psmatch2: only nearest-neighbour matching is implemented (kernel/radius/mahalanobis: run in Stata)")
            r, gen = mlx.psmatch2(y.astype(int), X, names, d["__y"].values.astype(float) if "__y" in d else None,
                                  "logit" if getopt(opts, "logit") is not None else "probit",
                                  int(nn) if nn and nn is not True else 1, float(cal) if cal and cal is not True else None,
                                  getopt(opts, "com:common") is not None, getopt(opts, "norepl:noreplacement") is None,
                                  getopt(opts, "ties") is not None)
            for k, v in gen.items():
                col = pd.Series(np.nan, index=df.index); col[d.index] = v
                if k == "_n1":
                    pass
                df[k] = col
            self.warnings.append(f"L{ln}: psmatch2 re-implemented (nearest neighbour); matched sample can differ slightly from Stata's because Stata's tie-breaking follows its own random sort order")
        r.pop("_b", None)
        r.update({"dep": ycol, "omitted_collinear": dropped, "cluster": [extra["__cl"]] if "__cl" in extra else None})
        sample = pd.Series(False, index=df.index); sample[d.index] = True
        if c != "psmatch2":
            df["__esample"] = sample.astype(float)
        return r

    def est_ppml(self, E, ycol, exn, opts, vce, clv, msk):
        import pyfixest as pf
        df = self.df
        ab = getopt(opts, "a:absorb")
        fe = [self.resolve_name(x) for x in (ab.split() if ab and ab is not True else [])]
        for f in fe:
            E[f] = df[f].values
        clc = [self.resolve_name(x) for x in clv]
        for col in clc:
            E[col] = df[col].values
        d = E[msk.values]
        ren = {n: f"v{i}" for i, n in enumerate([ycol] + exn + fe + clc)}
        dd = d.rename(columns=ren)
        fml = f"{ren[ycol]} ~ " + " + ".join(ren[n] for n in exn) + (" | " + " + ".join(ren[f] for f in fe) if fe else "")
        vc = {"CRV1": "+".join(ren[c] for c in clc)} if vce == "cluster" else ("hetero" if vce == "robust" else "iid")
        r = pf.fepois(fml, data=dd, vcov=vc)
        inv = {v: k for k, v in ren.items()}
        coefs = {inv.get(k, k): {"coef": float(r.coef()[k]), "se": float(r.se()[k]), "t": float(r.tstat()[k]),
                                 "p": float(r.pvalue()[k])} for k in r.coef().index}
        return {"estimator": "ppml", "dep": ycol, "N": int(r._N), "coefs": coefs, "names": list(coefs), "vce": vce,
                "fe": fe, "cluster": clc or None}

    def est_xtre(self, ln, line, E, ycol, exn, pid, vce, clv, msk):
        from linearmodels.panel import RandomEffects
        df = self.df
        t = self.panel[1]
        E[pid] = df[pid].values; E["__t"] = df[t].values if t else np.arange(len(df))
        d = E[msk.values][[pid, "__t", ycol] + exn].apply(pd.to_numeric, errors="coerce").dropna()
        d = d.set_index([pid, "__t"])
        X = d[exn].copy(); X["_cons"] = 1.0
        from .hdfe import _drop_collinear
        Xa, names, dropped = _drop_collinear(X.values, list(X.columns))
        X = pd.DataFrame(Xa, index=X.index, columns=names)
        mod = RandomEffects(d[ycol], X)
        r = mod.fit(cov_type="clustered", cluster_entity=True) if vce != "iid" else mod.fit()
        coefs = {n: {"coef": float(r.params[n]), "se": float(r.std_errors[n]), "t": float(r.tstats[n]),
                     "p": float(r.pvalues[n])} for n in names}
        sample = pd.Series(False, index=df.index)
        self.warnings.append(f"L{ln}: xtreg re via linearmodels (Swamy-Arora-type); SEs may differ slightly from Stata")
        return {"estimator": "re", "dep": ycol, "N": int(r.nobs), "coefs": coefs, "names": names, "vce": vce,
                "r2_overall": float(r.rsquared_overall), "r2_within": float(r.rsquared_within),
                "omitted_collinear": dropped}

    def est_mediation(self, p, opts, msk, vce, clv):
        """sgmediation2-style Sobel/Aroian/Goodman tests via three OLS regressions."""
        y = self.tokens(p["body"])[0]
        mv = getopt(opts, "mv"); xv = getopt(opts, "iv"); cv = getopt(opts, "cv") or ""
        E = pd.DataFrame(index=self.df.index)
        ycol = self.term_columns(y, E)[0]; m = self.term_columns(mv, E)[0]; x = self.term_columns(xv, E)[0]
        ctr = self.design(self.tokens(cv), E)
        clcols = [[self.resolve_name(c)] for c in clv]
        for cc in clcols:
            E[cc[0]] = self.df[cc[0]].values
        need = [ycol, m, x] + ctr + sum(clcols, [])
        Ed = E[msk.values][need].apply(pd.to_numeric, errors="coerce").dropna()
        a_ = estimate(Ed, m, [x] + ctr, vce=vce, cluster=clcols, convention="reg")
        b_ = estimate(Ed, ycol, [m, x] + ctr, vce=vce, cluster=clcols, convention="reg")
        c_ = estimate(Ed, ycol, [x] + ctr, vce=vce, cluster=clcols, convention="reg")
        a, sa = a_["coefs"][x]["coef"], a_["coefs"][x]["se"]
        b, sb = b_["coefs"][m]["coef"], b_["coefs"][m]["se"]
        ind = a * b
        out = {"estimator": "mediation (Baron-Kenny + Sobel)", "dep": ycol, "N": a_["N"], "vce": vce,
               "coefs": {f"a: {x}->{m}": a_["coefs"][x], f"b: {m}->{ycol}|{x}": b_["coefs"][m],
                         f"c': direct {x}": b_["coefs"][x], f"c: total {x}": c_["coefs"][x]}}
        for nm, se in (("Sobel", math.sqrt(b * b * sa * sa + a * a * sb * sb)),
                       ("Aroian", math.sqrt(b * b * sa * sa + a * a * sb * sb + sa * sa * sb * sb)),
                       ("Goodman", math.sqrt(max(b * b * sa * sa + a * a * sb * sb - sa * sa * sb * sb, 0)))):
            z = ind / se if se > 0 else np.nan
            out["coefs"][f"indirect ({nm})"] = {"coef": ind, "se": se, "t": z, "p": 2 * stats.norm.sf(abs(z))}
        tot = c_["coefs"][x]["coef"]
        out["proportion_mediated"] = ind / tot if tot else None
        out["names"] = list(out["coefs"])
        for k in ("_index", "_b"):
            for r in (a_, b_, c_):
                r.pop(k, None)
        return out

    # ------------------------------------------------------------------ output
    def summarize(self, keep_dummies=False):
        lines = ["# ek stata results", ""]
        if self.errors:
            lines += ["## Errors", ""] + [f"- {e}" for e in self.errors] + [""]
        if self.warnings:
            lines += ["## Warnings", ""] + [f"- {w}" for w in dict.fromkeys(self.warnings)] + [""]
        lines += ["## Estimates", ""]
        for r in self.estimates:
            lines += self.format_est(r, keep_dummies) + [""]
            if r.get("vce_scan"):
                lines += [f"**SE-convention scan for [{r['id']}]** (same coefficients, different inference):", "",
                          "| variant | id | key coef (se) t | weak-ID F |", "|---|---|---|---|"]
                for v in r["vce_scan"]:
                    if "error" in v:
                        lines.append(f"| {v['variant']} | - | error: {v['error']} | |"); continue
                    cells = "; ".join(f"{n} {d['coef']:.4g} ({d['se']:.4g}) t={d['t']:.3f}" for n, d in v.items()
                                      if isinstance(d, dict) and d.get("se") is not None and d.get("t") is not None)
                    wf = v.get("kp_rk_Wald_F") or ""
                    lines.append(f"| {v['variant']} | {v['id']} | {cells} | {wf if wf == '' else round(wf, 3)} |")
                lines.append("")
        if self.descriptives:
            lines += ["## Descriptives / tests", ""]
            for d in self.descriptives:
                lines.append(f"### L{d['line']}: `{d['cmd'][:150]}`")
                if "summary" in d:
                    lines.append("| var | N | mean | sd | min | p50 | max |")
                    lines.append("|---|---|---|---|---|---|---|")
                    for row in d["summary"]:
                        lines.append("| " + " | ".join(str(row[k]) for k in ("var", "N", "mean", "sd", "min", "p50", "max")) + " |")
                else:
                    lines.append("```json\n" + json.dumps(_clean({k: v for k, v in d.items() if k not in ("line", "cmd")}), ensure_ascii=False)[:3000] + "\n```")
                lines.append("")
        data_ev = [e for e in self.events if e["kind"] in ("data", "display")]
        if data_ev:
            lines += ["## Data log", ""] + [f"- L{e['line']}: {e.get('msg')}  `{e['cmd'][:100]}`" for e in data_ev]
        return "\n".join(lines)

    @staticmethod
    def is_dummy(n):
        return bool(re.match(r"^\d+[bo]?\.", n) or re.search(r"#\d+[bo]?\.", n))

    def format_est(self, r, keep_dummies=False):
        hdr = f"### [{r['id']}] L{r['line']}: `{r['command'][:220]}`"
        meta = [f"N={r.get('N')}"]
        for k, lab in (("r2", "R2"), ("adj_r2", "adjR2"), ("r2_within", "withinR2"), ("pseudo_r2", "pseudoR2")):
            if r.get(k) is not None:
                meta.append(f"{lab}={r[k]:.4f}")
        if r.get("fe"):
            meta.append("FE: " + " ".join(r["fe"]))
        vce = r.get("vce")
        if vce == "cluster":
            meta.append(f"cluster: {' '.join(r.get('cluster') or [])} (G={','.join(map(str, r.get('n_clusters') or []))})")
        else:
            meta.append(f"vce: {vce}")
        if r.get("singletons_dropped"):
            meta.append(f"singletons dropped: {r['singletons_dropped']}")
        if r.get("stored_as"):
            meta.append("stored: " + ",".join(r["stored_as"]))
        if r.get("omitted_collinear"):
            om = [o for o in r["omitted_collinear"] if not self.is_dummy(o)]
            nd = len(r["omitted_collinear"]) - len(om)
            meta.append("omitted (collinear): " + ", ".join(om[:8]) + (f" +{nd} dummies" if nd else ""))
        out = [hdr, "", " | ".join(meta), "", "| var | coef | se | t/z | p | |", "|---|---|---|---|---|---|"]
        nd = 0
        for n in r.get("names", []):
            c = r["coefs"].get(n)
            if c is None:
                continue
            if not keep_dummies and self.is_dummy(n):
                nd += 1; continue
            f = lambda v, d=4: "" if v is None else f"{v:.{d}f}"
            out.append(f"| {n} | {f(c['coef'])} | {f(c.get('se'))} | {f(c.get('t'), 3)} | {f(c.get('p'), 4)} | {stars(c.get('p'))} |")
        if "_cons" in r["coefs"] and "_cons" not in r.get("names", []):
            out.append(f"| _cons | {r['coefs']['_cons']['coef']:.4f} | | | | |")
        if nd:
            out.append(f"| ({nd} factor dummies hidden) | | | | | |")
        iv = r.get("iv")
        if iv:
            s = [f"KP rk LM={iv.get('kp_rk_LM', float('nan')):.4f} (p={iv.get('kp_rk_LM_p', float('nan')):.4f})",
                 f"KP rk Wald F={iv.get('kp_rk_Wald_F', float('nan')):.4f}",
                 f"CD Wald F={iv.get('cragg_donald_F') or float('nan'):.4f}",
                 f"AR F={iv['AR_F']:.4f} (chi2 p={iv['AR_p']:.4f})"]
            ov = iv.get("overid", {})
            if ov.get("test") not in (None, "none"):
                s.append(f"{ov['test']}={ov['stat']:.4f} (p={ov['p']:.4f})")
            else:
                s.append("exactly identified")
            s.append(f"endog test p={iv['endogeneity_test']['p']:.4f}")
            out += ["", "IV: " + "; ".join(s)]
            for fs in iv["first_stage"]:
                cs = ", ".join(f"{k}={v:.4f}(se {fs['se'][k]:.4f})" for k, v in fs["coefs"].items())
                out.append(f"- first stage {fs['endog']}: {cs}; partial F={fs['partial_F']:.2f}; SW F={fs['SW_F']:.2f}; partial R2={fs['partial_R2']:.4f}")
        for t in r.get("tests", []):
            out.append(f"- post-estimation `{t.get('test')} {t.get('expr')}`: " + ", ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in t.items()
                if k in ("estimate", "se", "t", "p", "F", "chi2", "df")))
        if r.get("proportion_mediated") is not None:
            out.append(f"\nproportion mediated = {r['proportion_mediated']:.4f}")
        if r.get("margins_dydx"):
            out.append("\nAverage marginal effects: " + ", ".join(f"{k}={v['dydx']:.4f}({v['se']:.4f})" for k, v in list(r["margins_dydx"].items())[:10] if not self.is_dummy(k)))
        return out


def stata_pctile(s, p):
    """Stata _pctile/summarize default percentile definition."""
    x = np.sort(np.asarray(pd.to_numeric(s, errors="coerce").dropna(), dtype=float))
    n = len(x)
    if n == 0:
        return np.nan
    P = n * p / 100.0
    i = int(math.floor(P))
    if P == i:
        if i <= 0:
            return x[0]
        if i >= n:
            return x[-1]
        return (x[i - 1] + x[i]) / 2
    return x[min(i, n - 1)]


def run(do=None, cmds=None, data=None, data_dirs=None, out_dir="ek_out/stata", est_filter=None, keep_dummies=False,
        vce_scan=None):
    t0 = time.time()
    S = Session(base_dirs=data_dirs or [], out_dir=out_dir, est_filter=est_filter)
    S.vce_variants = [v.strip() for v in (vce_scan or "").split(";") if v.strip()]
    if data:
        S.base_dirs.append(os.path.dirname(os.path.abspath(data)))
        S.df = S.load(data)
    if do:
        S.run_file(do)
    for c in cmds or []:
        S.run_text(c)
    iv_w = []
    for r in S.estimates:
        dropped = [x[len("instrument "):] for x in (r.get("omitted_collinear") or []) if str(x).startswith("instrument ")]
        if dropped:
            iv_w.append(f"L{r['line']}: excluded instrument(s) {', '.join(dropped)} dropped as collinear - identified by "
                        f"{(r.get('iv') or {}).get('n_excluded_instruments')} instrument(s), not as written (lags of a "
                        "time-invariant variable are identical within a unit); use the Stock-Yogo value for that count")
    S.warnings[:0] = iv_w
    os.makedirs(out_dir, exist_ok=True)
    md = S.summarize(keep_dummies)
    mdp = os.path.join(out_dir, "stata_results.md")
    open(mdp, "w", encoding="utf-8").write(md)
    est_out = []
    for r in S.estimates:
        rr = {k: v for k, v in r.items() if not k.startswith("_")}
        if len(rr.get("names", [])) > 150:
            rr.pop("vcov", None)
        est_out.append(rr)
    jp = os.path.join(out_dir, "stata_estimates.json")
    open(jp, "w", encoding="utf-8").write(json.dumps(_clean(est_out), ensure_ascii=False))
    json.dump({"do": os.path.abspath(do) if do else None, "cmds": cmds, "data": data, "est_filter": est_filter,
               "n_estimates": len(S.estimates), "n_errors": len(S.errors),
               "time": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(os.path.join(out_dir, "run_info.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    brief = []
    for r in S.estimates:
        nm = [n for n in r.get("names", []) if n != "_cons" and not Session.is_dummy(n)]
        key = nm[:3] if r.get("cmd") != "sgmediation2" else [n for n in nm if n.startswith("indirect (Sobel")] + nm[:1]
        parts = []
        for n in key:
            c = r["coefs"][n]
            parts.append(f"{n}={c['coef']:.4g}{stars(c.get('p'))}({c['se']:.3g})" if c.get("se") is not None else f"{n}={c['coef']:.4g}")
        brief.append(f"[{r['id']}] L{r['line']} {r['cmd']} {r.get('dep')}: " + ", ".join(parts) + f" N={r.get('N')}")
    return {"ok": not S.errors, "n_estimates": len(S.estimates), "seconds": round(time.time() - t0, 1),
            "results_md": mdp, "estimates_json": jp, "errors": S.errors[:30], "warnings": list(dict.fromkeys(S.warnings))[:30],
            "estimates": brief}, S
