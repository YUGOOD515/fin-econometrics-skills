"""Stata expression -> numpy/pandas evaluator with Stata missing-value semantics
(missing compares as +inf, logical true = nonzero incl. missing)."""
import re
import numpy as np
import pandas as pd


def _isstr(x):
    return isinstance(x, pd.Series) and (x.dtype == object or pd.api.types.is_string_dtype(x.dtype))

TOK = re.compile(r"""\s*(?:
 (?P<num>\d+\.\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?|\d+(?:[eE][-+]?\d+)?)|
 (?P<str>"[^"]*"|`"[^"]*"')|
 (?P<op>==|!=|~=|>=|<=|&&|\|\||[-+*/^()<>&|!~,\[\]=])|
 (?P<miss>\.[a-z]?)|
 (?P<name>(?:[LDFS]\d*\.)?[A-Za-z_][A-Za-z0-9_]*)
)""", re.X)


def tokenize(s):
    pos, out = 0, []
    s = s.strip()
    while pos < len(s):
        m = TOK.match(s, pos)
        if not m or m.end() == pos:
            raise ValueError(f"cannot parse expression near: {s[pos:pos+20]!r}")
        pos = m.end()
        kind = m.lastgroup
        out.append((kind, m.group(kind)))
    return out


class Parser:
    def __init__(self, toks):
        self.t, self.i = toks, 0

    def peek(self, k=0):
        return self.t[self.i + k] if self.i + k < len(self.t) else (None, None)

    def eat(self, val=None):
        tok = self.peek()
        if val is not None and tok[1] != val:
            raise ValueError(f"expected {val!r}, got {tok[1]!r}")
        self.i += 1
        return tok

    def parse(self):
        node = self.or_()
        if self.i != len(self.t):
            raise ValueError(f"unexpected token {self.peek()[1]!r}")
        return node

    def or_(self):
        n = self.and_()
        while self.peek()[1] in ("|", "||"):
            self.eat(); n = ("or", n, self.and_())
        return n

    def and_(self):
        n = self.not_()
        while self.peek()[1] in ("&", "&&"):
            self.eat(); n = ("and", n, self.not_())
        return n

    def not_(self):
        if self.peek()[1] in ("!", "~"):
            self.eat(); return ("not", self.not_())
        return self.rel()

    def rel(self):
        n = self.add()
        while self.peek()[1] in ("==", "!=", "~=", ">", "<", ">=", "<="):
            op = self.eat()[1]; n = ("cmp", "!=" if op == "~=" else op, n, self.add())
        return n

    def add(self):
        n = self.mul()
        while self.peek()[1] in ("+", "-"):
            op = self.eat()[1]; n = ("bin", op, n, self.mul())
        return n

    def mul(self):
        n = self.unary()
        while self.peek()[1] in ("*", "/"):
            op = self.eat()[1]; n = ("bin", op, n, self.unary())
        return n

    def unary(self):
        if self.peek()[1] == "-":
            self.eat(); return ("neg", self.unary())
        if self.peek()[1] == "+":
            self.eat(); return self.unary()
        return self.pow()

    def pow(self):
        n = self.atom()
        if self.peek()[1] == "^":
            self.eat(); n = ("bin", "^", n, self.unary())
        return n

    def atom(self):
        kind, v = self.peek()
        if kind == "num":
            self.eat(); return ("num", float(v))
        if kind == "str":
            self.eat(); return ("str", v.strip('`"\''))
        if kind == "miss":
            self.eat(); return ("num", np.nan)
        if v == "(":
            self.eat(); n = self.or_(); self.eat(")"); return n
        if kind == "name":
            self.eat()
            if self.peek()[1] == "(":
                self.eat(); args = []
                if self.peek()[1] != ")":
                    args.append(self.or_())
                    while self.peek()[1] == ",":
                        self.eat(); args.append(self.or_())
                self.eat(")"); return ("fn", v.lower(), args)
            if self.peek()[1] == "[":
                self.eat(); idx = self.or_(); self.eat("]"); return ("sub", v, idx)
            return ("var", v)
        raise ValueError(f"unexpected token {v!r}")


def _num(x):
    if isinstance(x, pd.Series):
        return x.astype(float) if (not _isstr(x)) else x
    return x


def _inf(x):
    if isinstance(x, pd.Series) and (not _isstr(x)):
        return x.astype(float).fillna(np.inf)
    if isinstance(x, float) and np.isnan(x):
        return np.inf
    return x


def _truth(x):
    if isinstance(x, pd.Series):
        if x.dtype == bool:
            return x
        return (x.astype(float) != 0) | x.isna()
    return bool(x != 0 or (isinstance(x, float) and np.isnan(x)))


def _f(b):
    return b.astype(float) if isinstance(b, pd.Series) else float(b)


class Evaluator:
    """env: resolve(name) -> Series (supports L./D./F. via callback); n_index: Series of _n (1-based within group)."""

    def __init__(self, df, resolve, groups=None):
        self.df, self.resolve, self.groups = df, resolve, groups
        idx = df.index
        if groups is not None:  # Series of integer group codes aligned with df.index
            g = df.groupby(groups.values, sort=False).cumcount()
            self._n = pd.Series(g.values + 1, index=idx)
            self._N = pd.Series(pd.Series(groups.values).map(pd.Series(groups.values).value_counts()).values, index=idx)
        else:
            self._n = pd.Series(np.arange(1, len(df) + 1), index=idx)
            self._N = pd.Series(len(df), index=idx)

    def ev(self, n):
        k = n[0]
        if k == "num":
            return n[1]
        if k == "str":
            return n[1]
        if k == "var":
            if n[1] == "_n":
                return self._n.astype(float)
            if n[1] == "_N":
                return self._N.astype(float)
            return _num(self.resolve(n[1]))
        if k == "neg":
            return -self.ev(n[1])
        if k == "not":
            v = self.ev(n[1])
            return _f(~_truth(v)) if isinstance(v, pd.Series) else float(not _truth(v))
        if k in ("and", "or"):
            a, b = _truth(self.ev(n[1])), _truth(self.ev(n[2]))
            r = (a & b) if k == "and" else (a | b)
            return _f(r) if isinstance(r, pd.Series) else float(r)
        if k == "cmp":
            op, a, b = n[1], self.ev(n[2]), self.ev(n[3])
            if not (isinstance(a, str) or isinstance(b, str) or
                    (isinstance(a, pd.Series) and _isstr(a)) or (isinstance(b, pd.Series) and _isstr(b))):
                a, b = _inf(a), _inf(b)
            r = {"==": lambda: a == b, "!=": lambda: a != b, ">": lambda: a > b, "<": lambda: a < b,
                 ">=": lambda: a >= b, "<=": lambda: a <= b}[op]()
            return _f(r)
        if k == "bin":
            op, a, b = n[1], self.ev(n[2]), self.ev(n[3])
            if op == "+":
                return a + b
            if op == "-":
                return a - b
            if op == "*":
                return a * b
            if op == "/":
                with np.errstate(all="ignore"):
                    r = a / b
                return r.replace([np.inf, -np.inf], np.nan) if isinstance(r, pd.Series) else (np.nan if b == 0 else r)
            if op == "^":
                return a ** b
        if k == "sub":
            name, idx = n[1], n[2]
            s = self.resolve(name)
            off = self._offset(idx)
            if off is None:
                raise ValueError("only x[_n+k] / x[_n-k] subscripts are supported")
            g = self.groups
            return s.groupby(g.values, sort=False).shift(-off) if g is not None else s.shift(-off)
        if k == "fn":
            return self.fn(n[1], n[2])
        raise ValueError(f"bad node {k}")

    def _offset(self, idx):
        if idx == ("var", "_n"):
            return 0
        if idx[0] == "bin" and idx[2] == ("var", "_n") and idx[3][0] == "num":
            return int(idx[3][1]) * (1 if idx[1] == "+" else -1)
        return None

    def fn(self, name, args):
        A = [self.ev(a) for a in args]
        npf = {"ln": np.log, "log": np.log, "log10": np.log10, "exp": np.exp, "sqrt": np.sqrt, "abs": np.abs,
               "floor": np.floor, "ceil": np.ceil, "int": np.trunc, "trunc": np.trunc, "sin": np.sin, "cos": np.cos}
        if name in npf:
            with np.errstate(all="ignore"):
                r = npf[name](A[0])
            return r.replace([np.inf, -np.inf], np.nan) if isinstance(r, pd.Series) else r
        if name == "round":
            u = A[1] if len(A) > 1 else 1.0
            return np.round(A[0] / u) * u
        if name in ("max", "min"):
            fr = pd.concat([a if isinstance(a, pd.Series) else pd.Series(a, index=self.df.index) for a in A], axis=1)
            return fr.max(axis=1) if name == "max" else fr.min(axis=1)
        if name == "missing":
            r = None
            for a in A:
                m = a.isna() if isinstance(a, pd.Series) else pd.Series(pd.isna(a), index=self.df.index)
                if a is not None and isinstance(a, pd.Series) and _isstr(a):
                    m = m | (a == "")
                r = m if r is None else (r | m)
            return _f(r)
        if name == "inrange":
            x, lo, hi = _inf(A[0]), A[1], A[2]
            return _f((x >= lo) & (x <= hi))
        if name == "inlist":
            x = A[0]
            r = None
            for v in A[1:]:
                m = (x == v)
                r = m if r is None else (r | m)
            return _f(r)
        if name == "cond":
            c = _truth(A[0])
            a, b = A[1], A[2]
            out = pd.Series(np.where(c, a, b), index=self.df.index)
            if len(A) > 3:
                cond_raw = A[0]
                if isinstance(cond_raw, pd.Series):
                    out[cond_raw.isna()] = A[3]
            return out
        if name == "sum":  # running sum (Stata gen sum())
            s = A[0].fillna(0)
            return s.groupby(self.groups.values, sort=False).cumsum() if self.groups is not None else s.cumsum()
        if name in ("normal", "normalden", "invnormal", "ttail", "chi2tail", "mod", "sign", "runiform", "rnormal",
                    "logit", "invlogit", "lnnormal", "comb"):
            from scipy import stats as _st
            def _ser(r):
                return pd.Series(r, index=self.df.index) if np.ndim(r) else r
            if name == "normal":
                return _ser(_st.norm.cdf(A[0]))
            if name == "normalden":
                return _ser(_st.norm.pdf(A[0]) if len(A) == 1 else _st.norm.pdf(A[0], 0, A[1]) if len(A) == 2 else _st.norm.pdf(A[0], A[1], A[2]))
            if name == "invnormal":
                return _ser(_st.norm.ppf(A[0]))
            if name == "lnnormal":
                return _ser(_st.norm.logcdf(A[0]))
            if name == "ttail":
                return _ser(_st.t.sf(A[1], A[0]))
            if name == "chi2tail":
                return _ser(_st.chi2.sf(A[1], A[0]))
            if name == "mod":
                return A[0] % A[1]
            if name == "sign":
                return np.sign(A[0])
            if name == "logit":
                return np.log(A[0] / (1 - A[0]))
            if name == "invlogit":
                return 1 / (1 + np.exp(-A[0]))
            if name == "runiform":
                return pd.Series(np.random.default_rng(12345).uniform(size=len(self.df)), index=self.df.index)
            if name == "rnormal":
                return pd.Series(np.random.default_rng(12345).normal(size=len(self.df)), index=self.df.index)
        if name in ("strlen", "length"):
            return A[0].astype(str).str.len().astype(float)
        if name == "substr":
            st, ln = int(A[1]), int(A[2])
            return A[0].astype(str).str.slice(st - 1, st - 1 + ln if ln >= 0 else None)
        if name in ("upper", "strupper"):
            return A[0].astype(str).str.upper()
        if name in ("lower", "strlower"):
            return A[0].astype(str).str.lower()
        if name in ("trim", "strtrim"):
            return A[0].astype(str).str.strip()
        if name == "strpos":
            return A[0].astype(str).str.find(A[1]).astype(float) + 1
        if name == "real":
            return pd.to_numeric(A[0], errors="coerce")
        if name == "string":
            return A[0].astype(str)
        if name in ("year", "month", "day"):
            return getattr(pd.to_datetime(A[0]).dt, name).astype(float)
        raise ValueError(f"function {name}() not supported")


def evaluate(expr, df, resolve, groups=None):
    tree = Parser(tokenize(expr)).parse()
    r = Evaluator(df, resolve, groups).ev(tree)
    if not isinstance(r, pd.Series):
        r = pd.Series(r, index=df.index)
    return r
