import json, math, os, sys
import numpy as np
import pandas as pd

ALPHA = 0.05


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else float(f"{f:.6g}")
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return o


def emit(obj, path=None):
    """Print JSON to stdout (and optionally save it)."""
    obj = _clean(obj)
    txt = json.dumps(obj, ensure_ascii=False, indent=2)
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(txt)
    sys.stdout.write(txt + "\n")
    return obj


def fail(msg, **extra):
    emit({"ok": False, "error": msg, **extra})
    sys.exit(1)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def pval_flag(p, alpha=ALPHA):
    return None if p is None else bool(p < alpha)
