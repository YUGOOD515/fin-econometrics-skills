"""Cinelli & Hazlett (2020) omitted-variable-bias sensitivity analysis (sensemakr)."""
import numpy as np
from scipy import stats


def ols(y, X):
    XtX = X.T @ X
    Mi = np.linalg.pinv(XtX)
    b = Mi @ X.T @ y
    e = y - X @ b
    df = len(y) - np.linalg.matrix_rank(X)
    s2 = e @ e / df
    se = np.sqrt(np.diag(Mi) * s2)
    return b, se, df


def partial_r2(t, df):
    return t ** 2 / (t ** 2 + df)


def group_partial_r2(y, X, cols):
    """Partial R2 of a group of columns via the F statistic."""
    b, se, df = ols(y, X)
    keep = [j for j in range(X.shape[1]) if j not in cols]
    e1 = y - X @ b
    br = np.linalg.lstsq(X[:, keep], y, rcond=None)[0]
    e0 = y - X[:, keep] @ br
    k = len(cols)
    F = ((e0 @ e0 - e1 @ e1) / k) / (e1 @ e1 / df)
    return F * k / (F * k + df)


def rv(t, df, q=1.0, alpha=None):
    f = q * abs(t / np.sqrt(df))
    if alpha is not None:
        tc = stats.t.ppf(1 - alpha / 2, df - 1)
        f = f - tc / np.sqrt(df - 1)
        if f <= 0:
            return 0.0
    return 0.5 * (np.sqrt(f ** 4 + 4 * f ** 2) - f ** 2)


def bounds(r2dxj_x, r2yxj_dx, kd, ky):
    r2dz_x = kd * (r2dxj_x / (1 - r2dxj_x))
    if r2dz_x >= 1:
        return None
    r2zxj_xd = kd * (r2dxj_x ** 2) / ((1 - kd * r2dxj_x) * (1 - r2dxj_x))
    if r2zxj_xd >= 1:
        return None
    r2yz_dx = ((np.sqrt(ky) + np.sqrt(r2zxj_xd)) / np.sqrt(1 - r2zxj_xd)) ** 2 * (r2yxj_dx / (1 - r2yxj_dx))
    return r2dz_x, min(r2yz_dx, 1.0)


def adjusted(est, se, df, r2dz_x, r2yz_dx, reduce=True, alpha=0.05, h0=0.0):
    bias = se * np.sqrt(r2yz_dx * r2dz_x / (1 - r2dz_x)) * np.sqrt(df)
    adj = (est - np.sign(est) * bias) if reduce else (est + np.sign(est) * bias)
    adj_se = se * np.sqrt((1 - r2yz_dx) / (1 - r2dz_x)) * np.sqrt(df / (df - 1))
    t = (adj - h0) / adj_se
    tc = stats.t.ppf(1 - alpha / 2, df - 1)
    return adj, adj_se, t, adj - tc * adj_se, adj + tc * adj_se
