"""Dynamic panel GMM: Arellano-Bond difference GMM and Blundell-Bond system GMM.

Follows xtabond2 (Roodman 2009) conventions: GMM-style and IV-style instrument groups,
`collapse`, `laglimits`, `equation(diff|level|both)`, `passthru`, `mz`, first differences
or forward orthogonal deviations (stored one period later, so lag limits keep their meaning),
one-step weight matrix H = D D' (h(3)), two-step efficient GMM with the Windmeijer (2005)
correction, Arellano-Bond AR(m) tests, Sargan / Hansen and difference-in-Hansen tests.
Official xtabond / xtdpdsys are mapped onto the same engine (see stata.py).

Data come in as rectangular (n_units x T) arrays; NaN = missing. Everything is a sum over
units, computed in chunks so that wide (uncollapsed) instrument sets stay in memory.
"""
import numpy as np
from scipy import stats


def _pinv(A):
    """Inverse of a symmetric PSD matrix; generalized inverse (like Stata's invsym) only when singular.
    Columns are scaled first so that badly scaled regressors (e.g. `year` next to `_cons`) are not truncated."""
    A = (A + A.T) / 2
    d = np.sqrt(np.clip(np.abs(np.diag(A)), 0, None))
    d = np.where(d > 0, d, 1.0)
    As = A / np.outer(d, d)
    try:
        c = np.linalg.cond(As)
        if np.isfinite(c) and c < 1e13:
            return np.linalg.inv(As) / np.outer(d, d)
    except np.linalg.LinAlgError:
        pass
    return np.linalg.pinv(As, rcond=1e-12, hermitian=True) / np.outer(d, d)


def _independent(G, tol=1e-9):
    """Indices of a maximal set of linearly independent columns (in order) from a Gram matrix G = Z'Z."""
    d = np.sqrt(np.clip(np.diag(G), 0, None))
    d = np.where(d > 0, d, 1.0)
    C = G / np.outer(d, d)
    keep = []
    for j in range(C.shape[0]):
        if not keep:
            if C[j, j] > tol:
                keep.append(j)
            continue
        A = C[np.ix_(keep, keep)]
        c = C[keep, j]
        try:
            r = C[j, j] - c @ np.linalg.solve(A, c)
        except np.linalg.LinAlgError:
            r = 0.0
        if r > tol:
            keep.append(j)
    return keep


class Spec:
    """One instrument group. kind: 'gmm' | 'iv'. arr: (n, T, m) level values of the base variables."""

    def __init__(self, kind, arr, names, lag=(1, None), collapse=False, eq="both", passthru=False, mz=False):
        self.kind, self.arr, self.names = kind, arr, list(names)
        self.a, self.b = lag
        self.collapse, self.eq, self.passthru, self.mz = collapse, eq, passthru, mz
        self.cols = []  # filled by build: list of (block, ...) descriptors


def _fod(A, valid):
    """Forward orthogonal deviations of A (n,T,...) using valid future observations; result at t (not shifted).
    Returns (A_fod, ok) where ok[i,t] = valid[i,t] and at least one valid future obs."""
    n, T = valid.shape
    v = valid.astype(float)
    Ar = np.where(valid.reshape(valid.shape + (1,) * (A.ndim - 2)), np.nan_to_num(A), 0.0)
    # reverse cumulative sums of future values
    cs = np.flip(np.cumsum(np.flip(Ar, 1), 1), 1)          # sum_{s>=t}
    cn = np.flip(np.cumsum(np.flip(v, 1), 1), 1)
    fut_sum = np.concatenate([cs[:, 1:], np.zeros_like(cs[:, :1])], 1)
    fut_n = np.concatenate([cn[:, 1:], np.zeros_like(cn[:, :1])], 1)
    ok = valid & (fut_n > 0)
    with np.errstate(all="ignore"):
        mean = fut_sum / np.where(fut_n > 0, fut_n, 1).reshape(fut_n.shape + (1,) * (A.ndim - 2))
        c = np.sqrt(fut_n / (fut_n + 1)).reshape(fut_n.shape + (1,) * (A.ndim - 2))
    out = c * (Ar - mean)
    return out, ok, fut_n


class DPD:
    def __init__(self, y, X, xnames, specs, system=True, fod=False, constant=True, h=3, artests=2,
                 chunk_cells=8_000_000):
        """y: (n,T); X: (n,T,K) with NaN for missing; xnames: K names. constant: add _cons to the level
        equation (system GMM; also xtabond's level-equation constant when system=False and constant='level')."""
        self.n, self.T = y.shape
        self.system, self.fod, self.h, self.artests = system, fod, h, artests
        self.cons_level = bool(constant) and (system or constant == "level")
        n, T = self.n, self.T
        K0 = X.shape[2]
        valid = ~np.isnan(y) & ~np.isnan(X).any(axis=2)
        # iv-style instruments without mz: missing -> drop the observation
        for s in specs:
            if s.kind == "iv" and not s.mz:
                valid &= ~np.isnan(s.arr).any(axis=2)
        self.valid = valid
        Rd = T - 1
        self.Rd = Rd
        R = Rd + (T if self.system or self.cons_level else 0)
        self.R = R
        y0 = np.where(valid, y, 0.0); X0 = np.where(valid[:, :, None], X, 0.0)
        # ---- transformed (diff/FOD) block, row r <-> "time" r+1
        if fod:
            yf, okf, futn = _fod(y[:, :, None], valid)
            Xf, _, _ = _fod(X, valid)
            vd = okf[:, :-1]
            yd = yf[:, :-1, 0]; Xd = Xf[:, :-1, :]
        else:
            vd = valid[:, 1:] & valid[:, :-1]
            yd = y0[:, 1:] - y0[:, :-1]; Xd = X0[:, 1:] - X0[:, :-1]
        self.vd = vd
        K = K0 + (1 if self.cons_level else 0)
        self.names = list(xnames) + (["_cons"] if self.cons_level else [])
        Y = np.zeros((n, R)); XX = np.zeros((n, R, K)); VR = np.zeros((n, R), bool)
        Y[:, :Rd] = np.where(vd, yd, 0); XX[:, :Rd, :K0] = np.where(vd[:, :, None], Xd, 0); VR[:, :Rd] = vd
        if R > Rd:
            vl = valid.copy()
            if not self.system:  # xtabond-style constant: level rows only for obs that are in the diff sample
                vl = valid & np.concatenate([np.zeros((n, 1), bool), vd], 1)
            Y[:, Rd:] = np.where(vl, y0, 0); XX[:, Rd:, :K0] = np.where(vl[:, :, None], X0, 0); VR[:, Rd:] = vl
            if self.cons_level:
                XX[:, Rd:, K0] = vl.astype(float)
            self.vl = vl
        self.Y, self.X, self.VR = Y, XX, VR
        # ---- D matrices (maps level errors -> equation rows), used for H = D D'
        self.D_diff = self._D_diff(valid, vd)
        # ---- instruments
        self.specs = specs
        self._build_instruments()
        self.chunk = max(1, int(chunk_cells // max(1, R * max(self.L, 1))))
        # drop all-zero instrument columns
        nz = np.zeros(self.L, bool)
        for ix in self._chunks():
            Z = self.Z(ix)
            nz |= (np.abs(Z) > 0).any(axis=(0, 1))
        self.zkeep = np.where(nz)[0]
        self.L = len(self.zkeep)
        # drop linearly dependent instrument columns (in order), like xtabond2's reported instrument count
        ZZ = np.zeros((self.L, self.L))
        for ix in self._chunks():
            Z = self.Z(ix)
            ZZ += np.einsum("nrl,nrm->lm", Z, Z)
        keep = _independent(ZZ)
        self.z_collinear = [self.zlabels_all[self.zkeep[j]] for j in range(self.L) if j not in set(keep)]
        self.zkeep = self.zkeep[keep]
        self.zlabels = [self.zlabels_all[j] for j in self.zkeep]
        self.L = len(self.zkeep)

    # --------------------------------------------------------------- structure
    def _D_diff(self, valid, vd):
        n, T = valid.shape
        if self.fod:
            v = valid.astype(float)
            cn = np.flip(np.cumsum(np.flip(v, 1), 1), 1)
            fut_n = np.concatenate([cn[:, 1:], np.zeros((n, 1))], 1)
            return ("fod", fut_n)
        return ("fd", None)

    def Dmat(self, ix):
        """(len(ix), R, T) map from level errors to equation errors (rows zeroed when invalid)."""
        n, T, Rd = len(ix), self.T, self.Rd
        D = np.zeros((n, self.R, T))
        vd = self.vd[ix]
        if self.D_diff[0] == "fd":
            r = np.arange(Rd)
            D[:, r, r + 1] = 1.0; D[:, r, r] = -1.0
        else:
            fut_n = self.D_diff[1][ix]
            val = self.valid[ix].astype(float)
            for t in range(Rd):
                c = np.sqrt(fut_n[:, t] / (fut_n[:, t] + 1)) if True else None
                with np.errstate(all="ignore"):
                    w = np.where(fut_n[:, t] > 0, 1.0 / fut_n[:, t], 0)
                D[:, t, t] = c
                D[:, t, t + 1:] = -(c * w)[:, None] * val[:, t + 1:]
        D[:, :Rd] *= vd[:, :, None]
        if self.R > Rd:
            vl = self.vl[ix]
            for t in range(T):
                D[:, Rd + t, t] = vl[:, t]
        return D

    def _build_instruments(self):
        """Create column descriptors: each column = function filling Z[:, rows] for a chunk."""
        T, Rd = self.T, self.Rd
        cols, labels = [], []
        has_level = self.system
        for s in self.specs:
            m = s.arr.shape[2]
            if s.kind == "gmm":
                a = s.a if s.a is not None else 1
                b = s.b
                for j in range(m):
                    nm = s.names[j]
                    if s.eq in ("both", "diff"):
                        # diff rows r (time tau=r+1): v[tau-l], l in [a, b]
                        pairs = []
                        for r in range(Rd):
                            tau = r + 1
                            lmax = tau if b is None else min(b, tau)
                            for l in range(a, lmax + 1):
                                if tau - l >= 0:
                                    pairs.append((r, tau - l, l))
                        if s.collapse:
                            for l in sorted(set(p[2] for p in pairs)):
                                cols.append(("gd", s, j, [(r, src) for r, src, ll in pairs if ll == l]))
                                labels.append(f"gmm {nm} L{l} (diff, collapsed)")
                        else:
                            for r, src, l in pairs:
                                cols.append(("gd", s, j, [(r, src)]))
                                labels.append(f"gmm {nm} L{l} t{r + 1} (diff)")
                    if has_level and s.eq in ("both", "level"):
                        la = a - 1  # xtabond2: level equation uses lag a-1 of the differenced variable
                        rows = [(t, t - la) for t in range(T) if t - la - 1 >= 0 and t - la <= T - 1]
                        if s.collapse:
                            cols.append(("gl", s, j, rows)); labels.append(f"gmm D.{nm} L{la} (level, collapsed)")
                        else:
                            for t, src in rows:
                                cols.append(("gl", s, j, [(t, src)])); labels.append(f"gmm D.{nm} L{la} t{t} (level)")
            else:
                for j in range(m):
                    nm = s.names[j]
                    use_d = s.eq in ("both", "diff")
                    use_l = has_level and s.eq in ("both", "level")
                    if s.eq == "level" and not has_level:
                        continue
                    if use_d or use_l:
                        cols.append(("iv", s, j, (use_d, use_l)))
                        labels.append(f"iv {nm} ({'both' if use_d and use_l else ('diff' if use_d else 'level')})")
        if self.cons_level:
            cols.append(("cons", None, None, None)); labels.append("_cons (level)")
        self.zcols, self.zlabels_all = cols, labels
        self.L = len(cols)
        # transformed iv-style data
        self._ivd = {}
        for s in self.specs:
            if s.kind == "iv":
                A = s.arr
                A0 = np.where(np.isnan(A), 0.0, A) if s.mz else A
                if s.passthru:
                    d = A0[:, 1:]
                elif self.fod:
                    f, _, _ = _fod(A0, ~np.isnan(A0).any(axis=2))
                    d = f[:, :-1]
                else:
                    d = A0[:, 1:] - A0[:, :-1]
                self._ivd[id(s)] = (np.nan_to_num(d), np.nan_to_num(A0))

    def _chunks(self):
        for st in range(0, self.n, self.chunk):
            yield np.arange(st, min(self.n, st + self.chunk))

    def Z(self, ix, keep=None):
        n, Rd = len(ix), self.Rd
        Z = np.zeros((n, self.R, len(self.zcols)))
        for c, (kind, s, j, info) in enumerate(self.zcols):
            if kind == "gd":
                A = s.arr[ix, :, j]
                for r, src in info:
                    Z[:, r, c] = A[:, src]
            elif kind == "gl":
                A = s.arr[ix, :, j]
                for t, src in info:
                    Z[:, Rd + t, c] = A[:, src] - A[:, src - 1]
            elif kind == "iv":
                d, lv = self._ivd[id(s)]
                use_d, use_l = info
                if use_d:
                    Z[:, :Rd, c] = d[ix, :, j]
                if use_l:
                    Z[:, Rd:, c] = lv[ix, :, j]
            else:
                Z[:, Rd:, c] = 1.0
        Z = np.nan_to_num(Z)
        Z *= self.VR[ix][:, :, None]
        if hasattr(self, "zkeep"):
            Z = Z[:, :, self.zkeep]
        if keep is not None:
            Z = Z[:, :, keep]
        return Z

    # --------------------------------------------------------------- estimation
    def moments(self, keep=None):
        L = self.L if keep is None else len(keep)
        K = self.X.shape[2]
        ZX = np.zeros((L, K)); Zy = np.zeros(L); ZHZ = np.zeros((L, L))
        for ix in self._chunks():
            Z = self.Z(ix, keep)
            X, Y = self.X[ix], self.Y[ix]
            ZX += np.einsum("nrl,nrk->lk", Z, X); Zy += np.einsum("nrl,nr->l", Z, Y)
            if self.h == 1:
                ZHZ += np.einsum("nrl,nrm->lm", Z, Z)
            else:
                D = self.Dmat(ix)
                if self.h == 2 and self.R > self.Rd:
                    for blk in (slice(0, self.Rd), slice(self.Rd, self.R)):
                        DZ = np.einsum("nrt,nrl->ntl", D[:, blk], Z[:, blk]); ZHZ += np.einsum("ntl,ntm->lm", DZ, DZ)
                else:
                    DZ = np.einsum("nrt,nrl->ntl", D, Z); ZHZ += np.einsum("ntl,ntm->lm", DZ, DZ)
        return ZX, Zy, ZHZ

    def scores(self, b, keep=None):
        """g_i = Z_i' e_i  (n, L) and residuals e (n, R)."""
        E = self.Y - self.X @ b
        E *= self.VR
        G = np.zeros((self.n, self.L if keep is None else len(keep)))
        for ix in self._chunks():
            G[ix] = np.einsum("nrl,nr->nl", self.Z(ix, keep), E[ix])
        return G, E

    def fit(self, twostep=False, robust=False, small=False, keep=None, full=True):
        ZX, Zy, ZHZ = self.moments(keep)
        XZ = ZX.T
        W1 = _pinv(ZHZ)
        A1 = XZ @ W1 @ ZX
        M1 = _pinv(A1)
        b1 = M1 @ XZ @ W1 @ Zy
        G1, E1 = self.scores(b1, keep)
        S1 = G1.T @ G1
        V1r = M1 @ XZ @ W1 @ S1 @ W1 @ ZX @ M1
        out = {"b1": b1, "M1": M1, "W1": W1, "V1r": V1r, "XZ": XZ, "ZX": ZX, "Zy": Zy, "ZHZ": ZHZ, "E1": E1, "G1": G1}
        W2 = _pinv(S1)
        A2 = XZ @ W2 @ ZX
        M2 = _pinv(A2)
        b2 = M2 @ XZ @ W2 @ Zy
        G2, E2 = self.scores(b2, keep)
        g2 = G2.sum(0)
        out.update({"W2": W2, "M2": M2, "b2": b2, "E2": E2, "G2": G2})
        out["hansen"] = float(g2 @ W2 @ g2)
        if not full:
            return out
        # sigma^2 of the level error (one-step, homoskedastic): from the transformed-equation residuals
        nd = self.VR[:, :self.Rd].sum()
        ed = E1[:, :self.Rd]
        Kx = len(b1)
        if self.fod:
            s2 = float((ed * ed).sum() / max(nd - Kx, 1))
        else:
            s2 = float((ed * ed).sum() / (2.0 * max(nd - Kx, 1)))
        out["sig2"] = s2
        g1 = G1.sum(0)
        out["sargan"] = float(g1 @ W1 @ g1) / s2
        if twostep:
            # Windmeijer correction
            K = len(b2)
            Dw = np.zeros((K, K))
            Q = np.zeros((K, self.L if keep is None else len(keep), self.L if keep is None else len(keep)))
            for ix in self._chunks():
                Zc = self.Z(ix, keep)
                ZXi = np.einsum("nrl,nrk->nkl", Zc, self.X[ix])
                Q += np.einsum("nkl,nm->klm", ZXi, G1[ix])
            Ze2 = g2
            for k in range(K):
                dS = Q[k] + Q[k].T
                Dw[:, k] = M2 @ XZ @ W2 @ dS @ W2 @ Ze2
            Vc = M2 + Dw @ M2 + M2 @ Dw.T + Dw @ V1r @ Dw.T
            out.update({"b": b2, "V": Vc if robust else M2, "M": M2, "W": W2, "E": E2, "G": G2, "Vnr": M2, "Vc": Vc})
        else:
            out.update({"b": b1, "V": V1r if robust else s2 * M1, "M": M1, "W": W1, "E": E1, "G": G1})
        return out

    # --------------------------------------------------------------- AR tests
    def ar_tests(self, fit, robust, maxm=2):
        """Arellano-Bond test for AR(m) in first-differenced residuals (computed from level residuals)."""
        b, V, M, W, G = fit["b"], fit["V"], fit["M"], fit["W"], fit["G"]
        n, T = self.n, self.T
        K0 = self.X.shape[2] - (1 if self.cons_level else 0)
        # level residuals of the structural equation (any _cons cancels in differences)
        Xl = self._Xlevel
        yl = self._ylevel
        el = yl - Xl @ b[:K0]
        v = self.valid
        dv = v[:, 1:] & v[:, :-1]
        ed = np.where(dv, el[:, 1:] - el[:, :-1], 0.0)
        Xd = np.where(dv[:, :, None], Xl[:, 1:] - Xl[:, :-1], 0.0)
        if self.cons_level:
            Xd = np.concatenate([Xd, np.zeros((n, T - 1, 1))], 2)
        XZ, XZW = fit["XZ"], M @ fit["XZ"] @ W
        res = []
        s2 = fit.get("sig2", 1.0)
        for m in range(1, maxm + 1):
            w = np.zeros_like(ed)
            if m < T - 1:
                w[:, m:] = ed[:, :-m]
            w *= dv
            d0 = float((w * ed).sum())
            EX = np.einsum("nr,nrk->k", w, Xd)
            if robust:
                we = (w * ed).sum(1)
                d1 = float((we ** 2).sum())
                zw = G.T @ we
            else:
                # homoskedastic: Omega_i = s2 * D_i D_i'
                d1 = 0.0; zw = np.zeros(G.shape[1])
                for ix in self._chunks():
                    Dfd = np.zeros((len(ix), T - 1, T)); r = np.arange(T - 1)
                    Dfd[:, r, r + 1] = 1; Dfd[:, r, r] = -1
                    Dfd *= dv[ix][:, :, None]
                    Dw_ = np.einsum("nrt,nr->nt", Dfd, w[ix])
                    d1 += float((Dw_ ** 2).sum())
                    DZ = np.einsum("nrt,nrl->ntl", self.Dmat(ix), self.Z(ix))
                    zw += np.einsum("ntl,nt->l", DZ, Dw_)
                d1 *= s2; zw *= s2
            d2 = -2.0 * float(EX @ XZW @ zw)
            d3 = float(EX @ V @ EX)
            den = d1 + d2 + d3
            z = d0 / np.sqrt(den) if den > 0 else np.nan
            res.append({"order": m, "z": z, "p": float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else None})
        return res


def run(y, X, xnames, specs, system=True, fod=False, twostep=False, robust=False, small=False, constant=True,
        h=3, artests=2, diffsargan=True, ids=None):
    m = DPD(y, X, xnames, specs, system=system, fod=fod, constant=constant, h=h, artests=artests)
    m._ylevel = np.where(m.valid, y, 0.0)
    m._Xlevel = np.where(m.valid[:, :, None], np.nan_to_num(X), 0.0)
    # collinear regressors (e.g. time dummies after differencing): drop, in order
    from .hdfe import _drop_collinear
    # collinearity is judged on the level equation in system GMM (as xtabond2's _rmcoll), else on the
    # transformed equation; _cons is checked first so that a redundant dummy, not the constant, is dropped
    rows = np.zeros(m.VR.shape, bool)
    if system:
        rows[:, m.Rd:] = m.VR[:, m.Rd:]
    else:
        rows[:, :m.Rd] = m.VR[:, :m.Rd]
    Xs = m.X.reshape(-1, m.X.shape[2])[rows.reshape(-1)]
    cons_first = system and "_cons" in m.names
    order = ([m.names.index("_cons")] if cons_first else []) + [i for i, nm in enumerate(m.names) if nm != "_cons"]
    _, kept, dropped = _drop_collinear(Xs[:, order], [m.names[i] for i in order])
    kept = [nm for nm in m.names if nm in kept or (nm == "_cons" and not system)]
    if dropped:
        idx = [m.names.index(k) for k in kept]
        m.X = m.X[:, :, idx]
        k0 = [i for i, nm in enumerate(xnames) if nm in kept]
        m._Xlevel = m._Xlevel[:, :, k0]
        m.names = kept
        if "_cons" not in kept:
            m.cons_level = False
    f = m.fit(twostep=twostep, robust=robust, small=small)
    b, V = f["b"], f["V"]
    K = len(b)
    N = int(m.VR[:, m.Rd:].sum()) if system else int(m.VR[:, :m.Rd].sum())
    per = (m.VR[:, m.Rd:] if system else m.VR[:, :m.Rd]).sum(1)
    G = int((per > 0).sum())
    if small:
        V = V * ((N - 1) / (N - K)) * (G / (G - 1)) if robust else V * N / (N - K)
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    with np.errstate(all="ignore"):
        z = b / se
    if small:
        hc = 1 if "_cons" in m.names else 0
        dfr = (G - hc) if (robust or twostep) else (N - (K - hc) - hc)
        p = 2 * stats.t.sf(np.abs(z), dfr); crit = stats.t.ppf(0.975, dfr)
    else:
        dfr = None
        p = 2 * stats.norm.sf(np.abs(z)); crit = stats.norm.ppf(0.975)
    coefs = {nm: {"coef": float(b[i]), "se": float(se[i]), "t": float(z[i]), "p": float(p[i]),
                  "ci_low": float(b[i] - crit * se[i]), "ci_high": float(b[i] + crit * se[i])} for i, nm in enumerate(m.names)}
    sl = [i for i, nm in enumerate(m.names) if nm != "_cons"]
    try:
        wald = float(b[sl] @ np.linalg.solve(V[np.ix_(sl, sl)], b[sl]))
    except np.linalg.LinAlgError:
        wald = float(b[sl] @ np.linalg.pinv(V[np.ix_(sl, sl)]) @ b[sl])
    L = m.L
    dfo = L - K
    out = {"coefs": coefs, "names": list(m.names), "vcov": V.tolist(), "N": N, "n_groups": G,
           "obs_per_group": {"min": int(per[per > 0].min()) if G else 0, "avg": float(per[per > 0].mean()) if G else 0,
                             "max": int(per.max()) if G else 0},
           "n_instruments": L, "omitted_collinear": dropped, "small": small, "df_r": dfr,
           "wald": ({"F": wald / len(sl), "df": [len(sl), dfr], "p": float(stats.f.sf(wald / len(sl), len(sl), dfr))}
                    if small else {"chi2": wald, "df": len(sl), "p": float(stats.chi2.sf(wald, len(sl)))}),
           "sargan": {"chi2": f["sargan"], "df": dfo, "p": float(stats.chi2.sf(f["sargan"], dfo)) if dfo > 0 else None,
                      "note": "not robust, but not weakened by many instruments"},
           "hansen": {"chi2": f["hansen"], "df": dfo, "p": float(stats.chi2.sf(f["hansen"], dfo)) if dfo > 0 else None,
                      "note": "robust, but weakened by many instruments"},
           "instruments": m.zlabels}
    out["ar_tests"] = m.ar_tests(f, robust, artests)
    warns = []
    if L > G:
        warns.append(f"too many instruments: {L} instruments > {G} groups (Hansen test weakened; consider collapse or "
                     "laglimits)")
    if not robust and twostep:
        warns.append("two-step without robust: SEs are downward biased; use robust (Windmeijer-corrected)")
    # difference-in-Hansen tests for each instrument group (and for all level-equation instruments in system GMM)
    dh = []
    if diffsargan and dfo > 0:
        groups = {}
        for j, lab in enumerate(m.zlabels):
            key = lab.split(" t")[0].split(" L")[0] if lab.startswith("gmm") else lab
            if lab.startswith("gmm"):
                key = lab.split(" L")[0] + (" (level)" if "(level" in lab else " (diff)")
            groups.setdefault(key, []).append(j)
        if system:
            groups["all level-equation GMM instruments"] = [j for j, lab in enumerate(m.zlabels)
                                                            if lab.startswith("gmm") and "(level" in lab]
        for key, js in groups.items():
            keep = [j for j in range(L) if j not in js]
            if not js or len(keep) < K:
                continue
            try:
                fk = m.fit(twostep=True, keep=np.array(keep), full=False)
                Jx = fk["hansen"]; dfx = len(keep) - K
                C = f["hansen"] - Jx
                dh.append({"group": key, "n_instr": len(js), "hansen_excluding": Jx, "df_excl": dfx,
                           "p_excl": float(stats.chi2.sf(Jx, dfx)) if dfx > 0 else None,
                           "diff": C, "df_diff": len(js), "p_diff": float(stats.chi2.sf(C, len(js))) if C >= 0 else None})
            except Exception:
                continue
    out["diff_hansen"] = dh
    out["dpd_warnings"] = warns
    return out
