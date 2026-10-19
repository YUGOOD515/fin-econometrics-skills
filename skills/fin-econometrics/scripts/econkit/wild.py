"""Wild (cluster) bootstrap tests for linear regression (Roodman, MacKinnon, Nielsen & Webb 2019 `boottest`).

Restricted (WCR, null imposed, default) or unrestricted (WCU) wild cluster bootstrap-t for one or several
linear restrictions R b = r after OLS / areg / reghdfe / xtreg,fe. Fixed effects nested in the bootstrap
clusters are partialled out (exact); other fixed effects are entered as dummies. Multiway clustering is
supported (CGM variance in each replication, bootstrap clusters = intersection of the clustering variables).
All replications are computed in closed form from cluster-level sums, so cost is O(G K B), not O(N B).
"""
import itertools

import numpy as np
import pandas as pd
from scipy import stats

from .hdfe import demean, group_codes


def _codes(df, cols):
    return pd.factorize(pd.MultiIndex.from_frame(df[cols].astype(str)) if len(cols) > 1 else df[cols[0]])[0]


def _nested(a, b):
    """True if every level of a lies within a single level of b."""
    d = pd.DataFrame({"a": a, "b": b}).drop_duplicates()
    return not d["a"].duplicated().any()


def weights_draw(kind, G, B, rng):
    kind = (kind or "rademacher").lower()
    if kind.startswith("r"):
        return rng.choice([-1.0, 1.0], size=(G, B))
    if kind.startswith("w"):
        vals = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
        return rng.choice(vals, size=(G, B))
    if kind.startswith("m"):
        s5 = np.sqrt(5)
        p = (s5 + 1) / (2 * s5)
        return np.where(rng.random((G, B)) < p, (1 - s5) / 2, (1 + s5) / 2)
    if kind.startswith("n"):
        return rng.standard_normal((G, B))
    raise ValueError(f"unknown weight type {kind}")


class WildBoot:
    def __init__(self, spec, boot_cluster=None, max_dummies=2000):
        E = spec["E"]
        y = E[spec["y"]].astype(float).values
        X = E[spec["x"]].astype(float).values if spec["x"] else np.zeros((len(E), 0))
        names = list(spec["x"])
        if spec["weights"]:
            sw = np.sqrt(E[spec["weights"]].astype(float).values)
        else:
            sw = None
        clus = [ _codes(E, c) for c in spec["cluster"] ] if spec["vce"] == "cluster" and spec["cluster"] else []
        if not clus:
            clus = [np.arange(len(E))]
            self.hetero = True
        else:
            self.hetero = False
        if boot_cluster is not None:
            bc = _codes(E, boot_cluster)
        elif len(clus) == 1:
            bc = clus[0]
        else:
            bc = pd.factorize(pd.MultiIndex.from_arrays(clus))[0]
        self.notes = []
        fe_codes, dummies = [], []
        for f in spec["fe"] or []:
            code = _codes(E, f)
            if _nested(code, bc):
                fe_codes.append(code)
            else:
                lv = np.unique(code)
                if len(lv) > max_dummies:
                    raise ValueError(f"absorbed FE {'#'.join(f)} is not nested in the bootstrap clusters and has "
                                     f"{len(lv)} levels; cannot enter it as dummies")
                D = (code[:, None] == lv[None, 1:]).astype(float)
                dummies.append(D)
                self.notes.append(f"FE {'#'.join(f)} not nested in bootstrap clusters: entered as {D.shape[1]} dummies")
        constant = spec["constant"] and not fe_codes
        self.fe_partialled = bool(fe_codes)
        if dummies:
            Xd = np.column_stack(dummies)
        else:
            Xd = np.zeros((len(E), 0))
        if constant or (dummies and not fe_codes and spec["constant"]):
            Xd = np.column_stack([Xd, np.ones(len(E))])
        Xall = np.column_stack([X, Xd])
        if fe_codes:
            M = demean(np.column_stack([y, Xall]), fe_codes, w=None if sw is None else sw ** 2)
            y, Xall = M[:, 0], M[:, 1:]
        if sw is not None:
            y = y * sw; Xall = Xall * sw[:, None]
        self.k_main = X.shape[1]
        # drop columns that became zero / collinear (e.g. dummies collinear with nested FE)
        from .hdfe import _drop_collinear
        _, kn, _ = _drop_collinear(Xall, list(range(Xall.shape[1])))
        keep = list(kn)
        self.col_map = {j: keep.index(j) for j in keep if j < self.k_main}
        self.X = Xall[:, keep]
        self.y = y
        self.names = names
        self.N, self.K = self.X.shape
        self.bc = pd.factorize(bc)[0]
        self.Gb = self.bc.max() + 1
        self.clus = [pd.factorize(c)[0] for c in clus]
        XtX = self.X.T @ self.X
        self.M = np.linalg.pinv(XtX)
        self.b = self.M @ (self.X.T @ self.y)
        self.u = self.y - self.X @ self.b
        # CGM terms: all non-empty subsets of the clustering dimensions, sign (-1)^(|s|+1)
        self.terms = []
        D = len(self.clus)
        for r in range(1, D + 1):
            for sub in itertools.combinations(range(D), r):
                code = self.clus[sub[0]] if r == 1 else pd.factorize(pd.MultiIndex.from_arrays([self.clus[i] for i in sub]))[0]
                # cells = intersections of the term's clusters with the bootstrap clusters (= bootstrap
                # clusters when these are nested; general when bootcluster() is coarser, e.g. one dimension)
                cell, cu = pd.factorize(pd.MultiIndex.from_arrays([code, self.bc]))
                cell_c = np.asarray(cu.get_level_values(0), int); cell_b = np.asarray(cu.get_level_values(1), int)
                Gc = code.max() + 1
                H = np.zeros((Gc, self.K, self.K))
                np.add.at(H, code, np.einsum("ni,nj->nij", self.X, self.X))
                self.terms.append({"sign": (-1) ** (r + 1), "cell": cell, "cell_c": cell_c, "cell_b": cell_b,
                                   "Gc": Gc, "H": H})
        # per bootstrap-cluster X'X sums for score updates are folded into term H matrices
        self.Xb = None

    def _Acell(self, tm, u):
        A = np.zeros((len(tm["cell_c"]), self.K))
        np.add.at(A, tm["cell"], self.X * u[:, None])
        return A

    def _A(self, u):
        A = np.zeros((self.Gb, self.K))
        np.add.at(A, self.bc, self.X * u[:, None])
        return A  # (Gb, K)

    def _stats(self, R, r, u_dgp, center, V):
        """Bootstrap Wald/t statistics for draws V (Gb, B) with DGP residuals u_dgp.
        center: value of R b* under the DGP (r for WCR, R b_hat for WCU)."""
        q = R.shape[0]
        A = self._A(u_dgp)                       # Gb x K
        RM = R @ self.M                          # q x K
        delta = self.M @ (A.T @ V)               # K x B  (b* - b_dgp)
        num = R @ delta                          # q x B
        # cluster scores of bootstrap residuals, projected: RM X_c' u*_c = sum_{b in c} RM A_b v_b - RM H_c delta
        Vs = np.zeros((q, q, V.shape[1]))
        for tm in self.terms:
            Gc = tm["Gc"]
            a = self._Acell(tm, u_dgp) @ RM.T    # cells x q
            Vc = V[tm["cell_b"]]                 # cells x B
            s = np.zeros((Gc, q, V.shape[1]))
            for j in range(q):
                np.add.at(s[:, j, :], tm["cell_c"], a[:, j:j + 1] * Vc)
            HRM = np.einsum("cij,qj->cqi", tm["H"], RM)    # Gc x q x K  (H_c symmetric)
            s -= np.einsum("cqk,kb->cqb", HRM, delta)
            Vs += tm["sign"] * np.einsum("cqb,cpb->qpb", s, s)
        if q == 1:
            se = np.sqrt(np.clip(Vs[0, 0], 1e-300, None))
            return num[0] / se
        W = np.empty(V.shape[1])
        for bi in range(V.shape[1]):
            try:
                W[bi] = num[:, bi] @ np.linalg.solve(Vs[:, :, bi], num[:, bi]) / q
            except np.linalg.LinAlgError:
                W[bi] = np.nan
        return W

    def _observed(self, R, r):
        V1 = np.ones((self.Gb, 1))
        # observed statistic: unrestricted fit, cluster-robust variance with the same (unscaled) formula
        q = R.shape[0]
        A = self._A(self.u)
        RM = R @ self.M
        num = R @ self.b - r
        Vs = np.zeros((q, q))
        for tm in self.terms:
            s = np.zeros((tm["Gc"], q))
            np.add.at(s, tm["cell_c"], self._Acell(tm, self.u) @ RM.T)
            Vs += tm["sign"] * s.T @ s
        if q == 1:
            return float(num[0] / np.sqrt(Vs[0, 0]))
        return float(num @ np.linalg.solve(Vs, num) / q)

    def restricted(self, R, r):
        RM = R @ self.M
        adj = self.M @ R.T @ np.linalg.solve(RM @ R.T, R @ self.b - r)
        bt = self.b - adj
        return bt, self.y - self.X @ bt

    def test(self, R, r, reps=999, weight="rademacher", seed=None, null=True, ptype="symmetric", level=0.95,
             ci=True, chunk=None):
        R = np.atleast_2d(np.asarray(R, float)); r = np.atleast_1d(np.asarray(r, float))
        q = R.shape[0]
        rng = np.random.default_rng(seed)
        enum = False
        if (weight or "rademacher").lower().startswith("r") and 2 ** self.Gb <= reps:
            V = np.array(list(itertools.product([-1.0, 1.0], repeat=self.Gb))).T
            enum = True
        else:
            V = weights_draw(weight, self.Gb, reps, rng)
        B = V.shape[1]
        chunk = chunk or max(1, int(2e7 // max(1, self.Gb * max(t["Gc"] for t in self.terms) // max(1, self.Gb) * self.K * q + self.Gb)))
        stat = self._observed(R, r)

        def boot_stats(rr, Vm):
            if null:
                _, ud = self.restricted(R, rr)
            else:
                ud = self.u
            outs = []
            for st in range(0, Vm.shape[1], chunk):
                outs.append(self._stats(R, rr, ud, None, Vm[:, st:st + chunk]))
            return np.concatenate(outs)

        def pval(rr):
            s_obs = self._observed(R, rr)
            tb = boot_stats(rr, V)
            if not null:  # WCU: bootstrap stats are centered at the estimate already (R delta)
                pass
            if q > 1:
                return float(np.mean(tb >= s_obs)), s_obs, tb
            if ptype == "symmetric":
                return float(np.mean(np.abs(tb) >= abs(s_obs))), s_obs, tb
            if ptype == "equaltail":
                return float(2 * min(np.mean(tb <= s_obs), np.mean(tb >= s_obs))), s_obs, tb
            if ptype == "upper":
                return float(np.mean(tb >= s_obs)), s_obs, tb
            return float(np.mean(tb <= s_obs)), s_obs, tb

        p, s_obs, tb = pval(r)
        # report the statistic on the scale of the estimation command (Stata small-sample factor)
        if self.hetero:
            cfac = self.N / max(self.N - self.K, 1)
        else:
            Gm = min(t["Gc"] for t in self.terms)
            cfac = Gm / max(Gm - 1, 1) * (self.N - 1) / max(self.N - self.K - (1 if self.fe_partialled else 0), 1)
        s_rep = s_obs / np.sqrt(cfac) if q == 1 else s_obs / cfac
        out = {"stat": s_rep, "p": p, "reps": B, "enumerated": enum, "G_boot": int(self.Gb), "q": q,
               "weight": weight, "null_imposed": null, "ptype": ptype}
        if q == 1 and ci:
            # CI by test inversion over the scalar R b = r0
            est = float(R[0] @ self.b)
            se_naive = abs(est - r[0]) / abs(s_obs) if s_obs != 0 else None
            alpha = 1 - level
            if se_naive is None or not np.isfinite(se_naive) or se_naive == 0:
                se_naive = np.sqrt(abs(R[0] @ self.M @ R[0])) * np.std(self.u) or 1.0
            f = lambda r0: pval(np.array([r0]))[0] - alpha
            lo = hi = None
            for side in (-1, 1):
                inner = est
                step = 2 * se_naive
                outer = est + side * step
                k = 0
                while f(outer) > 0 and k < 30:
                    inner = outer; step *= 1.8; outer = est + side * step; k += 1
                if k >= 30:
                    continue
                a_, b_ = inner, outer
                for _ in range(40):
                    mid = (a_ + b_) / 2
                    if f(mid) > 0:
                        a_ = mid
                    else:
                        b_ = mid
                    if abs(b_ - a_) < 1e-6 * max(1.0, abs(se_naive)):
                        break
                if side < 0:
                    lo = (a_ + b_) / 2
                else:
                    hi = (a_ + b_) / 2
            out["ci"] = (lo, hi)
            out["level"] = level
        return out
