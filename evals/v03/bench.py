"""econkit v0.3 benchmark. Run:  python3 evals/v03/bench.py   (from the repo root)
Reference values: Stata manual [XT] xtabond; Roodman (2009, Stata Journal) xtabond2; R `did` / Python `differences`
(mpdta); eventstudyinteract.ado / did_multiplegt_dyn.ado formulas; wildboottest (Python port of boottest);
Cinelli & Hazlett (2020, JRSS-B) Darfur example."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "skills", "fin-econometrics", "scripts"))
from econkit import stata

os.chdir(HERE)
o, _ = stata.run(do="bench.do", out_dir="/tmp/econkit_v03")
assert o["ok"], o["errors"]
E = {e["id"]: e for e in json.load(open("/tmp/econkit_v03/stata_estimates.json"))}
fails, n = [], 0


def chk(name, got, ref, tol):
    global n
    n += 1
    ok = abs(got - ref) <= tol
    print(f"{'OK ' if ok else 'FAIL'} {name:<52} got {got:>11.6g}  ref {ref:>11.6g}")
    if not ok:
        fails.append(name)


c = lambda i, v, k="coef": E[i]["coefs"][v][k]
# [1] xtabond one-step (Stata manual)
chk("xtabond 1-step L.n", c(1, "L.n"), .6862261, 1e-6); chk("xtabond 1-step se(L.n)", c(1, "L.n", "se"), .1486163, 1e-6)
chk("xtabond Sargan", E[1]["sargan"]["chi2"], 65.81806, 1e-3)
chk("xtabond AR(1) z", E[1]["ar_tests"][0]["z"], -3.9394, 1e-3); chk("xtabond AR(2) z", E[1]["ar_tests"][1]["z"], -.54239, 1e-4)
# [2] two-step WC
chk("xtabond 2-step L.n", c(2, "L.n"), .6287089, 1e-6); chk("xtabond 2-step WC se(L.n)", c(2, "L.n", "se"), .1934138, 1e-6)
chk("xtabond 2-step Wald chi2", E[2]["wald"]["chi2"], 1104.72, .01)
# [3] pre()
chk("xtabond pre() #instruments", E[3]["n_instruments"], 83, 0); chk("xtabond pre() L.n", c(3, "L.n"), .8580958, 1e-6)
chk("xtabond pre() se(L.n)", c(3, "L.n", "se"), .1265515, 1e-6)
# [4] Roodman 2009 system GMM
chk("xtabond2 system N", E[4]["N"], 891, 0); chk("xtabond2 system #instruments", E[4]["n_instruments"], 113, 0)
chk("xtabond2 system L.n", c(4, "L.n"), .9356053, 1e-6); chk("xtabond2 system se(L.n)", c(4, "L.n", "se"), .026569, 1e-5)
chk("xtabond2 system F(12,139)", E[4]["wald"]["F"], 1154.35, .05)
# [5] CS never-treated, no covariates
ag = {k: {r["name"]: r for r in v} for k, v in E[5]["aggregations"].items()}
chk("csdid simple ATT", ag["simple"]["ATT"]["att"], -0.0400, 1e-4); chk("csdid simple se", ag["simple"]["ATT"]["se"], 0.0120, 1e-4)
chk("csdid group overall", ag["group"]["GAverage"]["att"], -0.0310, 1e-4)
chk("csdid group overall se", ag["group"]["GAverage"]["se"], 0.0124, 1e-4)
chk("csdid ATT(2004,2006)", c(5, "ATT(2004,2006)"), -0.1373, 1e-4)
# [7] Sun-Abraham post effects (= CS event-time effects under never-treated control)
for v, r in zip(["L0", "L1", "L2", "L3"], [-0.0199, -0.0510, -0.1373, -0.1008]):
    chk(f"eventstudyinteract {v}", c(7, v), r, 1e-4)
# [8] dCDH DID_l (= CS not-yet event effects for staggered absorbing designs)
for l, (r, s) in enumerate([(-0.0189, 0.0121), (-0.0536, 0.0170), (-0.1363, 0.0362), (-0.1008, 0.0351)], 1):
    chk(f"did_multiplegt_dyn Effect_{l}", c(8, f"Effect_{l}"), r, 1e-4); chk(f"did_multiplegt_dyn se(Effect_{l})", c(8, f"Effect_{l}", "se"), s, 1e-4)
# [9] wild cluster bootstrap (wildboottest reference)
T = E[9]["tests"]
chk("boottest t", T[0]["t"], 2.1551, 1e-4)
chk("boottest webb WCR p (Monte Carlo)", T[0]["p"], 0.1533, 0.012)
chk("boottest rademacher p (full enumeration)", T[1]["p"], 0.1597, 0.002)
# [10] sensemakr Darfur
s = E[10]["sensitivity"]; b = {x["kd"]: x for x in s["bounds"]}
chk("sensemakr estimate", c(10, "directlyharmed"), 0.0973, 1e-4); chk("sensemakr R2yd|x", s["r2yd_x"], 0.0219, 1e-4)
chk("sensemakr RV_q", s["rv_q"], 0.1388, 1e-4); chk("sensemakr RV_q,a", s["rv_qa"], 0.0763, 1e-4)
for kd, r in [(1, 0.0752), (2, 0.0529), (3, 0.0304)]:
    chk(f"sensemakr {kd}x female adjusted estimate", b[kd]["adjusted_estimate"], r, 1e-4)
# [11] two-way cluster + bootcluster(g): brute-force Rademacher enumeration (refit + CGM VCE per draw)
chk("boottest 2-way cluster, bootcluster(g) p", E[11]["tests"][0]["p"], 0.1162109375, 1e-6)
# [12] IV with two-way clustering (reference: linearmodels IV2SLS clusters=[f,t], scaled by ivreg2 cluster factor Gmin/(Gmin-1)*(N-1)/(N-K))
import numpy as np, pandas as pd
rng = np.random.default_rng(3); N = 2000
f = rng.integers(0, 60, N); t = rng.integers(0, 25, N)
af = rng.normal(size=60)[f]; at = rng.normal(size=25)[t]
z1 = rng.normal(size=N) + 0.5 * rng.normal(size=60)[f]; z2 = rng.normal(size=N) + 0.5 * rng.normal(size=25)[t]
u = af + at + rng.normal(size=N); w = rng.normal(size=N)
x = 0.3 * z1 + 0.2 * z2 + 0.5 * u + rng.normal(size=N) + 0.3 * af
pd.DataFrame(dict(y=1 + 0.5 * x + 0.3 * w + u, x=x, w=w, z1=z1, z2=z2, f=f, t=t)).to_csv("/tmp/econkit_iv2.csv", index=False)
open("/tmp/econkit_iv2.do", "w").write("import delimited /tmp/econkit_iv2.csv, clear\nivreg2 y (x = z1 z2) w, cluster(f t)\n")
o, _ = stata.run(do="/tmp/econkit_iv2.do", out_dir="/tmp/econkit_v03_iv")
e = json.load(open("/tmp/econkit_v03_iv/stata_estimates.json"))[0]
chk("ivreg2 2-way cluster coef x", e["coefs"]["x"]["coef"], 0.436992, 1e-6)
chk("ivreg2 2-way cluster se(x)", e["coefs"]["x"]["se"], 0.314554 * (25 / 24 * 1999 / 1998) ** .5, 1e-5)
chk("ivreg2 2-way cluster se(w)", e["coefs"]["w"]["se"], 0.037370 * (25 / 24 * 1999 / 1998) ** .5, 1e-5)
chk("ivreg2 2-way KP rk Wald F", e["iv"]["kp_rk_Wald_F"], 11.271436, 1e-4)
print(f"\n{n - len(fails)}/{n} checks passed" + (f"; FAILED: {fails}" if fails else ""))
sys.exit(1 if fails else 0)
