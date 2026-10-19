# Eval results — v0.3 (advanced estimators)

`python3 evals/v03/bench.py` runs `evals/v03/bench.do` through `ek stata` and checks **48 numbers**
against published output or reference implementations — **48/48 pass**. Data: `abdata.csv`
(Arellano-Bond), `mpdta.csv` (Callaway-Sant'Anna minimum-wage panel), `darfur.csv` (Cinelli-Hazlett),
`wb.csv` (simulated, 12 clusters); the IV case is simulated in the script.

| method | command | reference | result |
|---|---|---|---|
| Difference GMM | `xtabond` one-step / two-step WC / `pre()` / `endogenous()` / lag limits / constant | Stata [XT] xtabond manual examples | coefs & SEs exact to 1e-6 (L.n .6862261 (.1486163); two-step .6287089 (.1934138), Wald 1104.72; Sargan 65.818, AR(1) −3.9394, AR(2) −.54239; `pre()` 83 instruments .8580958 (.1265515)); `_cons` with constant within 1e-4 |
| System GMM | `xtabond2`, `xtdpdsys` | Roodman (2009) examples; pydynpd | N 891, 113 instruments, L.n .9356053 (.026569), F(12,139) 1154.35; diff GMM F(16,140) 85.29; FOD + collapse two-step exact (Hansen 72.882) |
| Callaway-Sant'Anna | `csdid` (dripw / reg / stdipw, never / not-yet, 4 aggregations) | R `did` / Python `differences` on mpdta | all ATT(g,t), SEs and aggregations to 4 dp (simple −0.0400 (0.0120), group −0.0310 (0.0124)) |
| Sun-Abraham | `eventstudyinteract` | identity with CS post effects; SE formula of eventstudyinteract.ado | L0..L3 −0.0199 −0.0510 −0.1373 −0.1008 |
| de Chaisemartin-D'Haultfœuille | `did_multiplegt_dyn`, `did_multiplegt` | identity with CS not-yet event effects; variance of did_multiplegt_dyn.ado | Effect_1..4 and SEs to 4 dp; DID_M = DID_1 |
| Wild cluster bootstrap | `boottest` (WCR/WCU, rademacher/webb/mammen/normal, joint tests, multiway, FE) | `wildboottest` (Python port of boottest); brute-force enumeration | t 2.1551; Webb p 0.159 vs 0.153 (Monte Carlo); Rademacher enumeration 0.1602 vs 0.1597; two-way cluster with `bootcluster(g)` p = brute force exactly |
| Sensitivity to omitted variables | `sensemakr` | Cinelli & Hazlett (2020) Darfur | RV 13.88 %, RV_α 7.63 %, R²_YD 2.19 %, 1×/2×/3× female adjusted 0.0752 / 0.0529 / 0.0304 |
| IV, multiway clustering | `ivreg2/ivreghdfe ..., cluster(a b)` | `linearmodels` IV2SLS two-way clusters | SEs exact after ivreg2 scaling (G = min); KP rk Wald F 11.2714 exact |

Known limits: `csdid` panel data only, no `wboot` uniform bands; `did_multiplegt_dyn` without
`controls()` / `trends_*`; `boottest` after OLS-type estimators only (no IV / WRE); `xtdpd` unsupported.
