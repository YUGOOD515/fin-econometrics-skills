# Advanced estimators in `ek stata` (v0.3)

All run through `$EK stata --do file.do` (or `--cmd`) with Stata syntax; results go to
`stata_results.md` / `stata_estimates.json` like any estimate (`est store`, `ek check`, `ek compare`
work). Validated numbers: `evals/v03/bench.py` (48 checks, see `evals/RESULTS-v0.3.md`).

## 1. Dynamic panel GMM — `xtabond`, `xtdpdsys`, `xtabond2`
```stata
xtset id year
xtabond  n l(0/1).w l(0/2).(k ys) yr1980-yr1984, lags(2) twostep vce(robust)      // Arellano-Bond
xtabond  n l(0/1).ys yr*, lags(2) twostep pre(w, lag(1,.)) endogenous(k, lag(2,3)) vce(robust)
xtdpdsys n L(0/1).(w k) yr*, lags(2) twostep vce(robust)                           // Blundell-Bond
xtabond2 n L.n L(0/1).(w k) yr*, gmm(L.(n w k), lag(1 3) collapse) iv(yr*, eq(level)) twostep robust small
```
`xtabond2` options: `gmm(vars, laglimits(a b) collapse eq(diff|level|both) passthru)`,
`iv(vars, eq() passthru mz)`, `twostep robust small noleveleq orthogonal nocons h(1|2|3) artests(#)`.
Output: Windmeijer-corrected SEs (two-step robust), AR(1)…AR(k) tests, Sargan, Hansen, difference-in-
Hansen per instrument group, instrument count and a list of instruments. `xtdpd` is not supported.

How to read it (and what to write in the report):
- **AR(2)** in first differences must not reject (p > 0.1); AR(1) rejecting is expected.
- **Instrument proliferation**: instruments ≥ groups (`n_instruments` vs `n_groups`) overfits the
  endogenous variables and makes **Hansen p ≈ 1.000** meaningless. Rule of thumb: instruments < groups;
  re-run with `collapse` and/or `lag(2 3)` and show the coefficient is stable. Hansen p between
  ~0.1 and ~0.9 is the comfortable range. Sargan is not robust; Hansen is robust but weakened by many instruments.
- **Bounds check**: the coefficient on L.y should lie between FE (biased down) and pooled OLS (biased up).
- System GMM needs the extra stationarity / mean-stationarity assumption; check difference-in-Hansen
  for the level-equation instruments.
- Report one-step robust or two-step + Windmeijer; never two-step SEs without the correction.

## 2. Heterogeneity-robust DID — `csdid`, `eventstudyinteract`, `did_multiplegt(_dyn)`
```stata
* gvar = first treatment year, 0 = never treated (panel data)
csdid y [x1 x2], ivar(id) time(year) gvar(first) [notyet] [method(dripw|reg|stdipw)] [agg(simple|group|calendar|event)]
* Sun-Abraham: relative-time dummies (omit -1), cohort var, control cohort indicator
eventstudyinteract y F4 F3 F2 L0 L1 L2 L3, cohort(first) control_cohort(never) absorb(i.id i.year) vce(cluster id)
* de Chaisemartin-D'Haultfoeuille (also handles treatment switching on/off)
did_multiplegt_dyn y id year D, effects(4) placebo(3) [cluster(c)]
did_multiplegt y id year D, breps(100)        // old DID_M
```
Choice: absorbing staggered binary treatment → `csdid` (default never-treated; `notyet` when few
never-treated); Sun-Abraham gives the same post-period effects with a regression interface;
non-absorbing / switching treatment → `did_multiplegt_dyn`. All aggregations carry analytic
influence-function SEs clustered by unit; csdid prints the pre/post averages for event aggregation.
Read: (1) pre-period effects ≈ 0 (joint placebo test printed); (2) compare with TWFE — if TWFE is
much larger/smaller, the staggered-timing bias is material and the paper's TWFE number is not the ATT;
(3) say which cohorts drive the effect (`agg(group)`).
Limits: panel data only (no repeated cross-sections), no `wboot` uniform bands; `did_multiplegt_dyn`
without `controls()`/`trends_*` options.

## 3. Few clusters — `boottest` (wild cluster bootstrap)
```stata
reg y x z, cluster(state)          // or reghdfe ..., absorb(...) cluster(state)
boottest x, reps(9999) weight(webb) seed(1)        // WCR (null imposed), default
boottest x = 0.2, nonull                           // WCU
boottest (x) (z)                                   // joint test
boottest x, cluster(state year) bootcluster(state) // multiway
```
Use when clusters < ~40 or very unbalanced. With G ≤ 13 use `weight(webb)` (Rademacher has only
2^G distinct draws; ek enumerates them when 2^G ≤ reps). Report the bootstrap p and CI next to the
conventional ones; a cluster-robust t that "passes" while the bootstrap p > 0.1 means the stars come
from too few clusters. OLS-type estimators only (no IV / WRE).

## 4. Omitted-variable sensitivity — `sensemakr`
```stata
sensemakr y d x1 x2 i.region, treat(d) benchmark(x1) kd(1 2 3)
sensemakr y d x1 x2 x3, treat(d) gbenchmark(x2 x3) gname(occupation)
```
Reports partial R² of the treatment with the outcome, the robustness value RV (confounding that would
drive the estimate to 0) and RV_α (to lose significance), and adjusted estimates/CI if a confounder
were k times as strong as the benchmark. Write: "a confounder explaining RV% of the residual variance
of both D and Y would eliminate the effect; that is k times stronger than <benchmark>." Pick a
benchmark the reader believes is among the strongest observed confounders.

## 5. IV with multiway clustering
`ivreg2/ivreghdfe y (x = z) w, cluster(firm year)` — Cameron-Gelbach-Miller VCE, first stage,
KP rk LM / Wald F, AR and Hansen J all multiway; small-sample factor uses G = min(G_firm, G_year).
