# Decision rules

Each section: **detect → decide → treat → interpret → pitfalls**. α = 0.05 unless noted.
Textbook pointers: Chen Q. (2023) *Econometrics and Stata Applications*, 2nd ed. (CQ), and Wooldridge, *Introductory Econometrics* (W).

## heteroskedasticity
- **Detect**: `diagnose` reports BP/Koenker on fitted values and on all regressors, and the White test. Look at the residual-vs-fitted plot: a funnel shape points to a variance tied to size.
- **Decide**: a rejection by any test → heteroskedasticity. In cross-section data, assume it is present anyway.
- **Treat**:
  1. Main model: same OLS coefficients with robust SE (`--cov robust`, HC1; use HC3 if n < 100).
  2. Efficiency comparison: FGLS (`--model fgls --het-vars Z`). Choose Z = the variables that drive the variance (often size, e.g. output). Use `--het-noconst` only when theory says Var ∝ exp(Z·γ) without a constant.
  3. If FGLS and OLS-robust differ a lot, the mean model may be misspecified. Check RESET.
- **Interpret**: heteroskedasticity does not bias coefficients. It only invalidates the classic SE.
- **Pitfall**: do not report classic SE in cross-section. WLS with robust SE is legitimate.
- Ref: CQ ch.7; W ch.8.

## autocorrelation
- **Detect** (time series, `--time` given): Durbin-Watson (≈2 means none; well below 2 means positive serial correlation), Breusch-Godfrey at lag 1 and at a higher lag, Ljung-Box Q, the residual ACF plot, and the lag-1 residual autocorrelation.
- **Decide**: BG p < 0.05 at any lag → serial correlation. DW alone is unreliable when there are lagged y regressors.
- **Treat**, in order:
  1. **Dynamic misspecification first.** Serial correlation often means a missing lag of x or y. Try `--x x L.x` (and/or `L.y`), then re-run diagnose. If BG no longer rejects, this is usually the best model.
  2. Keep OLS with Newey-West HAC SE (`--cov hac`). The default lag rule is floor(4(T/100)^(2/9)); report results at 2 lag choices as a robustness check.
  3. FGLS for AR(1) errors: Prais-Winsten (`--model prais`, keeps the first observation) or Cochrane-Orcutt (`--model corc`). This is only valid with strictly exogenous regressors.
- **Interpret**: with static regressors, OLS remains consistent and HAC fixes inference. With a lagged y plus serially correlated errors, OLS is inconsistent: fix the dynamics.
- Ref: CQ ch.8; W ch.12.

## nonstationarity
- **Detect**: `diagnose` with `--time` runs ADF (H0: unit root) and KPSS (H0: stationary) on y and each regressor. Verdict `unit_root` = ADF does not reject and KPSS rejects. When y has a unit root, it also runs an Engle-Granger cointegration test.
- **Decide / treat**:
  - y and x both I(1), cointegrated → the levels regression is a long-run relation. Use HAC SE; an ECM captures short-run dynamics.
  - I(1), not cointegrated → spurious regression risk. Use differences (`D.y`, `D.x`) or growth rates.
  - T < 50 → the tests have low power (`possible_nonstationarity`). Do not over-react; add a differenced specification as a robustness check.
- Ref: CQ ch.13–14; W ch.18.

## functional_form (specification)
- **Detect**: Ramsey RESET p < 0.05. Also look for curvature in the residual plots.
- **Treat**: add theory-motivated nonlinearity, such as a square of the key x (diminishing returns), logs of money/size variables, or interactions (heterogeneous effects). Re-run diagnose. Report the marginal effect at meaningful values (e.g. at the quartiles of x).
- **Pitfall**: RESET rejection can also mean omitted variables. Say which one you believe and why.
- Ref: CQ ch.9; W ch.9.

## multicollinearity
- **Detect**: VIF > 10. Look for large SEs together with a high R² and jointly significant groups.
- **Treat**: usually nothing. Do not drop theoretically required controls. Report a joint F-test for the collinear group, or combine variables only if theory allows it (e.g. impose a restriction such as homogeneity).
- Ref: CQ ch.9.

## outliers
- **Detect**: number of observations with Cook's D > 4/n, and the top rows.
- **Treat**: inspect the rows (data errors vs genuine extremes). Re-estimate without them **only as a robustness check**, and say how many rows you dropped and why.

## endogeneity
- **Sources to argue explicitly**: an omitted variable correlated with x (e.g. ability in a wage equation), simultaneity/reverse causality (price and quantity; policy and outcome), measurement error in x (attenuation bias), and sample selection.
- **When an instrument Z is available**, it must be:
  1. **Relevant**: first-stage F ≥ 10 (Staiger-Stock rule; `weak_iv_flag`). Report the partial R² as well.
  2. **Exogenous** (exclusion restriction): Z affects y only through x. This **cannot be tested** with exactly one instrument per endogenous variable. Argue it in words, and name the threat to it.
  3. **Over-identified** (more instruments than endogenous variables): Sargan / Wooldridge score test (H0: all instruments valid). A non-rejection is necessary, not sufficient.
- **Estimators**: 2SLS (main), LIML (robust to weak instruments; should be close to 2SLS when instruments are strong), GMM (efficient under heteroskedasticity with over-id).
- **Endogeneity test**: Durbin / Wu-Hausman, or the robust Wooldridge regression test (`wooldridge_regression_endog`). Rejection → prefer IV. Non-rejection → OLS is acceptable and more efficient, but report both.
- **Proxy variables**: a proxy for an omitted factor (e.g. IQ for ability) reduces omitted-variable bias. If the proxy is measured with error, instrument it with other indicators (e.g. another test score or family background).
- **Weak-ID statistic to quote**: with robust/cluster SE use the Kleibergen-Paap rk Wald F (`kp_rk_Wald_F`); with iid SE the Cragg-Donald F; with several endogenous regressors check every Sanderson-Windmeijer F (`SW_F`), not just the joint first-stage F. Compare with Stock-Yogo (10% maximal IV size: 16.38 for 1 endog/1 IV, 19.93 for 1/2) — the "F > 10" rule is a floor. Under weak IV report the Anderson-Rubin test (`AR_F`, `AR_p`), which is robust to weak instruments.
- **Red flags that need a sentence each**: (a) IV ≫ OLS (e.g. 2–3×): measurement error in x, LATE for compliers, or a violated exclusion restriction — say which you believe; (b) family-background, geography or "industry/region average" instruments: they may affect y directly (exclusion threat — name the channel); (c) instruments measured after the treatment (post-treatment) or lagged x when x is persistent and errors are serially correlated: not excludable; (d) an over-id test that "passes" with instruments sharing the same threat is not reassuring.
- **Proxy + IV**: when a proxy (IQ for ability) is itself an imperfect measure, include it and instrument it with a second indicator (e.g. KWW score) — treat the proxy as endogenous, not the variable of interest only.
- **Headline must match the model**: if the final model is IV, the headline number is the IV estimate (with its CI); OLS appears only as the comparison.
- **Interpret**: IV estimates a LATE when effects are heterogeneous. IV SEs are larger; a smaller t-stat is expected.
- **Command**: `fit --model 2sls --x <all regressors incl. endogenous> --endog E --instruments Z1 Z2 --cov robust`.
- Ref: CQ ch.10; W ch.15.

## panel
- **Treat**: estimate pooled, FE (`--model fe`), RE (`--model re`), and two-way FE (`--model twfe`) when common shocks matter. The default SE is clustered by entity.
- **Choose**: the FE poolability F (H0: no entity effects) and Hausman FE vs RE (H0: RE consistent). Rejection → FE. With clustered SE, the classic Hausman test is not valid; mention a Mundlak test as an alternative.
- **Pitfalls**: FE cannot identify effects of time-invariant regressors. In two-way FE DiD with staggered timing, flag possible bias from heterogeneous effects.
- Ref: CQ ch.12; W ch.13–14.

## binary
- y ∈ {0,1} → logit/probit. Report **average marginal effects** (`marginal_effects`), not raw coefficients, and the % correctly predicted. A linear probability model with robust SE can serve as a robustness check.
- Ref: CQ ch.11.

## causal_claim
- `credible`: a design whose identifying assumption you can state and defend (randomization, IV with strong + plausibly excludable instruments, DID with parallel pre-trends shown, RDD), plus robustness. Even then, state the population (LATE, treated firms).
- `suggestive`: FE/controls remove obvious confounders, or an IV/DID whose key assumption is arguable; results robust across specifications.
- `none`: cross-sectional OLS with possible omitted variables or reverse causality. Write "is associated with".
Identification beats efficiency: prefer the consistent estimator even with larger SEs.

## reviewing a paper's identification (replication mode)
- **TWFE DID with staggered adoption**: heterogeneous effects can bias the TWFE coefficient (even its sign). Check the event-study pre-trends (`F.x`/leads ≈ 0); mention Callaway-Sant'Anna / Sun-Abraham / de Chaisemartin-D'Haultfœuille as the fix.
- **Bartik / shift-share instruments**: exogeneity comes from the shares or the shocks — say which the paper relies on.
- **Lagged x as "endogeneity test"**: mitigates simple reverse causality only; it does not remove omitted persistent factors.
- **Mechanism (mediation) tables**: X→M and Y on X+M is descriptive; M is post-treatment, so the "indirect effect" is not causally identified without extra assumptions. Sobel tests inherit this problem.
- **PSM**: balances observables only; check the balance table and common support; matched N should be reported.
- **Heterogeneity by subgroup**: compare groups with an interaction term (or a Chow / SUR test), not by eyeballing two separate significance stars.
- **Multiple robustness tables all supporting H1** do not address the same untested assumption — identify the weakest link.

## Choosing and reporting the final model
1. The final model is the simplest specification that passes the relevant diagnostics, with valid SE for the remaining issues.
2. The comparison table always includes the naive OLS, so readers can see what the treatment changed.
3. If two defensible models disagree on sign or significance, say so. Do not pick one silently.
