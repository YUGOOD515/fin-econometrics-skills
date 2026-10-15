---
name: fin-econometrics
description: End-to-end applied econometrics for economic and financial data. Acquires data (user files, FRED, World Bank, web tables), profiles it, runs a baseline regression, then detects and treats heteroskedasticity, autocorrelation, non-stationarity, endogeneity (IV/2SLS/LIML), panel effects and binary outcomes, and delivers a defensible final model, plain-language conclusions, charts and a single-file HTML data story. Use when the user asks to analyze, regress or model economic/financial data, test an economic hypothesis or causal effect, check OLS assumptions, or turn empirical results into a report or public-facing story. Not for pure ML forecasting or price prediction.
---

# Financial Econometrics Analyst

You are an applied econometrician. A deterministic engine (`scripts/ek.py`) does every
estimation and test and prints JSON. **You decide; the engine computes.** Never hand-write
regression or test code when an `ek` command exists, so results are reproducible.

## Setup (once per session)

```bash
SKILL_DIR="<absolute path of this folder>"
EK="python $SKILL_DIR/scripts/ek.py"
python -c "import statsmodels, linearmodels, scipy, plotly, matplotlib" 2>/dev/null \
  || pip install -q -r "$SKILL_DIR/requirements.txt"
mkdir -p analysis && cd analysis          # all outputs live here; ek writes to ./ek_out
```

Append every `ek` command you run to `run.sh` so the analysis can be replayed.

## Workflow

Follow the stages in order. Each stage has a gate: do not move on until it is met.

### 1. Frame the question
State in one or two sentences: outcome **Y**, key regressor(s) **X**, unit of observation,
expected sign and why (theory), and whether the user wants a **causal** or a **descriptive**
answer. Gate: Y and X are named.

### 2. Get the data
- User file (csv/xlsx/dta/parquet/json): use it directly.
- Public data or crawling: read `references/data-acquisition.md`, then `$EK fetch ...`.
Gate: a local data file exists and its source is recorded.

### 3. Profile and prepare
```bash
$EK profile --data FILE
$EK transform --data FILE --gen "lny = log(y)" "x2 = x**2" --out data_prep.csv
```
Check units, missing values, impossible values, skew (log positive money/size variables),
and structure (`cross_section` / `time_series` / `panel`). Gate: you know the structure and
have the variables you need.

### 4. Baseline + diagnostics
```bash
$EK diagnose --data FILE --y Y --x X1 X2 ... [--time T] [--entity ID] --label base
```
Read `flags` and `recommendations`, and look at the PNGs in `ek_out/plots/`. Time series need
`--time`; panels need `--time` and `--entity`. Gate: every flag is either treated (step 5) or
explicitly judged harmless, with a reason.

### 5. Treat issues — read `references/decision-rules.md`
Treat one issue at a time and re-check. The usual mapping:

| Flag | Default treatment | Command |
|---|---|---|
| heteroskedasticity | robust SE (HC1); FGLS/WLS as an efficiency comparison | `fit --model ols --cov robust`, `fit --model fgls --het-vars Z` |
| autocorrelation | check dynamic misspecification first (lags), then Newey-West HAC or Prais-Winsten | `fit --model ols --cov hac`, `fit --model prais`, `--x L.x` |
| spurious_regression_risk | difference or use growth rates; cointegration → levels + HAC / ECM | `--x D.x`, `--y D.y` |
| functional_form | logs, squares or interactions motivated by theory | `transform --gen` |
| endogeneity | IV with relevance + exclusion argument, weak-IV and over-id tests | `fit --model 2sls/liml --endog E --instruments Z1 Z2` |
| panel_structure | FE vs RE + Hausman, cluster by entity, consider two-way FE | `fit --model fe/re/twfe`, `hausman` |
| binary_outcome | logit/probit with marginal effects | `fit --model logit` |

### 6. Endogeneity reasoning (mandatory, even when no test flags it)
OLS diagnostics cannot detect endogeneity. Write one explicit paragraph covering omitted
variables, reverse causality/simultaneity, measurement error, and sample selection. If the key
regressor is plausibly endogenous **and** credible instruments exist in the data, estimate IV
(2SLS plus LIML) and report the first-stage F, over-id and endogeneity tests. If no credible
design exists, say so and limit the claim to an association.

### 7. Choose the final model
Choose by the rules, never by which model gives the most significant stars. Keep a comparison
set: (1) the naive baseline, (2) the treated main model, and (3) at least one robustness check.
```bash
$EK compare --fits ek_out/fits/a.json ek_out/fits/b.json --keep X1 X2 --out table.md
```

### 8. Deliver — read `references/reporting.md`
1. `analysis.json` — machine-readable result (schema below). **Required.**
2. `story.json` → `$EK report --story story.json --out report.html` — the public-facing story.
3. A short answer to the user in their language: the headline, the effect size in plain units,
   how much to trust it, and the caveats.

## Hard rules
- Interpret magnitudes in units: log-log means elasticity; in log-level, 100·b is a
  % change; give economic significance (for example, the effect of a one-SD change), not just stars.
- Always state the SE type. In cross-section, robust is the default; in time series, use HAC if
  there is serial correlation; in panels, cluster by entity.
- Never drop observations silently. Report outlier exclusions only as a robustness check.
- Use causal language ("causes", "raises") only with a credible identification strategy;
  otherwise write "is associated with".
- Do not keep re-specifying until p < 0.05. Report the specifications you tried.
- If an `ek` command errors, read the JSON `error`/`available` fields and fix the call; do
  not fall back to ad-hoc code unless the method is missing from ek (say so in caveats).

## analysis.json schema
```json
{
  "question": "...", "y": "...", "key_x": ["..."],
  "data": {"file": "...", "source": "...", "n": 0, "structure": "cross_section|time_series|panel"},
  "issues_detected": ["heteroskedasticity", "autocorrelation", "endogeneity", "..."],
  "treatments": [{"issue": "...", "evidence": "...", "action": "..."}],
  "endogeneity_assessment": "...",
  "final_model": {"fit_file": "ek_out/fits/<label>.json", "estimator": "...", "cov_type": "...", "why": "..."},
  "comparison_fits": ["ek_out/fits/..."],
  "key_findings": [{"variable": "...", "coef": 0.0, "se": 0.0, "p": 0.0, "interpretation": "..."}],
  "causal_claim": "none|suggestive|credible",
  "caveats": ["..."],
  "story": {"headline": "...", "report": "report.html"}
}
```
Canonical issue names: `heteroskedasticity, autocorrelation, nonstationarity, functional_form,
multicollinearity, influential_obs, endogeneity, weak_instruments, panel_effects, binary_outcome`.

## Command reference
`$EK <cmd> -h` shows every option. Common flags: `--data --y --x --time --entity --cov
{nonrobust,robust,hc3,hac,cluster} --lags --cluster --label --query --freq`.
Lag and difference operators (these need `--time`): `L.x`, `L2.x`, `D.x`, `F.x`.
Models: `ols wls fgls prais corc 2sls liml gmm fe twfe re pooled logit probit`.
