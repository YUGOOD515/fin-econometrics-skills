# fin-econometrics-skills

A portable **Agent Skill** for applied (financial) econometrics and **replication of empirical
papers**. It works in any harness that loads `SKILL.md` folders: Claude Code, Codex, Cursor,
WorkBuddy and others.

Two modes:
- **Replicate a paper** — paper (+ data, + optional do-file) → extract the design and table numbers →
  run the do-file (or a do-file written from the paper) **without Stata** → `ek check` against the
  paper's numbers → explain deviations → judge identification and whether the conclusions hold →
  Chinese replication report.
- **Independent analysis** — data → profile → baseline OLS → diagnose → treat → choose model →
  conclusions → HTML data story.

## Engine (`scripts/ek.py`, every command prints JSON)
- `ek stata` — a Stata do-file interpreter in Python that follows Stata's conventions (dof,
  small-sample scaling, singleton dropping, reghdfe FE-nesting dof, ivreg2 weak-ID statistics).
  Data steps: `use/merge/append/save/preserve/restore, gen/egen/replace/drop/keep, by/bys, macros,
  foreach/forvalues, winsor2, encode/destring, tab gen()`, factor variables and `L./F./D.` operators.
  Estimators: `reg areg reghdfe xtreg(fe/re) ivreghdfe ivreg2 ivregress xtivreg probit logit poisson
  nbreg ppmlhdfe newey qreg tobit ologit oprobit mlogit clogit xtlogit(fe) psmatch2 sgmediation2`,
  plus `test testparm lincom nlcom predict est store/eststo`. IV output includes first-stage F,
  Sanderson-Windmeijer F, Cragg-Donald, Kleibergen-Paap LM/Wald F, Anderson-Rubin, Hansen J.
- `ek check` — compares estimates with the paper's printed numbers (keeps the paper's rounding and
  stars) and returns a verdict: replicated / largely / partially / not replicated, plus
  sign-and-significance agreement.
- `ek profile/transform/diagnose/fit/compare/hausman/report/fetch` — the independent-analysis
  pipeline: OLS, robust/HAC/cluster SE, WLS/FGLS, Prais-Winsten/Cochrane-Orcutt, 2SLS/LIML/GMM,
  panel FE/RE/TWFE with Hausman, logit/probit with marginal effects, ADF/KPSS/Engle-Granger, RESET,
  VIF, Cook's D; FRED/World Bank/web-table fetchers; single-file HTML report (zh/en labels).

Validated against six published Chinese empirical papers (中国工业经济 / 世界经济) with the authors'
data and code: all checked coefficients, t/SE, N, R² and IV statistics match the published tables
or Stata logs at the printed precision (see `evals/RESULTS-v0.2.md`).

## Install
```bash
git clone <repo> && cd fin-econometrics-skills
pip install -r skills/fin-econometrics/requirements.txt
./install.sh            # copies the skill to ~/.claude/skills and ~/.agents/skills (Codex)
```
For other harnesses, copy `skills/fin-econometrics/` into the folder where that harness looks for skills.

## Evals
`evals/cases/` holds blind tasks at three difficulty levels for each mode.

| case | mode | difficulty | core issue |
|---|---|---|---|
| easy_nerlove | analysis | easy | heteroskedasticity, returns to scale |
| medium_icecream | analysis | medium | serial correlation in a short time series |
| hard_grilic | analysis | hard | ability bias / endogeneity → IV |
| easy_lixiaoping | replication | easy | paper + data + do-file: reghdfe & ivreg2 tables, weak/odd IVs |
| medium_litao | replication | medium | paper + data, no code: staggered DID, clustering levels, singletons |
| hard_overtime | replication | hard | paper + data, no code: areg + xtivreg fe, weak IV, IV ≫ OLS |

`python evals/grade.py evals/cases/<case> <run_dir>` scores a run. Replication cases need the
papers' public data packages (not redistributed here; see each case's `expected.json` → `source`).
