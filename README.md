# fin-econometrics-skills

A portable **Agent Skill** for applied (financial) econometrics. It works in any harness that loads
`SKILL.md` folders: Claude Code, Codex, Cursor, and others.

Pipeline: **data → profile → baseline OLS → diagnose → treat → choose model → conclusions → HTML data story**

- Deterministic engine `scripts/ek.py` (statsmodels + linearmodels). Every command prints JSON, so
  the agent makes the decisions and the engine does the computing.
- Covers: OLS, robust/HAC/cluster SE, WLS/FGLS, Prais-Winsten/Cochrane-Orcutt, 2SLS/LIML/GMM with
  first-stage, over-id and endogeneity tests, panel FE/RE/two-way FE with Hausman, logit/probit with
  marginal effects, ADF/KPSS/Engle-Granger, RESET, VIF, Cook's D.
- Data: local files (csv/xlsx/dta/parquet/json), FRED, World Bank, URLs, HTML tables, plus a
  GitHub Actions template for scheduled crawling.
- Output: `analysis.json` (machine-readable contract) and a single-file `report.html` story.

## Install

```bash
git clone <repo> && cd fin-econometrics-skills
pip install -r skills/fin-econometrics/requirements.txt
./install.sh            # copies the skill to ~/.claude/skills and ~/.agents/skills (Codex)
```
For other harnesses, copy `skills/fin-econometrics/` into the folder where that harness looks for skills.

## Evals

`evals/` contains three blind tasks built on textbook datasets (Chen Qiang, 2023), at three
difficulty levels:

| case | difficulty | core issue |
|---|---|---|
| easy_nerlove | easy | heteroskedasticity, returns to scale |
| medium_icecream | medium | serial correlation in a short time series |
| hard_grilic | hard | ability bias / endogeneity → IV |

`python evals/grade.py evals/cases/<case> <run_dir>` scores the `analysis.json` contract against
golden values. `expected.json` also holds a qualitative rubric for an LLM or human judge.
