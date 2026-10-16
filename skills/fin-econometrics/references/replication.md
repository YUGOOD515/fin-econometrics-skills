# Replicating an empirical paper

Goal: given a paper and its data, reproduce the reported estimates and judge whether the
paper's conclusions hold. Numbers come from `ek`; judgement comes from you.

## 0. Inventory (5 minutes)
```bash
find PAPER_DIR -maxdepth 3 -type f | head -100
```
Typical package: `paper.pdf`, `附录.pdf` (appendix), `数据.dta`, `程序.do`, `*.log/*.smcl`, `*.rtf` tables.
- PDF → text: `pdftotext -layout paper.pdf paper.txt` (scanned PDF: `pdftoppm -r 200 -png` then
  OCR, or view the page images). Tables in RTF/DOCX: `soffice --headless --convert-to txt:Text file.rtf`.
- Stata logs (`.smcl/.log`) are the most precise targets when present; prefer them over the PDF.
- Inspect the data once: `$EK profile --data 数据.dta` (variable labels are shown when stored).
- **pdftotext pitfalls**: in multi-column tables the stars, t-values in parentheses and column
  numbers often shift by one column or one row. Before writing targets, view the table page as an
  image (`pdftoppm -f P -l P -r 150 -png paper.pdf pg`) and read the numbers column by column.

## 1. Extract the paper's design (write it down before estimating)
In `replication_plan.md` record, per hypothesis: outcome, key regressor(s), controls, fixed
effects, SE/cluster level, sample filters (years, industries, winsorization, dropping finance
firms/ST firms), and which table/column reports it. Then list the robustness, endogeneity (IV,
PSM, DID, lags), mechanism and heterogeneity tables. Note the paper's stated conclusion for each.

## 2. Encode targets
`targets.json` holds the numbers you will try to hit. Copy them as printed (keep rounding and stars):
```json
{"tolerance": {"rel": 0.02},
 "targets": [
  {"label": "表2(1)", "match": {"stored": "m1"}, "coefs": {"AI": {"coef": "0.0392***", "t": "4.60"}}, "N": 10563},
  {"label": "表5(2) IV", "match": {"id": 7}, "coefs": {"AI": "0.3023***"},
   "stats": {"kp_rk_LM": "40.21", "kp_rk_Wald_F": "25.90", "cd_F": "90.79", "AR_F": "9.27", "r2": "0.80"}}]}
```
`match` keys: `stored` (name from `est store`/`eststo`), `id` (order of estimation), `line`,
`dep`, `cmd`, `contains` (substring of the command). Stats: `N r2 adj_r2 r2_within pseudo_r2
kp_rk_LM kp_rk_Wald_F cd_F AR_F hansen_J`. Start with the baseline table, the main IV table and
one robustness table; add more only if time allows.

## 3. Run
**With the authors' do-file** (fastest, most exact):
```bash
$EK stata --do 程序.do --data-dir PAPER_DIR      # data paths in the do-file are resolved automatically
$EK check --targets targets.json                # -> ek_out/stata/check_report.md/.json
```
Read `errors` and `warnings` in the JSON. `--est-filter REGEX` runs only matching estimation lines
(data steps always run) — use it to iterate quickly on long do-files.

**Without a do-file** write your own `analysis.do` in Stata syntax from step 1 and run it the same
way. Supported: `use/import/merge/append/save/preserve/restore`, `gen/egen/replace/drop/keep/
rename/encode/destring/winsor2/tab gen()/duplicates`, macros, `foreach/forvalues/if`, `bys:`,
`xtset/tsset` with `L./F./D.` operators, factor variables `i. ib. c. # ##`;
estimation `reg areg reghdfe xtreg(fe/re) ivreghdfe ivreg2 ivregress xtivreg(fe) probit logit
poisson nbreg ppmlhdfe newey qreg tobit ologit oprobit mlogit clogit xtlogit(fe) psmatch2(nn)
sgmediation2`; post-estimation `est store/eststo, predict, test, testparm, lincom, nlcom`.
Small one-off commands: `$EK stata --data d.dta --cmd "reghdfe y x c1 c2, absorb(id year) vce(cl id)"`.
`--cmd`-only runs write to `ek_out/stata/cmd/` so they never overwrite the do-file run; use
`--tag NAME` (→ `ek_out/stata/NAME/`) to keep several variants, and put
`"estimates": "ek_out/stata/NAME/stata_estimates.json"` in `targets.json` to check that run. `ek check`
always prints which estimates file (and which do-file/commands) it compared.
Also available: `tempfile/tempvar` + `` `name' `` macros, `xtsum` (overall/between/within),
`pwcorr` (pairwise N and p-values), `L./F./D.` operators inside `gen`/`if` expressions.

### No do-file: write `analysis.do` from the paper
1. One line per table column, in the paper's order, each with `eststo`/`est store tX_cY` so
   targets can use `{"stored": "t3_c2"}`.
2. Translate the text literally first: the estimator the paper names (`reghdfe`, `areg`,
   `xtreg, fe`, `xtivreg, fe`, `ivreghdfe`), every control in the table notes, FE rows at the bottom
   of the table, the SE note ("聚类到企业层面" → `vce(cl id)`).
3. Run, `ek check`, then vary **one** convention at a time (cluster level, `keepsingletons`, `i.year`
   vs `absorb(year)`, winsorizing already in the data or not) with `--tag`; keep the closest.
4. Report what you had to guess. If coefficients match but t-values do not, the SE convention
   differs; say which one reproduces the paper.

Unsupported (warning, not silently skipped): GMM dynamic panels (`xtabond2`), modern DID
(`csdid`, `did_multiplegt`), `rdrobust`, `synth`, `heckman` (do the two-step by hand: probit →
`predict xb` → `gen imr = normalden(xb)/normal(xb)` → OLS with `imr`), `bootstrap`, quantile
panels, spatial models. If a key table needs one, say so and, if feasible, implement the method in a
short script and label it "approximate".

## 4. When numbers differ — check in this order
1. **Sample**: N first. Year range, missing-value handling, winsorization level (1%/99% by year?),
   dropped industries, singleton removal (reghdfe drops singletons; areg/xtreg do not).
2. **Variable construction**: logs vs levels, lag timing, scaling (×100), industry code version.
3. **Fixed effects**: firm vs industry×year; `absorb(id year)` vs `i.year`.
4. **SE**: robust vs cluster level; areg/xtreg/reghdfe use different dof → tiny SE differences,
   identical coefficients. Coefficients equal but t differs → SE convention, not a replication failure.
5. **Estimator conventions**: ivreg2 reports large-sample z; `small` gives t; xtivreg fe uses z.
6. **Reporting**: some tables print marginal effects (probit `margins, dydx(*)`) or standardized
   coefficients; stars may use different thresholds.
7. **R² conventions**: `reg`/`areg` report R² including the dummies; `reghdfe` prints both
   "R-squared" (`r2`) and "Within R-sq." (`r2_within`); `xtreg, fe` reports within R² (`r2_within`).
   Constants: `reghdfe`/`areg` report a constant (mean-adjusted), `xtreg, fe` too; some papers omit it.
8. **N**: `ek check` adds a `hint` when N differs, e.g. "N + singletons_dropped = paper N → add
   `keepsingletons`".
Record every deviation you cannot close, with the most likely reason.

## 4b. Always re-estimate under the inference the claim needs
Even when the numbers match, re-run the key columns with (a) SE clustered at the entity (firm /
city / country) level if the paper used robust or iid SE, and (b) for IV, the same clustering —
`xtivreg, fe` and `ivreg2` without `cluster()` assume iid errors, so their Cragg-Donald F overstates
instrument strength for panel data; report the clustered KP rk Wald F and the AR test. Few clusters
(< ~40, e.g. provinces): say so and suggest wild-cluster bootstrap (`boottest`, not supported).

## 4c. IV construction checklist
- **Time-invariant base-year instruments** (1970 books, historical distance) and their lags:
  lags of a time-invariant variable are identical within a unit, so `L3.z L4.z` is one
  instrument (ek warns "dropped as collinear"); with unit FE the instrument is absorbed entirely
  unless interacted with a time-varying series. Use the Stock-Yogo value for the real count.
- **Leave-one-out / peer means** (industry or region average of x excluding self): related to y
  through common shocks and sorting; with unit FE identification comes from time variation in the
  peer mean only.
- **Bartik / shift-share**: say whether exogeneity comes from the shares or the shocks; check
  pre-trends of shares; a few industries usually drive the variation.
- **IV ≫ OLS** (5×-200×): weak IV, invalid exclusion or a tiny complier group; never just call it LATE.

## 4d. Staggered DID / policy-pilot papers
1. Tabulate adoption timing: `bys id: egen first = min(cond(treat==1, year, .))`, `tab first`.
2. Event study with the engine (no negative factor levels in Stata syntax, so shift):
```stata
gen rel = year - first
replace rel = -5 if rel < -5 & rel < .
replace rel = 5 if rel > 5 & rel < .
gen relp = rel + 6            // -5..5 -> 1..11
replace relp = 5 if missing(first)   // never-treated join the reference period (rel = -1)
reghdfe y ib5.relp $controls, absorb(id year) vce(cl id)   // ib5 = rel -1 as base
test 1.relp 2.relp 3.relp 4.relp                          // joint pre-trend test
```
Run with `--keep-dummies` to list the event-time coefficients (factor dummies are hidden by default).
3. Heterogeneous-timing bias: TWFE compares late with early adopters. A cheap check is the
   stacked / clean-control estimate: keep never-treated + one cohort at a time, estimate, average.
   Name Callaway-Sant'Anna / Sun-Abraham / de Chaisemartin-D'Haultfœuille (`csdid`,
   `did_multiplegt` are not supported) as the full fix.
4. Selection into treatment: compare pre-period means of treated vs control; PSM-DID only
   balances observables.

## 5. Judge, not just match
Replication of numbers ≠ validation of claims. After `ek check`, assess:
- **Conclusion consistency**: same sign and significance for the key coefficient in each
  hypothesis? Report `sign_significance` from `ek check`.
- **Identification**: is the IV relevant (KP F vs Stock-Yogo 16.38 for 1 endog/1 IV at 10%) and
  plausibly excludable? Bartik/lagged/"industry average" instruments: say what the exclusion
  restriction requires and why it may fail. IV ≫ OLS: discuss LATE, measurement error or invalid IV.
- **Robustness you can add cheaply**: alternative clustering (firm vs industry), dropping
  2020 or special industries, alternative key-variable measure, placebo/lead test (F.x should be 0).
- **Magnitude**: one-SD change of X in units of Y's SD or mean — is it economically plausible?
- **Mechanism tables**: a "mediator" regression (X→M, then Y on X and M) is descriptive; note
  the post-treatment-control problem.
Use the decision rules in `decision-rules.md` for each concern.

## 6. Deliver
1. `replication_report.md` (Chinese for Chinese papers): data & code overview → targets and
   match summary (verdict counts, table) → deviations and reasons → conclusion consistency →
   critique & extra robustness → overall verdict and whether the paper's conclusions are supported.
   Verdict mapping (`ek check` → `verdict_cn`): replicated = 完全复现 (all numbers exact/close),
   largely replicated = 基本复现 (≥ 80 %), partially replicated = 部分复现, not replicated = 未能复现.
   Numbers replicated ≠ conclusions supported: give both, separately.
2. `analysis.json` (schema in SKILL.md) with `mode: "replication"` and a `replication` block:
   `{"verdict": "...", "counts": {...}, "sign_significance": {...}, "deviations": ["..."]}`.
3. Optional HTML: story sections accept `"check": "ek_out/stata/check_report.json"` (paper-vs-
   replication table) and `"table": {"fits": ["ek_out/stata/stata_estimates.json#m1", ...]}`.
