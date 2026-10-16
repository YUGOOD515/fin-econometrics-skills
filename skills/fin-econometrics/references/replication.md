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
Record every deviation you cannot close, with the most likely reason.

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
   critique & extra robustness → overall verdict (完全复现/基本复现/部分复现/未能复现) and whether the
   paper's conclusions are supported.
2. `analysis.json` (schema in SKILL.md) with `mode: "replication"` and a `replication` block:
   `{"verdict": "...", "counts": {...}, "sign_significance": {...}, "deviations": ["..."]}`.
3. Optional HTML: story sections accept `"check": "ek_out/stata/check_report.json"` (paper-vs-
   replication table) and `"table": {"fits": ["ek_out/stata/stata_estimates.json#m1", ...]}`.
