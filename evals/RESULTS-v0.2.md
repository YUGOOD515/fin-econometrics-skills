# Eval results — v0.2 (paper replication mode)

Three blind replication tasks on published Chinese empirical papers, each run by two models in a
clean harness (skill folder + working folder only; no Stata). Graded with `grade.py`
(key coefficients at printed precision, t/SE, N, verdict, report, critique keywords); the
`rubric` in each `expected.json` is for a human/LLM judge.

| case | materials | Sonnet 5.5 | DeepSeek V4.1 Flash |
|---|---|---|---|
| easy_lixiaoping — 李小平等(2026)《中国工业经济》表1 + 附表3 | paper, appendix, data, do-file | 100 % (36/36) | 100 % (36/36) |
| medium_litao — 李涛等(2026)《世界经济》表3 + one robustness table | paper, data (no code) | 100 % (31/31) | 100 % (31/31) |
| hard_overtime — 宣扬、武凯文(2023)《世界经济》表4 + 表5 | paper, data (no code) | 100 % (38/38) | 100 % (38/38) |

Numbers: every run reached `ek check` verdict **replicated / 完全复现** (44/44, 71/71, 111/111,
44/44, 44/44, 70/70 numbers exact or close), including the two no-code cases where the agent had
to infer the specification (keepsingletons, areg vs reghdfe dof, iid `xtivreg, fe` for the IV
table) from the paper text.

Judgement (all six said "numbers replicated, conclusions only partly supported"), e.g.:
- 李小平: the significance in Table 1 and 附表3 rests on iid SE; clustered by economy every column
  is insignificant. The 1970 instruments are time-invariant (their lags are one instrument), the
  Bartik IV is rank-1 and the peer-mean IV keeps <0.1 % of its variance after FE.
- 李涛: pre-trends are jointly significant, the lead of the treatment is positive and the effect
  comes from the 2020-21 cohorts (staggered TWFE); 33 province clusters → wild bootstrap advised.
- 宣扬、武凯文: IV significance comes from unclustered SE; the instrument varies at city×year,
  clustering by city gives KP F 7.9 and t −1.3; IV ≈ 17× OLS is not credibly a LATE.

## Changes made from the v0.2 friction logs
- `ek ivcheck`: time invariance, level of variation (→ clustering advice), rank-1 shift-share,
  variance left after FE, partial correlations, leave-one-out peer-mean test, instrument-set rank.
- `ek stata --vce-scan "iid; robust; cluster id; cluster city; iid small"`: coef/se/t per SE
  convention for every estimation line; `--tag`, `--cmd` runs in `ek_out/stata/cmd/`, `run_info.json`.
- IV: Stock-Yogo 10 % critical value + flag, AR 95 % confidence set (`AR_CI95`), `AR_F_p`,
  warning when instruments are dropped as collinear, explanatory underidentification message.
- `ek check`: `--list`, `nth`, several estimates files / per-target file, N hints with the
  singleton arithmetic, "coefficients match but se/t differ" flag, `verdict_cn`, no star check for
  `_cons` without SE.
- `ek diagnose`: Wooldridge panel serial-correlation test with `--entity --time`.
- `ek compare` accepts `stata_estimates.json#name`.
- Stata engine: `tempfile/tempvar` macros, `xtsum`, `pwcorr` (pairwise N, p), `import delimited
  f.csv, clear`, xtset kept through save/use, `log using` noted, L./F./D. in `gen`/`if`;
  collinearity check rewritten (reg with 2,000 firm dummies: 60 s → 10 s).
- Docs: stats-key mapping table, no-do-file workflow, IV checklist, staggered-DID/event-study
  template (tested), cohort and PSM-flag tips, dynamic-panel stance (Anderson-Hsiao approximation).

## Open items
Wild-cluster bootstrap (`boottest`), multiway clustering in IV, CS/Sun-Abraham DID, dynamic-panel
GMM, `sensemakr` — reported as unsupported; agents flagged them as the next most useful additions.
