# v0.1 eval results (2026-10-16)

Each case was run once by Sonnet 5.5 (high) and DeepSeek V4.1 Flash (high): 6 independent blind sessions.

| case | Sonnet 5.5 | DeepSeek V4.1 Flash |
|---|---|---|
| easy_nerlove | 87.5* — quadratic cost function, scale economies exhausted at q≈4.4k, homogeneity imposed | 87.5* — quadratic centered at median, RTS 1.15 |
| medium_icecream | **100** — static HAC + dynamic `L.temp` (textbook solution) | 87.5* — semi-log partial adjustment model, HAC |
| hard_grilic | 95.8 — IQ instrumented by med+kww, s = 6.1% (matches textbook) | 83.3 — **s instrumented by med only → 12.3%**; iq dropped; exclusion "tested" invalidly |

\* Grader false negatives: the key coefficient was renamed (`lnq_c`) or rescaled (semi-log). To be fixed in grader v0.2.

## Judge notes
- Both models followed the workflow, produced every artifact, and went beyond the baseline (RESET → nonlinearity).
- The only substantive error: DeepSeek, hard case. It used parental education as the instrument for schooling, which
  violates exclusion under ability bias. IV > OLS was not treated as a red flag, and the headline claims "net of IQ"
  although IQ is not in the model.
- Report quality is high. Sonnet's grilic report has an excellent plain-language "naive vs corrected" narrative.

## Top friction (aggregated from friction_log.md)
1. Setup `cd analysis` conflicts with the user-specified working dir (6/6).
2. No `lincom`/`nlcom`/Wald test and no vcov in the fit JSON. Elasticities, long-run effects and RTS had to be computed by hand (5/6).
3. `diagnose` does not save a baseline fit; `run.sh` has to be maintained by hand (4/6).
4. Report: the legend overlaps the x-title; English "obs / linear fit" appears in Chinese reports; coef-chart labels can't be mapped; y2 margin is too narrow (4/6).
5. IV: multi-endogenous weak-IV flag misleading (needs Sanderson-Windmeijer / Shea); diagnose with `--endog` not visible in output.
6. stderr warnings break `| json.tool`; p-values underflow to 0.0; FGLS rmse/r2 is on the weighted scale.
7. Missing: `_n` / mean() in transform, row-index semantics for `--query`, VIF false alarm for x & x².
