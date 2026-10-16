# Evals

Blind end-to-end tasks. The agent sees only `task.md` and the data file; it never sees `expected.json`.

## Run one case
1. `mkdir -p runs/<model>/<case> && cp data/<file>.csv runs/<model>/<case>/`
2. Start the harness (Claude Code / Codex / ...) in that folder with the skill installed, and paste the contents of `task.md`.
   Append this line: "完成后写 friction_log.md：skill 哪里不清楚、ek 报错/缺功能、你绕过了什么、改进建议。"
3. Grade: `python grade.py cases/<case> runs/<model>/<case>`. Then score the `rubric_for_judge` items manually or with an LLM judge.

Golden values in `expected.json` come from the textbook Stata output (Chen Qiang, 2023), cross-checked with econkit.

## Replication cases (v0.2)
The agent gets a folder with the paper PDF and the authors' data (and, for the easy case, the
do-file) — see `materials` in each `expected.json`. Paper packages are public replication files from
《中国工业经济》/《世界经济》 and are not redistributed in this repo.
1. Copy the listed materials into `runs/<model>/<case>/`, start the harness there with the skill
   installed, paste `task.md` plus the friction-log line above.
2. `python grade.py cases/<case> runs/<model>/<case>` — finds every `ek` estimate written in the
   run (`stata_estimates.json`, `fits/*.json`) and checks the paper's key coefficients (rounded as
   printed), t/SE and N; plus the report, verdict and critique keywords. The `rubric` is for a judge.
