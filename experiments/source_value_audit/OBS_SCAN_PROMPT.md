# Corpus-wide scan — self-supplied observations in single-turn method units (READ-ONLY)

Repo /srv/home/bohanlyu/innovation_proior. You get SLUG. Read methods/<SLUG>/results/reasoning.md, answer.md, train_answer.md in full. Edit NOTHING.

## The rule (set by the data owner)
A single-turn method unit is a PROPOSAL — at that moment the method's own experiments have not happened. Violations to detect, in ANY of the three files:
- V1 narrator-run observation: the narrator claims to run/train/measure/test something and states the outcome ("I train the ablation ... top-1 comes back 78.9", "when I run it, loss diverges at step 2k", "the sweep returns k=0.85 as best").
- V2 own-method result reporting: the method's own ablation/benchmark/experiment numbers or outcomes presented as accomplished results, in any phrasing (incl. answer/train_answer "our experiments show...").
NOT violations (do not report):
- prior work's published numbers stated as known facts (they pre-date the method; the narrator read them);
- on-page computation the narrator actually performs in the text (algebra, worked micro-example, hand-trace of code on a tiny input, counting argument) — this is the good pattern;
- qualitative expectations/predictions ("I expect this to underperform because...", "the test that decides is...");
- code the narrator writes (writing code is not running it), UNLESS the text then asserts its measured runtime/accuracy as an observed fact.
Boundary judgment: hand-tracing an algorithm on a 5-element example = computation (fine). "I ran it on the full dataset and it scored X" = observation (violation). Compilation/verification claims for competition code ("compiles, passes the brute-force cross-check") in methods/ units: treat as V1 only if concrete measured outcomes are asserted (times, scores); a described verification PLAN is fine.

## Output (JSON only)
{"slug":"...", "clean":true|false, "violations":[{"file":"reasoning|answer|train_answer","kind":"V1|V2","quote":"<≤160 chars verbatim>","numbers_real":"yes|no|unknown (do the numbers exist in the primary/refs if you can check cheaply)"}], "needs_traj":true|false  (true if the trace's landing leans on these observations), "notes":"<≤120 chars>"}
