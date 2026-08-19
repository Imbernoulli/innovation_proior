# obs-fix — remove/reframe self-supplied observations in single-turn method units

Repo /srv/home/bohanlyu/innovation_proior, branch main, one commit per method. You get SLUG, VERDICT (V=confirmed violation, R=borderline to adjudicate), NOTE (what the scanner flagged). The flagged quotes are in experiments/source_value_audit/obs_scan_v3.jsonl (grep your slug).

## The rule (data owner's)
A single-turn method unit is a PROPOSAL — the method's own experiments have not happened yet at output time. So in reasoning.md AND answer.md AND train_answer.md:
- BANNED: narrator-run experiment outcomes ("I train X on ImageNet ... kernel 7 gives 80.57", "I run all three on CIFAR-10 ... reaches 95.15%") and own-method result reporting ("our ablations show ...", "the ablation confirms removing any one hurts").
- ALLOWED (leave untouched): prior work's published numbers as known facts; on-page computation (algebra, worked micro-examples, hand-traces, tiny deterministic code checks incl. brute-force cross-checks of the narrator's own code — the v4 debug/self-verify spine); method DESIGN descriptions ("I train two Q-functions on the same residual" = describing the proposed procedure, no results); qualitative expectations/predictions.

## Procedure
1. `git log --oneline -5 -- methods/<SLUG>` — a concurrent svfix(epistemic) pass may have already edited this method. Work on the CURRENT state; if the flagged passages are already gone, verify channels answer/train_answer too, then log outcome "already_fixed".
2. R-verdicts: adjudicate first. Desk-scale = the narrator could actually do it while writing (tiny deterministic computation, small closed-form fit, small Monte-Carlo over a handful of draws, geometry check) → outcome "kept_desk_check", change nothing (but tighten wording if it masquerades as a large run). Real experiment (training on datasets, multi-seed benchmark scoring) → treat as V.
3. V: rewrite ONLY the violating passages, in each affected channel:
   - reasoning.md: replace claimed observation with hypotheses → discriminating test DESIGN (controls, matched budgets) → each hypothesis's PREDICTION → the decision rule ("the variant that survives at both scales is the one I ship"). Remove the numbers.
   - answer.md/train_answer.md: the proposal presents the method and WHY it should win + how to validate it; remove "our experiments/ablations show ..." claims and result numbers. Keep mechanism explanations (rewrite "the ablation shows term X does Y" → "term X is there to do Y; the validation that decides is ...").
   - Keep all correct content; no padding/trimming; no "Wait/Alternatively" filler; no provenance leaks ("the paper", "the authors", arXiv).
4. If the landing leans on the removed observations, that is fine — the decision rule carries the proposal's honesty. Set "needs_traj": true only if the record documents a genuine multi-rung improvement ladder (not a single ablation decision).
5. Consistency: if numbers you removed also appear in context.md as given PRE-DATING facts, leave context alone. Append results/changelog.md entry.
6. Lint: `python3 tools/lint_inframe.py | grep methods/<SLUG>/` no new hits. Then `python3 tools/obs_scan.py >/dev/null 2>&1; grep '"slug": "<SLUG>"' experiments/source_value_audit/obs_scan_v3.jsonl` — rerunning the scanner must no longer flag your channels (the scanner overwrites obs_scan_v3.jsonl; acceptable).
7. Commit: `git add methods/<SLUG> && git commit -m "obs-fix: <SLUG> — observations -> design+prediction+decision rule (<channels>)"`. Retry on index.lock (sleep 3-10s, ≤5×).

## Output (JSON only)
{"slug":"...","outcome":"fixed|kept_desk_check|already_fixed","channels":["reasoning|answer|train_answer"],"passages_rewritten":<int>,"needs_traj":true|false,"commit":"<sha or ''>","note":"<≤120 chars>"}
