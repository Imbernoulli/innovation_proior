# Epistemic correction — remove self-supplied observations introduced by earlier svfix passes

Repo /srv/home/bohanlyu/innovation_proior (branch main; one commit per method). You get SLUG. Scope: ONLY passages that svfix commits added/changed (find them: `git log --oneline --all --since="2026-08-17" -- methods/<SLUG>` and `git diff svfix-baseline-2026-08-17..HEAD -- methods/<SLUG>/results/`). Pre-existing corpus text is out of scope for this pass (a separate sweep will handle it) — but if the svfix diff is clean and the flagged pattern is pre-existing, log outcome "preexisting" and touch nothing.

## The rule being enforced (set by the data owner)
A single-turn method unit is a PROPOSAL: at that moment in the frame, the method's own experiments have not happened yet. Therefore:
- reasoning.md must NEVER have the narrator run an experiment and state its result ("I train X... the numbers come back 78.9") — real numbers or not.
- answer.md / train_answer.md must NOT report the method's own experimental results either (they are trained model voice; the proposal has no results yet).
- The method's own results live ONLY in trajectory observation turns (separate track).
- ALLOWED and untouched: prior work's published numbers stated as known facts ("the reported top-1 of AdamW on ViT-S is..." where that pre-dates the method and exists in the record); the narrator's own on-page computation (algebra, worked example, hand-trace, code trace); context.md carrying pre-dating facts.

## Procedure
1. Read the svfix diff for this method. Identify every added passage where (a) the narrator claims to run/train/measure and reports outcomes, or (b) the method's own ablation/benchmark numbers appear in reasoning/answer/train_answer.
2. Judge each: own-method observation (violates) vs prior-work known fact (fine) vs on-page computation (fine).
3. Rewrite the violating passages ONLY: keep the hypotheses, the discriminating-experiment DESIGN (controls, matched budgets), each hypothesis's PREDICTION, and the decision rule ("whichever survives at both scales is the one to ship"). Remove the claimed observations and the numbers. Keep every non-violating improvement the svfix pass made (algebra, real counterexamples, design reasons, self-account grounding). Do not shrink or grow the rest.
4. If the landing (final method choice) is now unjustified without the removed observation, that is EXPECTED — the decision rule sentence carries the proposal's honesty ("this is the test that decides; I ship the variant that survives it"). Log "needs_traj": true so the unit enters the trajectory-conversion queue.
5. Check answer.md/train_answer.md for the same violation in svfix diffs; fix consistently. Append results/changelog.md entry.
6. Lint: `python3 tools/lint_inframe.py | grep methods/<SLUG>/` — no new hits. No "the paper/the authors/arXiv" leaks.
7. Commit: `git add methods/<SLUG> && git commit -m "svfix(epistemic): <SLUG> — observations removed from proposal voice (design+prediction+decision rule kept)"`. Retry on index.lock (sleep 3-10s, up to 5×).

## Output (JSON only)
{"slug":"...", "outcome":"corrected|clean|preexisting", "violations_removed":<int>, "needs_traj":true|false, "channels":["reasoning|answer|train_answer"], "kept":"<≤120 chars: what good content was preserved>", "commit":"<sha or ''>"}
