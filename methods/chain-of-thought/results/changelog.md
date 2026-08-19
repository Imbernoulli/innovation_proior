# changelog — chain-of-thought

## 2026-08-18 (obs-fix: self-supplied observation)
`obs_scan_v3.jsonl` flagged own-method result reporting (`abl_shows`) in two channels:
`answer.md`'s "## What the ablations show" section stated the equation-only, dots-only, and
post-answer-chain controls' outcomes as accomplished facts ("helps on one- or two-step
datasets but not much on... GSM8K", "about baseline", "about baseline"); `train_answer.md`
had the matching sentence ("ablations show that emitting dots of the same length does not
help..."). `reasoning.md` was already clean — a prior svfix pass had already cast these three
controls as hypotheses with predictions and a decision rule, so the discrepancy was that
`answer.md`/`train_answer.md` still reported the (not-yet-run) outcomes while `reasoning.md`
did not.

Fixed by rewriting both flagged passages to match `reasoning.md`'s already-correct frame:
each of the three alternative-explanation controls (equation-only, dots-only/variable-compute,
chain-after-answer) restated as a control with its PREDICTED near-baseline-or-not outcome,
plus an explicit decision rule (the method is credited with genuine step-by-step reasoning
only if the full chain clears all three controls while each control itself lands near
baseline). No result numbers to begin with in these passages (they were qualitative), so
nothing needed stripping beyond the past-tense outcome claims; mechanism content (why each
control isolates what it isolates) kept intact. `context.md` untouched — its benchmark/model
list is the pre-existing evaluation setup, not a reported result.

Single ablation decision (3 controls, one round), not a multi-rung ladder — no
trajectory-observation turn required.
