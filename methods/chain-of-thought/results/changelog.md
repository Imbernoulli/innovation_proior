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

## 2026-08-18 (obs-fix repair pass: missed passage in `## Method`)
The prior pass fixed the two `obs_scan_v3.jsonl`-flagged `abl_shows` passages but left a
third instance of the same violation live: `answer.md`'s `## Method` section (untouched by
that commit) had an "Empirical behavior" bullet asserting the method's own scale-emergence
and robustness results as accomplished fact — "the gain is an emergent ability of scale —
CoT hurts or does not help most models below 10B parameters... is clearest around 100B+
parameters... arithmetic gains are robust to exemplar annotator, style, choice, order, and
count." This is exactly the scale-emergence/robustness content `reasoning.md` already hedges
as prediction ("I'd predict chain-of-thought prompting to do nothing — or even hurt — for
small models... I'd predict it to kick in only past some scale, maybe around 100B
parameters... If chain-of-thought beats the baseline by a wide margin under all of those
[checks]..."), and that `train_answer.md` already states as "is expected to be
scale-emergent" / "predicted" — but `answer.md`'s `## Method` bullet was still asserting it
as observed fact.

Fixed by relabeling the bullet "Predicted behavior" and rewriting it to match the
already-correct hedged frame in `reasoning.md`/`train_answer.md`: scale-emergence, the
100B+ threshold, and the multi-step-vs-one-step gain split are now stated as expectations
("is expected to be", "should hurt or not help", "should be clearest"), and the
annotator/style/order/count robustness claim is now framed as a precondition for crediting
the method ("only credited... if they hold up under...") rather than a reported outcome. No
numbers or mechanism content removed — `10B`/`100B` and the multi-step/one-step split are
kept as the predicted thresholds, matching `reasoning.md`'s own figures.
