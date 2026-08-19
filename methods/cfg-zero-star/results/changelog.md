# Changelog

## 2026-08-18 — svfix(epistemic)
Reverted an observation claim introduced by svfix(D_candidate) (checkpoint-convergence
evidence for the zero-init `K` step). A single-turn method unit is a proposal: at that
point in the frame this method's own experiments have not happened yet, so the narrator
cannot claim to have tracked the guided-vs-zero gap across checkpoints and report what it
found ("Early checkpoints show exactly the gap... As training proceeds the gap shrinks,
and past a clear turning point pure zero-init stops improving anything...", plus the
downstream claim that "the trick keeps paying off outside the toy setting"). No numbers
were involved, but the passage still reported a completed observation and its outcome in
proposal voice.

Rewrote the paragraph in `results/reasoning.md` (the "`K` must stay small" discussion) to
keep the good addition from that svfix pass — framing "fixed geometric fact" vs.
"underfitting symptom" as two competing, testable hypotheses — and the discriminating-test
DESIGN (track the guided-vs-zero gap across checkpoints of increasing convergence, on the
MoG closed form and on a real generator on ImageNet), plus each hypothesis's PREDICTION
(gap constant across checkpoints vs. gap shrinking to a turning point). Replaced the
reported outcome with an explicit decision rule: the sweep is what would decide between
the two readings, not yet run; the landing (zero-init as a short prefix, not a fixed
schedule fraction) is justified instead by what the toy Gaussian check already established
(the closed form is well-defined and the learned field has every chance to track it as `t`
grows), which was the pre-svfix justification. No change to the method or code. The
checkpoint-convergence sweep is queued for the trajectory-conversion track
(`needs_traj: true`) — this unit's own results belong there, not in reasoning.md.
