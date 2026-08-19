# changelog — deepseek-r1

## 2026-08-18 (epistemic correction pass)
Earlier svfix commit `7bdc87eb3` ("svfix(D_candidate): deepseek-r1 — DeepSeekMath (Shao et al.
2024, ancestor of GRPO) + unused appendix.tex ablation, PPO->GRPO step now grounded") had
`reasoning.md` narrate a *completed* pilot ablation in first person: "I train PPO on the MATH
task with the field-default GAE `λ = 0.95`, and it comes in considerably behind a critic-free
contender... Pushing `λ` up toward `1.0` closes most of the gap but not all of it... So the gap
isn't a hunch, it's measured." That reports the method's own experimental outcome as something
already observed in a single-turn PROPOSAL frame, where the method's own experiments have not
happened yet — a violation regardless of whether the underlying numbers are real (they trace to
DeepSeekMath's own published appendix ablation, which the narrator here is not entitled to claim
as its own in-frame run).

Fixed by rewriting the passage to keep the pilot's experiment DESIGN (same 16B MoE / 2.4B-active
model, MATH task, matched compute and sample budget, PPO started at the field-default GAE
`λ = 0.95` then swept toward `1.0`), the hypothesis and its mechanistic reason (sweeping `λ` up
trusts the critic's per-step estimate less and the raw return more, so the sweep is a direct
probe of how much the critic costs), the PREDICTION (PPO trails the critic-free run at default
`λ`, and the tuning sweep narrows but doesn't close the gap, at the cost of one full training run
per point on the sweep), and the decision rule (a critic that needs tuning just to partially
catch up isn't worth carrying into a long-context reasoning run) — while removing the claimed
observation and its stated outcome. The rest of the paragraph (the three analytical objections to
PPO's critic — memory/compute doubling, the critic having no direct training target for a
half-written response and having to interpolate a value it was never supervised on, and the dense
KL-to-reference term implicitly penalizing length) was left untouched; it is on-page reasoning
about PPO's mechanism, not a reported observation, and it alone still fully justifies the landing
on GRPO. The immediately following paragraph's addition (a correctness check is inherently
comparative, so a group baseline is the natural shape of the signal rather than a workaround) was
also left untouched — it is design reasoning, not an experimental claim.

Because the analytical case for choosing GRPO over PPO does not depend on the removed pilot
observation, the landing is still fully justified without it; this unit does not need a
trajectory-observation turn to supply the pilot's numbers. No changes to `answer.md` or
`train_answer.md` were needed — the svfix diff touched only `reasoning.md`, and neither of the
other two files mentions the pilot, the 16B MoE model, or a GAE `λ` sweep.
