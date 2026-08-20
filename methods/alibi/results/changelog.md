# Changelog

## 2026-08-19 — svfix repair (W3_ancestors_only)
Verifier rejected a prior fix pass: `reasoning.md:45` self-supplied the trainable-slopes
ablation's outcome inline in first person ("And indeed when I let the slopes be
trainable, the extrapolation comes out weak ... costs a few percent in training speed"),
violating the rule that a single-turn reasoning trace's think-voice may never state the
result of an experiment it claims to run mid-reasoning — even when, as here, the claim
matches the primary paper's own ablation (`src/03-ourmethod.tex`: "We initially
experimented with making the slopes trainable, but this did not yield strong
extrapolation results").

Rewrote the passage to hypothesis (gradient on m only ever sees d<L, so it's free to
overfit the training window) -> test design (matched fixed-vs-trainable run, same data/
seed/steps, compare extrapolation curve and per-step cost) -> prediction under each
hypothesis -> decision rule (wins on both extrapolation and cost, or doesn't ship). The
landing (fix the slopes before training) is unchanged; only the epistemics changed — no
inline outcome/numbers.

Also removed the same defect from `answer.md` and `train_answer.md`, which independently
stated the ablation's outcome ("making them trainable yields weak extrapolation (and a
small slowdown)" / "... weakens extrapolation and reintroduces length-dependent
overfitting"). Per the hard rule, the method's own empirical results may not appear in
any single-turn channel; both now state only the a priori design rationale (gradient on m
never sees distances past L) instead of a reported result.

No factual errors found or corrected; no new external sources introduced (class D,
primary-only per triage — see `notes/sources.md`).
