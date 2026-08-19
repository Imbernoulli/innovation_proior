# Changelog

## 2026-08-18 — svfix(W3_primary_plus_ancestors)

- **Fixed factual error (LayerNorm extrapolation bound).** `context.md`, `answer.md`, and
  `train_answer.md` stated the critic LayerNorm bound as `|Q| <= ||w||`, copying an imprecision
  in the ancestor source (`refs/rlpd_src/sections/method.tex`, which asserts
  `||psi(s,a)|| <= 1`). Standard LayerNorm normalizes each of the `d` hidden features to unit
  *variance*, not the vector to unit *norm*, so `||psi(s,a)|| = sqrt(d)` exactly and the correct
  bound is `|Q| <= ||w||*sqrt(d)` (a factor of 16 above the naive bound at the paper's hidden
  width `d=256`). `reasoning.md` already derives and self-corrects to this bound with a worked
  numeric check (`psi = [50,-80,120,-30]` layer-normalizes to norm `2.0 = sqrt(4)`); the other
  three files now match it. The mechanism (a fixed constant independent of how OOD the input
  is) is unchanged — only the constant.
- **Removed an unhedged empirical-outcome claim (answer.md).** "The actor penalty is the
  load-bearing one; the critic penalty ... contributes less on most tasks" restated ReBRAC's
  own decoupling ablation result (primary, line 517) as settled fact in the answer channel.
  Rewrote to state only the structural reason for decoupling (actor vs. critic penalty answer
  different questions, so one shared scalar collapses a 2-D trade-off), matching the properly
  hedged framing already used in `reasoning.md`/`train_answer.md` ("I'd expect ... but I'm not
  assuming that").
- Decisive-step derivations themselves (decoupling rationale, LayerNorm bound mechanism) were
  found genuinely derived on the page and were left unchanged; see `notes/sources.md` for the
  full quality-gate writeup.
