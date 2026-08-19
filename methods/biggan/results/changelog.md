# biggan changelog

## 2026-08-18 — obs-fix pass
- `results/answer.md` (large-scale instability paragraph): removed the claimed observation
  "freezing experiments show stability is a property of the G–D *interaction*" — a narrator-run
  experiment outcome not available at proposal time. Rewrote the paragraph as the diagnostic
  design (why σ₀ in G and D's spectra/train-val gap are the quantities to track), the
  discriminating test (freeze one network, keep training the other), the prediction (an asymmetric
  outcome would pin the cause on the G–D interaction rather than either network alone), and the
  decision rule (whether the strongly-constrain-D trade-off is worth its performance cost versus
  light conditioning + early stop). No numbers removed (none were present); mechanism content
  kept in full.
- `results/reasoning.md` was not touched by this pass — it was not the flagged channel for this
  slug's obs-fix task; its own extensive narrator-run-experiment content (batch/width ablations,
  the σ₀ clamp experiments, the R₁ sweep, the D memorization test, the freeze test itself) remains
  out of scope here and is a candidate for a separate reasoning-channel obs-fix pass.

## 2026-08-18 — obs-fix repair pass (reasoning.md)
- The prior pass's channel-scoping left `results/reasoning.md` untouched, and verification found the
  same rule violation category (narrator-run own-method experiment outcomes) present throughout the
  large-scale/instability arc, not just in the passage already fixed in `answer.md`. Rewrote every
  affected paragraph from "So my baseline is all of that assembled..." through the freeze-test
  conclusion (batch-size scaling, width/depth, shared-embedding + skip-z wiring, the truncation
  trick, the modified-orthogonal-regularization sweep, the σ₀ clamp experiments on G, the R₁
  gradient-penalty sweep on D, the D-memorization validation test, and the G/D freeze test) into
  hypothesis → discriminating-test design → prediction → decision-rule shape, in each case removing
  the narrated outcome and any reported numbers (IS/FID deltas, iteration-speed fractions, model
  fractions, accuracy percentages, loss magnitudes) while keeping every mechanism explanation, the
  algebra/gradient derivations, and the two small deterministic desk-checks (the log-softmax
  residual check and the truncated-z concentration check), which are on-page computations the rule
  allows and were left untouched. No provenance leaks, no "Wait/Alternatively" filler introduced.
  Code block untouched. `tools/lint_inframe.py` shows no new hits for methods/biggan; a fresh
  `tools/obs_scan.py` run (which currently writes `obs_scan_hits.jsonl`, not the `obs_scan_v3.jsonl`
  the audit task names — the latter is a stale prior-run snapshot) shows zero unguarded hits for
  biggan, with the only 3 remaining biggan/reasoning hits being the two allowed desk-checks and one
  guarded false positive, all `guarded: true`.
