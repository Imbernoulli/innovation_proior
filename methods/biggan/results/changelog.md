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
