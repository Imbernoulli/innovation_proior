# Progressive Growing of GANs changelog

## 2026-08-17 — source-value recheck
Two author self-accounts were located that earlier passes had concluded did not exist: Karras's 2018
Machine Learning Coffee Seminar talk and the ICLR 2018 OpenReview thread (quotes in
`notes/sources.md`). Two passages of `results/reasoning.md` now run through them:
- The per-pixel normalization: the trace asserted "I want a parameter-free brake in G" with no argument
  for why only G. It now carries the author's reasoning — every normalization on the table computes
  minibatch statistics, which need a large minibatch that 1024×1024 rules out; the reflex is to
  normalize both networks; but an escalation takes two participants, so disarming one is enough.
- The growth schedule: added the explored-alternatives result from the authors' OpenReview reply —
  starting at 2×2, 4×4, 8×8 or 16×16 makes no real difference, 4×4 is chosen as the natural fit for the
  block structure, and what does matter is keeping the two networks mirrored in structure, capacity and
  matched up/downsampling operators. This also removes the implicit suggestion that 4×4 is essential.
- Unchanged and explicitly re-verified: the real-image fade step. No self-account explains it (a wide
  search is recorded in `notes/sources.md`); the trace's own derivation stands and its numeric check
  (residual 0.65 → 0.32 → 0.0 as `old_weight` goes 0 → 0.5 → 1.0) is exact, since the one-octave blur is
  idempotent and the residual is therefore `(1-old_weight)` times the sharp-image residual.
- No factual errors found; the `lod` schedule, the fade formulae, the landing and the code are unchanged.
