# Changelog

- 2026-08-18 `results/reasoning.md` — grounded the decisive-step paragraph (naive
  separate-batch BatchNorm in the critic, traced with a self-generated numeric
  toy, "BatchNorm has a bad reputation in RL, though...") in the self-account
  material now on disk at `refs/bhatt2019_crossnorm_v1.pdf/.txt` — the same
  authors' 2019 predecessor paper under the same arXiv id (1902.05605v1,
  "CrossNorm: Normalization for Off-Policy TD Reinforcement Learning"). The
  obvious first move (forward `(s,a)` then separately forward `(s',a')`
  through the same BatchNorm critic, both in training mode) is now stated as
  the exact configuration that earlier paper names "a common pitfall,"
  diagnosed there with the specific, checkable mechanism "the statistics of
  each batch oscillate between two modes" — not a vague "bad reputation." The
  resolution (concatenate the two batches, one joint forward pass, then
  split) is now tied explicitly to that earlier paper's own resolution
  (there named CrossNorm), rather than reading as a clean forward derivation
  invented for this trace. The hand-computed numeric toy is kept as a
  secondary mechanistic check, not the sole justification. No factual errors
  found; landing (BatchRenorm twin critics, joint forward pass, no target
  network, UTD=1) unchanged, so answer.md/train_answer.md left as-is. See
  `notes/sources.md` for the quotes and provenance.
