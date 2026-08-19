# Changelog — dual-hilbert

## 2026-08-19 — svfix(W3_primary_plus_ancestors)
Decisive step: `V(s,g) = -||phi(s)-phi(g)||` (symmetric Hilbert approximation of the asymmetric
`d*`) fit via the twin-expectile offline TD backup.

- The symmetric-Hilbert-form half of the decisive step (why a symmetric `||.||` can only
  approximate an asymmetric `d*`, and why `l2`/Hilbert rather than an arbitrary metric) was
  already genuinely derived on the page via a worked directed-3-cycle counterexample (exact
  squared-error floor of 1.5) and the embeddability obstruction — left untouched.
- Fixed the twin-expectile half: `results/reasoning.md` previously asserted, with no forcing
  reason, that keying the expectile weight on the online residual "would chase its own tail."
  Replaced with the actual mechanism — grounded in the HIQL author's own GitHub reply
  (`refs/self_accounts/hiql_github_issue6_compute_value_loss.txt`,
  https://github.com/seohongpark/HIQL/issues/6): keying the classification on the same online
  head being updated creates a self-reinforcing coupling between "which transitions are
  optimistic" and "how wrong the head currently is," the same pathology Double Q-learning
  diagnoses for argmax/max coupling in ordinary Q-learning; reading the sign off the target nets
  instead breaks the loop. Framed as mechanism + explicit deferral of the effect size to the
  downstream success rate (author: "we tested several variants ... slightly improves
  performance"), not as a fabricated derived/observed result.
- No factual errors found; landing (method + code) unchanged.
- New source added: `refs/self_accounts/hiql_github_issue6_compute_value_loss.txt`.
