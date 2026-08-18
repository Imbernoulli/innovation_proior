# UniPC changelog

## 2026-08-17 — source-value recheck
The NeurIPS 2023 OpenReview thread, recorded by the earlier pass as unreachable (bot challenge), was
retrieved this pass and saved at `refs/self_accounts/openreview_thread_hrkmlPhp1u.md`. Its author
rebuttals supply exactly the reasoning the paper compresses, and two passages of
`results/reasoning.md` now run through it (quotes in `notes/sources.md`):
- Opening: the assertion "just raising the order of a predictor solver buys less and less" is
  replaced by the mechanism — a predictor buys order `p` by feeding in more *previous* model outputs,
  the oldest and least accurate quantities in the loop, so each added order imports a staler error —
  plus the structural obstacle the authors state directly, that the analytical coefficients for these
  solvers have only ever been written explicitly for orders ≤ 3.
- Order-schedule passage: added the measured sweep that settles the question, NFE=10 on CIFAR10,
  FID50K per per-step-order schedule — `1223433321` 4.07, `1233343321` 4.14, `1234544321` 4.76,
  `1234554322` 5.41, `1234565432` 18.23, `1234444443` 6.84 — and the authors' own reading, that the
  extra orders are bought with progressively older previous points whose error propagates, whereas the
  corrector buys its order from the *current* point. Best low-NFE schedules `123432` (6) and `1223334`
  (7) added.
No factual errors found; the "corrector is free" derivation, the Vandermonde solve, the worked toy
integral, the landing and the code are unchanged.
