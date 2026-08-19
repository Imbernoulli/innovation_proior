# Changelog — hc

## 2026-08-18 — svfix(W3_ancestors_only) arithmetic-typo fix, sourcing verdict = sound_as_is
- `results/reasoning.md`: K2 sanity-check worked example (counts `(4,2,3)`, `r=3`, `N_ij=9`)
  had an internally-inconsistent intermediate fraction: `2 / 39916800 · 24·2·6 = 5760 / 79833600`.
  `2 * 24*2*6 = 576`, and the denominator stays `39916800` (not doubled), so the correct
  intermediate fraction is `576 / 39916800`. Verified numerically: `576/39916800 ≈ 1.44300e-5`,
  matching the passage's own final printed decimal (which was already correct — only the
  unreduced intermediate fraction was wrong). Fixed `5760 / 79833600` → `576 / 39916800`.
  Confirmed the number does not recur in `answer.md`/`train_answer.md`/`code/` (no propagation
  needed).
- Sourcing audit (track W3_ancestors_only, triage class D): re-verified the decisive step
  ("drop K2's topological-ordering requirement → search the full DAG space with add/delete/
  reverse ops + an explicit acyclicity guard, scored by score-equivalent BDeu") against
  `refs/cooper_herskovits.pdf` and `refs/chickering1995_search.pdf`, already on disk and
  already logged in `notes/synthesis.md`. Cooper & Herskovits explicitly ran K2 with an
  ordering "consistent with the partial order of the nodes as specified by ALARM" and state
  they "are exploring methods that do not require an ordering" — the trace's stated failure
  reason for the ordering-dependent heuristic is real and checkable, not asserted. Chickering,
  Geiger & Heckerman's search-methods paper independently supplies the resolution verbatim:
  the add/delete/reverse operator set with the acyclicity constraint, the decomposability-based
  one-or-two-local-recompute cost accounting, the Chow-Liu/max-branching k=1 special case, and
  "the general case (l > 1) is NP-hard, even for a score-equivalent metric." Also re-derived
  numerically the score-equivalence claims used to justify BDeu over K2 (K2 orientation gap
  ≈0.0083 on the worked `[[7,3],[2,8]]` example; BDeu equal to machine precision at N'=1 and
  N'=10) — both reproduce the trace's printed numbers exactly. Outcome: **sound_as_is** — the
  decisive step is genuinely derived on the page and its failure/resolution reasons are backed
  by the ancestor papers already in `refs/`. No sources grafted; no rewrite of the decisive
  step.
