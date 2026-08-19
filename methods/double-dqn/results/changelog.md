# Changelog

## 2026-08-19 — svfix(W3_primary_plus_ancestors) verification pass
- Quality-gate review of the decisive step (Jensen/convexity overestimation
  argument -> tied-state Theorem-1 lower bound, worked and numerically
  checked on the page -> decouple selection (online net) from evaluation
  (target net) in the bootstrap target, reasoning.md lines ~1-247): confirmed
  genuinely derived (the trace re-derives the tight lower bound by
  contradiction from scratch, verifies the tight witness and the i.i.d.
  uniform-error expectation $(m-1)/(m+1)$ numerically, and lands on the exact
  target formula) and fully backed by the primary already on disk
  (`refs/primary/double-dqn-arxiv-1509.06461.txt` Theorem 1 statement and
  appendix proof, `arxiv_source/DoubleDQN_aaai2016_total.tex` lines 350-354
  for the target formula).
- TRIAGE (class D) pointed at `refs/self_accounts` (van Hasselt's own blog
  post) and `refs/explainers` (two implementation-walkthrough HTMLs) as
  on-disk-but-unused material. Both were read in full: the self-account's
  actual body (stripped of blog boilerplate) is ~2.8 KB and is nothing but
  the paper's own abstract restated in an announcement post; the explainers
  are hindsight tutorials of the finished algorithm. None contain
  documented-struggle content beyond what the primary already states, so no
  source was grafted — bolting a decorative citation onto an
  already-grounded derivation would be damage, not improvement.
- Bonus check: the primary paper's own appendix displays a double-estimator
  zero-bound witness that, taken literally, does not satisfy its own
  mean-squared-error hypothesis ($\sum_a\epsilon_a^2=C$ instead of $mC$
  at that one spot, per `arxiv_source` tex lines 350-354). reasoning.md uses
  a different, internally consistent witness that does satisfy the theorem's
  hypothesis and checks out numerically for $m=2,3,5$ — no error to log
  against reasoning.md; it is quietly more correct than the primary's
  displayed formula at that one point.
- No rewrite. No factual errors found in reasoning.md, answer.md, or
  train_answer.md (formulas, target-copy period, and code all cross-checked
  consistent). See `notes/sources.md` for the full gate write-up and quotes.
