An imputer can win on one of this benchmark's two yardsticks while quietly
losing the other. Filling every hole with a conditional mean is the
entry-by-entry optimal move for reconstruction error, yet it shrinks marginal
variance, flattens feature interactions, and hands the downstream
gradient-boosted learner a completed matrix whose joint structure no longer
resembles the truth; conversely, completions that faithfully restore spread and
dependence can drift away from the masked values pointwise. This variant is
about closing that disagreement, not averaging over it.

Your objective is agreement between reconstruction and utility on the weakest
evaluated dataset. The settings differ by more than an order of magnitude in
sample count, mix classification with regression targets, and span roughly 8 to
30 features; because the task score is a geometric mean over settings, a method
that collapses on any one of them — typically the small-sample,
correlated-feature regime where aggressive iterative refinement overfits —
forfeits nearly everything. Treat whichever dataset scores lowest under your
method as the object of study, and improve it without giving back the others.

Constraints that define the variant:

- **One fixed configuration serves every dataset.** No branching on matrix
  shape or any dataset fingerprint to select a different imputer; adaptation
  must flow through statistics estimated from the NaN-bearing matrix itself
  (per-column spread, dependence structure, missingness fraction) and be used
  continuously.
- **Both metrics must move together.** A change that lowers masked-entry error
  while degrading the downstream model's accuracy or R² — or the reverse — does
  not count as progress; structure the refinement loop so each pass can be
  judged against both criteria at once.
- **Stay deterministic and finite.** Given the provided seed, transform must
  return the same fully finite matrix every time, within the fixed evaluation
  time limits.

The deliverable is a mechanism, not a tuning result: an explanation of why
pointwise accuracy and preserved joint structure stopped trading off on the
dataset where they conflicted most.
