Boosting fails under label noise in a characteristic way: the update meets
a mislabeled point it can never fit and responds by escalating, until a
corrupted example commands an outsized share of the sample distribution.
This variant asks for a strategy designed as if a fraction of the training
labels -- class flips on the classification side, heavy-tailed target
corruption on the regression side -- could not be trusted, with the
protection built into the update rules themselves.

The non-negotiable property is bounded influence: no single training
point, however violently it disagrees with the current ensemble, may exert
more than a bounded effect on the pseudo-targets, on any learner's
coefficient, or on the weight distribution. Beyond boundedness lies the
interesting part, discrimination: separating hard-but-genuine samples,
which deserve continued effort, from likely-corrupt ones, which deserve
suspicion. Sustained disagreement over many consecutive rounds, or
residuals that sit far outside a robust scale of their peers, are natural
raw signals; the scaffold seeds both (a clipped-residual target rule and a
consecutive-miss tracker) without yet acting decisively on either.

Honesty constraint: the evaluation data is what it is, and the reported
metrics do not change, so the robustness machinery must cost approximately
nothing if the training labels happen to be clean. Insurance bought by
discarding information that should have been used will show up as a worse
score. What must be argued is that trust-aware updates are nearly free on
clean data and protective under contamination -- the same mechanism, both
regimes.
