Batches are the unit of purchase here, and batches are where single-score
acquisition quietly fails. All 50 or 100 picks of a round are scored by the
same frozen model before a single new label arrives, so whatever that model
finds confusing, it finds confusing many times over: the same tangled region
of letter's 26-way boundary, the same borderline spam template, can fill an
entire batch with near-copies of one query. The oracle then answers what is
effectively one question at the price of a hundred, the retrained model moves
barely at all, and the learning-curve area — which integrates accuracy over
every round — records the waste permanently.

This variant makes within-batch redundancy the central object of design.
Construct an acquisition rule that treats the batch as a set to be optimized,
not a top-k cut of a pointwise score: candidates should compete not only on
informativeness but on their distance — geometric, gradient-based, or
predictive — from the other members of the same batch and from what is
already labeled. Submodular coverage objectives, determinantal repulsion,
clustering with per-cluster quotas, or greedy farthest-point schemes are all
admissible shapes; the requirement is that adding a candidate that duplicates
an existing pick must cost it the slot. The tension to manage is that
diversity pushed to its extreme collapses into uniform sampling — the rule
still has to concentrate labels where the classifier is genuinely unsettled,
and one fixed formulation must handle a 26-class, a 3-class, and a binary
pool without retuning.

Judgment comes from the unchanged harness — final-round accuracy and the
accuracy-versus-labels area on the same three tabular datasets — and the case
to make is comparative: at an identical per-round budget, the de-duplicated
batch should beat its own pointwise-scored ablation, with the gap widest on
the many-class dataset where redundant picks are most likely.
