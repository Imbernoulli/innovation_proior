A single fit on a single draw of the data is an anecdote. The score this benchmark assigns is
dominated by settings where the anecdote misleads — four hundred samples against a dense
graph, noise at two and a half times the unit scale — and where the edge set chosen by one
pass of any estimator swings visibly when the rows are reshuffled. This variant makes that
swing the object of study: membership in the reported graph must be earned by reappearance
across perturbed versions of the dataset, not by clearing a statistic once.

Concretely, the estimator's own selections are the raw material. Run the base learner on
subsamples, half-samples, or bootstrap replicates; record, per candidate edge, how often it is
selected; report the edges whose selection frequency clears a bar; and let every downstream
decision — orientation included — consume those frequencies rather than the single-fit
statistics they replace. The knobs that matter are the resampling design and the frequency
bar, and both must be fixed once for all five regimes: a frequency threshold re-tuned per
setting is exactly the dataset-specific constant this benchmark forbids. Runtime is part of
the contract too, since the resampling loop must fit comfortably inside the same time budget
the single-fit baselines already use.

The defense rests on a trade this variant claims is favorable. Stability selection discards
real but fragile edges, costing some adjacency recall; in exchange, the edges that remain —
and the orientations built on them — hold their precision in the regimes where single-fit
methods scatter. If the frequency filter does not visibly stabilize the noisy and
sample-starved settings relative to its own base learner, the added machinery has not paid
for itself.
