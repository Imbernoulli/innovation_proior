How many groups are in this data? Every reference method dodges that
question in its own way: K-Means is told the answer outright, and the density
methods smuggle it in through radius and minimum-size knobs. This variant
makes the question itself the deliverable. The count metadata your
constructor receives must be treated as untrustworthy — possibly wrong,
possibly absent — and the fitted model has to commit to a number of clusters
it derived from the data alone.

What to optimize: recover partitions whose agreement with the hidden labels
survives the removal of the count oracle. ARI and NMI are the binding scores
here, because they collapse fastest when a model over- or under-splits;
silhouette still counts toward the total and serves as a sanity check that
the discovered count produced coherent shapes.

Rules that define the variant:
- Model selection must be one rule evaluated on every input — a criterion
  (stability under perturbation, gap statistic, eigengap, density-mode
  counting, information-theoretic penalty) scanned over candidate counts,
  never a table keyed on what the input looks like.
- The supplied hint may be recorded for reference, but no code path may use
  it to set the fitted number of groups, to seed centroids, or to prune the
  final labelling.
- Degrade gracefully under ambiguity: when the criterion cannot decide,
  prefer the coarser answer, since spurious splits poison label agreement
  faster than conservative merges do.

The deliverable to defend: a stated selection rule, the count it chose on
each input geometry, and evidence from the scored metrics that self-chosen
counts cost little compared to being handed the truth.
