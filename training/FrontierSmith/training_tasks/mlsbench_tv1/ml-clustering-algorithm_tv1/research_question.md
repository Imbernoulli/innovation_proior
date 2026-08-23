A partition can look excellent by one yardstick and worthless by another: a
method that recovers the ground-truth grouping (high ARI/NMI) often produces
geometrically diffuse clusters (near-zero or negative silhouette) in high
dimension, while a method tuned for compactness carves convex chunks out of
shapes that no label would agree with. This task is about that disagreement,
not about raising an average.

Your objective is **worst-geometry robustness under a single fixed
configuration**: design one clusterer whose *weakest* evaluated setting is as
strong as possible, where a setting counts as strong only when the extrinsic
agreement and the intrinsic geometry agree. Concretely, treat the per-setting
result as limited by its weakest component, and treat any setting where
label-agreement is high but compactness collapses (or vice-versa) as an
unsolved setting, because that gap is the symptom of a representation the
algorithm never adapted to.

Constraints that define the variant:

- **One configuration, all inputs.** The same object, with the same
  hyperparameters, must handle every input it is given. Branching on shapes,
  dimensionality thresholds, or any dataset fingerprint to select a different
  algorithm is out of scope — adaptation must come from a statistic the method
  computes about the data, and that statistic must be used continuously, not as
  an if/else switch between named methods.
- **Earn the geometry.** If your assignments only score well extrinsically,
  change the space you measure distance in (learned metric, local scaling,
  affinity re-weighting, an embedding you fit yourself) rather than post-hoc
  relabelling. The `custom_distance` hook exists for exactly this.
- **No free `k`.** `n_clusters` may be supplied as metadata but a method that
  is only correct when it is supplied is a weak answer; the interesting
  algorithm degrades gracefully when the count is wrong or absent.

The claim to defend at the end is not "my average went up" but "here is the
mechanism that stopped the weakest geometry from being weak, and here is why it
did not cost me the others".
