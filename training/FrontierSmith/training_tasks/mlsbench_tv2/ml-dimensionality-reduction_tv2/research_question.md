A 2-D map earns its keep in this variant only through what a classifier can
do with it. Among the reported numbers, the 7-NN accuracy in the embedded
space is the one to maximize: after your method compresses raw pixel images
or SVD-compressed text down to two coordinates, points of the same class
must still be each other's nearest neighbors. Trustworthiness and
continuity remain on the scoreboard and should not be wrecked, but wherever
a design decision trades pure neighbor-set fidelity against class-structure
survival, this brief says take the class structure.

The hard part: no labels are visible when fit_transform runs. The method
therefore needs an unsupervised surrogate for "same class" — density modes,
cluster structure, mutual-neighbor consistency, anything computable from X
alone that correlates with the hidden partition — exploited so the 2-D
layout separates those surrogate groups widely enough for a 7-NN vote to
succeed at evaluation time.

Terms of the variant:
- Labels are off-limits at fit time in any form; pseudo-supervision must be
  manufactured from the input matrix itself.
- The same machinery faces every input; whatever adapts must adapt through
  quantities the method measures on X.
- Runtime stays within the pipeline's per-dataset CPU allowance.

Defend the surrogate: show that the structure your method chose to separate
is the structure the hidden labels reward, using the reported accuracy as
evidence and the two fidelity scores as guard rails against degenerate
layouts.
