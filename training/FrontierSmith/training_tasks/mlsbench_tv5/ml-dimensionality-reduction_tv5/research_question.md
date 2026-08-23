High-dimensional inputs arrive padded with coordinates that carry no
structure: border pixels blank in every image, trailing SVD directions
encoding residue rather than topics, dimensions that are effectively noise.
Neighbor computations in the raw space average over all of them, and each
irrelevant coordinate dilutes the distances a 2-D embedding is trying to
honor. This variant targets that dilution directly: the method must decide,
from the data alone, which directions of the input deserve to shape the map
— and mute the rest.

Optimize the standard scores by making the embedding operate on a cleaned
version of the geometry. Feature-relevance estimates, smoothness over a
neighborhood graph, spectral gaps, or any other unsupervised statistic may
drive a weighting or selection of input dimensions before or during the
embedding stage. The bet to validate: neighborhoods computed in the
structure-bearing subspace agree better with class structure than
neighborhoods computed raw, and that advantage shows up in the 2-D
classifier accuracy without sacrificing either fidelity score.

Variant boundaries:
- Relevance must be estimated, not assumed — no hardcoded pixel masks, no
  "keep the first m dimensions" constant tuned to a known dataset.
- The estimator must be harmless on clean inputs: when every dimension
  carries signal, the weighting should approach uniform rather than
  butchering informative coordinates.
- Everything runs unsupervised, under the usual per-dataset CPU allowance,
  with one recipe for all inputs.

Close by defending the cleanup with the scoreboard: which coordinates were
muted, why the surviving geometry is the right one, and how the reported
numbers reflect it.
