## The Embedding Bottleneck

The Johnson-Lindenstrauss lemma gives a powerful promise: a finite set of high-dimensional points can move into far fewer Euclidean coordinates while nearly preserving every distance. Dense Gaussian or Rademacher projections make this promise routine. For a fixed vector, concentration of the projected norm gives a failure probability, and a union bound over pairwise differences handles a data set.

The algorithmic cost is the pressure point. A dense target matrix with `k = Theta(epsilon^-2 log(1/delta))` rows touches every output coordinate for every nonzero input coordinate. On a sparse input vector, the multiplication cost is still `O(k ||x||_0)`, and the output is immediately dense.

## What Dense Randomness Already Solves

The classical proof already explains the right target dimension. It does not depend on a special structure of the input set; it only needs a distribution whose rows preserve the norm of any fixed vector with a sharp tail. Discrete sign matrices can replace Gaussians, and simple database-friendly distributions can even put zeros in many entries without changing the order of the target dimension.

Those simplifications reduce constants and random bits, but they do not change the central update-time scaling. A projection with a constant fraction of nonzero entries is still dense at the level of sparse-vector multiplication.

## Why Naive Sparsity Is Dangerous

Sparse matrices introduce a new failure mode that dense concentration hides. A sparse input coordinate may land in only a few output rows, so its own contribution can fluctuate. Two large input coordinates may collide in the same output rows, and the resulting cross terms can change the squared norm.

Fast transform methods address a related problem by first spreading mass with a random sign transform and a Hadamard or Fourier matrix, then sampling. That preconditioning makes later sparsification safer for dense vectors, but it requires a global pass over the ambient dimension. For streaming and sparse-update settings, paying roughly `d log d` before exploiting sparsity defeats the point.

## The Hashing Temptation

Signed hashing looks like the natural primitive. It updates only a few counters per nonzero coordinate and had already powered streaming sketches. Repetition plus a median can give strong norm or frequency estimates.

The difficulty is that such an estimator is not automatically a Johnson-Lindenstrauss embedding. A median over sketches is nonlinear, while the geometric goal asks for one linear map followed by an ordinary Euclidean norm. Earlier sparse embedding work moved toward this goal with hashed copies and local densification, but extra logarithmic factors remained in the number of nonzeros per column.

## The Open Shape Of A Better Map

The desired object has to satisfy several constraints at once. It must keep the same target dimension order as dense Johnson-Lindenstrauss projections, apply to a vector in time proportional to its input sparsity, remain linear, use manageable randomness, and preserve norms for arbitrary fixed vectors rather than only for well-spread ones.

The hardest cases are not necessarily high-entropy vectors. A basis vector or a two-coordinate vector can expose diagonal fluctuation or collision error. Any successful construction has to make those cases harmless without giving up the concentration proof that makes the classical lemma work.
