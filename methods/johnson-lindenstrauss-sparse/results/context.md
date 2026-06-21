## The Embedding Bottleneck

The Johnson-Lindenstrauss lemma gives a powerful promise: a finite set of high-dimensional points can move into far fewer Euclidean coordinates while nearly preserving every distance. Dense Gaussian or Rademacher projections make this promise routine. For a fixed vector, concentration of the projected norm gives a failure probability, and a union bound over pairwise differences handles a data set.

A dense target matrix with `k = Theta(epsilon^-2 log(1/delta))` rows touches every output coordinate for every nonzero input coordinate. On a sparse input vector, the multiplication cost is `O(k ||x||_0)`, and the output is immediately dense.

## What Dense Randomness Already Solves

The classical proof already explains the right target dimension. It does not depend on a special structure of the input set; it only needs a distribution whose rows preserve the norm of any fixed vector with a sharp tail. Discrete sign matrices can replace Gaussians, and simple database-friendly distributions can put zeros in many entries without changing the order of the target dimension.

Those simplifications reduce constants and random bits, but they do not change the central update-time scaling. A projection with a constant fraction of nonzero entries is still dense at the level of sparse-vector multiplication.

## Sparse Matrices and Concentration

Sparse matrices interact with concentration arguments differently from dense ones. A sparse input coordinate may land in only a few output rows, and two input coordinates may share rows, introducing cross terms in the squared norm. Understanding how these collision terms behave under random sign choices is central to the analysis.

Fast transform methods address a related problem by first spreading mass with a random sign transform and a Hadamard or Fourier matrix, then sampling. That preconditioning makes sparsification tractable for dense vectors and requires a global pass over the ambient dimension of cost roughly `d log d`.

## The Hashing Temptation

Signed hashing looks like the natural primitive. It updates only a few counters per nonzero coordinate and had already powered streaming sketches. Repetition plus a median can give strong norm or frequency estimates.

Such an estimator is not automatically a Johnson-Lindenstrauss embedding. A median over sketches is nonlinear, while the geometric goal asks for one linear map followed by an ordinary Euclidean norm. Earlier sparse embedding work moved toward this goal with hashed copies and local densification, and extra logarithmic factors remained in the number of nonzeros per column.

## The Setting

The question is how to construct a linear map from `R^d` to `R^k` that applies to a sparse input vector in time proportional to its input sparsity, while keeping the target dimension on the same order as dense Johnson-Lindenstrauss projections.
