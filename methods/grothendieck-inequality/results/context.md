## Research question

The question concerns how much a bilinear optimization problem over signs changes when it is relaxed to one over unit vectors. Given a real matrix `A=(a_ij)`, the discrete problem is

`OPT_sign(A)=max_{epsilon_i,delta_j in {-1,1}} sum_{i,j} a_ij epsilon_i delta_j`.

The relaxed problem replaces each sign by a unit vector and each product by an inner product:

`OPT_vec(A)=max_{||x_i||=||y_j||=1} sum_{i,j} a_ij <x_i,y_j>`.

The relaxation is always at least as large, because signs are one-dimensional unit vectors, so `OPT_sign(A) <= OPT_vec(A)`. The question is how `OPT_vec(A)` compares to `OPT_sign(A)` from above: by what factor can passing from signs to unit vectors enlarge the optimum, as the matrix size and entries vary.

## Background

The statement sits at the meeting point of bilinear forms, Banach spaces, and Hilbert space geometry. The sign problem is the norm of a bilinear form tested on extreme points of `ell_infty` balls. The vector problem tests the same coefficients after embedding the variables into Hilbert space, where correlations can be represented by inner products.

In Banach-space language, the two quantities are norms of different origin: one coming from the geometry of `ell_infty` and `ell_1`, and another coming from factorization through Hilbert space. Grothendieck's original setting used tensor products of Banach spaces; later elementary formulations made the finite matrix inequality visible.

Algorithmically, the vector relaxation is the natural semidefinite-programming relaxation of the sign problem. A Gram matrix records all inner products among the vectors, the unit-vector constraints become diagonal constraints, and positive semidefiniteness captures exactly the feasibility of such inner products.

## Baselines

- **Exact sign search.** Enumerating all choices of two sign vectors directly solves the discrete bilinear problem, and is exponential in the number of variables.

- **Naive continuous relaxation.** Replacing signs by real numbers in `[-1,1]` keeps the variables scalar. For a bilinear form, extrema still occur at signs.

- **Spectral or Euclidean bounds.** Matrix norms based on Euclidean vectors are easy to compute or approximate.

- **Generic SDP relaxation.** The vector program is efficiently approximable as an SDP and upper-bounds the sign optimum.

- **Dimension-dependent rounding.** Many geometric rounding arguments produce comparisons that depend on dimension or problem size.

## Evaluation settings

The core finite setting is a matrix `A` and the comparison between `OPT_sign(A)` and `OPT_vec(A)`, varying over all matrix sizes and entries. The precise value of the best real or complex constant in such a comparison is a separate, difficult problem.

The functional-analytic setting views `A` as a bilinear form on finite-dimensional `ell_infty` spaces and asks whether boundedness on scalar signs controls boundedness after Hilbert-space substitution. This is a question about how Banach-space geometry behaves under tensor norms and Hilbertian factorization.

The approximation setting uses the vector optimum as an SDP upper bound for the discrete optimum. Cut-norm estimation is a canonical example: an apparently combinatorial matrix norm is approached through the SDP relaxation, and the quality of that approach depends on how `OPT_vec(A)` relates to `OPT_sign(A)`.

## Proof artifact

The final artifact should state the relation between the two finite-dimensional optima, explain what each one measures, and quantify how the vector relaxation compares to the sign optimum:

`OPT_sign(A) <= OPT_vec(A)`.

It should then explain the bridge across the three settings. In functional analysis, the question is whether testing a bilinear form on signs controls its Hilbert-space amplification. In Banach-space geometry, it concerns the relation between `ell_infty`/`ell_1` behavior and Hilbertian factorization. In approximation algorithms, it concerns whether the SDP vector relaxation is a reliable surrogate for the hard sign optimization.

The comparison to make precise is what happens when scalar signs are replaced by vectors: this greatly enlarges the feasible set, and the goal is to quantify how much the bilinear objective can exploit that enlargement.
