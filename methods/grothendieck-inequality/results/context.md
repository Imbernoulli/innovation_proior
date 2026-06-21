## Research question

Grothendieck's inequality asks how much is lost when a bilinear optimization problem over signs is relaxed to one over unit vectors. Given a real matrix `A=(a_ij)`, the discrete problem is

`OPT_sign(A)=max_{epsilon_i,delta_j in {-1,1}} sum_{i,j} a_ij epsilon_i delta_j`.

The relaxed problem replaces each sign by a unit vector and each product by an inner product:

`OPT_vec(A)=max_{||x_i||=||y_j||=1} sum_{i,j} a_ij <x_i,y_j>`.

The relaxation is always at least as large, because signs are one-dimensional unit vectors. The question is whether it can be arbitrarily larger. Grothendieck's insight is that it cannot: there is a universal constant `K_G`, independent of the matrix size and entries, such that `OPT_vec(A) <= K_G OPT_sign(A)`.

## Background

The statement sits at the meeting point of bilinear forms, Banach spaces, and Hilbert space geometry. The sign problem is the norm of a bilinear form tested on extreme points of `ell_infty` balls. The vector problem tests the same coefficients after embedding the variables into Hilbert space, where correlations can be represented by inner products.

In Banach-space language, the theorem compares norms that look very different: one coming from the geometry of `ell_infty` and `ell_1`, and another coming from factorization through Hilbert space. Grothendieck's original setting used tensor products of Banach spaces; later elementary formulations made the finite matrix inequality visible.

Algorithmically, the vector relaxation is the natural semidefinite-programming relaxation of the sign problem. A Gram matrix records all inner products among the vectors, the unit-vector constraints become diagonal constraints, and positive semidefiniteness captures exactly the feasibility of such inner products.

## Baselines

- **Exact sign search.** Enumerating all choices of two sign vectors directly solves the discrete bilinear problem. Gap: it is exponential in the number of variables and gives no useful convex relaxation.

- **Naive continuous relaxation.** Replacing signs by real numbers in `[-1,1]` does not expose the Hilbert geometry that makes SDP methods powerful. For a bilinear form, extrema still occur at signs, so this relaxation is not the useful one.

- **Spectral or Euclidean bounds.** Matrix norms based on Euclidean vectors are easy to compute or approximate, but without Grothendieck's inequality they need not certify a constant-factor approximation to the sign optimum.

- **Generic SDP relaxation.** The vector program is efficiently approximable as an SDP and upper-bounds the sign optimum. Gap: an arbitrary SDP relaxation can have a large integrality gap; the theorem says this particular relaxation has a universal constant gap.

- **Dimension-dependent rounding.** Many geometric rounding arguments degrade with dimension or problem size. Gap: Grothendieck's constant is universal, so the comparison is stable across all finite matrix sizes.

## Evaluation settings

The core finite setting is a matrix `A` and the comparison between `OPT_sign(A)` and `OPT_vec(A)`. Success means a dimension-free inequality with a universal multiplicative constant. The precise value of the best real or complex constant is a separate, difficult problem; the conceptual contribution is the existence of such a constant.

The functional-analytic setting views `A` as a bilinear form on finite-dimensional `ell_infty` spaces and asks whether boundedness on scalar signs controls boundedness after Hilbert-space substitution. This is a question about how Banach-space geometry behaves under tensor norms and Hilbertian factorization.

The approximation setting uses the vector optimum as an SDP upper bound and then interprets Grothendieck's inequality as a constant-factor certificate for the discrete optimum. Cut-norm estimation is a canonical example: an apparently combinatorial matrix norm can be approximated through the SDP relaxation because the relaxation cannot exceed the sign optimum by more than `K_G`.

## Proof artifact

The final artifact should state the real finite-dimensional inequality, explain the two optima, and emphasize the constant-gap claim:

`OPT_sign(A) <= OPT_vec(A) <= K_G^R OPT_sign(A)`.

It should then explain the bridge. In functional analysis, the theorem says that testing a bilinear form on signs already controls its Hilbert-space amplification up to a universal constant. In Banach-space geometry, it identifies a robust relation between `ell_infty`/`ell_1` behavior and Hilbertian factorization. In approximation algorithms, it says that the SDP vector relaxation is not merely a heuristic upper bound; it is a constant-factor surrogate for the hard sign optimization.

The proof details need not be reconstructed here. The important insight is the invariant comparison: replacing scalar signs by vectors greatly enlarges the feasible set, but the bilinear objective can exploit that enlargement only by a universal factor.
