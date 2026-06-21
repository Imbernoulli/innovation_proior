The method is Grothendieck's inequality. The analysis starts from a simple but puzzling comparison. Given a real matrix A with entries a_ij, the sign optimization problem asks for the largest value of the bilinear form sum a_ij epsilon_i delta_j when every epsilon_i and delta_j is restricted to {-1, 1}. This is a hard combinatorial problem: the number of assignments grows exponentially, and the optimum measures how strongly the matrix can correlate two independent sign vectors. The natural relaxation replaces each sign by a unit vector in a Hilbert space and replaces the product epsilon_i delta_j by the inner product <x_i, y_j>. Because one-dimensional unit vectors are exactly signs, every feasible sign assignment is also a feasible vector assignment, so the vector optimum is always at least the sign optimum. The question is whether the vector optimum can be arbitrarily larger. A generic relaxation often blows up; vectors can encode many pairwise correlations at once, and a positive semidefinite Gram matrix has far more degrees of freedom than two sign vectors. Existing ideas do not close this gap. Exact enumeration over signs is infeasible for large instances and gives no convex certificate. Replacing signs by scalars in [-1, 1] merely reproduces the original bilinear problem at the corners, so it does not expose the Hilbert geometry that makes semidefinite programming useful. Spectral or Euclidean matrix norms are easy to compute, but without an extra structural theorem they do not certify a constant-factor approximation to the sign optimum. A generic semidefinite relaxation can have an unbounded integrality gap, and dimension-dependent rounding arguments would not explain why the comparison stays uniform across all matrix sizes.

Grothendieck's inequality is the statement that the vector relaxation is not arbitrarily loose. For every real m-by-n matrix A, if OPT_sign(A) denotes the sign optimum and OPT_vec(A) denotes the vector optimum, then there exists a universal constant K_G^R, independent of m, n, and the entries of A, such that OPT_sign(A) <= OPT_vec(A) <= K_G^R OPT_sign(A). The same phenomenon holds over the complex numbers with a generally different constant. The inequality does not claim that signs and vectors are equivalent; it says that a bilinear form can exploit Hilbert-space correlations only by a fixed multiplicative factor beyond what it can already achieve on scalar signs. That dimension-free bound is the key insight: the cube geometry of ell_infty variables and the round geometry of Hilbert-space unit vectors are universally coupled through bilinear forms.

The theorem works in several languages at once. In functional analysis, it says that a bounded bilinear form on ell_infty spaces remains bounded, up to K_G, when scalar variables are replaced by vectors and multiplication is replaced by inner product. This is why Grothendieck originally formulated it through tensor products and factorization through Hilbert space. In Banach-space geometry, it compares the extremal geometry of cubes with the smooth geometry of spheres: a priori the two spaces could behave very differently, but bilinear forms cannot separate them by more than a fixed factor in this setting. In approximation algorithms, the vector optimum is exactly the semidefinite-programming relaxation obtained by writing a Gram matrix of inner products and imposing unit diagonal constraints. Grothendieck's inequality turns that SDP value into a constant-factor certificate for the original discrete sign optimum, which is the mechanism behind applications such as cut-norm approximation. The constant-gap guarantee is what lifts the result from a special rounding trick to a structural theorem.

The practical message is that the vector program is a reliable surrogate for the sign program. Whenever a combinatorial objective can be cast as a bilinear optimization over signs, solving or bounding the SDP vector relaxation gives a value that is at most a universal constant away from the true discrete optimum. Constructive proofs of the inequality also supply rounding schemes that convert near-optimal vectors back into near-optimal signs, making the bridge algorithmic. The precise value of the best real constant K_G^R is a separate and difficult problem, but the existence of such a universal constant is the main content of the theorem.

```python
import itertools
import numpy as np
import cvxpy as cp


def sign_optimum(A):
    """Exact sign optimum by exhaustive search (only for small matrices)."""
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    best = -np.inf
    best_pair = None
    for eps in itertools.product([-1, 1], repeat=m):
        eps = np.array(eps, dtype=float)
        # For fixed eps, the best delta_j is the sign of (A^T eps)_j.
        v = A.T @ eps
        delta = np.sign(v)
        delta[delta == 0] = 1
        val = eps @ A @ delta
        if val > best:
            best = val
            best_pair = (eps.copy(), delta.copy())
    return best, best_pair


def vector_optimum_sdp(A):
    """SDP relaxation for the vector optimum via a Gram matrix."""
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    # Gram matrix for m x-vectors and n y-vectors on the unit sphere.
    G = cp.Variable((m + n, m + n), symmetric=True)
    constraints = [G >> 0, cp.diag(G) == 1]
    XY = G[:m, m:]
    objective = cp.sum(cp.multiply(A, XY))
    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    return prob.value, G.value


def empirical_grothendieck_ratio(A):
    opt_sign, _ = sign_optimum(A)
    opt_vec, _ = vector_optimum_sdp(A)
    if opt_sign <= 0:
        return np.inf
    return opt_vec / opt_sign


if __name__ == "__main__":
    np.random.seed(0)
    A = np.random.randn(6, 5)
    opt_sign, _ = sign_optimum(A)
    opt_vec, _ = vector_optimum_sdp(A)
    ratio = empirical_grothendieck_ratio(A)
    print(f"OPT_sign = {opt_sign:.4f}")
    print(f"OPT_vec  = {opt_vec:.4f}")
    print(f"Empirical OPT_vec / OPT_sign = {ratio:.4f}")
```
