I want the cheapest way to reshuffle mass from one probability distribution into another. The canonical formulation is the Monge–Kantorovich optimal transport problem, and the move that makes it tractable is Kantorovich's relaxation from transport maps to transport plans. Suppose I have two probability measures, mu on a source space X and nu on a target space Y, and a cost function c(x, y) that tells me the price of moving a unit of mass from x to y. Monge asked for a map T pushing mu forward to nu, T_# mu = nu, that minimizes the total cost integral of c(x, T(x)) with respect to mu. The difficulty is that such a map may not exist: if mu is a Dirac mass and nu is uniform, no deterministic map can split the single source point. Kantorovich solved this by allowing mass at one source location to divide and spread over many targets. Instead of a map T, he optimizes over couplings, joint probability measures pi on X times Y whose first marginal is mu and second marginal is nu. The Kantorovich problem is to minimize the expected cost, the integral of c(x, y) with respect to pi, over all such couplings. This is a linear program, and because the feasible set of couplings is convex and weak-* compact under mild assumptions, an optimal coupling always exists.

In the finite discrete case, which is the one I usually compute, mu and nu become histograms a and b in the probability simplex over n and m bins, and the cost becomes a matrix C with entries c_ij. A coupling is then a matrix P with nonnegative entries, row sums equal to a, and column sums equal to b. The Kantorovich problem is the linear program minimize the sum over i and j of C_ij P_ij subject to P times the all-ones vector equaling a, P transpose times the all-ones vector equaling b, and every entry of P being nonnegative. This is the formulation I propose as the canonical method: Kantorovich optimal transport, or the Kantorovich relaxation of the Monge problem. Its importance is not just that it exists where Monge fails; it is that it exposes the structure of the problem through duality. The Kantorovich dual problem is to maximize over potentials f on X and g on Y the quantity the integral of f with respect to mu plus the integral of g with respect to nu, subject to the constraint f(x) + g(y) is at most c(x, y) for all x and y. In the discrete case this becomes maximize the sum over i of a_i f_i plus the sum over j of b_j g_j subject to f_i + g_j at most c_ij. Weak duality is immediate: for any feasible P and any feasible pair (f, g), the primal cost is at least the dual objective, because the sum over i and j of C_ij P_ij is at least the sum over i and j of (f_i + g_j) P_ij, which equals the sum over i of f_i a_i plus the sum over j of g_j b_j by the marginal constraints. Strong duality also holds under mild conditions, so the optimal primal and dual values coincide. This dual view is what makes optimal transport useful in statistics, economics, and machine learning: it turns a problem over high-dimensional couplings into a problem over scalar potentials, and it is the foundation of Wasserstein distances and their applications.

The Wasserstein distance of order p is simply the p-th root of the optimal Kantorovich cost when the ground cost is a metric raised to the p-th power. For p equal to one on a metric space, the dual simplifies further through the Kantorovich–Rubinstein theorem to a supremum over one-Lipschitz functions, which is why Wasserstein-1 distances appear so often in generative modeling and robust statistics. But even without specializing to metrics, the Kantorovich formulation gives a principled way to compare probability distributions that respects the geometry of the underlying space, unlike KL divergence or total variation which treat the bins as labels rather than locations.

To make this concrete, I solve a small discrete instance and check that the primal optimum matches the dual optimum. I take two histograms over four bins and a cost matrix built from the absolute differences of bin indices, so the cost respects a one-dimensional geometry. I solve the primal linear program with a standard LP solver, then solve the dual linear program, and verify that the two optimal values agree up to numerical tolerance. I also recover an optimal coupling and inspect where it places mass. This is not a production solver; it is a direct verification that the Kantorovich formulation and its dual are correctly stated and numerically consistent.

```python
import numpy as np
from scipy.optimize import linprog


def solve_kantorovich(a, b, C):
    """Solve the discrete Kantorovich optimal transport problem and its dual."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    C = np.asarray(C, dtype=float)
    n, m = C.shape
    assert np.allclose(a.sum(), 1.0) and np.allclose(b.sum(), 1.0)
    assert np.all(a >= 0) and np.all(b >= 0)

    # Primal: minimize <C, P> subject to row sums = a, column sums = b, P >= 0.
    c = C.ravel()
    A_eq = np.zeros((n + m, n * m))
    for i in range(n):
        A_eq[i, i * m:(i + 1) * m] = 1.0
    for j in range(m):
        A_eq[n + j, j::m] = 1.0
    b_eq = np.concatenate([a, b])
    res_primal = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    P_opt = res_primal.x.reshape(n, m)
    primal_value = res_primal.fun

    # Dual: maximize a^T f + b^T g subject to f_i + g_j <= C_ij.
    # linprog minimizes, so minimize -a^T f - b^T g.
    c_dual = np.concatenate([-a, -b])
    A_dual = np.zeros((n * m, n + m))
    for i in range(n):
        for j in range(m):
            A_dual[i * m + j, i] = 1.0
            A_dual[i * m + j, n + j] = 1.0
    b_dual = C.ravel()
    res_dual = linprog(
        c_dual, A_ub=A_dual, b_ub=b_dual, bounds=(None, None), method="highs"
    )
    dual_value = -res_dual.fun
    f = res_dual.x[:n]
    g = res_dual.x[n:]

    return P_opt, primal_value, f, g, dual_value


if __name__ == "__main__":
    a = np.array([0.4, 0.3, 0.2, 0.1])
    b = np.array([0.1, 0.2, 0.3, 0.4])
    C = np.abs(
        np.arange(4)[:, None] - np.arange(4)[None, :]
    ).astype(float)

    P_opt, primal_value, f, g, dual_value = solve_kantorovich(a, b, C)
    print(f"Primal optimum: {primal_value:.6f}")
    print(f"Dual optimum:   {dual_value:.6f}")
    print(f"Duality gap:    {abs(primal_value - dual_value):.2e}")
    print("Optimal coupling P:")
    print(np.round(P_opt, 6))
    print("Dual potentials f, g:")
    print(np.round(f, 6), np.round(g, 6))

    # Spot-check: active cells satisfy f_i + g_j == C_ij up to tolerance.
    complement = C - (f[:, None] + g[None, :])
    active = P_opt > 1e-6
    max_violation = np.max(np.abs(complement[active]))
    print(f"Max complementary slackness violation: {max_violation:.2e}")
```

Running this script on the small four-bin example confirms that the primal and dual objectives coincide to within numerical precision, and that the optimal coupling only places mass on cells where the dual constraint is tight, exactly as complementary slackness predicts. The Kantorovich formulation is therefore not an approximation; it is the correct convex relaxation of the Monge problem, and its dual provides the certificates that make optimal transport a practical tool for comparing distributions.
