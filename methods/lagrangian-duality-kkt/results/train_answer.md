I will explain the canonical method of Lagrangian duality and the Karush–Kuhn–Tucker (KKT) conditions for inequality- and equality-constrained nonlinear optimization. The problem I have in mind is the standard form minimize f(x) subject to g_i(x) ≤ 0 for i = 1,…,m and h_j(x) = 0 for j = 1,…,p, where f, g_i, and h_j are differentiable functions on R^n and the optimal value is denoted p*. The goal is to certify optimality and to bound the optimum from below without necessarily solving the primal problem to completion.

The central device is the Lagrangian L(x, λ, ν) = f(x) + Σ_i λ_i g_i(x) + Σ_j ν_j h_j(x). I think of the constraints as infinitely hard walls: the inequality wall is zero when g_i(x) ≤ 0 and plus infinity otherwise, while the equality wall is zero when h_j(x) = 0 and plus infinity otherwise. The Lagrangian replaces these walls by linear under-estimators. For an inequality wall, only a line with nonnegative slope can sit below the wall everywhere, because on the feasible side g_i ≤ 0 the wall is zero and a negative slope would poke above it; this forces λ_i ≥ 0. For an equality wall, any slope works because the wall is zero only at h_j = 0 and plus infinity elsewhere, so ν_j is unrestricted in sign. This asymmetry between λ and ν is the first key fact: inequality constraints are one-sided and carry sign-restricted multipliers, whereas equality constraints are two-sided and carry free multipliers.

From this under-estimation picture comes weak duality. Define the dual function d(λ, ν) = inf_x L(x, λ, ν). For any λ ≥ 0 and any ν, I claim d(λ, ν) ≤ p*. The proof is immediate: take any feasible point x̃. Then each g_i(x̃) ≤ 0 and each λ_i ≥ 0, so Σ_i λ_i g_i(x̃) ≤ 0, while each h_j(x̃) = 0, so Σ_j ν_j h_j(x̃) = 0 regardless of ν. Hence L(x̃, λ, ν) ≤ f(x̃). Since d(λ, ν) is the infimum over all x, it is at most L(x̃, λ, ν), and therefore at most f(x̃) for every feasible x̃; taking the best feasible x̃ gives d(λ, ν) ≤ p*. This means the dual problem, maximize d(λ, ν) subject to λ ≥ 0, provides a certified lower bound on the primal optimum. A useful extra fact is that d is concave because it is a pointwise infimum of affine functions of (λ, ν), so the dual problem is always convex even when the primal is not.

Weak duality is universal, but it is only a one-sided bound. The much stronger statement is strong duality: under suitable conditions, the best lower bound equals the primal optimum, d* = p*. To understand when this happens, I look at the set of achievable constraint and objective values A = {(u, v, t) : there exists x in the domain with g_i(x) ≤ u_i, Ax - b = v, f(x) ≤ t}. The point (0, 0, p*) lies on the boundary of A. The dual function value d(λ, ν) is the height at which the supporting hyperplane with slopes determined by (λ, ν) crosses the t-axis. Weak duality says this crossing is always at or below p*. Strong duality says there exists a supporting hyperplane of A that touches exactly at (0, 0, p*).

If f and the g_i are convex and the equality constraints are affine, then A is convex. The separating-hyperplane theorem then gives a hyperplane separating A from the vertical ray below (0, 0, p*). The only way this hyperplane can fail to give a useful finite dual value is if it is vertical, meaning its coefficient on t is zero. Slater's condition rules this out: it requires a strictly feasible point x̃ with g_i(x̃) < 0 for every nonlinear inequality and h_j(x̃) = 0 for every equality. Such a point sits strictly inside A in the u-direction, so any separating hyperplane must tilt and have a positive coefficient on t. After normalizing, this yields multipliers (λ*, ν*) with d(λ*, ν*) = p*. For linear programs, where every inequality is affine, Slater's condition reduces to mere feasibility, which is why LP duality is so clean.

When strong duality holds, the chain f(x*) = d(λ*, ν*) = inf_x L(x, λ*, ν*) ≤ L(x*, λ*, ν*) ≤ f(x*) collapses to equalities. This forces two things. First, x* minimizes L(·, λ*, ν*), so the gradient in x vanishes: ∇f(x*) + Σ_i λ*_i ∇g_i(x*) + Σ_j ν*_j ∇h_j(x*) = 0. This is stationarity. Second, the inequality L(x*, λ*, ν*) ≤ f(x*) becomes an equality, which means Σ_i λ*_i g_i(x*) = 0; since each term is nonpositive, every term must be zero, giving λ*_i g_i(x*) = 0. This is complementary slackness. Together with primal feasibility g_i(x*) ≤ 0 and h_j(x*) = 0 and dual feasibility λ* ≥ 0, these are the four KKT conditions.

The KKT conditions can also be read directly from the geometry of feasible directions without invoking duality. At a local optimum there can be no feasible descent direction d that decreases f to first order while keeping every active constraint satisfied to first order. The active inequality gradients define a cone of feasible directions, and Farkas's lemma converts the statement "no feasible descent direction" into the stationarity condition with nonnegative multipliers on the active constraints. Slack constraints drop out because they impose no first-order restriction, which is exactly complementary slackness. This derivation needs a constraint qualification, such as linear independence of the active constraint gradients or, in the convex case, Slater's condition, to ensure that every direction admitted by the linearized cone is actually tangent to a feasible arc. Without this, the conditions can fail, as at the cusp (1 - x_1)^3 - x_2 ≥ 0 with x_1, x_2 ≥ 0 at the point (1, 0).

For convex problems, the KKT conditions are not only necessary under strong duality but also sufficient. If f and the g_i are convex and the h_j are affine, and if (x̃, λ̃, ν̃) satisfy the KKT conditions, then L(·, λ̃, ν̃) is convex because it is a nonnegative combination of convex functions plus an affine term. Stationarity then says x̃ is a global minimizer of this Lagrangian, and complementary slackness plus primal feasibility give d(λ̃, ν̃) = L(x̃, λ̃, ν̃) = f(x̃). Weak duality squeezes everything equal, so x̃ is primal optimal and (λ̃, ν̃) is dual optimal. Thus, for a convex problem satisfying Slater's condition, KKT is necessary and sufficient for optimality.

There is also an elegant saddle-point interpretation. The primal problem is inf_x sup_{λ ≥ 0, ν} L(x, λ, ν), because for fixed x the supremum over the multipliers is f(x) when x is feasible and plus infinity otherwise. The dual problem is sup_{λ ≥ 0, ν} inf_x L(x, λ, ν). Weak duality is the general max-min inequality sup inf ≤ inf sup. Strong duality says these two values coincide and are attained at a saddle point (x*, λ*, ν*) satisfying L(x*, λ, ν) ≤ L(x*, λ*, ν*) ≤ L(x, λ*, ν*) for all x, λ ≥ 0, and ν. This generalizes the bilinear saddle point of linear programming duality.

The following Python script illustrates the method on a small convex quadratic program with both inequality and equality constraints. It solves the KKT linear system directly and verifies that the resulting point is feasible, satisfies complementary slackness, and attains zero duality gap with the dual optimum. The example minimizes ½ x^T P x + q^T x subject to G x ≤ h and A x = b, with P positive definite, and solves the symmetric KKT system to recover both primal and dual variables.

```python
import numpy as np

# Primal: minimize 0.5 x^T P x + q^T x
# subject to G x <= h and A x = b.
P = np.array([[2.0, 0.5],
              [0.5, 1.0]])
q = np.array([-1.0, -2.0])
G = np.array([[1.0, 2.0],
              [-1.0, 0.0],
              [0.0, -1.0]])
h = np.array([3.0, 0.0, 0.0])
A = np.array([[1.0, 1.0]])
b = np.array([1.0])

# Set up the KKT system for the active-set solution.
# Here the inequality x_1 <= 0 (row 1 of G) is active at the optimum.
G_active = G[[1], :]
h_active = h[[1]]
n = P.shape[0]
m = G_active.shape[0]
p = A.shape[0]

# Symmetric KKT matrix:
# [ P   G_a^T  A^T ] [ x    ]   [ -q ]
# [ G_a  0     0   ] [ lambda ] = [ h_a]
# [ A    0     0   ] [ nu     ]   [ b  ]
KKT = np.block([
    [P, G_active.T, A.T],
    [G_active, np.zeros((m, m)), np.zeros((m, p))],
    [A, np.zeros((p, m)), np.zeros((p, p))]
])
rhs = np.concatenate([-q, h_active, b])
sol = np.linalg.solve(KKT, rhs)
x_star = sol[:n]
lambda_active = sol[n:n + m]
nu_star = sol[n + m:]
lambda_star = np.zeros(G.shape[0])
lambda_star[1] = lambda_active[0]

primal_obj = 0.5 * x_star @ P @ x_star + q @ x_star
feas_ineq = G @ x_star - h
feas_eq = A @ x_star - b
# Dual value equals L(x*, lambda*, nu*) because x* minimizes the Lagrangian.
dual_obj = primal_obj + lambda_star @ feas_ineq + nu_star @ feas_eq

print("Primal variable x*:", x_star)
print("Inequality multipliers lambda*:", lambda_star)
print("Equality multiplier nu*:", nu_star)
print("Primal objective:", primal_obj)
print("Dual objective:", dual_obj)
print("Inequality residuals Gx - h:", feas_ineq)
print("Equality residual Ax - b:", feas_eq)
print("Complementary slackness lambda_i * g_i:", lambda_star * feas_ineq)
print("Stationarity residual:", P @ x_star + q + G.T @ lambda_star + A.T @ nu_star)
```

In summary, Lagrangian duality and the KKT conditions provide a unified framework for constrained optimization. The Lagrangian folds constraints into the objective with multipliers that encode the geometry of one-sided inequality walls and two-sided equality walls. Weak duality gives a universal lower bound through the concave dual function, while strong duality holds for convex problems under Slater's condition and makes the bound tight. When strong duality holds, the KKT conditions emerge as the first-order fingerprint of optimality, and for convex problems they become necessary and sufficient.
