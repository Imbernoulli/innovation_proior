We are given a convex problem whose objective separates into two blocks tied together by a single linear constraint: minimize f(x) + g(z) subject to Ax + Bz = c. The difficulty is that the two natural requirements pull in opposite directions. We want robustness, meaning convergence even when f or g is nonsmooth, affine, or takes the value plus infinity so that constraints can be folded in as indicators, with no need for strict convexity or full rank of A or B. We also want decomposition, because the data and variables are large and distributed, so each iteration should split into independent pieces with only a cheap gather or broadcast between them.

The classical tools each satisfy only one of these demands. Dual decomposition is beautifully parallel: when f is separable, the x-minimization shatters into independent subproblems solved on separate workers, tied only by a shared price vector. But its x-update is the unconstrained minimization of f(x) plus a linear term, which is unbounded below for affine or merely convex f, so it is fragile. The method of multipliers, or augmented Lagrangian, fixes this by adding a quadratic penalty on the constraint residual. That makes the inner minimizations well posed and yields convergence under weak assumptions. However, the penalty introduces cross terms that couple the blocks, so the joint x-minimization can no longer be split across processors. Robustness and decomposition are incompatible unless we change the algorithmic shape.

The method that resolves this tension is the Alternating Direction Method of Multipliers, or ADMM. Instead of minimizing the augmented Lagrangian jointly over (x, z), ADMM takes a single Gauss-Seidel sweep: minimize over x with z fixed, then minimize over z with the new x fixed, then update the dual variable. Freezing one block turns the bilinear cross term in the penalty into a linear term in the other block, so each subproblem is self-contained. The alternation keeps the robustness bought by the quadratic penalty while restoring the per-block decomposition that the joint minimization had destroyed. Abstractly, ADMM is Douglas-Rachford splitting applied to the dual problem, which is itself an instance of Rockafellar's proximal point algorithm; this is why the strong convergence theory carries over.

Form the augmented Lagrangian L_rho(x, z, y) = f(x) + g(z) + y^T (Ax + Bz - c) + (rho/2) ||Ax + Bz - c||^2. In unscaled form, ADMM iterates x^{k+1} = argmin_x L_rho(x, z^k, y^k), then z^{k+1} = argmin_z L_rho(x^{k+1}, z, y^k), then y^{k+1} = y^k + rho (Ax^{k+1} + Bz^{k+1} - c). The dual step size is rho itself, not arbitrary: because z^{k+1} minimizes L_rho(x^{k+1}, z, y^k), this choice makes the second dual-feasibility condition, 0 in partial g(z^{k+1}) + B^T y^{k+1}, hold automatically at every iterate.

It is usually cleaner to work in scaled form. Complete the square in the residual r = Ax + Bz - c to get y^T r + (rho/2)||r||^2 = (rho/2)||r + u||^2 - (rho/2)||u||^2 with u = y / rho the scaled dual variable. Dropping the constant gives the scaled iteration: x^{k+1} = argmin_x { f(x) + (rho/2)||Ax + Bz^k - c + u^k||^2 }, z^{k+1} = argmin_z { g(z) + (rho/2)||Ax^{k+1} + Bz - c + u^k||^2 }, and u^{k+1} = u^k + Ax^{k+1} + Bz^{k+1} - c. Here u^k is the running sum of primal residuals, which is exactly the accumulated constraint violation that a price should encode.

The natural stopping criteria come from the optimality conditions. Primal feasibility requires r^{k+1} = Ax^{k+1} + Bz^{k+1} - c = 0. The first dual-feasibility condition fails by s^{k+1} = rho A^T B (z^{k+1} - z^k), which measures how much z moved. The second dual condition holds for free. A Lyapunov argument with V^k = (1/rho)||y^k - y*||^2 + rho ||B(z^k - z*)||^2 shows V^{k+1} <= V^k - rho||r^{k+1}||^2 - rho||B(z^{k+1} - z^k)||^2, so r^k and s^k converge to zero, the objective value converges to optimal, and y^k converges to a dual optimum. A practical stopping rule is ||r^k|| <= epsilon_pri and ||s^k|| <= epsilon_dual, with tolerances mixing absolute and relative terms scaled by the dimensions and magnitudes of the iterates.

The simplest concrete instance is the lasso: minimize (1/2)||Cx - b||^2 + lambda ||x||_1. Introduce a copy z and split via x - z = 0, so f(x) = (1/2)||Cx - b||^2 and g(z) = lambda ||z||_1. The x-update becomes a ridge regression, x^{k+1} = (C^T C + rho I)^{-1}(C^T b + rho(z^k - u^k)), where the matrix is factorized once and reused. The z-update is the prox of the l1 norm, which is soft-thresholding: z^{k+1} = S_{lambda/rho}(x^{k+1} + u^k) with S_kappa(a) = sign(a) max(|a| - kappa, 0). Then u^{k+1} = u^k + x^{k+1} - z^{k+1}. This turns a nonsmooth regularized problem into a sequence of well-conditioned linear solves and separable thresholdings.

The same pattern extends to distributed consensus. To minimize sum_i f_i(x) over a shared parameter, give each block a private copy x_i and enforce x_i = z for a global consensus variable. The x-updates decouple into independent prox-like fits on each worker, the z-update is just the average of the local copies plus their scaled duals, and the duals are updated locally. The only communication is a gather of the average and a broadcast back. Swap the regularizer or constraint and only the corresponding prox changes: an indicator becomes a projection, a group norm becomes block soft-thresholding.

```python
import numpy as np
from numpy.linalg import cholesky, norm


def shrinkage(a, kappa):
    # prox of kappa * ||.||_1 : soft-threshold
    return np.maximum(0.0, a - kappa) - np.maximum(0.0, -a - kappa)


def factor(C, rho):
    # Cache a Cholesky for the x-update; use the matrix-inversion-lemma form when short & fat.
    q, d = C.shape
    if q >= d:
        L = cholesky(C.T @ C + rho * np.eye(d))
    else:
        L = cholesky(np.eye(q) + (1.0 / rho) * (C @ C.T))
    return L, L.T.conj()


def lasso(C, b, lam, rho=1.0, alpha=1.0, n_iter=1000, abstol=1e-4, reltol=1e-2):
    # Solve min 1/2 ||C x - b||^2 + lam ||x||_1 via ADMM with the split x - z = 0.
    q, d = C.shape
    Ctb = C.T @ b
    L, U = factor(C, rho)
    x = np.zeros(d)
    z = np.zeros(d)
    u = np.zeros(d)  # scaled dual u = y / rho

    for k in range(n_iter):
        # x-update: ridge regression via cached factors.
        rhs = Ctb + rho * (z - u)
        if q >= d:
            x = np.linalg.solve(U, np.linalg.solve(L, rhs))
        else:
            tmp = np.linalg.solve(U, np.linalg.solve(L, C @ rhs))
            x = rhs / rho - (C.T @ tmp) / (rho ** 2)

        z_old = z
        x_hat = alpha * x + (1.0 - alpha) * z_old  # over-relaxation, alpha in [1, 2)
        z = shrinkage(x_hat + u, lam / rho)        # z-update: prox of l1 norm
        u = u + (x_hat - z)                        # scaled dual: running residual sum

        r_norm = norm(x - z)                       # primal residual
        s_norm = norm(-rho * (z - z_old))          # dual residual (A^T B = -I here)
        eps_pri = np.sqrt(d) * abstol + reltol * max(norm(x), norm(-z))
        eps_dual = np.sqrt(d) * abstol + reltol * norm(rho * u)
        if r_norm < eps_pri and s_norm < eps_dual:
            break

    return z


def solve_split_problem(x_update, z_update, A, B, c, n, m, p,
                        rho=1.0, n_iter=1000, abstol=1e-4, reltol=1e-2):
    # Generic scaled-form ADMM for min f(x) + g(z) s.t. A x + B z = c.
    # x_update(z, u, rho) returns argmin_x f(x) + (rho/2)||A x + B z - c + u||^2.
    # z_update(x, u, rho) returns argmin_z g(z) + (rho/2)||A x + B z - c + u||^2.
    x = np.zeros(n)
    z = np.zeros(m)
    u = np.zeros(p)

    for k in range(n_iter):
        z_old = z
        x = x_update(z, u, rho)
        z = z_update(x, u, rho)
        r = A @ x + B @ z - c               # primal residual
        u = u + r                           # scaled dual update
        s = rho * (A.T @ (B @ (z - z_old))) # dual residual

        eps_pri = np.sqrt(p) * abstol + reltol * max(norm(A @ x), norm(B @ z), norm(c))
        eps_dual = np.sqrt(n) * abstol + reltol * norm(rho * (A.T @ u))
        if norm(r) < eps_pri and norm(s) < eps_dual:
            break

    return x, z


def consensus(prox_fi, N, d, rho=1.0, n_iter=1000, abstol=1e-4, reltol=1e-2):
    # Solve min sum_i f_i(x) by global consensus: private copies x_i agree with global z.
    # prox_fi[i](v, rho) returns argmin_xi f_i(xi) + (rho/2)||xi - v||^2.
    X = np.zeros((N, d))
    z = np.zeros(d)
    U = np.zeros((N, d))

    for k in range(n_iter):
        z_old = z
        for i in range(N):                    # x-updates run independently / in parallel
            X[i] = prox_fi[i](z - U[i], rho)
        z = (X + U).mean(axis=0)              # z-update: gather and average
        U = U + X - z                         # local scaled dual updates

        r_norm = np.sqrt(((X - z) ** 2).sum())
        s_norm = np.sqrt(N) * rho * norm(z - z_old)
        eps_pri = np.sqrt(N * d) * abstol + reltol * max(norm(X), np.sqrt(N) * norm(z))
        eps_dual = np.sqrt(N * d) * abstol + reltol * norm(rho * U)
        if r_norm < eps_pri and s_norm < eps_dual:
            break

    return z, X
```
