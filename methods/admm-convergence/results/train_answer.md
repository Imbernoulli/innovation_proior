The problem is to minimize f(x) + g(z) subject to Ax + Bz = c, where f and g are closed, proper, and convex, possibly nonsmooth or even infinite-valued, and the matrices A and B need not have full rank. The two classical routes each give only one of the two properties we want. Dual ascent, or dual decomposition, separates the x- and z-updates when f or g is a sum, but it breaks as soon as f is affine in a coordinate, because the inner Lagrangian is then unbounded below, and it relies on a delicate dual stepsize when the dual is nondifferentiable. The method of multipliers repairs this by adding a quadratic penalty to the Lagrangian, which makes the inner problem well posed and lets a dual step of size rho keep every iterate dual feasible. That gives robustness, but the penalty's cross term couples x and z, so the joint x-minimization can no longer be split across blocks or machines.

The method that captures both properties is the Alternating Direction Method of Multipliers, ADMM. Instead of minimizing the augmented Lagrangian jointly, ADMM takes a single Gauss-Seidel sweep: minimize over x with z and the multiplier y fixed, then minimize over z with the newly computed x and y fixed, and finally update the multiplier by the constraint residual scaled by rho. In the scaled form, with u = y / rho, the iterations are x^{k+1} in argmin_x ( f(x) + (rho/2)||Ax + Bz^k - c + u^k||^2 ), z^{k+1} in argmin_z ( g(z) + (rho/2)||Ax^{k+1} + Bz - c + u^k||^2 ), and u^{k+1} = u^k + Ax^{k+1} + Bz^{k+1} - c. Freezing one block turns the penalty's cross term into a constant in the other block, so each subproblem is a single-block penalized minimization that preserves separability. The z-minimization followed immediately by the dual update of size rho enforces the z-block dual-feasibility condition at every iteration, so convergence reduces to driving the primal residual r^{k+1} = Ax^{k+1} + Bz^{k+1} - c and the dual residual s^{k+1} = rho A^T B(z^{k+1} - z^k) to zero.

ADMM converges under exactly the assumptions one would hope for: f and g closed, proper, and convex, plus existence of a saddle point of the ordinary Lagrangian. There is no need for strict convexity, finite values everywhere, or full rank of A or B. The certificate is the Lyapunov function V^k = (1/rho)||y^k - y*||^2 + rho||B(z^k - z*)||^2, which decreases by rho(||r^{k+1}||^2 + ||B(z^{k+1} - z^k)||^2) each iteration; telescoping this inequality gives r^k -> 0, s^k -> 0, and f(x^k) + g(z^k) -> p* with a worst-case O(1/k) rate. The same conclusion can be read from the underlying structure: ADMM is Douglas-Rachford splitting applied to the dual problem, and Douglas-Rachford is the proximal-point algorithm on a maximal-monotone splitting operator with stepsize one, which explains why any positive rho works and why the Gauss-Seidel order factors the joint resolvent into two sequential block solves.

A practical stopping rule checks the two residuals against mixed absolute-relative tolerances. The primal tolerance scales with the square root of the constraint dimension p and the size of Ax, Bz, and c, while the dual tolerance scales with the square root of the x-dimension n and the size of A^T y. When both residuals are small, the suboptimality bound f(x^k) + g(z^k) - p* <= ||y^k|| ||r^k|| + d ||s^k|| guarantees that the objective is near optimal.

```python
import numpy as np


def admm(prox_f, prox_g, A, B, c, rho,
         x0, z0, y0, eps_abs=1e-6, eps_rel=1e-4, max_iter=10000):
    """
    minimize f(x) + g(z) subject to A x + B z = c.
    prox_f(z, y): argmin_x f(x) + (rho/2)*||A x + B z - c + y/rho||^2
    prox_g(x, y): argmin_z g(z) + (rho/2)*||A x + B z - c + y/rho||^2
    """
    x = np.array(x0, dtype=float)
    z = np.array(z0, dtype=float)
    y = np.array(y0, dtype=float)
    p, n = c.size, x.size
    for _ in range(max_iter):
        z_prev = z.copy()
        x = prox_f(z, y)
        z = prox_g(x, y)
        r = A @ x + B @ z - c
        y = y + rho * r
        s = rho * (A.T @ (B @ (z - z_prev)))
        eps_pri = np.sqrt(p) * eps_abs + eps_rel * max(
            np.linalg.norm(A @ x), np.linalg.norm(B @ z), np.linalg.norm(c))
        eps_dual = np.sqrt(n) * eps_abs + eps_rel * np.linalg.norm(A.T @ y)
        if np.linalg.norm(r) <= eps_pri and np.linalg.norm(s) <= eps_dual:
            break
    return x, z, y


# Example: Lasso  min 0.5*||C x - b||^2 + lam*||x||_1  via  x - z = 0.
if __name__ == "__main__":
    np.random.seed(0)
    n, d = 50, 20
    C = np.random.randn(n, d)
    x_true = np.random.randn(d)
    x_true[np.abs(x_true) < 0.8] = 0.0
    b = C @ x_true + 0.1 * np.random.randn(n)
    lam = 0.5

    rho = 1.0
    L = np.linalg.cholesky(C.T @ C + rho * np.eye(d))

    def prox_f(z, y):
        q = C.T @ b + rho * z - y
        return np.linalg.solve(L.T, np.linalg.solve(L, q))

    def soft_threshold(v, kappa):
        return np.sign(v) * np.maximum(np.abs(v) - kappa, 0.0)

    def prox_g(x, y):
        return soft_threshold(x + y / rho, lam / rho)

    A = np.eye(d)
    B = -np.eye(d)
    c = np.zeros(d)
    x, z, y = admm(prox_f, prox_g, A, B, c, rho,
                   np.zeros(d), np.zeros(d), np.zeros(d))
    obj = 0.5 * np.linalg.norm(C @ z - b)**2 + lam * np.linalg.norm(z, 1)
    print("objective:", obj)
```
