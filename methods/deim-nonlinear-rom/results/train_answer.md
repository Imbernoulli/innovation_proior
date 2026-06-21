A discretized nonlinear PDE typically yields a large ODE system dy/dt = A y + F(y) in R^n, where n is the number of grid points and F is a nonlinearity. Projection-based reduced-order models replace the state by a low-rank POD basis V_k so that y ≈ V_k ỹ and the linear term reduces to a cheap k×k matrix. The trouble is the nonlinear term: evaluating V_k^T F(V_k ỹ) each step still requires lifting to R^n, evaluating F at all n components, and projecting back, so the per-step cost remains O(n). Plain POD-Galerkin therefore reduces the number of variables without reducing the cost of a step, and for sparse A the reduced model can even be slower than the full one. The empirical interpolation method handles non-affine parametrized functions in continuous space but does not directly address the discrete reduced-order setting with a POD basis and SVD-style error analysis, while gappy POD reconstructs from partial samples without prescribing which indices to keep or targeting cheap nonlinear-term evaluation inside dynamics.

The method that solves this is the Discrete Empirical Interpolation Method (DEIM). It builds a second POD basis U = [u_1, ..., u_m] from snapshots of the nonlinearity F(y), approximating f := F(V_k ỹ) ≈ U c. Rather than computing the optimal orthogonal coefficients c = U^T f, which would again need all n entries of f, DEIM fixes the m unknowns in c by demanding the approximation be exact at m selected rows ℘_1, ..., ℘_m. With P = [e_{℘_1}, ..., e_{℘_m}], this gives P^T f = (P^T U) c and the interpolant f̂ = U (P^T U)^{-1} P^T f. For componentwise F, the key identity P^T F(V_k ỹ) = F(P^T V_k ỹ) means we never lift to R^n: we evaluate F at only those m rows. The reduced system becomes dỹ/dt = Ã ỹ + B F(V_℘ ỹ), with Ã = V_k^T A V_k, B = V_k^T U (P^T U)^{-1}, and V_℘ = P^T V_k, all precomputed offline. Each step now costs O(α(m) + 4 m k), independent of n. The interpolation error is bounded by ||f − f̂||_2 ≤ ||(P^T U)^{-1}||_2 · ||(I − U U^T) f||_2, so the cheap interpolant is nearly as accurate as the expensive orthogonal projection as long as the selected rows keep P^T U well-conditioned.

The row selection is therefore crucial. DEIM chooses indices greedily: seed at the largest-magnitude entry of u_1, then at each step solve (P^T U) c = P^T u_ℓ for the coefficients that interpolate mode u_ℓ through the previous modes at the chosen rows, form the residual r = u_ℓ − U c, and place the next index where |r| is largest. This maximizes the residual entry that governs the growth of ||(P^T U)^{-1}||_2, keeps indices distinct and hierarchical, and guarantees nonsingularity as long as the modes are linearly independent. Modes are fed in descending singular-value order, so the most energetic nonlinear structures are resolved first and the discarded singular value σ_{m+1} estimates the unavoidable projection error.

```python
import numpy as np


def pod_basis(S, r):
    """Leading r left singular vectors of snapshot matrix S (n x n_s)."""
    U, sv, _ = np.linalg.svd(S, full_matrices=False)
    return U[:, :r], sv


def deim_indices(U):
    """Greedy DEIM point selection."""
    n, m = U.shape
    P = np.empty(m, dtype=int)
    P[0] = np.argmax(np.abs(U[:, 0]))
    for ell in range(1, m):
        Ue, Pe = U[:, :ell], P[:ell]
        c = np.linalg.solve(Ue[Pe, :], U[Pe, ell])
        r = U[:, ell] - Ue @ c
        P[ell] = np.argmax(np.abs(r))
    return P


# Full-order model: 1-D cubic reaction-diffusion, dy/dt = A y + F(y)
n = 1024
x = np.linspace(0.0, 1.0, n)
h = x[1] - x[0]
eps = 0.015
main, off = -2.0 * np.ones(n), np.ones(n - 1)
Lap = (np.diag(main) + np.diag(off, 1) + np.diag(off, -1)) / h**2
Lap[0, 1] = 2.0 / h**2
Lap[-1, -2] = 2.0 / h**2
A = eps**2 * Lap


def F(v, t=0.0):
    return v * (v - 0.1) * (1.0 - v) + 0.05


dt, nt = 1e-5, 4000
y0 = np.zeros(n)
y0[:50] = 0.5

# Collect state and nonlinear snapshots
yy, SY, SF = y0.copy(), [], []
for j in range(nt):
    yy = yy + dt * (A @ yy + F(yy))
    if j % 20 == 0:
        SY.append(yy.copy())
        SF.append(F(yy).copy())
full = yy.copy()
SY, SF = np.array(SY).T, np.array(SF).T

# Offline assembly: two POD bases, DEIM points, n-independent operators
k, m = 10, 12
Vk, _ = pod_basis(SY, k)
U, svf = pod_basis(SF, m)
P = deim_indices(U)
At = Vk.T @ A @ Vk
B = Vk.T @ U @ np.linalg.inv(U[P, :])
Vp = Vk[P, :]

# Online integration: POD-DEIM touches only m rows per step
yd = Vk.T @ y0
for j in range(nt):
    yd = yd + dt * (At @ yd + B @ F(Vp @ yd))

error = np.linalg.norm(Vk @ yd - full) / np.linalg.norm(full)
magn = np.linalg.norm(np.linalg.inv(U[P, :]), 2)
print(f"n={n} k={k} m={m}  DEIM points: {P}")
print(f"||(P^T U)^-1||_2 = {magn:.3g}  sigma_(m+1)/sigma_1 = {svf[m]/svf[0]:.2e}")
print(f"POD-DEIM rel. error = {error:.3e}  (F at m={m} rows/step)")
```
