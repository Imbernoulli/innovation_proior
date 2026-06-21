The problem is to prove existence of a canonical Kähler metric with prescribed Ricci curvature in a fixed Kähler class. On a compact Kähler manifold, prescribing the Ricci form directly is a tensorial existence question: one must find a whole metric satisfying a curvature constraint while staying inside a given cohomology class. Attempting to solve for the metric components directly is overdetermined and coordinate-dependent, so a different formulation is needed.

The first key insight is that within a fixed Kähler class, every metric can be written as a potential perturbation: $\omega_\phi = \omega + i\partial\bar\partial\phi$. This reduces the unknown from a tensor to a single scalar function $\phi$. The Ricci form has the matching structure $\operatorname{Ric}(\omega_\phi) = -i\partial\bar\partial \log\det(g_\phi)$, so prescribing Ricci curvature becomes the complex Monge-Ampère equation $(\omega + i\partial\bar\partial\phi)^n = e^F \omega^n$. Calabi had already obtained uniqueness and set up the continuity method, but the equation is fully nonlinear in the highest derivatives. The remaining obstacle is closedness along the continuity path: solutions might degenerate, lose ellipticity, or blow up unless they are controlled uniformly.

The method that overcomes this is the Calabi-Yau theorem, proved by Yau through a priori estimates for the complex Monge-Ampère equation combined with the continuity method. The proof strategy is to start from an easily solvable equation and deform it toward the target equation along a parameter $t\in[0,1]$. Nonemptiness of the solution set is trivial, openness follows from elliptic linearization, and closedness follows from a chain of uniform estimates. The $C^0$ estimate controls the potential, the second-order estimate controls the trace of the metric relative to the background metric, and higher-order estimates follow from elliptic regularity once uniform ellipticity is secured. These estimates depend only on the background geometry and prescribed data, not on the deformation parameter, so the solution family is precompact and a smooth limit exists at $t=1$.

The theorem is often stated for the case $c_1=0$, where it produces a unique Ricci-flat Kähler metric in each Kähler class. The deeper contribution is the bridge between geometry and analysis: complex geometry supplies the scalar potential formulation and the determinant equation, topology supplies the cohomological compatibility condition, and nonlinear elliptic PDE estimates supply compactness. The proof does not construct the metric explicitly; it shows that no analytic blow-up is possible along the deformation path, so existence follows from continuity.

The following code illustrates the core algorithmic idea on a simplified real-Monge-Ampère problem on the periodic square. It uses finite-difference Hessians and a continuation path, applying Newton's method at each continuation step and monitoring a discrete analogue of the a priori estimates.

```python
import numpy as np
from scipy.sparse import diags, kron, eye, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# Solve a real Monge-Ampere equation on the periodic torus by continuation:
#   det(D^2 phi + I) = e^f,   with phi having zero mean.
# Discretized with central finite differences.

N = 64
L = 2 * np.pi
h = L / N
x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')

# Right-hand side f with zero mean and compatible volume
f = 0.3 * (np.sin(X) * np.cos(Y) + np.sin(2 * X + Y))
f -= f.mean()
F = np.exp(f)

# 1D periodic Laplacian and gradient matrices
off = np.ones(N)
D2 = diags([off, -2 * off, off], [-1, 0, 1], shape=(N, N))
D2 += diags([off, off], [N - 1, -(N - 1)])
D2 = D2 / (h ** 2)

D1 = diags([-off, off], [-1, 1], shape=(N, N)) / (2 * h)
D1 += diags([-off, off], [N - 1, -(N - 1)]) / (2 * h)

I_N = eye(N)
Lap = kron(I_N, D2) + kron(D2, I_N)
Dx = kron(I_N, D1)
Dy = kron(D1, I_N)

# Mean-zero projection
vec = np.ones(N * N)
P = eye(N * N) - csr_matrix(np.outer(vec, vec) / (N * N))
Lap = P @ Lap @ P

def hessian(u):
    uxx = (Dx @ Dx) @ u
    uyy = (Dy @ Dy) @ u
    uxy = (Dx @ Dy) @ u
    return uxx, uxy, uyy

def residual(u, target):
    uxx, uxy, uyy = hessian(u)
    det = (1 + uxx) * (1 + uyy) - uxy ** 2
    return det - target

def jacobian(u):
    uxx, uxy, uyy = hessian(u)
    # Linearization of det(I + D^2 u) around u
    J = diags(2 + uxx + uyy, 0, shape=(N * N, N * N)) @ Lap \
        - 2 * diags(uxy, 0) @ (Dx @ Dy)
    return P @ J @ P

def solve_step(u0, target, max_iter=20, tol=1e-8):
    u = u0.copy()
    for _ in range(max_iter):
        R = residual(u, target)
        du = spsolve(jacobian(u), -P @ R)
        u += du
        u -= u.mean()
        if np.linalg.norm(du) < tol:
            break
    return u

# Continuation path from t=0 (flat solution) to t=1 (target RHS)
t_steps = 11
u = np.zeros(N * N)
for t in np.linspace(0, 1, t_steps)[1:]:
    target = (1 - t) + t * F
    u = solve_step(u, target.ravel())
    u -= u.mean()
    # Monitor discrete analogues of C0 and Laplacian bounds
    print(f"t={t:.2f}, |u|_max={np.abs(u).max():.4f}, "
          f"lap_max={(Lap @ u).max():.4f}, "
          f"res_norm={np.linalg.norm(residual(u, target.ravel())):.2e}")

# Visualize the resulting potential
U = u.reshape((N, N))
plt.figure(figsize=(5, 4))
plt.contourf(X, Y, U, levels=40, cmap='RdBu_r')
plt.colorbar(label=r'$\phi$')
plt.title('Potential solving discrete Monge-Ampère')
plt.axis('equal')
plt.tight_layout()
plt.show()
```

The code mirrors the proof structure in miniature: a geometric curvature prescription is rewritten as a scalar determinant equation, a continuation parameter deforms an easy problem into the target one, and at each step Newton's method converges only because the linearization remains uniformly elliptic. In Yau's theorem, uniform ellipticity is guaranteed not by discretization but by the chain of a priori estimates, which is the heart of the method.
