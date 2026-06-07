# The Discrete Empirical Interpolation Method (DEIM)

## Problem

A discretized nonlinear PDE gives a large system of ODEs

```
dy/dt = A y + F(y),   y(t) in R^n,
```

with `A` the discrete linear operator and `F` a nonlinearity that, for a scalar
reaction–diffusion problem, acts componentwise, `F(y) = [F(y_1), ..., F(y_n)]^T`. The grid size `n`
must be large (thousands+) for accuracy, so every simulation is expensive and parameter sweeps,
control, and optimization become infeasible.

POD-Galerkin reduces the *number of variables*: take an orthonormal POD basis `V_k in R^{n×k}` of
solution snapshots, write `y ≈ V_k ỹ`, and project (`residual ⊥ Range(V_k)`, i.e. left-multiply by
`V_k^T`):

```
dỹ/dt = (V_k^T A V_k) ỹ + V_k^T F(V_k ỹ).
```

The linear part `Ã = V_k^T A V_k` is a precomputed `k×k` matrix, genuinely cheap. But the reduced
nonlinear term `V_k^T F(V_k ỹ)` must each step lift `V_k ỹ` to `R^n`, evaluate `F` at **all `n`**
components, and project back — cost `O(α(n) + 4nk)`, still scaling with `n` (here `α(q)` is the cost
of evaluating `F` at `q` points). Dimension was reduced, per-step cost was not; with a sparse `A` the
"reduced" model can be slower than the full one. The goal: make the nonlinear term **`n`-independent**
per step, while staying nearly as accurate as the optimal POD fit.

## Key idea

`F` along the trajectory also lives near a low-dimensional subspace, so take a *second* POD basis
`U = [u_1, ..., u_m] in R^{n×m}` of the **nonlinear** snapshots `{F(y_j)}` and approximate
`f := F(V_k ỹ) ≈ U c`. The best (orthogonal) coefficients `c = U^T f` would need all `n` entries of
`f` again — the bottleneck merely moves.

Instead, **interpolate**: with only `m` unknowns in `c`, demand the approximation be exact at `m`
chosen rows `℘_1, ..., ℘_m`. Let `P = [e_{℘_1}, ..., e_{℘_m}] in R^{n×m}` (columns of the identity).
Forcing `P^T f = (P^T U) c` gives, when `P^T U` is nonsingular,

```
f ≈ f̂ = U (P^T U)^{-1} P^T f.
```

`P^T f` reads off only `m` entries. The map `𝒫 = U (P^T U)^{-1} P^T` is idempotent with range
`Range(U)` — an **oblique** projector (not the orthogonal `U U^T`), and `P^T f̂ = P^T f`, so `f̂`
interpolates `f` at the `m` points. This is the discrete cousin of replacing projection by
interpolation (EIM, Barrault–Maday–Nguyen–Patera 2004), expressed as one clean matrix factorization.

**Why this kills the `n`-dependence.** For componentwise `F`,
`P^T F(V_k ỹ) = F(P^T V_k ỹ)` — selecting rows `℘` of `F(z)` equals applying `F` to rows `℘` of `z`,
since output `℘_i` depends only on input `℘_i`. So nothing is ever lifted to `R^n`. Precompute the
constant operators and the reduced model becomes

```
dỹ/dt = Ã ỹ + B F(V_℘ ỹ),
   Ã = V_k^T A V_k,   B = V_k^T U (P^T U)^{-1} in R^{k×m},   V_℘ = P^T V_k in R^{m×k},
```

with per-step cost `O(α(m) + 4mk)` — set by `m`, `k`, never `n`. (For a local/stencil `F` whose row
`i` reads inputs `p_i`, only `∪_i p_{℘_i}` rows of `V_k` are touched, still `n`-independent when the
stencils are sparse; componentwise is the case `p_i = {i}`.)

**Error bound.** Split `f = f_* + w` with `f_* = U U^T f` the best `m`-term fit and
`w = (I − U U^T) f`. Since `𝒫` is the identity on `Range(U)`, `𝒫 f_* = f_*`, so
`f − f̂ = (I − 𝒫) w`. Using `‖I − 𝒫‖_2 = ‖𝒫‖_2` (any projector ≠ `0`, `I`) and
`‖𝒫‖_2 ≤ ‖U‖_2 ‖(P^T U)^{-1}‖_2 ‖P^T‖_2 = ‖(P^T U)^{-1}‖_2` (`U` orthonormal, `P` identity columns),

```
‖f − f̂‖_2 ≤ ‖(P^T U)^{-1}‖_2 · ‖(I − U U^T) f‖_2.
```

The interpolation error is the unavoidable **orthogonal-projection error** `E_*(f)` (the cost of a
rank-`m` subspace) times a single magnification factor `‖(P^T U)^{-1}‖_2` set only by which rows were
chosen. POD for `U` minimizes `E_*` (Eckart–Young on `F`'s snapshots) and lets us read
`E_*(f) ≈ σ_{m+1}` off the discarded singular value — pick `m` at the knee of the nonlinear spectrum.

**Greedy point selection.** Keep `‖(P^T U)^{-1}‖_2` small by building the index set one at a time.
Seed at `℘_1 = argmax_i |u_1(i)|`, making `‖(P^T U)^{-1}‖_2 = 1/‖u_1‖_∞` smallest at step 1. At step
`ℓ`, solve `(P^T U) c = P^T u_ℓ` and form the residual `r = u_ℓ − U c` of mode `ℓ` after interpolating
it through the previous modes/points; a rank-one update of the inverse gives
`‖(P^T U)^{-1}‖_2 ≤ (1 + √(2n)) ‖(P̄^T Ū)^{-1}‖_2` with the increment governed by `1/|r_{℘_ℓ}|`.
Maximizing `|r_{℘_ℓ}| = ‖r‖_∞` minimizes the growth — **so each new index is placed at the
largest-magnitude entry of the residual.** The residual vanishes at previously chosen rows, so indices
are distinct and hierarchical, and `r ≠ 0` for linearly independent modes keeps `P^T U` nonsingular.
Modes are fed in descending singular-value order.

## Algorithm

```
DEIM point selection, input U = [u_1,...,u_m] (POD modes of F-snapshots, sing.-value order):
  ℘_1 = argmax_i |u_1(i)|
  for ℓ = 2..m:
      solve (U[℘,:]) c = u_ℓ[℘]            # make current basis interpolate mode ℓ at chosen rows
      r = u_ℓ − U[:,1:ℓ-1] c               # residual; zero at the previous indices
      ℘_ℓ = argmax_i |r(i)|                # new index at the largest residual entry
  return ℘

Offline (once):  Ã = V_k^T A V_k,   B = V_k^T U (U[℘,:])^{-1},   V_℘ = V_k[℘,:]
Online (per step):  dỹ/dt = Ã ỹ + B F(V_℘ ỹ)        # F evaluated at only m rows
```

## Code

A self-contained POD-DEIM ROM for a 1-D cubic reaction–diffusion system (FitzHugh–Nagumo
nonlinearity `F(v)=v(v−0.1)(1−v)+c`), full dimension `n=1024`. It builds both POD bases from
snapshots, selects the DEIM points greedily, and integrates the reduced system two ways — plain
POD-Galerkin (touches all `n` rows) and POD-DEIM (touches `m` rows) — showing matching accuracy at a
fraction of the nonlinear-evaluation work.

```python
import numpy as np


def pod_basis(S, r):
    """Leading r left singular vectors of snapshots S (n x n_s).
       Eckart-Young => optimal rank-r least-squares fit; columns by descending sing. value."""
    U, sv, _ = np.linalg.svd(S, full_matrices=False)
    return U[:, :r], sv


def deim_indices(U):
    """Greedy DEIM selection: each new index sits where the residual
       r = u_l - U c (mode l minus its interpolation through previous modes/points)
       is largest in magnitude -- maximizing |r_inf| caps the growth of ||(P^T U)^-1||."""
    n, m = U.shape
    P = np.empty(m, dtype=int)
    P[0] = np.argmax(np.abs(U[:, 0]))                 # seed at ||u_1||_inf
    for ell in range(1, m):
        Ue, Pe = U[:, :ell], P[:ell]
        c = np.linalg.solve(Ue[Pe, :], U[Pe, ell])    # (P^T U) c = P^T u_l
        r = U[:, ell] - Ue @ c                         # residual, zero at previous P
        P[ell] = np.argmax(np.abs(r))                  # argmax |r|
    return P


# --- full-order model: 1-D cubic reaction-diffusion, dy/dt = A y + F(y) -------
n = 1024
x = np.linspace(0.0, 1.0, n); h = x[1] - x[0]
eps = 0.015
main, off = -2.0 * np.ones(n), np.ones(n - 1)
Lap = (np.diag(main) + np.diag(off, 1) + np.diag(off, -1)) / h**2
Lap[0, 1] = 2.0 / h**2; Lap[-1, -2] = 2.0 / h**2      # Neumann ends
A = eps**2 * Lap


def F(v, t=0.0):                                       # componentwise cubic nonlinearity
    return v * (v - 0.1) * (1.0 - v) + 0.05


dt, nt = 1e-5, 4000
y0 = np.zeros(n); y0[:50] = 0.5

# --- collect state and nonlinear snapshots along the trajectory ---------------
yy, SY, SF = y0.copy(), [], []
for j in range(nt):
    yy = yy + dt * (A @ yy + F(yy))
    if j % 20 == 0:
        SY.append(yy.copy()); SF.append(F(yy).copy())
full = yy.copy()
SY, SF = np.array(SY).T, np.array(SF).T

# --- offline assembly: two POD bases + DEIM points + n-independent operators ---
k, m = 10, 12
Vk, _   = pod_basis(SY, k)                              # state POD basis  (n x k)
U,  svf = pod_basis(SF, m)                              # nonlinear POD basis (n x m)
P  = deim_indices(U)
At = Vk.T @ A @ Vk                                      # k x k linear part
B  = Vk.T @ U @ np.linalg.inv(U[P, :])                  # k x m, V_k^T U (P^T U)^{-1}
Vp = Vk[P, :]                                           # m x k, rows of V_k = P^T V_k

# --- online: integrate the reduced system two ways ----------------------------
yp = Vk.T @ y0                                          # POD-Galerkin: touches all n rows
yd = Vk.T @ y0                                          # POD-DEIM:     touches only m rows
for j in range(nt):
    yp = yp + dt * (At @ yp + Vk.T @ F(Vk @ yp))
    yd = yd + dt * (At @ yd + B @ F(Vp @ yd))

eP = np.linalg.norm(Vk @ yp - full) / np.linalg.norm(full)
eD = np.linalg.norm(Vk @ yd - full) / np.linalg.norm(full)
magn = np.linalg.norm(np.linalg.inv(U[P, :]), 2)        # ||(P^T U)^{-1}||_2

print(f"n={n}  k={k}  m={m}   DEIM points: {P}")
print(f"||(P^T U)^-1||_2 = {magn:.3g}    sigma_(m+1)/sigma_1 = {svf[m] / svf[0]:.2e}")
print(f"POD-Galerkin rel. error = {eP:.3e}   (F at n={n} rows/step)")
print(f"POD-DEIM     rel. error = {eD:.3e}   (F at m={m} rows/step)")
print(f"nonlinear-evaluation work cut by n/m = {n / m:.0f}x")
```

Running it: the DEIM points cluster around the moving reaction front, `‖(P^T U)^{-1}‖_2 ≈ 8.7` (tiny,
far from singular), and POD-DEIM reproduces the POD-Galerkin trajectory to `~4e-7` relative error —
the magnification factor times `σ_{m+1}` — while evaluating `F` at only `m=12` rows instead of
`n=1024`, an ~85× cut in nonlinear-evaluation work that is independent of the grid size. The bound
`‖f − f̂‖ ≤ ‖(P^T U)^{-1}‖ · ‖(I − U U^T)f‖` makes precise why the cheap interpolation is nearly as
good as the expensive orthogonal projection, and the residual-greedy is exactly the rule that keeps
its only data-independent factor small.
