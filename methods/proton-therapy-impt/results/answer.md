# IMPT inverse planning: spot-weight optimization with robust optimization

## Problem

Deliver a uniform prescription dose to a tumour while sparing neighbouring organs at risk (OARs), using a magnetically scanned proton pencil beam. Each proton spot — one (lateral position, energy) pencil — deposits a fixed Bragg-peaked dose pattern, scalable by its weight (number of protons). Intensity-modulated proton therapy (IMPT) chooses the weight of every individual spot in 3-D so the *summed* dose over all fields is uniform on the target and low on OARs. The sharp distal fall-off of the Bragg peak that makes protons attractive also makes the delivered dose acutely sensitive to range and setup uncertainty; robust optimization makes the plan hold up under those errors.

## Key idea

1. **Linear dose model.** Dose is linear in spot weights: `D = P w`, where `P` (the dose-influence matrix) gives the dose each unit-weight spot deposits in each voxel, precomputed once. `w ≥ 0` (a weight is a proton count).

2. **Penalized least-squares objective.** Minimize a weighted sum of per-voxel quadratic penalties:
   - Target: squared deviation `(D_i − d_ref)²` (both directions bad).
   - OAR: one-sided squared overdose `max(D_i − d_max, 0)²`.
   - Target floor: one-sided squared underdose `max(d_min − D_i, 0)²`.

   `f(w) = Σ_s (p_s/|V_s|) Σ_{i∈V_s} g_s(D_i)²`, with structure penalties `p_s` setting the clinical trade-off.

3. **Gradient by back-projection.** Since `f` depends on `w` only through `D = P w`, `∇_w f = Pᵀ (∂f/∂D)`, where `∂f/∂D` is the per-voxel delta read off each penalty (one-sided penalties give zero delta on their good side). Forward through `P`, backward through `Pᵀ`.

4. **Solve** the bound-constrained (`w ≥ 0`) convex program with a gradient-based optimizer (interior-point / L-BFGS-B).

5. **Robust optimization.** The nominal-optimal plan parks the steep distal edge behind serial OARs and deforms catastrophically under a few-percent range error (CT Hounsfield → stopping-power conversion) or a few-mm setup shift; a static geometric margin cannot fix this because the proton dose *reshapes* under a shift rather than translating. Instead optimize over a set of error scenarios `k`, each with its own influence matrix `P^{(k)}` and dose `D^{(k)} = P^{(k)} w`:
   - **Probabilistic / expected** (scenario probabilities `π_k`): `min_w Σ_k π_k f(w; D^{(k)})`; optionally add a dose-variance penalty `wᵀΩw`.
   - **Worst-case** (uncertainty set): `min_w max_k f(w; D^{(k)})`.
     - *Voxel-wise worst case (VWWC):* build one synthetic worst dose (min over scenarios in target, max in OAR), penalize that.
     - *Composite worst case (COWC):* take the worst whole-scenario objective; smooth the max (log-sum-exp / p-norm) for a usable gradient.

   The robust optimizer is the nominal one wrapped in a scenario loop and a combine rule; each scenario's gradient back-projects through its own `P^{(k)}`.

## Algorithm

```
precompute P^{(k)} for nominal + each error scenario (range ±%, setup shifts ±mm)
initialize w = 1, bounds w >= 0
minimize, via L-BFGS-B / IPOPT:
    for each scenario k:  D_k = P^{(k)} w
                          f_k = Σ_structures penalty(D_k[struct])
                          δ_k = Σ_structures penalty_grad(D_k[struct])  (per-voxel)
                          g_k = (P^{(k)})ᵀ δ_k
    combine: expected  -> f = Σ π_k f_k,    g = Σ π_k g_k
             COWC      -> f = softmax_τ(f_k), g = Σ softmax-weight_k · g_k
return w
```

## Code

Faithful to the structure of the matRad optimization toolkit (penalty objects with value+dose-gradient, a dose projection that maps `w↔D` through the influence matrix, an objective wrapper that loops structures and applies a robustness rule, a bound-constrained solver). Written here in NumPy/SciPy.

```python
import numpy as np
from scipy.optimize import minimize

# ---- voxel penalties: value + per-voxel dose gradient (the "delta") ----
class SquaredDeviation:                 # target: (D - dref)^2
    def __init__(self, dref, penalty=1.0): self.dref, self.p = dref, penalty
    def value(self, d):
        dev = d - self.dref
        return self.p / d.size * (dev @ dev)
    def grad(self, d):
        return self.p * 2.0 / d.size * (d - self.dref)

class SquaredOverdosing:                # OAR: max(D - dmax, 0)^2
    def __init__(self, dmax, penalty=1.0): self.dmax, self.p = dmax, penalty
    def value(self, d):
        over = np.maximum(d - self.dmax, 0.0)
        return self.p / d.size * (over @ over)
    def grad(self, d):
        over = np.maximum(d - self.dmax, 0.0)
        return self.p * 2.0 / d.size * over

class SquaredUnderdosing:               # target floor: max(dmin - D, 0)^2
    def __init__(self, dmin, penalty=1.0): self.dmin, self.p = dmin, penalty
    def value(self, d):
        under = np.minimum(d - self.dmin, 0.0)
        return self.p / d.size * (under @ under)
    def grad(self, d):
        under = np.minimum(d - self.dmin, 0.0)
        return self.p * 2.0 / d.size * under

# ---- dose projection: forward d = P w, backward grad = P^T delta ----
def dose(P, w):              return P @ w
def back_project(P, delta):  return P.T @ delta

# ---- per-scenario objective: loop structures ----
def scenario_obj_grad(P, w, structures):
    d = dose(P, w)
    f = 0.0
    voxel_delta = np.zeros_like(d)
    for s in structures:                 # s = {"idx": voxel indices, "pen": penalty}
        idx, pen = s["idx"], s["pen"]
        d_s = d[idx]
        f += pen.value(d_s)
        voxel_delta[idx] += pen.grad(d_s)
    return f, back_project(P, voxel_delta)

# ---- robustness wrapper: combine scenarios into one (f, grad) ----
def robust_obj_grad(w, P_scen, structures, mode="cowc", probs=None, tau=10.0):
    fs, gs = zip(*(scenario_obj_grad(P, w, structures) for P in P_scen))
    fs = np.asarray(fs); gs = np.asarray(gs)

    if mode == "nominal":
        return fs[0], gs[0]
    if mode == "expected":                       # probabilistic: E_k[f]
        pi = np.asarray(probs)
        return float(pi @ fs), (pi[:, None] * gs).sum(0)
    if mode == "cowc":                            # composite worst case (smoothed max)
        m = fs.max()
        ex = np.exp(tau * (fs - m))
        sm = ex / ex.sum()
        f = m + np.log(ex.sum()) / tau            # -> max_k f_k as tau -> inf
        return f, (sm[:, None] * gs).sum(0)
    raise ValueError(mode)

# ---- solve: bound-constrained, w >= 0 ----
def plan(P_scen, structures, n_spots, mode="cowc", probs=None):
    w0 = np.ones(n_spots)
    bounds = [(0.0, None)] * n_spots
    res = minimize(lambda w: robust_obj_grad(w, P_scen, structures, mode, probs),
                   w0, jac=True, bounds=bounds, method="L-BFGS-B",
                   options={"maxiter": 500})
    return res.x
```

`P_scen[0]` is the nominal dose-influence matrix; the rest are the range-/setup-error scenarios. `mode="nominal"` recovers the basic IMPT optimizer; `"expected"` is probabilistic robust optimization; `"cowc"` is worst-case robust optimization.
