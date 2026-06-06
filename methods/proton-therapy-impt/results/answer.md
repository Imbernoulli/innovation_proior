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

3. **Gradient by back-projection.** Since `f` depends on `w` only through `D = P w`, `∇_w f = Pᵀ (∂f/∂D)`, where `∂f/∂D` is the per-voxel delta read off each penalty. The underdose term has derivative `2 min(D_i − d_min, 0)`, so cold target voxels get a negative dose-gradient that pushes their dose upward. Forward through `P`, backward through `Pᵀ`.

4. **Solve** the bound-constrained (`w ≥ 0`) convex program with a gradient-based optimizer (interior-point / L-BFGS-B).

5. **Robust optimization.** The nominal-optimal plan parks the steep distal edge behind serial OARs and deforms catastrophically under a few-percent range error (CT Hounsfield → stopping-power conversion) or a few-mm setup shift; a static geometric margin cannot fix this because the proton dose *reshapes* under a shift rather than translating. Instead optimize over a set of error scenarios `k`, each with its own influence matrix `P^{(k)}` and dose `D^{(k)} = P^{(k)} w`:
   - **Stochastic / expected** (scenario probabilities `π_k`): `min_w Σ_k π_k f(w; D^{(k)})`.
   - **Probabilistic expectation-plus-variance:** optimize the objective on the expected dose and add a dose-variance penalty `wᵀΩw`, whose gradient contribution is `2Ωw`.
   - **Worst-case** (uncertainty set): `min_w max_k f(w; D^{(k)})`.
     - *Voxel-wise worst case (VWWC):* build one synthetic worst dose (min over scenarios in target, max in OAR), penalize that.
     - *Composite worst case (COWC):* take the worst whole-scenario objective; use a hard max or smooth it with log-sum-exp / p-norm for a usable gradient.
     - *Objective-wise worst case (OWC):* take the worst scenario per objective term.

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
    combine: STOCH -> f = Σ π_k f_k, g = Σ π_k g_k
             PROB  -> f = f(E[D]) + p wᵀΩw, g = P_expᵀ δ_exp + 2pΩw
             VWWC  -> choose each voxel's target-min / OAR-max scenario
             COWC  -> choose or smooth max_k f_k
             OWC   -> choose or smooth max per objective term
return w
```

## Code

The code mirrors the matRad optimization structure: objective objects return unweighted values and dose-gradients, the wrapper applies penalties and robustness modes, the dose projection maps `w` to dose and maps dose-gradients back with the transpose influence matrix, and the solver enforces `w >= 0`.

```python
import numpy as np
from scipy.optimize import minimize

# ---- dose objectives: unweighted value + unweighted dose-gradient ----
class DoseObjective:
    def __init__(self, penalty=1.0, robustness="none"):
        self.penalty = penalty
        self.robustness = robustness

class SquaredDeviation(DoseObjective):   # target: (D - dref)^2
    def __init__(self, dref, penalty=1.0, robustness="none"):
        super().__init__(penalty, robustness)
        self.dref = dref

    def value(self, d):
        dev = d - self.dref
        return (dev @ dev) / d.size

    def dose_grad(self, d):
        return 2.0 * (d - self.dref) / d.size

class SquaredOverdosing(DoseObjective):  # OAR: max(D - dmax, 0)^2
    def __init__(self, dmax, penalty=1.0, robustness="none"):
        super().__init__(penalty, robustness)
        self.dmax = dmax

    def value(self, d):
        over = np.maximum(d - self.dmax, 0.0)
        return (over @ over) / d.size

    def dose_grad(self, d):
        over = np.maximum(d - self.dmax, 0.0)
        return 2.0 * over / d.size

class SquaredUnderdosing(DoseObjective): # target floor: max(dmin - D, 0)^2
    def __init__(self, dmin, penalty=1.0, robustness="none"):
        super().__init__(penalty, robustness)
        self.dmin = dmin

    def value(self, d):
        under = np.minimum(d - self.dmin, 0.0)
        return (under @ under) / d.size

    def dose_grad(self, d):
        under = np.minimum(d - self.dmin, 0.0)
        return 2.0 * under / d.size       # negative while the voxel is cold

# ---- projection: forward dose and transpose back-projection ----
class DoseProjection:
    def __init__(self, physical_dose, scenario_prob=None,
                 physical_dose_exp=None, omega=None):
        self.P = physical_dose
        self.scenario_prob = scenario_prob
        self.P_exp = physical_dose_exp
        self.omega = omega

    def dose(self, scen, w):
        return self.P[scen] @ w

    def expected_dose(self, w):
        return self.P_exp @ w

    def back_project(self, scen, dose_grad):
        return self.P[scen].T @ dose_grad

    def back_project_prob(self, dose_grad_exp, omega_grad):
        return self.P_exp.T @ dose_grad_exp + 2.0 * omega_grad

def max_value_and_weights(values, approx="logsumexp", tau=10.0):
    values = np.asarray(values, dtype=float)
    if approx == "none":
        weights = np.zeros_like(values)
        weights[np.argmax(values)] = 1.0
        return float(values.max()), weights
    if approx == "logsumexp":
        m = values.max()
        ex = np.exp(tau * (values - m))
        weights = ex / ex.sum()
        return float(m + np.log(ex.sum()) / tau), weights
    if approx == "pnorm":
        p = max(values.size, 2)
        norm = np.linalg.norm(values, ord=p)
        if norm == 0:
            return 0.0, np.zeros_like(values)
        weights = values ** (p - 1) / (norm ** (p - 1))
        return float(norm), weights
    raise ValueError(approx)

def objective_and_gradient(w, projection, structures, scenarios,
                           max_approx="logsumexp", tau=10.0):
    doses = {k: projection.dose(k, w) for k in scenarios}
    dose_grads = {k: np.zeros_like(doses[k]) for k in scenarios}
    f = 0.0

    cowc_values = np.zeros(len(scenarios))
    cowc_grads = {k: np.zeros_like(doses[k]) for k in scenarios}
    dose_grad_exp = None
    omega_grad = np.zeros_like(w)

    for st in structures:
        idx = st["idx"]
        role = st["role"]
        objectives = st["objectives"]

        for obj_ix, obj in enumerate(objectives):
            rob = obj.robustness

            if rob == "none":
                k = scenarios[0]
                d = doses[k][idx]
                f += obj.penalty * obj.value(d)
                dose_grads[k][idx] += obj.penalty * obj.dose_grad(d)

            elif rob == "STOCH":
                for local, k in enumerate(scenarios):
                    d = doses[k][idx]
                    scale = projection.scenario_prob[local] * obj.penalty
                    f += scale * obj.value(d)
                    dose_grads[k][idx] += scale * obj.dose_grad(d)

            elif rob == "PROB":
                d_exp = projection.expected_dose(w)
                if dose_grad_exp is None:
                    dose_grad_exp = np.zeros_like(d_exp)
                f += obj.penalty * obj.value(d_exp[idx])
                dose_grad_exp[idx] += obj.penalty * obj.dose_grad(d_exp[idx])
                key = st.get("omega_key")
                if obj_ix == 0 and projection.omega is not None and key is not None:
                    Omega = projection.omega[key]
                    omega_w = Omega @ w
                    scale = obj.penalty / idx.size
                    f += scale * float(w @ omega_w)
                    omega_grad += scale * omega_w

            elif rob in ("VWWC", "VWWC_INV"):
                stack = np.column_stack([doses[k][idx] for k in scenarios])
                use_max = (role in ("OAR", "EXTERNAL") and rob == "VWWC") or \
                          (role == "TARGET" and rob == "VWWC_INV")
                wc_dose = stack.max(axis=1) if use_max else stack.min(axis=1)
                winners = stack.argmax(axis=1) if use_max else stack.argmin(axis=1)
                f += obj.penalty * obj.value(wc_dose)
                delta = obj.penalty * obj.dose_grad(wc_dose)
                for local, k in enumerate(scenarios):
                    m = winners == local
                    dose_grads[k][idx[m]] += delta[m]

            elif rob == "COWC":
                for local, k in enumerate(scenarios):
                    d = doses[k][idx]
                    cowc_values[local] += obj.penalty * obj.value(d)
                    cowc_grads[k][idx] += obj.penalty * obj.dose_grad(d)

            elif rob == "OWC":
                values = []
                local_grads = []
                for k in scenarios:
                    d = doses[k][idx]
                    values.append(obj.penalty * obj.value(d))
                    local_grads.append(obj.penalty * obj.dose_grad(d))
                f_owc, weights = max_value_and_weights(values, max_approx, tau)
                f += f_owc
                for local, k in enumerate(scenarios):
                    dose_grads[k][idx] += weights[local] * local_grads[local]

            else:
                raise ValueError(rob)

    if np.any(cowc_values):
        f_cowc, weights = max_value_and_weights(cowc_values, max_approx, tau)
        f += f_cowc
        for local, k in enumerate(scenarios):
            dose_grads[k] += weights[local] * cowc_grads[k]

    weight_grad = sum(projection.back_project(k, dose_grads[k]) for k in scenarios)
    if dose_grad_exp is not None:
        weight_grad += projection.back_project_prob(dose_grad_exp, omega_grad)
    return f, weight_grad

def plan(P_scen, structures, n_spots, scenario_prob=None):
    w0 = np.ones(n_spots)
    bounds = [(0.0, None)] * n_spots
    projection = DoseProjection(P_scen, scenario_prob=scenario_prob)
    scenarios = list(range(len(P_scen)))
    res = minimize(lambda x: objective_and_gradient(x, projection, structures, scenarios),
                   w0, jac=True, bounds=bounds, method="L-BFGS-B",
                   options={"maxiter": 500})
    return res.x
```

`P_scen[0]` is the nominal dose-influence matrix; the rest are the range-/setup-error scenarios. Each objective's `robustness` field selects nominal (`"none"`), stochastic expectation (`"STOCH"`), expected-dose-plus-variance (`"PROB"`), voxel-wise worst case (`"VWWC"` / `"VWWC_INV"`), composite worst case (`"COWC"`), or objective-wise worst case (`"OWC"`).
