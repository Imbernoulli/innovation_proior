# Electrical Flows for Approximate Maximum Flow

## Problem

Compute a `(1−ε)`-approximate maximum `s-t` flow (and a `(1+ε)`-approximate minimum
`s-t` cut) in an undirected graph `G = (V, E)` with `n` vertices, `m` edges, integer
capacities `u_e`. The yardstick to beat is the `O(m^{3/2})` / `Õ(m√n)`-style running time of
path-augmentation and blocking-flow methods, which sit inside the `Ω(mn)` flow-decomposition
barrier.

## Key idea

Replace path-based augmentation with a non-combinatorial primitive: the **electrical flow**,
the minimum-energy (`ℓ_2`) `s-t` flow, computed by a single Laplacian linear-system solve in
nearly-linear time. Max flow is an `ℓ_∞` problem (bound the worst congestion); electrical flow
is the `ℓ_2` relaxation, which can overload an edge by up to `√m`. **Multiplicative weights**
turns this capacity-oblivious oracle into a feasible flow, with iteration count proportional to
the oracle's **width** (worst congestion). The width is `√m` for a plain electrical flow, but
the overloaded edges are *fragile*: removing any edge that exceeds a target width `ρ` and
recomputing — while tracking the **effective resistance** as a monotone potential that jumps
each time such an edge is cut — keeps the number of removals small. Balancing iterations against
removals gives `ρ ≈ m^{1/3}` and running time `Õ(m^{4/3} ε^{-3})`.

## The electrical-flow primitive

Assign resistance `r_e > 0`, conductance `c_e = 1/r_e`, `C = diag(c_e)`, incidence matrix `B`,
Laplacian `L = B C Bᵀ`. The electrical `s-t` flow of value `F` minimizes the energy
`E_r(f) = Σ_e r_e f(e)²` over flows with `Bf = Fχ_{s,t}`; it is a potential flow `f = C Bᵀφ`
with `φ = L⁺(Fχ_{s,t})`, and `E_r(f) = F² R_eff(r)` where `R_eff(r) = χᵀL⁺χ` is the effective
`s-t` resistance. `L` is symmetric diagonally dominant, so `Lφ = Fχ` is solved approximately in
`Õ(m log(1/δ))` time (Koutis-Miller-Peng; Spielman-Teng), returning `φ̂` with
`‖φ̂ − φ‖_L ≤ δ'‖φ‖_L`.

Two facts that drive the analysis:

- **Effective conductance / Thomson's principle.**
  `C_eff(r) = 1/R_eff(r) = min_{φ: φ_s=1, φ_t=0} Σ_{(u,v)} (φ_u − φ_v)²/r_{uv}`,
  minimized by the electrical potentials. Hence **Rayleigh monotonicity**: `r' ≥ r ⇒ R_eff(r') ≥ R_eff(r)`.

- **Effect of a resistance increase.** If edge `h` carries a `β` fraction of the energy
  (`f(h)² r_h = β E_r(f)`) and `r_h → γ r_h`, then
  `R_eff(r') ≥ (γ/(β + γ(1−β))) R_eff(r)`. In particular cutting `h` (`γ = ∞`) gives
  `R_eff(r') ≥ R_eff(r)/(1−β)`; a bump `γ = 1+ε` gives `R_eff(r') ≥ (1 + εβ/2) R_eff(r)`.

  *Proof.* Normalize `f` to the unit-conductance flow, `φ_s = 1, φ_t = 0`; then
  `C_eff(r) = Σ (φ_u−φ_v)²/r_{uv}`, the `h`-term is `βC_eff(r)`, the rest `(1−β)C_eff(r)`.
  Plugging this same `φ` into the min for `r'` (an upper bound):
  `C_eff(r') ≤ (β/γ)C_eff(r) + (1−β)C_eff(r) = C_eff(r)(β + γ(1−β))/γ`. Invert. ∎

## Multiplicative-weights oracle (width `3√(m/ε)`)

Maintain weights `w_e ≥ 1`. Given `w` and target value `F`, set
`r_e = (1/u_e²)(w_e + ε‖w‖_1/(3m))`, compute the `(ε/3)`-approximate electrical flow `f̃` of
value `F`, fail if `E_r(f̃) > (1+ε)‖w‖_1`, else return `f̃`. The `w_e` term ties energy to the
*weighted-average* congestion; the additive floor `ε‖w‖_1/3m` caps the *worst* congestion.

**Lemma (oracle).** This is an `(ε, 3√(m/ε))`-oracle: when `F ≤ F*` it never fails, and any
returned `f̃` satisfies `Σ_e w_e cong(f̃,e) ≤ (1+ε)‖w‖_1` and `max_e cong(f̃,e) ≤ 3√(m/ε)`.

*Proof.* For feasible `f*`, `cong ≤ 1` so `E_r(f*) ≤ (1+ε/3)‖w‖_1`; since electrical flow
minimizes energy and the approximation costs `(1+ε/3)`, `E_r(f̃) ≤ (1+ε/3)²‖w‖_1 ≤ (1+ε)‖w‖_1`
when `F ≤ F*`. Given `E_r(f̃) ≤ (1+ε)‖w‖_1`: dropping the floor, `Σ w_e cong² ≤ (1+ε)‖w‖_1`, and
Cauchy-Schwarz gives `Σ w_e cong ≤ √(1+ε)‖w‖_1`. Keeping only the floor,
`(ε‖w‖_1/3m)cong_e² ≤ (1+ε)‖w‖_1`, so `cong_e ≤ √(3m(1+ε)/ε) ≤ 3√(m/ε)`. ∎

## Multiplicative-weights theorem

```
MaxFlowMW(G, F, oracle O with width ρ, ε):
  w_e ← 1 for all e;  N ← 2 ρ ln m / ε²
  for i = 1..N:
    f^i ← O(w, F);  if fail: return FAIL
    w_e ← w_e · (1 + (ε/ρ) cong(f^i, e))   for all e
  return  f̄ ← ((1−ε)² / ((1+ε) N)) · Σ_i f^i
```

**Theorem.** Given an `(ε, ρ)`-oracle of running time `T`, `MaxFlowMW` computes a
`(1−O(ε))`-approximate maximum flow in `Õ(ρ ε^{-2} · T)` time.

*Proof.* Potential `μ_i = ‖w^i‖_1`. From the average bound,
`μ_{i+1} ≤ μ_i(1 + ε(1+ε)/ρ) ≤ μ_i exp(ε(1+ε)/ρ)`, so `μ_N ≤ m exp(ε(1+ε)N/ρ)`. Per edge,
using `1 + εx ≥ exp((1−ε)εx)` for `x = cong/ρ ∈ [0,1]`,
`w_e^N ≥ exp((1−ε)(ε/ρ) Σ_i cong(f^i,e)) ≥ exp((1−ε)(ε/ρ) N cong(f̄_+,e))`, where `f̄_+` is the
average flow. Since `w_e^N ≤ μ_N`, taking logs and using `N = 2ρ ln m/ε²`, the per-edge
congestion of the scaled return `f̄` satisfies `cong(f̄,e) ≤ 1 − ε + ε(1−ε)/(2(1+ε)) ≤ 1`, so
`f̄` is feasible. Each `f^i` has value `F`, so `f̄` has value `(1−ε)²/(1+ε)·F ≥ (1−O(ε))F`. ∎

Combining with the `√(m/ε)`-oracle gives a **`(1−ε)`-approximate max flow in
`Õ(m^{3/2} ε^{-5/2})`** time.

## Width reduction → `Õ(m^{4/3} ε^{-3})`

The bad graph (`k` parallel length-`k` paths plus one direct `s-t` edge, `m = Θ(k²)`) forces
`Θ(√m)` flow on the direct edge — `ρ = Θ(√m)` is tight for a single electrical flow. But that
edge is fragile. Use the modified oracle:

```
ImprovedOracle(w, F, H, ρ):                      # ρ = 8 m^{1/3} ln^{1/3}m / ε
  r_e ← (1/u_e²)(w_e + ε|w|₁/(3m))   for e ∈ E∖H
  f̃ ← approx electrical flow on G_H = (V, E∖H), value F, accuracy δ=ε/3
  if E_r(f̃) > (1+ε)|w|₁  or  s,t disconnected in G_H:  return FAIL
  if ∃ e with cong(f̃,e) > ρ:  add e to H;  restart
  return f̃, H
```

**Lemma (removals).** Throughout the algorithm `|H| ≤ 30 m ln m/(ε²ρ²)` and
`u(H) ≤ 30 m F ln m/(ε²ρ³)`. With `ρ = 8 m^{1/3}ln^{1/3}m/ε`: `|H| ≤ (15/32)(m ln m)^{1/3}` and
`u(H) < εF/12`.

*Proof sketch.* Potential `Φ(j) = R_eff(r^j)` (removed edges at `r = ∞`). (1) `Φ` never
decreases (Rayleigh). (2) `Φ(1) ≥ m^{-4}F^{-2}` (the unit flow sends `≥ 1/m` across the min cut,
whose edges have `r ≥ 1/F*²`, and `F* ≤ mF`). (3) Each removed edge has `cong > ρ`, so via the
floor it carries `> ερ²/(5m)` of the energy (transferred to the exact flow by the solver's
per-edge guarantee), and cutting it multiplies `Φ` by `≥ 1/(1−ερ²/5m)`. Combining with
`Φ(j) ≤ (1+ε)‖w‖_1/F² ≤ 2m⁵ exp(3ε^{-1}ln m)·Φ(1)` and `ln(1−c) < −c` yields the cardinality
bound; `u_e < F/ρ` (since `> ρu_e` units flow over a removed edge but never `> F`) gives the
capacity bound. ∎

Because `u(H) < εF/12`, a feasible flow of value `(1−ε/12)F` always survives — the oracle never
wrongly fails. Total electrical solves `≤ N + |H| = Õ(ρ/ε²) + Õ(m^{1/3}) = Õ(m^{1/3}ε^{-3})`,
each `Õ(m)`:

**Theorem.** A `(1−ε)`-approximate maximum `s-t` flow is computable in `Õ(m^{4/3} ε^{-3})` time;
with Karger's graph smoothing, in **`Õ(m n^{1/3} ε^{-11/3})`** time. The min `s-t` cut value is
`(1+ε)`-approximable in `Õ(m + n^{4/3} ε^{-8/3})` time.

## Dual cut algorithm (`Õ(m + n^{4/3} ε^{-8/3})`)

No oracle/averaging: repeatedly solve an electrical flow, raise resistances by congestion, and
read a cut from the potentials by threshold sweep.

```
MinCut(G, F, ε≤1/7):                              # ρ = 3 m^{1/3} ε^{-2/3}, N = 5 ε^{-8/3} m^{1/3} ln m, δ=ε²
  w_e ← 1
  for i = 1..N:
    f̃, φ̃ ← approx electrical flow & potentials with r_e = w_e/u_e², value F, accuracy δ
    μ ← Σ_e w_e
    w_e ← w_e + (ε/ρ) cong(f̃,e) w_e + (ε²/(mρ)) μ        # extra floor term keeps w_e ≥ (ε/m)μ
    rescale φ̃ so φ̃_s = 1, φ̃_t = 0;  S_x = {v : φ̃_v > x}
    S ← arg min_x cap(S_x, V∖S_x)
    if cap(S) < F/(1−7ε):  return S
  return FAIL
```

A random threshold cut has expected capacity `Σ_e |φ_u−φ_v| u_e`, and by Cauchy-Schwarz with
`μ = Σ_e u_e² r_e` this is `≤ √(μ/R_eff)`. The contradiction argument (total weight bounded; the
geometric mean `ν` of min-cut weights grows on low-congestion steps; `R_eff` grows on
high-congestion steps; both sums `< N`) shows `R_eff` reaches `(1−7ε)μ/F²` within `N` steps, so
the best threshold cut has capacity `≤ F/(1−7ε)`. On a Benczúr-Karger sparsifier with
`O(n log n/ε²)` edges, this gives a `(1+ε)`-cut in `Õ(m + n^{4/3} ε^{-8/3})`.

## Reference implementation

```python
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def incidence_matrix(n, edges):
    m = len(edges)
    rows, cols, vals = [], [], []
    for e, (a, b) in enumerate(edges):
        rows += [a, b]; cols += [e, e]; vals += [1.0, -1.0]
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, m))

def electrical_flow(B, conduct, src, snk, F):
    # min-energy s-t flow of value F: L phi = F*chi, f = C B^T phi
    n, _ = B.shape
    L = (B @ sp.diags(conduct) @ B.T).tolil()
    chi = np.zeros(n); chi[src] = 1.0; chi[snk] = -1.0
    keep = list(range(1, n))                       # ground vertex 0
    phi = np.zeros(n)
    phi[keep] = spla.spsolve(L[keep, :][:, keep].tocsr(), (F * chi)[keep])
    return conduct * (B.T @ phi), phi

def oracle(B, u, w, src, snk, F, eps, active):
    m = len(u); w1 = float(w.sum())
    res = np.full(m, np.inf)
    res[active] = (w[active] + eps * w1 / (3 * m)) / (u[active] ** 2)  # avg term + floor term
    conduct = np.where(np.isinf(res), 0.0, 1.0 / res)
    f, phi = electrical_flow(B, conduct, src, snk, F)
    E = float(np.sum(np.where(np.isinf(res), 0.0, res) * f * f))
    return f, phi, E <= (1 + eps) * w1                                # fail-test on energy

def approx_max_flow(n, edges, u, src, snk, F, eps, rho=None):
    B = incidence_matrix(n, edges); m = len(edges); u = np.asarray(u, float)
    rho = 3.0 * np.sqrt(m / eps) if rho is None else rho
    w = np.ones(m); active = np.ones(m, dtype=bool)
    N = int(np.ceil(2 * rho * np.log(m) / eps ** 2))
    acc = np.zeros(m)
    for _ in range(N):
        f, phi, ok = oracle(B, u, w, src, snk, F, eps, active)
        if not ok:
            return None                                              # certifies F > F*
        w = w * (1 + (eps / rho) * (np.abs(f) / u))                  # reweight by congestion
        acc += f
    return (1 - eps) ** 2 / ((1 + eps) * N) * acc                    # feasibility-scaled average

# Improved oracle: add `if cong>rho: active[e]=False; recompute` and reuse this loop.
```

Running this on two parallel unit-capacity edges (`F = 2, ε = 0.2`) returns a feasible flow with
each edge near `0.53` (max congestion `< 1`); the value gap from `F` is the conservative
`(1−ε)²/(1+ε)` feasibility-scaling, which the analysis only needs up to `1 − O(ε)`.
