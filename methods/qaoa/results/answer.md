# QAOA — the Quantum Approximate Optimization Algorithm

## Problem

Given a combinatorial objective `C(z) = Σ_{α=1}^m C_α(z)` over `n` bits (each clause `C_α`
local, `C_α(z) ∈ {0,1}`), produce a string `z` with `C(z)` close to `max_z C(z)`. MaxCut and
Max E3LIN2 are the running examples. The goal is a quantum procedure with a tunable depth knob
`p` that provably improves with `p`, compiles to gates no more nonlocal than the clauses, and
whose parameters are classically cheap to set for fixed `p`.

## Key idea

View `C` as a diagonal operator (`C|z⟩ = C(z)|z⟩`); the optimum is its extremal eigenstate.
Adiabatic evolution from the easy ground state `|s⟩ = |+⟩^{⊗n}` of the transverse-field
mixer `B = Σ_j σˣ_j` toward `C` would reach it, but the required runtime scales like
`1/g_min²` and is one long coherent analog evolution. Trotterize that path into `p` discrete
alternating layers, then **free the Trotter angles from the adiabatic schedule and make them
variational**: define a `2p`-parameter state and maximize the expectation of `C` in it,
classically optimizing the angles against a quantum-measured objective.

## Final algorithm

Cost unitary (exact, since the `C_α` commute; locality = clause locality):
```
U(C, γ) = e^{-iγC} = ∏_α e^{-iγ C_α},        γ ∈ [0, 2π].
```
Mixer unitary (single-qubit RX per qubit; moves amplitude between basis states):
```
U(B, β) = e^{-iβB} = ∏_j e^{-iβ σˣ_j},        β ∈ [0, π].
```
Depth-`p` ansatz on `|s⟩ = |+⟩^{⊗n}`:
```
|γ, β⟩ = U(B, β_p) U(C, γ_p) ⋯ U(B, β_1) U(C, γ_1) |s⟩,
F_p(γ, β) = ⟨γ, β| C |γ, β⟩,     M_p = max_{γ,β} F_p(γ, β).
```
Properties: `M_p ≥ M_{p−1}` (extra layers can be made trivial) and `lim_{p→∞} M_p = max_z C(z)`
(the faithful Trotterized adiabatic path lies inside the parameter family). By the variational
principle `F_p ≤ max C`, so every increase in `F_p` is real progress.

Procedure: pick `p`; find good `(γ, β)` — for fixed `p` and bounded degree by exact classical
preprocessing (each edge term depends only on its distance-`p` subgraph, so
`F_p = Σ_g w_g f_g(γ,β)` with the `f_g` on `n`-independent Hilbert spaces of size `≤ 2^{q_tree}`,
`q_tree = 2[((v−1)^{p+1}−1)/((v−1)−1)]`), otherwise by a classical optimizer that queries the
quantum computer for `F_p`; prepare `|γ,β⟩`; measure in the computational basis; repeat and keep
the best string. The measured `C(z)` concentrates: `Var(C) = O(m)`, so a sample mean of `O(m²)`
shots is within `1` of `F_p` with probability `1 − 1/m`.

Analytic guarantees the construction yields. MaxCut on any 3-regular graph at `p=1`: the
worst-case ratio is `0.6924`, from minimizing `M_1(1,s,t)/(3/2 − s − t)` over crossed-square and
triangle densities `s, t` (`4s+3t ≤ 1`), minimized at `s=t=0`. On the ring (2-regular),
`M_p = n(2p+1)/(2p+2)`, ratio `→ 1` at constant depth `3p`. Max E3LIN2 with each bit in `≤ D+1`
equations at `p=1`, `β=π/4`: there is a `γ ∈ [−1/(10D^{1/2}), 1/(10D^{1/2})]` (found among
`5 ln D` values) giving `(½ + 1/(101 D^{1/2} ln D)) m` satisfied equations; for the typical
random-sign instance, `γ = 1/(√3 D^{1/2})` gives `(½ + 1/(2√(3e) D^{1/2})) m`.

## Working code (MaxCut, PennyLane)

```python
import networkx as nx
import pennylane as qml
from pennylane import numpy as np

n_wires = 4
graph = [(0, 1), (0, 3), (1, 2), (2, 3)]
wires = range(n_wires)

# cost layer  U(C, gamma) = e^{-i gamma C},  C = sum_edges 1/2 (1 - Z_j Z_k)
def U_C(gamma):
    for j, k in graph:
        qml.CNOT(wires=[j, k])
        qml.RZ(gamma, wires=k)
        qml.CNOT(wires=[j, k])

# mixer layer  U(B, beta) = prod_j e^{-i beta X_j}
def U_B(beta):
    for w in wires:
        qml.RX(2 * beta, wires=w)

dev = qml.device("lightning.qubit", wires=n_wires)

@qml.qnode(dev)
def circuit(gammas, betas, return_samples=False):
    for w in wires:
        qml.Hadamard(wires=w)                 # |s> = |+>^n
    for gamma, beta in zip(gammas, betas):
        U_C(gamma)
        U_B(beta)                              # p alternating layers
    if return_samples:
        return qml.sample()
    C = qml.sum(*(qml.Z(j) @ qml.Z(k) for j, k in graph))
    return qml.expval(C)                       # F_p up to the cut affine map

def objective(params):
    gammas, betas = params[0], params[1]
    return -0.5 * (len(graph) - circuit(gammas, betas))   # maximize cut = minimize negative

def qaoa_maxcut(p=1, steps=30):
    params = 0.01 * np.random.rand(2, p, requires_grad=True)   # 2p variational angles
    opt = qml.GradientDescentOptimizer(stepsize=0.5)
    for _ in range(steps):
        params = opt.step(objective, params)
    # sample many shots at the optimized angles; the most frequent bitstring is the cut
    bitstrings = [circuit(params[0], params[1], return_samples=True) for _ in range(100)]
    return params, bitstrings
```

For a general cost Hamiltonian the same structure is built directly:
`H_C = Σ_edges ½(Z_i Z_j − I)` and `H_B = Σ_i X_i`, with each layer `e^{-iγ H_C}` then
`e^{-iβ H_B}` (PennyLane's `qaoa.cost_layer`/`qaoa.mixer_layer` apply exactly these via
`ApproxTimeEvolution`), the objective the expectation of `H_C`, and the `2p` angles optimized
classically.
