# QAOA — the Quantum Approximate Optimization Algorithm

## Problem

Given a combinatorial objective `C(z) = Σ_{α=1}^m C_α(z)` over `n` bits (each clause `C_α`
local, `C_α(z) ∈ {0,1}`), produce a string `z` with `C(z)` close to `max_z C(z)`. MaxCut and
Max E3LIN2 are the running examples. The goal is a quantum procedure with a tunable depth knob
`p` that provably improves with `p`, compiles to gates no more nonlocal than the clauses, and
whose parameters are classically cheap to set for fixed `p`.

## Key idea

View `C` as a diagonal operator (`C|z⟩ = C(z)|z⟩`); the optimum is its extremal eigenstate.
Adiabatic evolution from the easy transverse-field state `|s⟩ = |+⟩^{⊗n}` — the ground state
of `Σ_j ½(1−σˣ_j)`, equivalently the top state of `B = Σ_j σˣ_j` — toward `C` would reach it,
but the required runtime scales like
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

graph = nx.Graph([(0, 1), (0, 3), (1, 2), (2, 3)])
wires = list(graph.nodes)
n_wires = len(wires)

def build_objective_and_driver(graph):
    # PennyLane's MaxCut cost Hamiltonian is H_C = 1/2 sum_edges (Z_i Z_j - I) = -C_cut.
    return qml.qaoa.maxcut(graph)

def prepare_start(wires):
    for w in wires:
        qml.Hadamard(wires=w)

def state_preparation(gammas, betas, cost_h, mixer_h):
    for gamma, beta in zip(gammas, betas):
        qml.qaoa.cost_layer(gamma, cost_h)
        qml.qaoa.mixer_layer(beta, mixer_h)

cost_h, mixer_h = build_objective_and_driver(graph)
dev = qml.device("default.qubit", wires=n_wires)
shot_dev = qml.device("default.qubit", wires=n_wires, shots=100)

@qml.qnode(dev)
def expectation_circuit(gammas, betas):
    prepare_start(wires)
    state_preparation(gammas, betas, cost_h, mixer_h)
    return qml.expval(cost_h)                  # minimize H_C to maximize the cut

def objective(params):
    gammas, betas = params[0], params[1]
    return expectation_circuit(gammas, betas)

@qml.qnode(shot_dev)
def sampling_circuit(gammas, betas):
    prepare_start(wires)
    state_preparation(gammas, betas, cost_h, mixer_h)
    return qml.sample(wires=wires)

def cut_value(bitstring):
    assignment = dict(zip(wires, map(int, bitstring)))
    return sum(assignment[u] != assignment[v] for u, v in graph.edges)

def qaoa_maxcut(p=1, steps=30):
    params = np.array(0.01 * np.random.rand(2, p), requires_grad=True)
    opt = qml.GradientDescentOptimizer(stepsize=0.5)
    for _ in range(steps):
        params = opt.step(objective, params)
    samples = sampling_circuit(params[0], params[1])
    best = max(samples, key=cut_value)
    return params, best, cut_value(best)
```

For MaxCut, `qml.qaoa.maxcut` returns `H_C = Σ_edges ½(Z_i Z_j − I)` and `H_B = Σ_i X_i`.
`qml.qaoa.cost_layer` and `qml.qaoa.mixer_layer` apply `e^{-iγ H_C}` and `e^{-iβ H_B}` via
`ApproxTimeEvolution`; minimizing `⟨H_C⟩` is the same as maximizing the cut count.
