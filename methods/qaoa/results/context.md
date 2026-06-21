# Context: approximate combinatorial optimization on a quantum computer (circa 2000–2014)

## Research question

A combinatorial optimization instance is specified by `n` bits and `m` clauses. Each clause
`C_α` is a constraint on a small subset of the bits — satisfied (`C_α(z) = 1`) for some
assignments of those bits and unsatisfied (`C_α(z) = 0`) for the rest. The objective is the
number of satisfied clauses,

```
C(z) = Σ_{α=1}^m C_α(z),   z = z_1 z_2 … z_n ∈ {0,1}^n.
```

Satisfiability asks whether some `z` satisfies every clause; MaxSat asks for the `z` that
maximizes `C`; approximate optimization asks only for a `z` whose `C(z)` is *close* to the
maximum. MaxCut is the canonical example: given a graph, two-color the vertices so as to
maximize the number of edges whose endpoints get different colors. These problems are NP-hard
in general, and even constant-factor approximation is hard for some of them, so the realistic
goal is a guaranteed *approximation ratio* `C(z)/max C` rather than the exact optimum.

The question is how to design a quantum circuit procedure for approximate combinatorial
optimization that is practical for near-term gate-model hardware.

## Background

By the early 2010s the dominant quantum proposal for optimization was **adiabatic quantum
computation / quantum annealing** (Farhi, Goldstone, Gutmann, Sipser, 2000,
`quant-ph/0001106`). Its premises are the load-bearing facts the rest of this landscape rests
on.

*Encoding the optimum as a ground state.* Replace each bit `z_i` by a qubit. The objective
becomes a diagonal operator `C` on the `2^n`-dimensional Hilbert space with computational
basis `|z⟩`, `C|z⟩ = C(z)|z⟩`. Finding the best string is then finding the extremal eigenstate
of `C`. A "problem Hamiltonian" `H_P` is built so that its ground state encodes the optimal
assignment.

*The easy starting Hamiltonian.* Alongside `H_P` one takes a "beginning" Hamiltonian whose
ground state is trivial to prepare. The standard choice is the transverse field
`H_B = Σ_i ½(1 − σˣ_i)`, a sum of single-qubit terms. Its ground state is the uniform
superposition over all basis states,

```
|s⟩ = (1/√(2^n)) Σ_z |z⟩ = |+⟩_1 |+⟩_2 … |+⟩_n,
```

because `|+⟩` is the `+1` eigenstate of `σˣ`. A uniform superposition is a single layer of
Hadamards deep.

*Adiabatic interpolation and the adiabatic theorem.* Define a time-dependent Hamiltonian that
slowly drags `H_B` into `H_P`,

```
H(t) = (1 − t/T) H_B + (t/T) H_P,   0 ≤ t ≤ T,
```

equivalently `H̃(s) = (1−s)H_B + s H_P` with `s = t/T`. Start in the ground state of `H̃(0) =
H_B`. The **adiabatic theorem** states that if the spectral gap `E_1(s) − E_0(s)` between the
ground and first-excited instantaneous eigenstates stays strictly positive for all `s`, then
the evolved state stays arbitrarily close to the instantaneous ground state as `T → ∞`, so it
ends in the ground state of `H_P`. Quantitatively, success to high fidelity needs

```
T ≫ ℰ / g_min²,   g_min = min_s (E_1(s) − E_0(s)),   ℰ = max_s |⟨1;s| dH̃/ds |0;s⟩|.
```

With `ℰ` typically of order a single eigenvalue, the **required runtime is governed by
`1/g_min²`** — it scales as the inverse square of the smallest gap encountered along the
path. The adiabatic procedure's success probability also varies non-monotonically with `T` for
some instances (Crosson, Farhi, Lin, Lin, Shor, 2014, `arXiv:1401.7320`). There are also
instances where the adiabatic evolution gets trapped for subexponential runtimes — for a
Hamming-weight-symmetric objective the evolution is stuck at weight `w = n` while the true
optimum is at `w = 0` (Farhi, Goldstone, Gutmann, 2002, `quant-ph/0201031`).

*Three standard mathematical tools sit on the table.* **Trotterization** (the Lie–Trotter product formula): for non-commuting `A` and `B`,
`e^{−i(A+B)t} = (e^{−iA t/N} e^{−iB t/N})^N + O(t²/N)`, so a continuous evolution under a sum of
two terms can be approximated by alternating short evolutions under each term separately, with
error controllable by taking more, smaller steps. **The Perron–Frobenius theorem**: the
transverse-field driver `B = Σ_i σˣ_i` has non-negative off-diagonal entries and connects the
hypercube of bit strings, so its top eigenstate is non-degenerate and separated from the rest;
equivalently, the shifted Hamiltonian `Σ_i ½(1 − σˣ_i)` has the same easy state as its
non-degenerate ground state. That is the gap-positivity structure an extremal-state adiabatic
path needs. **The variational principle**: for any Hermitian `C` and any normalized state `|ψ⟩`,
`⟨ψ|C|ψ⟩ ≤ C_max`, the largest eigenvalue. Any state's expected objective is thus a rigorous
lower bound on the true optimum — one can never exceed it, and an expectation value is an honest
quantity to report.

*Classical yardsticks for the example problems.* For MaxCut, the Goemans–Williamson semidefinite-
programming rounding achieves ratio `0.878` on general graphs; for cubic (3-regular) graphs
there are dedicated combinatorial algorithms (Halperin, Livnat, Zwick, 2004). For the
bounded-occurrence linear problem Max E3LIN2 (each constraint is a mod-2 sum of exactly three
bits equal to 0 or 1, each bit in at most `D+1` equations), a random assignment already
satisfies half the equations; Håstad (2000) gave a classical algorithm reaching
`½ + constant/D`, and Trevisan (2001) showed that beating `½ + constant/D^{1/2}` for large
enough constant would imply P = NP, while general-instance `½ + ε` is itself NP-hard (Håstad,
2001).

## Baselines

**Quantum adiabatic algorithm (QAA) / quantum annealing** (Farhi et al, 2000). *Idea:* encode
the optimum in `H_P`'s ground state, prepare `H_B`'s easy ground state `|s⟩`, and evolve under
the slowly varying `H(t) = (1−t/T)H_B + (t/T)H_P` for a long time `T`, relying on the adiabatic
theorem to track the ground state into the answer. *Math:* one continuous Schrödinger
evolution `i d|ψ⟩/dt = H(t)|ψ⟩`; correctness governed by `T ≫ ℰ/g_min²`.

**Continuous-time quantum walks / quantum decision trees** (Farhi, Gutmann, 1997,
`quant-ph/9706062`). *Idea:* evolve `e^{−ibB}` where `B` is the adjacency matrix of a graph
whose vertices are configurations and whose edges connect configurations differing by one
move; amplitude spreads across the graph like a wave. *Math:* a single `e^{−ibB}` applied to a
localized starting vertex.

**Simulated annealing (classical).** *Idea:* a thermal random walk over assignments with a
temperature lowered toward zero, settling into low-energy (good) configurations. *Math:*
Metropolis transitions with a cooling schedule.

**Classical approximation algorithms for the example problems.** Goemans–Williamson SDP
rounding (`0.878` for MaxCut), Halperin–Livnat–Zwick for cubic MaxCut, and Håstad's
bounded-occurrence E3LIN2 algorithm (`½ + constant/D`). *Idea/math:* polynomial-time rounding
of relaxations or combinatorial arguments.

## Evaluation settings

The natural testbeds are bounded-degree combinatorial problems where locality is explicit.
**MaxCut on regular graphs**: 2-regular (rings) and 3-regular (cubic) graphs, with the
approximation ratio `C(z)/max C` as the metric; for cubic graphs the natural structural
parameters are the counts of small subgraphs (isolated triangles, crossed squares). **Max
E3LIN2 with bounded occurrence**: each equation a mod-2 sum of three bits, each bit in at most
`D+1` equations, the metric being the fraction of equations satisfied, with `½` (random
guessing) as the floor and the `D`-dependence (`1/D` versus `1/D^{1/2}`) of the excess
over `½` as the figure of merit. The protocol is: prepare a parameterized state on the quantum
computer, measure in the computational basis to read off a candidate string, evaluate its
objective, and repeat to estimate the mean objective; the parameters are either fixed by
offline classical analysis or tuned by repeated quantum evaluations.

## Code framework

The primitives already available are a quantum-circuit library that can place single-qubit
rotations and entangling gates and measure in the computational basis, plus a classical
optimizer. The scaffold keeps objective construction, the easy starting state, expectation
measurement, sampling, and classical parameter search explicit; the state-preparation routine
is the design slot.

```python
import networkx as nx
import pennylane as qml
from pennylane import numpy as np

def build_objective_and_driver(graph):
    # TODO: assemble the diagonal objective from local clauses and choose a simple driver.
    objective_op = None
    driver_op = None
    return objective_op, driver_op

def prepare_start(wires):
    for w in wires:
        qml.Hadamard(wires=w)   # |s> = |+>^n, the easy transverse-field state

def state_preparation(params, objective_op, driver_op):
    # TODO: the parameterized gate sequence whose measured strings are good candidates.
    pass

dev = qml.device("default.qubit", wires=...)

@qml.qnode(dev)
def expectation_circuit(params, graph, wires):
    objective_op, driver_op = build_objective_and_driver(graph)
    prepare_start(wires)
    state_preparation(params, objective_op, driver_op)
    return qml.expval(objective_op)

@qml.qnode(dev)
def sampling_circuit(params, graph, wires):
    objective_op, driver_op = build_objective_and_driver(graph)
    prepare_start(wires)
    state_preparation(params, objective_op, driver_op)
    return qml.sample(wires=wires)                   # read a candidate string

def objective(params, graph, wires):
    # maximize the objective; optimizers minimize, so return the chosen sign convention
    return -expectation_circuit(params, graph, wires)

def optimize(graph, wires):
    params = init_params()                           # TODO: shape set by the slot above
    opt = qml.GradientDescentOptimizer(stepsize=0.5)
    for _ in range(n_steps):
        params = opt.step(lambda p: objective(p, graph, wires), params)
    return params
```
