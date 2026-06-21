# Context: Routing logical circuits onto sparsely-coupled NISQ devices

## Research question

A quantum algorithm in the circuit model is written as if any two-qubit gate (a CNOT) may act on any pair of logical qubits. Real Noisy Intermediate-Scale Quantum (NISQ) hardware does not honor that assumption. On a superconducting chip the physical qubits sit on a planar layout and a two-qubit gate can only be applied between two qubits joined by a physical coupler. The coupling graph is sparse — on the kind of 20-qubit device available, a qubit touches only four to six neighbors.

So a logical circuit cannot, in general, be run directly. Before it can execute we must (1) choose an **initial mapping** π from the n logical qubits onto the N physical qubits, and (2) **change that mapping over time** by inserting extra gates, so that whenever a two-qubit gate is reached, its two logical qubits currently sit on adjacent physical qubits. The standard move to change the mapping is a SWAP gate, which exchanges the states of two qubits and itself costs three CNOTs.

Each extra two-qubit gate adds error (CNOT error rates are around 1e-2, and a SWAP is three of them), and extra gates deepen the circuit, which matters because qubit coherence time bounds how many sequential operations can run before the state decays. The goal is therefore: given a circuit and a device coupling graph, produce a hardware-compliant circuit that computes the same function while adding as few gates and as little depth as possible. This is the qubit mapping (or routing) problem, and it has been shown NP-complete (Siraichi et al. 2018), so an exact optimum is out of reach at device scale; a good heuristic is what is needed.

## Background

**The circuit model and the gate set.** A qubit's state is α|0⟩+β|1⟩ with |α|²+|β|²=1; multi-qubit states are vectors over the 2ⁿ basis states, and two-qubit gates create entanglement. Barenco et al. (1995) showed that single-qubit gates together with CNOT form a universal set, and this is exactly the elementary set the superconducting cloud devices expose. The mapping problem concerns only the CNOTs: single-qubit gates act on one qubit and impose no locality constraint, so they can be ignored when reasoning about which qubits must be adjacent.

**The device.** On the relevant superconducting hardware (e.g. the 20-qubit IBM Q20 Tokyo model), qubits are placed on a planar grid; a coupler physically connects a qubit only to its on-chip neighbors. On this generation the coupling is symmetric — a CNOT can be applied in either direction between any connected pair. Reported device characteristics: average single-qubit-gate error ≈ 4.43e-3, measurement error ≈ 8.47e-2, CNOT error ≈ 3.00e-2; average qubit lifetime ≈ 50 µs, with state-of-the-art coherence reaching ≈ 100 µs. Earlier 5- and 16-qubit chips (QX2/QX3/QX5) had **asymmetric** couplings (CNOT allowed in only one direction even between connected qubits), which forced extra transforms; physical work (Chen et al. 2014) showed superconducting couplings can be made symmetric, and on the symmetric 20-qubit device only state-moving SWAPs are needed.

**The SWAP and the distance.** Inserting one SWAP on an edge of the coupling graph exchanges the two qubits there, so a chain of SWAPs can carry a logical qubit to any physical location. With every coupling-graph edge weighted 1, the shortest-path distance between physical qubits is the number of edge-SWAP moves needed to move one logical state from one endpoint to the other. A two-qubit gate only needs adjacency, so the exact SWAP count to make a pair executable is one less than this distance; nearest-neighbor-cost heuristics usually keep the raw shortest-path distance as the score. This shortest-path quantity is the **Nearest Neighbor Cost (NNC)**, a heuristic measure used widely in earlier routing work (Saeedi et al. 2011; Wille et al. 2016; Zulehner et al. 2018).

**Complexity.** Stripped of quantum specifics, moving occupants around a graph by adjacent swaps is the token-swapping problem, which is NP-hard; the time-extended routing version inherits that hardness (it is NP-complete, Siraichi et al. 2018). The initial mapping has a large, proven effect on the final overhead (Siraichi et al. 2018; Zulehner et al. 2018).

## Baselines

**Solver / exact formulations.** One line recasts mapping as a known optimization problem and calls a generic solver: graph isomorphism (Maslov et al. 2008), graph partitioning (Chakrabarti et al. 2011), Minimum Linear Arrangement and Mixed-Integer Programming (Shafaei et al. 2013, 2014), pseudo-Boolean optimization (Wille et al. 2014), temporal planning (Venturelli et al. 2017, 2018).

**Heuristic routers on ideal lattices.** A second line builds heuristic searches assuming an idealized 1D or 2D nearest-neighbor lattice (Saeedi et al. 2011; Lin et al. 2015 (PAQCS); Wille et al. 2016; Kole et al. 2016, 2018).

**Siraichi et al. 2018 ("Qubit Allocation," CGO).** Formalized the problem and proved it NP-complete. Provides an exact dynamic-programming optimum that is exponential and only runs for circuits of ≤8 qubits. Its heuristic determines the initial mapping by counting, for each pair of logical qubits, how many two-qubit gates act on them and matching busy logical qubits to high-degree physical qubits. For movement it resolves one CNOT at a time greedily by gate counts and uses Reverse/Bridge/Swap transforms suited to the asymmetric chips.

**IBM QISKit mapper.** Partitions the circuit into layers of mutually non-overlapping gates and, for each layer, randomly searches (heuristic-guided) for a satisfying mapping, realizing it with SWAPs.

**Zulehner, Paler, Wille 2018 ("Efficient Mapping to IBM QX," DATE/TCAD).** Partitions the circuit into depth layers l₀,l₁,… (each a set of gates on disjoint qubits; the number of layers is the depth). Between consecutive layers it inserts a permutation layer of SWAPs, and finds that permutation with an **A\*** search over combinations of concurrently-applicable SWAPs, minimizing added gates and depth jointly. The A\* node cost is f(x)=g(x)+h(x): g the SWAPs used so far, h an **admissible** estimate equal to the maximum over the layer's CNOTs of (shortest-path-length − 1)·7 gates (plus 4 if a direction reversal is needed) — the max (not the sum) keeps h from overestimating, since one SWAP can shorten several CNOTs. A look-ahead scheme biases toward mappings good for upcoming layers, and the initial mapping is determined by the gates at the **beginning** of the circuit.

## Evaluation settings

The natural yardstick is a set of quantum circuits of varying size and function drawn from existing benchmark suites: small arithmetic circuits and library functions from RevLib, algorithms compiled from Quipper and ScaffCC, quantum-simulation circuits (e.g. Ising-model circuits), and Quantum Fourier Transform circuits of several widths, plus example programs shipped with QISKit. The hardware target is the coupling graph of a real superconducting device (the symmetric 20-qubit IBM Q20 Tokyo model). The metrics are the **total number of gates** and the **circuit depth** of the final hardware-compliant circuit, together with **runtime** and **memory** as scalability measures. The prior strongest method serves as the comparison point, run on the same device model and machine.

## Code framework

The primitives that already exist: a circuit representation, a device `CouplingMap` with methods to test adjacency, list neighbors, and compute pairwise distances, a layout object mapping logical qubits to physical qubits, and an all-pairs-shortest-path routine (Floyd-Warshall). A routing pass consumes a circuit plus a coupling map and emits a hardware-compliant circuit; a layout pass consumes a circuit plus a coupling map and emits an initial mapping. The empty bodies below are the generic slots where the routing strategy and the initial-mapping strategy still have to be designed.

```python
from collections import defaultdict

# --- distance: precompute once from the coupling graph ---
def all_pairs_shortest_path(coupling_map):
    # Floyd-Warshall on G(V,E), every edge weight 1.
    # D[i][j] = shortest-path edge distance between physical qubits i and j.
    pass  # TODO


class Router:
    """Consume circuit + coupling map, emit a hardware-compliant circuit."""
    def __init__(self, coupling_map):
        self.coupling_map = coupling_map
        self.D = all_pairs_shortest_path(coupling_map)

    def run(self, circuit, initial_layout):
        pass  # TODO


class InitialMapper:
    """Consume circuit + coupling map, emit an initial mapping."""
    def __init__(self, coupling_map, router):
        self.coupling_map = coupling_map
        self.router = router

    def run(self, circuit):
        pass  # TODO
```
