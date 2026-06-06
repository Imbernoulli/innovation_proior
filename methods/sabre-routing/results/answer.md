# SABRE — SWAP-based bidirectional heuristic routing for the qubit mapping problem

## Problem

A logical quantum circuit assumes any two-qubit gate (CNOT) can act on any pair of qubits. A NISQ superconducting device only allows a two-qubit gate between physically coupled (adjacent) qubits, and the coupling graph G(V,E) is sparse. To run the circuit, one must choose an **initial mapping** π from logical to physical qubits and **change it during execution** by inserting SWAP gates (each = 3 CNOTs), so that every CNOT's two qubits are adjacent when it runs — while minimizing the number of added gates and the depth. The problem is NP-complete, so the method is a greedy heuristic.

## Key ideas

1. **SWAP-based greedy search instead of mapping/permutation search.** Represent the two-qubit gates as a dependency DAG; the **front layer** F is the set of gates with no unexecuted predecessor. Loop: execute every F-gate whose qubits are already adjacent (single-qubit gates are free); when none is executable, insert one SWAP and update π. Only SWAPs touching a qubit in F are candidates — SWAPs among idle qubits cannot help — which shrinks the per-step search space from O(exp(N)) to O(N), giving an exponential speedup over exhaustive per-layer search. Per two-qubit gate the cost is O(N^{2.5}).

2. **Distance-based heuristic cost with look-ahead and decay.** Precompute the all-pairs shortest-path distance matrix D on G (Floyd–Warshall, edge weight 1). D[i][j] is the shortest-path edge distance between physical qubits i and j; making a two-qubit gate executable would require D[i][j]−1 SWAPs for that pair, but the heuristic uses the raw nearest-neighbor distance D as its ranking signal. Score a candidate SWAP by the layout it would produce:

   - basic: H = Σ_{gate∈F} D[π(q₁)][π(q₂)].
   - look-ahead: add an **extended set** E of the nearest near-future successor gates, normalize each nonempty set by its own size, and down-weight E by W (0 ≤ W < 1):
     A_F = (1/|F|) Σ_F D, A_E = 0 if E=∅ else (1/|E|) Σ_E D, H = A_F + W·A_E.
   - decay: multiply by max(decay(SWAP.q₁), decay(SWAP.q₂)), where each qubit's decay starts at 1 and is bumped by δ when swapped, reset periodically. This penalizes piling SWAPs on the same qubit, biasing toward parallel (depth-minimizing) SWAPs; δ tunes the gate-count-versus-depth trade-off:

     H = max(decay(q₁), decay(q₂)) · (A_F + W·A_E).

   Pick the minimum-H candidate (random tie-break), apply, update π.

3. **Reverse-traversal initial mapping.** Quantum circuits are reversible: the reversed circuit has the same two-qubit gates in reverse order. Routing forward from a random π₀ ends at a final mapping good for the *end* of the circuit; feeding that as the initial mapping to route the *reversed* circuit ends at a mapping good for the *beginning* of the original — globally informed but front-weighted. The three traversals are forward, backward, and the final real forward routing pass; the layout-refinement traversals can run in fake mode, emitting no SWAPs. Run from several random seeds and keep the best.

## Algorithm (routing, one traversal)

```
build DAG of 2-qubit gates; F = indegree-0 gates; given initial mapping π
while F not empty:
    ready = []                              # executable now
    for g in F: if π(g.q1), π(g.q2) adjacent in G: ready.append(g)
    if ready:
        for g in ready: emit g; remove g from F; add newly-resolved successors to F
        continue
    ext_set = closest 2-qubit successors of F, capped at EXTENDED_SET_SIZE
    candidates = { SWAP(a,b) : a or b is a front-layer qubit, (π(a),π(b)) is an edge }
    for s in candidates: score[s] = H(F, ext_set, π after s, s)
    s* = argmin score   (random tie-break);  emit s*;  π = π after s*;  update decay
```

## Code

```python
from collections import defaultdict
from copy import copy, deepcopy
import numpy as np

from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.circuit.quantumregister import Qubit
from qiskit.converters import dag_to_circuit
from qiskit.transpiler.basepasses import AnalysisPass, TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.passes.layout.apply_layout import ApplyLayout
from qiskit.transpiler.passes.layout.enlarge_with_ancilla import EnlargeWithAncilla
from qiskit.transpiler.passes.layout.full_ancilla_allocation import FullAncillaAllocation
from qiskit.transpiler.passes.layout.set_layout import SetLayout
from qiskit.dagcircuit import DAGNode

EXTENDED_SET_SIZE = 20       # |E|: look-ahead window size
EXTENDED_SET_WEIGHT = 0.5    # W: weight of E relative to F
DECAY_RATE = 0.001           # delta: per-use decay bump
DECAY_RESET_INTERVAL = 5     # reset all decays every few steps


class SabreSwap(TransformationPass):
    """Route onto a coupling map via SWAP insertion."""

    def __init__(self, coupling_map, heuristic="basic", seed=None, fake_run=False):
        super().__init__()
        if coupling_map.is_symmetric:
            self.coupling_map = coupling_map
        else:
            self.coupling_map = deepcopy(coupling_map)
            self.coupling_map.make_symmetric()
        self.heuristic = heuristic
        self.seed = seed
        self.fake_run = fake_run
        self.applied_predecessors = None
        self.qubits_decay = None
        self._bit_indices = None

    def run(self, dag):
        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("Sabre swap runs on physical circuits only.")
        if len(dag.qubits) > self.coupling_map.size():
            raise TranspilerError("More virtual qubits exist than physical.")

        rng = np.random.default_rng(self.seed)
        mapped_dag = None if self.fake_run else dag._copy_circuit_metadata()
        canonical_register = dag.qregs["q"]
        current_layout = Layout.generate_trivial_layout(canonical_register)
        self._bit_indices = {bit: idx for idx, bit in enumerate(canonical_register)}
        self.qubits_decay = {qubit: 1 for qubit in dag.qubits}

        num_search_steps = 0
        front_layer = dag.front_layer()
        self.applied_predecessors = defaultdict(int)
        for _, input_node in dag.input_map.items():
            for successor in self._successors(input_node, dag):
                self.applied_predecessors[successor] += 1

        while front_layer:
            execute_gate_list = []
            for node in front_layer:
                if len(node.qargs) == 2:
                    v0, v1 = node.qargs
                    if self.coupling_map.graph.has_edge(current_layout[v0], current_layout[v1]):
                        execute_gate_list.append(node)
                else:  # single-qubit gates / barriers are free
                    execute_gate_list.append(node)

            if execute_gate_list:
                for node in execute_gate_list:
                    self._apply_gate(mapped_dag, node, current_layout, canonical_register)
                    front_layer.remove(node)
                    for successor in self._successors(node, dag):
                        self.applied_predecessors[successor] += 1
                        if self._is_resolved(successor):
                            front_layer.append(successor)
                    if node.qargs:
                        self._reset_qubits_decay()
                continue

            extended_set = self._obtain_extended_set(dag, front_layer)
            swap_candidates = self._obtain_swaps(front_layer, current_layout)
            swap_scores = dict.fromkeys(swap_candidates, 0)
            for swap_qubits in swap_scores:
                trial_layout = current_layout.copy()
                trial_layout.swap(*swap_qubits)
                swap_scores[swap_qubits] = self._score_heuristic(
                    self.heuristic, front_layer, extended_set, trial_layout, swap_qubits)
            min_score = min(swap_scores.values())
            best_swaps = [k for k, v in swap_scores.items() if v == min_score]
            best_swaps.sort(key=lambda x: (self._bit_indices[x[0]], self._bit_indices[x[1]]))
            best_swap = rng.choice(best_swaps)
            swap_node = DAGNode(op=SwapGate(), qargs=best_swap, type="op")
            self._apply_gate(mapped_dag, swap_node, current_layout, canonical_register)
            current_layout.swap(*best_swap)

            num_search_steps += 1
            if num_search_steps % DECAY_RESET_INTERVAL == 0:
                self._reset_qubits_decay()
            else:
                self.qubits_decay[best_swap[0]] += DECAY_RATE
                self.qubits_decay[best_swap[1]] += DECAY_RATE

        self.property_set["final_layout"] = current_layout
        return dag if self.fake_run else mapped_dag

    def _apply_gate(self, mapped_dag, node, current_layout, canonical_register):
        if self.fake_run:
            return
        new_node = _transform_gate_for_layout(node, current_layout, canonical_register)
        mapped_dag.apply_operation_back(new_node.op, new_node.qargs, new_node.cargs)

    def _reset_qubits_decay(self):
        self.qubits_decay = {k: 1 for k in self.qubits_decay.keys()}

    def _successors(self, node, dag):
        for _, successor, edge_data in dag.edges(node):
            if successor.type != "op":
                continue
            if isinstance(edge_data, Qubit):
                yield successor

    def _is_resolved(self, node):
        return self.applied_predecessors[node] == len(node.qargs)

    def _obtain_extended_set(self, dag, front_layer):
        extended_set, incremented = [], []
        tmp_front_layer = front_layer
        done = False
        while tmp_front_layer and not done:
            new_tmp_front_layer = []
            for node in tmp_front_layer:
                for successor in self._successors(node, dag):
                    incremented.append(successor)
                    self.applied_predecessors[successor] += 1
                    if self._is_resolved(successor):
                        new_tmp_front_layer.append(successor)
                        if len(successor.qargs) == 2:
                            extended_set.append(successor)
                if len(extended_set) >= EXTENDED_SET_SIZE:
                    done = True
                    break
            tmp_front_layer = new_tmp_front_layer
        for node in incremented:
            self.applied_predecessors[node] -= 1
        return extended_set

    def _obtain_swaps(self, front_layer, current_layout):
        candidate_swaps = set()
        for node in front_layer:
            for virtual in node.qargs:
                physical = current_layout[virtual]
                for neighbor in self.coupling_map.neighbors(physical):
                    virtual_neighbor = current_layout[neighbor]
                    swap = sorted([virtual, virtual_neighbor], key=lambda q: self._bit_indices[q])
                    candidate_swaps.add(tuple(swap))
        return candidate_swaps

    def _compute_cost(self, layer, layout):
        cost = 0
        for node in layer:
            cost += self.coupling_map.distance(layout[node.qargs[0]], layout[node.qargs[1]])
        return cost

    def _score_heuristic(self, heuristic, front_layer, extended_set, layout, swap_qubits=None):
        first_cost = self._compute_cost(front_layer, layout)
        if heuristic == "basic":
            return first_cost
        first_cost /= len(front_layer)
        second_cost = 0
        if extended_set:
            second_cost = self._compute_cost(extended_set, layout) / len(extended_set)
        total_cost = first_cost + EXTENDED_SET_WEIGHT * second_cost
        if heuristic == "lookahead":
            return total_cost
        if heuristic == "decay":
            return max(self.qubits_decay[swap_qubits[0]],
                       self.qubits_decay[swap_qubits[1]]) * total_cost
        raise TranspilerError("Heuristic %s not recognized." % heuristic)


def _transform_gate_for_layout(op_node, layout, device_qreg):
    mapped_op_node = copy(op_node)
    mapped_op_node.qargs = [device_qreg[layout[x]] for x in op_node.qargs]
    return mapped_op_node
```

The reverse-traversal initial mapping wraps this router, running it forward/backward and keeping only the settled layout:

```python
class SabreLayout(AnalysisPass):
    """Choose an initial mapping by forward/backward routing (fake_run)."""

    def __init__(self, coupling_map, routing_pass=None, seed=None, max_iterations=3):
        super().__init__()
        self.coupling_map = coupling_map
        self.routing_pass = routing_pass
        self.seed = seed
        self.max_iterations = max_iterations

    def run(self, dag):
        if len(dag.qubits) > self.coupling_map.size():
            raise TranspilerError("More virtual qubits exist than physical.")
        if self.seed is None:
            self.seed = np.random.randint(0, np.iinfo(np.int32).max)
        rng = np.random.default_rng(self.seed)

        physical_qubits = rng.choice(self.coupling_map.size(), len(dag.qubits), replace=False)
        physical_qubits = rng.permutation(physical_qubits)
        initial_layout = Layout({q: dag.qubits[i] for i, q in enumerate(physical_qubits)})

        if self.routing_pass is None:
            self.routing_pass = SabreSwap(self.coupling_map, "decay", seed=self.seed, fake_run=True)
        else:
            self.routing_pass.fake_run = True

        circ = dag_to_circuit(dag)
        rev_circ = circ.reverse_ops()
        for _ in range(self.max_iterations):
            for _ in ("forward", "backward"):
                pm = self._layout_and_route_passmanager(initial_layout)
                new_circ = pm.run(circ)
                final_layout = self._compose_layouts(
                    initial_layout, pm.property_set["final_layout"], new_circ.qregs)
                initial_layout = final_layout
                circ, rev_circ = rev_circ, circ   # alternate forward / reverse

        for qreg in dag.qregs.values():
            initial_layout.add_register(qreg)
        self.property_set["layout"] = initial_layout
        self.routing_pass.fake_run = False

    def _layout_and_route_passmanager(self, initial_layout):
        layout_and_route = [
            SetLayout(initial_layout),
            FullAncillaAllocation(self.coupling_map),
            EnlargeWithAncilla(),
            ApplyLayout(),
            self.routing_pass,
        ]
        return PassManager(layout_and_route)

    def _compose_layouts(self, initial_layout, pass_final_layout, qregs):
        trivial_layout = Layout.generate_trivial_layout(*qregs)
        qubit_map = Layout.combine_into_edge_map(initial_layout, trivial_layout)
        final_layout = {
            v: pass_final_layout[qubit_map[v]]
            for v, _ in initial_layout.get_virtual_bits().items()
        }
        return Layout(final_layout)
```

Typical configuration: |E| ≈ 20, W = 0.5, δ starting at 0.001, decay reset every 5 search steps or whenever a gate executes; run several random initial mappings, each with a forward–backward–forward traversal, and keep the best.
