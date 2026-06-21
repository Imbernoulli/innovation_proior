A quantum algorithm is written as if any CNOT may act on any pair of logical qubits, but a real NISQ superconducting device honors no such freedom: its physical qubits sit on a planar chip and a two-qubit gate can fire only between two qubits joined by a coupler. The coupling graph $G(V,E)$ is sparse — on a 20-qubit device each qubit touches only four to six neighbors — so most CNOTs in a circuit are simply not executable as written, because the two logical qubits they name do not currently sit on adjacent physical qubits. To run such a circuit one must do two things: choose an initial mapping $\pi$ from logical to physical qubits, and then change $\pi$ over time, because no single fixed mapping can keep every CNOT in a long circuit local. The only legal move that changes which physical qubit holds which logical state is a SWAP between two adjacent qubits; chained, SWAPs can walk a logical qubit anywhere on the chip. But SWAPs are not free — a SWAP is three CNOTs, each erring at about a percent, and every inserted SWAP both raises the total error and deepens the circuit, which matters because coherence lasts only tens of microseconds. The goal is therefore not merely a legal SWAP schedule but the one that adds the fewest gates and the least depth. Stripped of its quantum dressing this is time-extended token swapping on a graph with a chosen initial arrangement, which is NP-complete, so an exact optimum at device scale is off the table and a good greedy heuristic is what is needed.

The prior approaches each hit a definite wall. Recasting the problem as a textbook optimization and handing it to a generic solver — graph isomorphism, partitioning, minimum linear arrangement, MIP, pseudo-Boolean, temporal planning — discards all of the routing structure (that demands arrive in a temporal stream, that a SWAP far from the action buys nothing) and so only finishes on tiny circuits. Heuristic searches built on idealized 1D/2D lattices do not transfer, because a NISQ chip is irregular, not a clean lattice. Siraichi and colleagues formalized the problem and gave an exact dynamic-programming optimum that is exponential and dies past eight qubits; their fast heuristic picks the initial mapping from a frequency count of which logical pairs interact, with no sense of *time*, and moves qubits one CNOT at a time with no look-ahead. The strongest prior method, Zulehner, Paler and Wille, slices the circuit into depth layers and between consecutive layers searches with admissible-heuristic A\* for the best *permutation* of qubits — a whole bundle of concurrent SWAPs at once — but the per-layer search ranges in the worst case over $m!/(m-n)!$ arrangements, so it is exponential in the number of qubits and runs out of memory on 16- to 20-qubit circuits, and its initial mapping reflects only the gates at the very beginning. Two pain points fall out: the fast methods are myopic, the good method is exponential, and *everybody's* initial mapping is local. What is wanted is a search whose per-step cost is polynomial and a genuinely global way to pick $\pi$.

I propose SABRE — a SWAP-based bidirectional heuristic router. The first idea is to stop searching for a *mapping* or a whole permutation and instead search for *one SWAP at a time*, among only the SWAPs that could plausibly help. The structure that governs adjacency is the dependency among the two-qubit gates: single-qubit gates impose no locality constraint and can be emitted whenever their qubit is free, so I build a directed acyclic graph with a node per CNOT and an edge $g_i\to g_j$ when they share a qubit and $g_i$ comes first. The gates runnable right now are exactly the DAG nodes with no unexecuted predecessor — indegree zero — which I call the front layer $F$. The main loop then writes itself: while $F$ is nonempty, scan it and execute every gate whose two logical qubits currently map to adjacent physical qubits, dropping each executed gate and pulling in any successor whose predecessors are now all done; if at least one gate ran, loop again, since that may have freed more. Only when nothing in $F$ is executable am I genuinely stuck and must insert a SWAP. The key pruning is in which SWAPs are candidates: only those touching a qubit that appears in some front-layer gate. A SWAP among idle qubits cannot help the gates I am blocked on — it merely burns a gate shuffling occupants — so for each logical qubit in $F$ I look at its physical location's neighbors on the coupling graph, and the SWAP exchanging it with each neighbor is a candidate. Every such candidate is legal by construction and moves a blocked qubit. In the worst case all qubits are in $F$, each contributing its handful of neighbors, so the candidate list is $O(N)$ rather than $O(\exp(N))$; I pay a little back by committing one SWAP at a time and so taking more steps — at most the chip diameter, about $\sqrt{N}$ — for a per-two-qubit-gate cost of order $N^{2.5}$, which is polynomial. That is the scalability win.

What decides whether this greedy search is any good is how I score a candidate SWAP. The natural measure of how far a CNOT is from executable is the distance between its two qubits on the coupling graph, so I run Floyd–Warshall once, $O(N^3)$ and a one-time cost since $N$ is at most a few hundred, to get the matrix $D$ where $D[i][j]$ is the shortest-path edge distance between physical qubits $i$ and $j$. Moving a logical state all the way from $Q_i$ to $Q_j$ takes $D[i][j]$ edge SWAPs and making a single CNOT executable needs only $D[i][j]-1$; but the score is a ranking signal, not an exact routed-SWAP count, and the nearest-neighbor-cost convention keeps the raw distance $D$, which also keeps already-adjacent near-future gates visible at distance 1 rather than vanishing to zero when look-ahead and decay are mixed in. The simplest score sums front-layer distances under the layout the candidate SWAP would produce,
$$H_{\text{basic}} = \sum_{g\in F} D[\pi(g.q_1)][\pi(g.q_2)],$$
and I keep the SWAP minimizing it — small sum means the stuck gates are collectively close to executable. I sum over the whole front layer rather than scoring a single worst gate because I am being greedy about total remaining work, not one bottleneck; the A\* method used a max over its layer only because it needed an *admissible* underestimate to stay optimal, and I am not running A\*.

$H_{\text{basic}}$ is too myopic. If the front layer is $\{\text{CNOT}(q_1,q_2), \text{CNOT}(q_3,q_7)\}$ with a $\text{CNOT}(q_2,q_7)$ waiting right behind, two candidates can reduce the front-layer sum equally while one of them also lines up $q_2$ and $q_7$ for the next gate and the other shoves them apart — and $H_{\text{basic}}$, which sees only $F$, cannot tell them apart, so it sometimes resolves the immediate blockage only to undo itself one step later. The fix is to let the score peek a bounded distance past the front layer. I define an extended set $E$: up to a small constant number (in practice $|E|=20$) of the *closest* DAG successors of the front-layer gates — those about to become front-layer next — and fold their distance sum into the score. The two sums must be combined carefully. $F$ and $E$ generally differ in size, so I normalize each by its own cardinality lest the larger set dominate for no reason, taking $E$'s average to be zero when $E$ is empty; and the two are not equally urgent, since every front-layer gate *must* execute before anything in $E$ can even become runnable while $E$ is only a hint, so I down-weight $E$ by a factor $W$ with $0\le W<1$ (in practice $W=0.5$, present but clearly secondary):
$$A_F = \frac{1}{|F|}\sum_{g\in F} D[\pi(q_1)][\pi(q_2)],\qquad A_E = \begin{cases}0 & E=\varnothing\\ \frac{1}{|E|}\sum_{g\in E} D[\pi(q_1)][\pi(q_2)] & \text{otherwise}\end{cases},\qquad H = A_F + W\cdot A_E.$$
$E$ is deliberately shallow: the further out I look, the more the mapping will have drifted by the time those gates arrive, so far-future distances under today's tentative layout are noise, and a bigger $E$ also costs more every step — a couple-dozen-gate window breaks the myopia without paying for unreliable estimates.

That handles gate count but nothing in the score yet knows about depth. Depth grows unnecessarily when I pile SWAP after SWAP onto the *same* qubits, forcing them to serialize; spreading the needed SWAPs across disjoint qubits lets them run in parallel so depth barely grows. So I give every qubit a decay value starting at 1, bump a qubit's decay by a small $\delta$ each time it is involved in a chosen SWAP, and multiply the whole candidate cost by the larger of its two qubits' decays:
$$H = \max\!\big(\text{decay}(\text{SWAP}.q_1),\ \text{decay}(\text{SWAP}.q_2)\big)\cdot\big(A_F + W\cdot A_E\big).$$
The max — not the sum or product — is right because a SWAP is bad for parallelism if *either* endpoint was just moved, so the more-recently-used endpoint should set the penalty. A SWAP touching a freshly moved qubit is nudged up, so among comparable candidates the search drifts toward untouched qubits, which parallelize and hold depth down. To stop the penalty from accumulating into a permanent obsession with avoiding a few qubits, I reset all decays to 1 periodically — every few search steps (in practice every 5) and whenever a gate actually executes, which begins a fresh phase. The rate $\delta$ (in practice $0.001$) is then a genuine, tunable knob: larger $\delta$ fights harder for parallel, depth-minimizing SWAPs at the cost of perhaps a few more of them; smaller $\delta$ minimizes gate count more aggressively at the cost of depth. Pushed too far it preferentially moves untouched qubits instead of resolving the gate it is stuck on, inserting pointless SWAPs and worsening both metrics — so there is a sweet spot, not a free lunch. Evaluating the cost is cheap, $O(N)$ to sum over $F$ and $E$, across $O(N)$ candidates, which is exactly the per-gate complexity quoted above.

This still assumes I have an initial mapping $\pi$, and that is precisely where everyone falls short. I want a $\pi$ informed by the *whole* circuit but weighted toward the beginning, since the beginning is what runs first and what $\pi$ directly determines — yet scoring a mapping against all gates is again a global search. The structure I exploit is a symmetry of the routing constraints: for routing I only need the sequence of logical two-qubit pairs, not the semantic inverse of the algorithm, and reversing the operation list gives the same pairs in opposite order — a free mirror image of the problem. My router, handed an initial mapping and a circuit, returns the *final* mapping the qubits are in when the last gate runs, which by construction suits the *end* of the circuit because the router worked the whole way to make it so. Turn that around: route forward from a random $\pi_0$ to a final layout good for the end, then use that layout as the initial mapping for a traversal over the *reversed* order; that traversal finishes at the beginning of the original order, so the layout it settles on is good for the *beginning* — and it is not myopic, because it was produced by routing a full pass of every gate, with the gates nearest the original start (last in the reverse pass, closest to where the layout settles) influencing it most and the rest influencing it less but never ignored. That is exactly the global-but-front-loaded weighting I wanted, obtained from the constraint symmetry rather than any extra search. So the initial mapping is not computed up front but *refined* by running the router back and forth: from a random $\pi_0$, a fake forward traversal, then a fake backward traversal over the reversed order, repeated for `max_iterations` rounds, keeping only the settled layout (not the SWAPs) as the start for the real forward route. Because the router is greedy and seeded by a random start, I run the whole thing from several $\pi_0$ samples and keep the best SWAP count.

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
