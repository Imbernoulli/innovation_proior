OK, let me think this through from scratch. I have a quantum circuit written the way everyone writes them — a list of gates, where each two-qubit gate (a CNOT) acts on whatever pair of logical qubits the algorithm wants. And I have a real superconducting device, where the qubits sit on a planar chip and a two-qubit gate can physically only happen between two qubits that share a coupler. The coupling graph is sparse: on the 20-qubit device a qubit touches four to six neighbors, not all nineteen others. So most of the CNOTs in my circuit are simply not executable as written, because the two logical qubits they name don't currently sit on adjacent physical qubits.

So I need two things. First an initial assignment of logical qubits to physical qubits — call it π. And second, because no single fixed π can possibly make every CNOT in a long circuit local (the circuit will want q3 next to q7 here and next to q1 there), I need to *change* π as the circuit runs. The only legal move that changes which physical qubit holds which logical state is a SWAP: it exchanges the states of two adjacent physical qubits. Chain a few SWAPs and I can walk a logical qubit anywhere on the chip. So the plan is: run the circuit, and whenever the next CNOT's two qubits aren't adjacent, insert SWAPs to drag them together, then do the CNOT.

But SWAPs are not free, and this is the whole tension. A SWAP is three CNOTs, and CNOTs on this hardware err at around a percent each. Every SWAP I insert raises the total error and lengthens the circuit, and the circuit can't get too long because the qubits only stay coherent for tens of microseconds. So I'm not just looking for *a* legal schedule of SWAPs — I want the one that adds the fewest gates and the least depth. Minimize added SWAPs, basically, with an eye on depth too.

Now, how hard is this really? Forget the quantum dressing for a second. I have tokens (logical qubits) sitting on the vertices of a graph (the chip), and I'm allowed to swap tokens across edges, and I have a schedule of "these two tokens must be adjacent at this time" constraints to satisfy with as few swaps as possible. The static version of that — permute tokens to a target arrangement using fewest adjacent swaps — is token swapping, and that's NP-hard. My version is worse: it's time-extended, with a sequence of adjacency demands, and the initial arrangement is also mine to choose. It's been proven NP-complete. So an exact optimum at device scale is off the table. I'm going to have to be greedy and clever, not optimal.

Let me look at what people have already tried, because I want to know exactly where each approach hits a wall — that tells me what to fix.

One camp recasts the whole thing as some textbook optimization and hands it to a generic solver: graph isomorphism, graph partitioning, minimum linear arrangement, mixed-integer programming, pseudo-Boolean, temporal planning. The trouble is uniform: a general solver doesn't know anything about the structure of *this* problem — that the demands come in a temporal stream, that a SWAP near where the action is matters and a SWAP far away doesn't — so it grinds, and only finishes on tiny circuits.

Another camp builds heuristic searches but assumes an idealized 1D or 2D lattice, where distances are just grid coordinates. Clean, but a NISQ chip isn't a clean lattice; the coupling is irregular and more restricted. Those methods don't transfer.

Then the ones that actually target real chips. Siraichi and colleagues formalized the problem and gave an exact dynamic-programming optimum — exponential, dies past eight qubits. Their fast heuristic picks the initial mapping by counting how many two-qubit gates touch each pair of logical qubits and matching the busy ones onto high-degree physical qubits — but that's a frequency count with no sense of *time*: a pair that interacts a lot at the very end of the circuit gets the same vote as one that interacts at the very start. And their movement resolves one CNOT at a time, greedily, by gate counts, never looking at what a local move does to the gates just downstream.

The strongest one is Zulehner, Paler, and Wille. They slice the circuit into depth layers — each layer a batch of gates on disjoint qubits — and between consecutive layers they search for the best *permutation* of the qubits, a whole bundle of SWAPs to apply at once, using A\* with an admissible heuristic. It's good, and it's the best results around. But look at the cost: finding the permutation for one layer means, in the worst case, considering m!/(m−n)! arrangements — exponential in the number of qubits. That's not a constant they can optimize away; it's the search space itself. So on 16- and 20-qubit circuits it runs out of memory. And their initial mapping is set by the gates at the *beginning* of the circuit only — again, no global view.

Two clear pain points fall out of this survey. The fast methods are myopic; the good method is exponential and can't scale; and *everybody's* initial mapping is local — either a count with no time, or just the first few gates. So if I want something that's both scalable and good, I need (a) a search whose per-step cost is polynomial, not exponential, and (b) a genuinely global way to pick the initial mapping. Let me attack the search first.

Why is the strongest method exponential? Because it searches over *combinations of concurrent SWAPs* — entire permutation layers at once — and the number of those blows up. Stare at one of these layer searches for a second. At any moment, the gates I actually need to make progress on are a small handful: the ones whose predecessors are all done, the ones that are "next." A SWAP that moves two qubits neither of which is involved in any of those next gates — what can it possibly buy me right now? Nothing for the gates I'm stuck on. It might pre-arrange something for later, but I have no reliable way to know that, and meanwhile it's burning a gate. So most of the combinations that exhaustive search dutifully enumerates are doing exactly that: shuffling idle qubits. They're redundant.

That reframes the whole thing. Instead of searching for a *mapping* (or a whole permutation), I should search for *one SWAP at a time*, and only among the SWAPs that could plausibly help the gates I'm currently blocked on. Let me make that precise.

First I need to know what "the gates I'm currently blocked on" even are, in order. The single-qubit gates I can ignore entirely — they act on one qubit, so they impose no adjacency demand; I can always emit them whenever their qubit is free. The structure that matters is the dependency among the *two-qubit* gates. So I build a directed acyclic graph: a node per CNOT, and an edge gᵢ→gⱼ when they share a qubit and gᵢ comes first. A CNOT can't run until every earlier CNOT on either of its qubits has run. The gates that are runnable *right now* are exactly the DAG nodes with no unexecuted predecessor — indegree zero. Call that set the **front layer** F.

Now the main loop almost writes itself. While F is nonempty: scan F and execute every gate whose two logical qubits currently map to adjacent physical qubits — emit it, drop it from F, and pull in any successor whose other predecessor is also now done. If at least one gate executed, loop again — maybe that freed up more. If *nothing* in F is executable, then I'm genuinely stuck: every front-layer gate needs its two qubits moved closer, and I must insert a SWAP. So I generate candidate SWAPs, score them, pick the best one, apply it, update π, and loop.

Which SWAPs are candidates? By the argument above, only the ones that touch a qubit I'm blocked on. So: for each logical qubit appearing in some front-layer gate, find its physical location, look at that physical qubit's neighbors on the coupling graph, and the SWAP that would exchange my qubit with whatever sits on each neighbor is a candidate. Every such candidate is a legal SWAP (the two physical qubits are adjacent by construction) and every one of them moves a blocked qubit. That's the candidate set.

How big is it? In the worst case all the qubits are in the front layer, and each contributes its handful of neighbors, so the candidate list is O(N). Compare that to O(exp(N)) for combination search — I've collapsed the search space from exponential to linear. Yes, I pay something back: because I commit one SWAP at a time instead of a whole permutation, I take more steps — I might need several SWAPs to bring one pair together. But how many? At most the distance between the two qubits, which is bounded by the diameter of the chip, about √N for a 2D layout. So per two-qubit gate: O(N) candidates, each scored in (I'll show) O(N), over O(√N) steps — order N^2.5 per gate. Polynomial. That's the scalability win, and it's the first design decision locked in.

Now the part that decides whether this greedy search is any good: how do I score a candidate SWAP? I need a number that says "this SWAP makes the stuck gates more executable."

The natural measure of how far a CNOT is from being executable is the distance between its two qubits on the coupling graph. I can get all these distances up front: run Floyd–Warshall on the coupling graph once, O(N³), giving a matrix D where D[i][j] is the shortest-path edge distance between physical qubits Qᵢ and Qⱼ. Moving one logical state all the way from Qᵢ to Qⱼ would take D[i][j] edge SWAPs; making a CNOT executable only needs the two states adjacent, so the exact count for that one pair would be D[i][j]−1. But this score is not an exact routed-SWAP count, it is a ranking signal, and the nearest-neighbor-cost convention keeps the raw shortest-path distance. That also keeps already-adjacent near-future gates visible as distance 1 rather than zero when the look-ahead and decay terms are mixed. N is at most a few hundred, so the O(N³) APSP preprocessing is fine, and it is a one-time cost.

So the simplest score: sum the distances of all the front-layer gates under the layout that the candidate SWAP would produce.

  H_basic = Σ_{gate ∈ F} D[π(gate.q₁)][π(gate.q₂)]

To score a candidate, tentatively apply it to π, compute this sum, and keep the SWAP with the smallest H_basic — small sum means the stuck gates are collectively close to executable. Why sum over the whole front layer and not, say, the single worst gate? Because I want to make progress on *all* of them; I'm being greedy about total remaining work, not about one bottleneck. (The A\* method used a max over the layer, but it needed an *admissible* underestimate to stay optimal; I'm not running A\*, I'm greedily picking the next move, so the sum — total work — is the honest objective.) That's H_basic.

Let me stress-test it. Picture being stuck with front layer {CNOT(q1,q2), CNOT(q3,q7)}, and right behind q7 in the DAG there's a CNOT(q2,q7). Suppose two candidate SWAPs both reduce the front-layer sum by the same amount, but one of them happens to also pull q2 and q7 closer, setting up that next gate, while the other doesn't. H_basic can't tell them apart — it only sees F. So it will sometimes pick the SWAP that resolves the immediate blockage but shoves apart the gate right behind it, and then I undo my own work on the next step. The score is too myopic.

The fix is to let the score peek a little past the front layer. I'll define an **extended set** E: a bounded number of the *closest* successors of the front-layer gates in the DAG — the gates that are about to become front-layer next. Then add their distance sum into the score so a SWAP that helps F *and* lines up E scores better.

I have to be careful combining the two sums, though. F and E generally have different sizes, and if I just add raw sums the bigger set dominates the score for no good reason. So normalize the front layer by its size, and normalize the look-ahead set only when it exists; if the peek finds no E, its average is zero. And the two sets aren't equally urgent — the front-layer gates *must* be executed before anything in E even becomes runnable, while E is only a hint about the near future. So I down-weight the E term by a factor W with 0 ≤ W < 1 (in practice around a half — present, but clearly secondary):

  A_F = (1/|F|) Σ_{gate ∈ F} D[π(q₁)][π(q₂)]
  A_E = 0 if E = ∅, otherwise (1/|E|) Σ_{gate ∈ E} D[π(q₁)][π(q₂)]
  H = A_F + W · A_E

How far should E look? Not far. The further out I look, the more the mapping will have drifted by the time those gates actually arrive, so their estimated distances under *today's* tentative layout become noise — and a bigger E costs more to compute every step. A modest window (a couple dozen gates) is enough to break the myopia of H_basic without paying for unreliable far-future estimates. So |E| is capped at a small constant.

That handles fidelity — minimizing added gates. But I claimed I also care about depth, and so far nothing in the score knows about depth at all. When does routing add depth unnecessarily? When I pile SWAP after SWAP onto the *same* qubits — those SWAPs are forced to run one after another, serializing. If instead, when I need several SWAPs, I spread them across *disjoint* qubits, they can run in parallel and depth barely grows. So I want the search to prefer SWAPs that don't reuse a qubit I just moved.

Here's a way to bias it. Give every qubit a **decay** value, starting at 1. Each time a qubit is involved in a chosen SWAP, bump its decay up by a small δ. Then multiply the whole cost of a candidate SWAP by the larger of its two qubits' decays:

  H = max(decay(SWAP.q₁), decay(SWAP.q₂)) · (A_F + W · A_E)

Why the max and not, say, the sum or product of the two decays? A SWAP is "overlapping" — bad for parallelism — if *either* of its endpoints was just moved; the worse (more-recently-used) of the two endpoints is what should set the penalty, and that's the max. The effect: a SWAP touching a freshly-moved qubit gets its score nudged up, so among otherwise-comparable candidates the search drifts toward ones on untouched qubits, which run in parallel and hold depth down.

I don't want this penalty to accumulate forever, though, or the search would become permanently obsessed with avoiding a few qubits and stop trying to satisfy the actual dependency. So I reset all decays back to 1 periodically — every few search steps, and whenever a gate actually executes (a fresh phase begins). And δ becomes a control knob: turn δ up and the search fights harder for parallel, depth-minimizing SWAPs at the cost of maybe using a few more of them; turn it down and it minimizes gate count more aggressively at the cost of depth. That's a genuine, tunable trade-off between the number of gates and the depth — exactly what you'd want to set differently for, say, a high-fidelity-but-slow technology versus a fast-but-noisier one. Push δ too far and both metrics worsen, because the search starts preferring to move untouched qubits over actually resolving the gate it's stuck on, inserting pointless SWAPs — so there's a sweet spot, not a free lunch.

The cost is cheap to evaluate: summing over F and E is O(N) in the worst case (all qubits in F), and there are O(N) candidates, which is where the per-gate complexity I quoted earlier comes from. Good. The routing engine is done. But I've been quietly assuming I *have* an initial mapping π to start from, and I argued at the very beginning that the initial mapping is where everybody falls short. Let me confront it.

The fast prior method picks π from a frequency count with no time; the strong one picks π from the first few gates. Both are local. What I want is a π informed by the *whole* circuit but weighted toward the beginning — because the beginning is what runs first and what π directly determines, but I don't want to ignore the rest. The problem is that "the whole circuit" is exactly what makes this hard: if I try to score an initial mapping against all gates I'm back to a global search.

Let me look for structure I haven't used. Here's something peculiar to quantum circuits that a classical scheduler never has: they're **reversible**. Reverse the order of all the gates and you get a perfectly valid circuit — the two-qubit gates are the very same gates, just in the opposite order. So I have, for free, a mirror image of my problem.

What does my router actually compute? I hand it an initial mapping and a circuit; it walks the gates inserting SWAPs and hands back a *final* mapping — the arrangement the qubits are in when the last gate runs. That final mapping is, by construction, a good arrangement for the *end* of the circuit: the router worked the whole way to make it so.

Now turn that around. Run the router *forward* on the original circuit starting from some random π₀. I end at a final mapping π_f that's well-suited to the end of the circuit. Take π_f and use it as the *initial* mapping for the **reversed** circuit. Route the reversed circuit. The reversed circuit ends where the original begins — so the final mapping I get out of *this* run is an arrangement well-suited to the *beginning* of the original circuit. And it's not myopic: it was produced by routing an entire pass of all the gates, so every gate had a say. The gates nearest the start of the original (which are last in the reverse pass, closest to where the mapping finally settles) influence it most; the rest influence it less but aren't ignored. That's precisely the weighting I wanted — global, but front-loaded — and it came out of the reversibility, not from any extra search.

So the initial mapping isn't something I compute up front at all; it's something the router *refines* by being run back and forth. Concretely: start from a random π₀, route forward through the original, take the final mapping; route that through the reverse; take that final mapping as the refined initial layout for the original; then route the original forward with that layout. Forward–backward–forward. The layout-refinement traversals can run in a "fake" mode that updates the mapping but doesn't bother building the routed circuit; the last forward pass is the one that emits the hardware-compliant circuit. Since the router is greedy and seeded by a random start, I run the whole thing from several random π₀ and keep the best result.

Let me put the pieces together in code, the routing engine first.

```python
from collections import defaultdict
from copy import copy, deepcopy
import numpy as np

EXTENDED_SET_SIZE = 20      # |E|: how many near-future gates to peek at
EXTENDED_SET_WEIGHT = 0.5   # W: E is a hint, weaker than F
DECAY_RATE = 0.001          # delta: per-use bump to a qubit's decay
DECAY_RESET_INTERVAL = 5    # forget decay penalties every few steps


class SabreSwap:
    """Route a circuit onto a coupling map by inserting SWAPs, greedily,
    one SWAP at a time, scored by a distance heuristic with look-ahead
    and a decay penalty for serial (depth-adding) SWAPs."""

    def __init__(self, coupling_map, heuristic="basic", seed=None, fake_run=False):
        # Couplings are symmetric on this device; a CNOT runs either way.
        if coupling_map.is_symmetric:
            self.coupling_map = coupling_map
        else:
            self.coupling_map = deepcopy(coupling_map)
            self.coupling_map.make_symmetric()
        self.heuristic = heuristic
        self.seed = seed
        self.fake_run = fake_run   # layout-refinement passes don't emit SWAPs

    def run(self, dag):
        rng = np.random.default_rng(self.seed)
        mapped_dag = None if self.fake_run else dag._copy_circuit_metadata()
        canonical_register = dag.qregs["q"]
        current_layout = Layout.generate_trivial_layout(canonical_register)
        self._bit_indices = {bit: i for i, bit in enumerate(canonical_register)}

        # decay[q] = 1, bumped when q is swapped, reset periodically.
        self.qubits_decay = {q: 1 for q in dag.qubits}

        # The blocking set is the 2-qubit front layer; the full Qiskit DAG
        # also surfaces 1-qubit ops here, and those are emitted immediately.
        front_layer = dag.front_layer()
        self.applied_predecessors = defaultdict(int)
        for _, input_node in dag.input_map.items():
            for successor in self._successors(input_node, dag):
                self.applied_predecessors[successor] += 1

        num_search_steps = 0
        while front_layer:
            execute_gate_list = []
            # Anything in F whose qubits are already adjacent: run it now.
            for node in front_layer:
                if len(node.qargs) == 2:
                    v0, v1 = node.qargs
                    if self.coupling_map.graph.has_edge(
                            current_layout[v0], current_layout[v1]):
                        execute_gate_list.append(node)
                else:
                    execute_gate_list.append(node)  # 1-qubit gates / barriers free

            if execute_gate_list:
                for node in execute_gate_list:
                    self._apply_gate(mapped_dag, node, current_layout, canonical_register)
                    front_layer.remove(node)
                    for successor in self._successors(node, dag):
                        self.applied_predecessors[successor] += 1
                        if self._is_resolved(successor):   # all input wires ready
                            front_layer.append(successor)
                    if node.qargs:
                        self._reset_qubits_decay()   # a gate ran: new phase
                continue   # executing may have freed more gates; re-check F

            # Stuck: nothing in F is adjacent. Score candidate SWAPs, pick best.
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
            best_swap = rng.choice(best_swaps)   # break ties randomly, seeded

            swap_node = DAGNode(op=SwapGate(), qargs=best_swap, type="op")
            self._apply_gate(mapped_dag, swap_node, current_layout, canonical_register)
            current_layout.swap(*best_swap)

            # Bump decay of the two swapped qubits; reset every few steps.
            num_search_steps += 1
            if num_search_steps % DECAY_RESET_INTERVAL == 0:
                self._reset_qubits_decay()
            else:
                self.qubits_decay[best_swap[0]] += DECAY_RATE
                self.qubits_decay[best_swap[1]] += DECAY_RATE

        self.property_set["final_layout"] = current_layout   # harvested by the layout pass
        return dag if self.fake_run else mapped_dag

    def _obtain_swaps(self, front_layer, current_layout):
        # Only SWAPs touching a blocked (front-layer) qubit can help.
        candidate_swaps = set()
        for node in front_layer:
            for virtual in node.qargs:
                physical = current_layout[virtual]
                for neighbor in self.coupling_map.neighbors(physical):
                    virtual_neighbor = current_layout[neighbor]
                    swap = sorted([virtual, virtual_neighbor],
                                  key=lambda q: self._bit_indices[q])
                    candidate_swaps.add(tuple(swap))   # dedup (i,j)==(j,i)
        return candidate_swaps

    def _obtain_extended_set(self, dag, front_layer):
        # E = up to EXTENDED_SET_SIZE of the closest 2-qubit successors of F.
        extended_set, incremented = [], []
        tmp_front_layer = front_layer
        done = False
        while tmp_front_layer and not done:
            new_tmp = []
            for node in tmp_front_layer:
                for successor in self._successors(node, dag):
                    incremented.append(successor)
                    self.applied_predecessors[successor] += 1
                    if self._is_resolved(successor):
                        new_tmp.append(successor)
                        if len(successor.qargs) == 2:
                            extended_set.append(successor)
                if len(extended_set) >= EXTENDED_SET_SIZE:
                    done = True
                    break
            tmp_front_layer = new_tmp
        for node in incremented:      # undo bookkeeping: this was only a peek
            self.applied_predecessors[node] -= 1
        return extended_set

    def _compute_cost(self, layer, layout):
        # Sum of coupling-graph distances of the gates in `layer`.
        return sum(self.coupling_map.distance(layout[n.qargs[0]], layout[n.qargs[1]])
                   for n in layer)

    def _score_heuristic(self, heuristic, front_layer, extended_set, layout, swap_qubits=None):
        first_cost = self._compute_cost(front_layer, layout)
        if heuristic == "basic":
            return first_cost                       # H_basic: just F
        first_cost /= len(front_layer)
        second_cost = 0
        if extended_set:
            second_cost = self._compute_cost(extended_set, layout) / len(extended_set)
        total_cost = first_cost + EXTENDED_SET_WEIGHT * second_cost   # look-ahead
        if heuristic == "lookahead":
            return total_cost
        if heuristic == "decay":                    # penalize serial SWAPs
            return max(self.qubits_decay[swap_qubits[0]],
                       self.qubits_decay[swap_qubits[1]]) * total_cost
        raise TranspilerError("Heuristic %s not recognized." % heuristic)

    def _reset_qubits_decay(self):
        self.qubits_decay = {k: 1 for k in self.qubits_decay}

    def _is_resolved(self, node):
        return self.applied_predecessors[node] == len(node.qargs)
```

And the reverse-traversal layer that refines the initial mapping by running that router forward and backward, harvesting only the layout:

```python
class SabreLayout:
    """Pick a good initial mapping by routing forward/backward; the final
    forward routing pass then consumes the settled layout."""

    def __init__(self, coupling_map, routing_pass=None, seed=None, max_iterations=3):
        self.coupling_map = coupling_map
        self.routing_pass = routing_pass
        self.seed = seed
        self.max_iterations = max_iterations

    def run(self, dag):
        rng = np.random.default_rng(self.seed)
        # Start from a random mapping.
        physical_qubits = rng.choice(self.coupling_map.size(), len(dag.qubits), replace=False)
        initial_layout = Layout({q: dag.qubits[i] for i, q in enumerate(physical_qubits)})

        if self.routing_pass is None:
            # fake_run: we only want the layout it lands on, not the SWAPs.
            self.routing_pass = SabreSwap(self.coupling_map, "decay",
                                          seed=self.seed, fake_run=True)

        circ = dag_to_circuit(dag)
        rev_circ = circ.reverse_ops()          # the mirror circuit, gates reversed
        for _ in range(self.max_iterations):
            for _ in ("forward", "backward"):
                pm = self._layout_and_route_passmanager(initial_layout)
                new_circ = pm.run(circ)
                # The final layout this pass settled on becomes the next
                # pass's starting layout; then flip to the reversed circuit.
                final_layout = self._compose_layouts(
                    initial_layout, pm.property_set["final_layout"], new_circ.qregs)
                initial_layout = final_layout
                circ, rev_circ = rev_circ, circ
        self.property_set["layout"] = initial_layout
```

So the whole thing, end to end: a long circuit can't run on a sparse chip because its CNOTs demand adjacencies the chip doesn't provide, and shuffling occupants by SWAPs to satisfy those demands optimally is NP-complete, so I go greedy. I build a DAG of the two-qubit gates and track the front layer of runnable ones; whenever I'm stuck I consider only the SWAPs touching a blocked qubit, collapsing an exponential search down to a linear one; I score each candidate by the summed coupling-graph distance of the front-layer gates, plus a down-weighted, size-normalized peek at the near-future extended set to kill myopia, all multiplied by a decay penalty that steers away from piling SWAPs on the same qubits so the result stays shallow and parallel — with the decay rate as a knob trading gate count against depth. And because quantum circuits are reversible, I get the initial mapping by running this same router forward and backward to carry information from deeper in the circuit back to the front, harvesting the settled layout rather than the gates; the final forward traversal then emits the hardware-compliant circuit.
