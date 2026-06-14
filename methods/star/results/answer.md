# MacNet (Multi-Agent Collaboration Network), distilled

MacNet organizes many LLM-backed agents as nodes of a directed acyclic graph and orchestrates
their interactive reasoning in topological order to solve a task autonomously. Each **node**
holds an *actor* that produces an artifact; each **edge** holds a *critic* that reflects on the
upstream artifact and instructs a refinement. Agents act in topological order; only the refined
*artifact* (never the full dialogue) propagates, which keeps context linear rather than
quadratic and lets collaboration scale to over a thousand agents. The **star** topology is one
instance of the DAG family: a hub broadcasts to all other nodes (`0 -> i`), the widest, most
divergent single-layer shape.

## Problem it solves

General, scalable multi-agent collaboration. Prior systems use few agents (<10, a few dozen) in
fixed hand-built structures, blow up to quadratic context when every agent reads the whole
conversation, and cannot answer how performance scales with agent count. MacNet provides a
topology-agnostic way to organize an arbitrary number of agents, a memory mechanism that keeps
cost linear, and — through it — the first measurement of a *collaborative scaling law*.

## Key ideas

- **Agents as a DAG.** `G = (V, E)`, edges directed, no cycles. Acyclicity prevents information
  backflow (no task-specific cycle-breaking) and guarantees a topological order.
- **Functional bipartition.** An actor `a_i = ρ(v_i)` on every node; a critic `a_ij =
  ρ(<v_i,v_j>)` on every edge, where `ρ` agentizes an element (foundation model + role + memory
  + tools). A graph with `|V|` nodes and `|E|` edges deploys `|V|+|E|` agents.
- **Topological-order interactive reasoning.** Control flow follows a topological order
  satisfying `I(a_i) < I(a_ij) < I(a_j)` for every edge `<v_i,v_j>`; data flow follows the
  original edges. Each edge runs a dual-agent, multi-turn reflect-then-refine loop (default 3
  turns): the critic reflects/instructs, the downstream actor refines. At least `2|E|`
  interaction rounds. Implemented by Kahn peeling: process the zero-in-degree wavefront, then
  delete it and repeat. Input/output sentinels give a single source and sink.
- **Memory control.** Short-term memory holds the working context within one interaction;
  long-term memory keeps **only the final artifact** of each dialogue, not the transcript. Only
  artifacts propagate, decoupling context length from quadratic to linear growth.
- **Hierarchical aggregation at convergent nodes.** A node with several parents merges incoming
  artifacts by a pairwise tournament (merge in small units, halve each round) — a "non-linear"
  strength-aggregation, not a flat concatenation.

## Memory-control token complexity

For the sink agent under maximum context pressure (mesh, `n = |V|`), with lengths `t` task,
`p` profile, `i` instruction, `s` artifact, `m` rounds per adjacent pair:

```
without memory control:  O(n)_wo = t + p + s + (2m-1)(i+s)( n(n-1)/2 + 2(n-2) )  ≈ C n^2
with memory control:     O(n)_w  = t + p + s +     m(i+s)( (n-1)   + 2(n-2) )  ≈ C̄ n
where                    C  = (2m-1)(i+s)/2 ,   C̄ = 3m(i+s)
```

Without control the sink transitively inherits the dialogue of essentially every edge, so its
context grows with the number of edges `n(n-1)/2` — quadratic. With control it pays only for
the artifacts on its own `n-1` incoming edges — linear. The difference is exactly the
`n(n-1)/2` versus `(n-1)` term; `(n-1)+2(n-2) = 3n-5 ≈ 3n` gives `C̄ = 3m(i+s)`.

## Collaborative scaling law

Sweeping `|V|` geometrically, average quality follows a logistic curve in `log|V|`:

```
f(|V|) = γ / ( 1 + e^{ -β ( log|V| - α ) } ) + δ
```

with `{α, β, γ, δ}` real numbers per topology: `δ` the single-agent floor, `γ` the range, `α`
the inflection (in log-scale), `β` the steepness. Logistic, not power-law, because a fixed task
has finitely many improvable aspects, so collaboration saturates. Collaborative emergence
appears at far smaller scale than neural emergence (which needs ~billion-parameter models and
~`10^22` FLOPs), because agents already carry foundation-model knowledge and only need to
*coordinate* it — a "shortcut" to higher effective intelligence without retraining. A node count
around `2^4` is a reasonable default near the inflection.

## Why scaling surfaces new capability (long-tail / Zipf)

Refinement "aspects" raised in interactions are long-tailed (Zipf: rank-`r` aspect probability
`p(t) ∝ 1/r(t)`). The number of samples is proportional to interaction density, `n ∝ |V|^2`.
The probability a tail aspect appears at least once:

```
p^n(t) = 1 - (1 - p(t))^n  ∝  1 - (1 - 1/r(t))^{|V|^2}  -> 1  as |V| -> ∞
```

So scaling drives coverage of rare aspects toward certainty; more covered aspects yield more
comprehensive (and longer) artifacts — the mechanism behind the S-curve.

## Topology guidance (the dial)

Representative DAGs: **chain** (deep, no diversity), **star** (widest divergence, hub-to-all),
**tree** (branching with depth), **mesh** (densest, `n(n-1)/2` edges), **layered** (MLP-like,
balanced), **random** (irregular). Derived tendencies:

- Higher interaction **density** generally helps (more refinement angles), but not monotonically:
  very dense convergent nodes face hard aggregation, and aggressive forgetting makes deep paths
  lose distant agents (artifact rollback).
- At equal density, **wider beats deeper** (short paths under memory control).
- **Irregular/small-world** topologies often beat regular ones: a few random shortcut edges cut
  the average path length (Watts–Strogatz), reducing long-distance artifact invisibility, at
  lower density (cheaper) than a mesh.
- **Divergent beats convergent**: fan-out is smooth; merging many artifacts is hard and lossy.
  Reversing a divergent topology into its convergent mirror degrades it. Rule: maximize
  divergence, minimize forced convergence — which places the **star** at the divergent extreme.
- No universal winner: closed-domain step-by-step tasks favor chain-like depth; open-ended,
  breadth-hungry tasks favor divergent shapes (star/tree). Choose shape and scale by task
  openness and compute budget.

## Working code

Grounded in the canonical topology generators and the DAG runtime (Node with
predecessors/successors/pre_solutions/solution; topological execution with per-edge
optimize and per-node aggregate; long-term memory = only `solution`).

```python
import math
import random


# ---- topology family: directed edges (source, target), source < target ----

def generate_chain(n):
    return [(i, i + 1) for i in range(n - 1)]

def generate_star(n):
    # hub-and-spoke: node 0 broadcasts to all others. n-1 edges, depth 1.
    # the divergent extreme: one seed, n-1 parallel sibling refinements.
    return [(0, i) for i in range(1, n)]

def generate_tree(n):
    edges, i = [], 0
    while len(edges) < n - 1:
        edges.append((i, 2 * i + 1))
        if len(edges) >= n - 1:
            break
        edges.append((i, 2 * i + 2))
        i += 1
    return edges

def generate_mesh(n):
    return [(u, v) for u in range(n) for v in range(n) if u < v]

def generate_layered(n):
    layer_num = int(math.log(n, 2))
    sizes = [n // layer_num for _ in range(layer_num)]
    sizes[0] += n % layer_num
    starts, ends = [0], [sizes[0]]
    for k in range(1, layer_num):
        starts.append(ends[-1]); ends.append(ends[-1] + sizes[k])
    edges = []
    for k in range(layer_num - 1):
        for u in range(starts[k], ends[k]):
            for v in range(starts[k + 1], ends[k + 1]):
                edges.append((u, v))
    return edges

def generate_random(n):
    space = [(u, v) for u in range(n) for v in range(n) if u < v]
    random.shuffle(space)
    edge_num = random.randint(n - 1, n * (n - 1) // 2)
    return space[:edge_num]


# ---- actors on nodes, critics on edges ----

class Node:
    """Actor on a node. Long-term memory = ONLY self.solution (the artifact);
    the dialogue is forgotten, keeping context linear."""
    def __init__(self, node_id, model):
        self.id = node_id
        self.predecessors, self.successors = [], []
        self.pre_solutions = {}            # parent_id -> incoming artifact
        self.solution = Artifact()         # the only thing propagated onward
        self.depth = 0
        self.graph_depth = 1
        self.model = model

    @property
    def temperature(self):
        # diverge early (shallow, hot), converge late (deep, cold)
        return 1 - self.depth / self.graph_depth

    def optimize(self, task, pre_solution):
        """Edge reflect-then-refine: edge-critic instructs, this actor refines."""
        suggestion = "None."
        if pre_solution:
            suggestion = critic_agent(self.model).step(
                instructor_prompt(task, pre_solution)).content
        response = actor_agent(self.model, self.temperature).step(
            assistant_prompt(task, pre_solution, suggestion)).content
        return Artifact(response), suggestion

    def aggregate(self, task, unit_num):
        """Convergent node: hierarchical pairwise tournament merge."""
        pool = list(self.pre_solutions.values())
        while len(pool) > 1:
            if unit_num >= 4:
                unit_num = unit_num // 2
            pool = [Artifact(merge_agent(self.model, self.temperature)
                             .step(cooperate_prompt(task, pool[k:k + unit_num])).content)
                    for k in range(0, len(pool), unit_num)]
        self.solution = pool[0]


class MacNet:
    """Organize agents as a DAG and run them in topological order."""
    def __init__(self, num_agents, task_prompt, model_name, topology="star",
                 aggregate_unit_num=2):
        self.task = task_prompt
        self.unit_num = aggregate_unit_num
        gen = {"chain": generate_chain, "star": generate_star, "tree": generate_tree,
               "mesh": generate_mesh, "layered": generate_layered,
               "random": generate_random}[topology]
        self.nodes = {i: Node(i, model_name) for i in range(num_agents)}
        self.IN, self.OUT = -1, -2
        self.nodes[self.IN] = Node(self.IN, model_name)
        self.nodes[self.OUT] = Node(self.OUT, model_name)
        for u, v in gen(num_agents):
            self._add_edge(u, v)
        for nd in list(self.nodes.values()):              # single source / single sink
            if nd.id not in (self.IN, self.OUT) and not nd.predecessors:
                self._add_edge(self.IN, nd.id)
            if nd.id not in (self.IN, self.OUT) and not nd.successors:
                self._add_edge(nd.id, self.OUT)
        assert not self._has_cycle(), "topology must be a DAG"
        self._assign_depths()
        self.nodes[self.IN].solution = Artifact(task_prompt)

    def _add_edge(self, u, v):
        self.nodes[u].successors.append(self.nodes[v])
        self.nodes[v].predecessors.append(self.nodes[u])

    def _assign_depths(self):
        # BFS layer index from the sources = each node's depth; total depth scales temperature
        layer, work, depth = 0, {nd.id: len(nd.predecessors) for nd in self.nodes.values()}, {}
        frontier = [i for i, d in work.items() if d == 0]
        while frontier:
            nxt = []
            for nid in frontier:
                depth[nid] = layer
                for s in self.nodes[nid].successors:
                    work[s.id] -= 1
                    if work[s.id] == 0:
                        nxt.append(s.id)
            frontier, layer = nxt, layer + 1
        total = max(layer - 1, 1)
        for nid, d in depth.items():
            self.nodes[nid].depth = d
            self.nodes[nid].graph_depth = total

    def _has_cycle(self):
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {nid: WHITE for nid in self.nodes}
        def dfs(nid):
            color[nid] = GRAY
            for s in self.nodes[nid].successors:
                if color[s.id] == GRAY or (color[s.id] == WHITE and dfs(s.id)):
                    return True
            color[nid] = BLACK
            return False
        return any(color[nid] == WHITE and dfs(nid) for nid in self.nodes)

    def run(self):
        nodes = dict(self.nodes)
        while True:
            wave = [nd for nd in nodes.values() if not nd.predecessors]   # Kahn wavefront
            if not wave:
                break
            advanced = set()
            for cur in wave:
                for nxt in cur.successors:                 # data flow along original edges
                    artifact, _ = nxt.optimize(self.task, cur.solution.as_text())
                    nxt.pre_solutions[cur.id] = artifact
                    advanced.add(nxt.id)
            for nid in advanced:
                nd = self.nodes[nid]
                if len(nd.pre_solutions) >= 2:            # convergent -> aggregate
                    nd.aggregate(self.task, self.unit_num)
                else:                                     # single parent -> take refinement
                    nd.solution = next(iter(nd.pre_solutions.values()))
            for cur in wave:                              # peel the wavefront
                for nxt in list(cur.successors):
                    nxt.predecessors.remove(cur)
                del nodes[cur.id]
        return self.nodes[self.OUT].solution
```

The `generate_topology` slot used as a MacNet baseline is exactly `generate_star`: hub-and-spoke
edges `(0, i)` for `i = 1 .. n-1`, fed to the same DAG runtime.
