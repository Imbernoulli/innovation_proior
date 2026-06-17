# MacNet (Multi-Agent Collaboration Network), distilled

MacNet organizes many LLM-backed agents as nodes of a directed acyclic graph and orchestrates
their interactive reasoning in topological order to solve a task autonomously. Conceptually,
each **node** holds an *actor* that produces an artifact and each **edge** holds a *critic* that
reflects on the upstream artifact and instructs a refinement. In the public `macnet` code this
edge critic is realized inside the successor node's `optimize` call: first an instructor-style
prompt produces suggestions, then an assistant-style prompt produces the refined code. Only the
refined *artifact* (never the full dialogue) propagates, keeping context linear rather than
quadratic. The **star** topology is one DAG instance: a hub broadcasts to all other interior
nodes (`0 -> i`), the widest divergent single-hop shape before the output sentinel aggregates.

## Problem it solves

General, scalable multi-agent collaboration. Prior systems use few agents (<10, a few dozen) in
fixed hand-built structures, blow up to quadratic context when every agent reads the whole
conversation, and cannot answer how performance scales with agent count. MacNet provides a
topology-agnostic way to organize an arbitrary number of agents, a memory mechanism that keeps
cost linear, and — through it — the first measurement of a *collaborative scaling law*.

## Key ideas

- **Agents as a DAG.** `G = (V, E)`, edges directed, no cycles. Acyclicity prevents information
  backflow (no task-specific cycle-breaking) and guarantees a topological order.
- **Functional bipartition.** In the paper, an actor `a_i = ρ(v_i)` is assigned to every node
  and a critic `a_ij = ρ(<v_i,v_j>)` to every edge, where `ρ` agentizes an element
  (foundation model + role + memory + tools). A graph with `|V|` nodes and `|E|` edges thus
  has `|V|+|E|` conceptual agents. The reference code stores structural edges separately but
  implements the critic/actor exchange as two prompts in `Node.optimize`.
- **Topological-order interactive reasoning.** Control flow follows a topological order
  satisfying `I(a_i) < I(a_ij) < I(a_j)` for every edge `<v_i,v_j>`; data flow follows the
  original edges. The paper allows a dual-agent, multi-turn reflect-then-refine loop, capped
  at three exchange rounds in experiments; the exposed reference implementation performs one
  suggestion prompt and one refinement prompt per traversed edge, plus aggregation where needed.
  Implemented by Kahn peeling: process the zero-in-degree wavefront, then delete it and repeat.
  Input/output sentinels give a single source and sink.
- **Memory control.** Short-term memory holds the working context within one interaction;
  long-term memory keeps **only the final artifact** of each dialogue, not the transcript. Only
  artifacts propagate, decoupling context length from quadratic to linear growth.
- **Pooled aggregation at convergent nodes.** Once all predecessor solutions have arrived and
  their count is at least `Aggregate_unit_num` (default `2`), the reference calls `Pool` to
  merge candidates in small batches, prune large queues, and reduce them to one code bundle.
  If there are too few inputs or aggregation fails after retries, it forwards the first
  predecessor solution.

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
`p(t) ∝ 1/r(t)`). Let `N_s` be the effective sample count; it is proportional to interaction
density, so `N_s ∝ |V|^2`. The probability a tail aspect appears at least once:

```
p^{N_s}(t) = 1 - (1 - p(t))^{N_s}  ∝  1 - (1 - 1/r(t))^{|V|^2}  -> 1  as |V| -> ∞
```

So scaling drives coverage of rare aspects toward certainty; more covered aspects yield more
comprehensive (and longer) artifacts — the mechanism behind the S-curve.

## Topology guidance (the dial)

Representative DAGs: **chain** (deep, no diversity), **star** (widest divergence, hub-to-all),
**tree** (branching with depth), **mesh/net** (densest, `n(n-1)/2` edges), **layered/MLP**
(balanced), **random** (irregular sampled subset of forward mesh edges; the paper describes
maintaining connectivity, but `generate_random` does not explicitly test it). Derived tendencies:

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

Grounded in `OpenBMB/ChatDev` branch `macnet` at
`e7a35824fd683ffe8fc237e28ecc47d7b1a5da63`. The canonical files are
`generate_graph.py` for topology generation, `graph.py` for Kahn-style execution and node
state, and `chatdev/waiting.py` for pooled aggregation.

```python
import math
import random


class Edge:
    def __init__(self, source: int, target: int):
        self.source = source
        self.target = target


class TopologyGraph:
    """Faithful shape of generate_graph.py. Names in config are chain/star/tree/net/mlp/random."""
    def __init__(self, node_num: int):
        self.node_num = node_num
        self.edges = []

    def generate_star(self):
        for i in range(1, self.node_num):
            self.edges.append(Edge(0, i))
        assert len(self.edges) == self.node_num - 1
        return self

    def generate_net(self):
        for u in range(self.node_num):
            for v in range(self.node_num):
                if u < v:
                    self.edges.append(Edge(u, v))
        assert len(self.edges) == self.node_num * (self.node_num - 1) / 2
        return self

    def generate_mlp(self):
        layer_num = int(math.log(self.node_num, 2))
        layers = [self.node_num // layer_num for _ in range(layer_num)]
        layers[0] += self.node_num % layer_num
        starts, ends = [0], [layers[0]]
        for i in range(1, len(layers)):
            starts.append(ends[-1])
            ends.append(ends[-1] + layers[i])
        for i in range(len(layers) - 1):
            for u in range(starts[i], ends[i]):
                for v in range(starts[i + 1], ends[i + 1]):
                    self.edges.append(Edge(u, v))
        return self

    def generate_random(self):
        # Source code samples a forward-edge subset; it does not explicitly enforce connectivity.
        # The paper/code intent is an integer upper bound n(n-1)/2.
        edge_num = random.randint(self.node_num - 1, self.node_num * (self.node_num - 1) // 2)
        space = [(u, v) for u in range(self.node_num) for v in range(self.node_num) if u < v]
        random.shuffle(space)
        for i in range(edge_num):
            self.edges.append(Edge(*space[i]))
        return self


class RuntimeNode:
    """Faithful state fields from graph.py."""
    def __init__(self, node_id: int, model=None):
        self.id = node_id
        self.predecessors = []
        self.successors = []
        self.pre_solutions = {}   # predecessor id -> Codes artifact
        self.solution = Codes()   # only artifact propagated as long-term state
        self.depth = 0
        self.temperature = 0.2
        self.model = model

    def optimize(self, task_prompt: str, pre_solution: str, config: dict, name: str):
        suggestion = "None."
        if pre_solution != "":
            suggestion = llm(config["Agent"]["instructor_prompt"].format(task_prompt, pre_solution))
        response = llm(config["Agent"]["assistant_prompt"].format(task_prompt, pre_solution, suggestion))
        return response, Codes(response), suggestion

    def aggregate(self, prompt: str, retry_limit: int, unit_num: int, layer_dir: str,
                  graph_depth: int, store_dir: str):
        pool = Pool(len(self.pre_solutions), unit_num, layer_dir, self.model)
        for _ in range(retry_limit):
            new_codes = pool.state_pool_add(
                layer_dir, cc_prompt(), 6000000, prompt, Codes(), store_dir,
                temperature=1 - self.depth / graph_depth,
            )
            if new_codes is not None:
                self.solution = new_codes
                return 0
        return 1


def execute(graph, prompt, name):
    """Kahn-style execution from graph.py::Graph.execute."""
    while True:
        input_nodes = graph.get_input_layer()
        if len(input_nodes) == 0:
            break

        visited_edges, next_nodes = set(), set()
        for cur_node in input_nodes:
            for next_node in cur_node.successors:
                response, codes, suggestion = next_node.optimize(
                    task_prompt=prompt,
                    pre_solution=cur_node.solution._get_codes(),
                    config=graph.config,
                    name=name,
                )
                next_node.pre_solutions[cur_node.id] = codes
                visited_edges.add((cur_node.id, next_node.id))
                next_nodes.add(next_node.id)

        for node_id in next_nodes:
            node = graph.nodes[node_id]
            ready = len(node.pre_solutions) == len(node.predecessors)
            enough = len(node.pre_solutions) >= graph.aggregate_unit_num
            if ready and enough:
                error = node.aggregate(prompt, graph.aggregate_retry_limit,
                                       graph.aggregate_unit_num, layer_dir(),
                                       graph.depth, store_dir())
                if error:
                    node.solution = node.pre_solutions[next(iter(node.pre_solutions))]
            else:
                node.solution = node.pre_solutions[next(iter(node.pre_solutions))]

        for u, v in visited_edges:
            graph.delete_edge(u, v)
        for cur_node in input_nodes:
            graph.delete_node(cur_node.id)
```

The `generate_topology` slot used as a MacNet baseline is exactly `generate_star`: hub-and-spoke
edges `(0, i)` for `i = 1 .. n-1`, then the runtime adds the input sentinel to the hub and the
output sentinel after all leaves.
