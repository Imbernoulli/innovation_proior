# MacNet and the layered (MLP-shaped) topology, distilled

MacNet organizes many LLM-backed agents as a directed acyclic graph and orchestrates
their interaction in topological order so that adding more agents scales collaboration
the way adding neurons scales a network. Each node carries an *actor* that produces an
artifact; each edge carries a *critic* that takes the upstream artifact and instructs the
downstream actor to refine it. Only the final refined artifact of each interaction
propagates (transcripts are pruned), which keeps per-agent context linear instead of
quadratic in the number of agents; at a convergent node the several incoming artifacts
are aggregated into one before refining. The collaboration *structure* is a swappable
edge set over the same machinery, so different topologies (chain, star, tree, mesh,
layered, random) can be compared and scaled on equal footing. The **layered** topology is
the MLP-shaped member of this family: partition agents into ~log2(N) ordered layers and
fully connect adjacent layers, balancing depth (iterative refinement) with width
(parallel diversity and per-layer synthesis).

## Problem it solves

Multi-agent LLM collaboration beats a single agent, but prior systems fixed a small agent
count (typically <10) and offered no general, task-agnostic, scalable way to wire up
arbitrarily many agents, fix their interaction order, and keep per-agent context from
exploding as the population grows. MacNet provides that mechanism; the layered topology
is the concrete wiring that balances depth and width.

## Key idea

- **Agents on a DAG.** `G = (V, E)`, acyclic. Acyclicity gives a clean, task-independent
  interaction order and forbids information backflow (no per-task cycle-breaking).
- **Functional bipartition.** `a_i = rho(v_i)` is an actor on node `v_i` (produces an
  artifact); `a_ij = rho(<v_i, v_j>)` is a critic on edge `<v_i, v_j>` (instructs the
  refinement). A graph deploys `|V| + |E|` agents.
- **Topological-order interactive reasoning.** For every edge, `I(a_i) < I(a_ij) <
  I(a_j)`. Each edge runs a dual-agent multi-turn loop (request/reply, capped at a few
  rounds), i.e. multi-agent self-refinement; ~`2|E|` interaction rounds total.
- **Memory control.** Short-term memory within an interaction; long-term memory keeps
  only the final *artifact*, not the transcript. Only artifacts propagate.
- **Aggregation.** A convergent node (several predecessors) fuses its incoming artifacts
  into one ("non-linear strength aggregation") before refining — this turns parallel
  breadth into synthesized quality.

## Why memory control: the n^2 -> n token bound

For the sink agent (maximum context pressure) of a mesh of `n` nodes, with `t` task
length, `p` profile length, `i` avg instruction length, `s` avg artifact length, `m` max
rounds per adjacent pair:

```
without memory control:
  O(n)_{w/o} = t + p + s + (2m-1)(i+s) ( n(n-1)/2 + 2(n-2) )  ~  C  n^2,    C    = (2m-1)(i+s)/2
with memory control (only the artifact propagates):
  O(n)_{w/}  = t + p + s +   m  (i+s) ( (n-1)   + 2(n-2) )    ~  Cbar n,    Cbar = 3m(i+s)
```

Reading full transcripts makes the sink inherit ~`n(n-1)/2` dialogues (quadratic);
propagating only the distilled artifact replaces that with a linear edge factor.
Decoupling context from quadratic to linear is what makes hundreds-to-thousands of agents
feasible (and explains why transcript-sharing hierarchical systems broke past a few dozen).

## The layered (MLP-shaped) topology

Goal: a shape that is neither pure depth (chain: width 1, no aggregation) nor pure width
(star: depth ~2, no iterative refinement) nor pure density (mesh: ~`n^2/2` edges,
information overload). The MLP layout interleaves both:

- **Layer count** `L = floor(log2 N)` — depth grows only logarithmically. Memory control
  penalizes *deep* topologies (distant agents are lost, causing artifact rollbacks), so
  shallow-and-wide is preferred: depth ~ `log N`, width ~ `N / log N`.
- **Equal split, remainder to the front.** Each layer gets `N // L` agents; the remainder
  `N % L` goes to layer 0, making the source layer the broadest parallel-exploration
  stage (most independent initial drafts entering the network).
- **Full bipartite between adjacent layers.** Every node in layer `i` feeds every node in
  layer `i+1`, so every non-source node is convergent over its *entire* previous layer
  and aggregates that layer's full diversity before refining. Edge count is
  `sum_i |layer_i| * |layer_{i+1}|` ~ `N^2 / log N` — denser than a chain (`N`), cheaper
  than a mesh (`N^2 / 2`).
- **Index-by-layer numbering** makes every edge run low-index -> high-index, so the
  numbering is itself a valid topological order and the DAG property is automatic.

Worked sizes: `N=4` -> `L=2`, layers `[2,2]`, edges `{0->2,0->3,1->2,1->3}`; `N=8` ->
`L=3`, `[4,2,2]`, 12 edges; `N=16` -> `L=4`, `[4,4,4,4]`, 48 edges. Degeneracy to note:
for `N<4`, `floor(log2 N)=1` gives a single layer and hence *no* internal edges (the
agents connect to the task only via the runtime's input/output sentinels) — vacuous, but
not a crash; the interesting regime is `N>=4`.

Because every non-source node is heavily convergent, the layered shape revises artifacts
far more often and produces longer, more complete artifacts than a chain — the upside of
leaning into convergence; the matching cost is that aggregation is harder than divergence,
so this is also where the quality risk concentrates.

## Collaborative scaling law

Sweeping the agent count `|V|` geometrically, average quality follows a logistic
(sigmoid-in-`log|V|`) growth pattern:

```
f(|V|) = gamma / (1 + e^{-beta(log|V| - alpha)}) + delta
```

with `{alpha, beta, gamma, delta}` topology-specific. Collaborative emergence appears at
much smaller scale than neural emergence (most topologies saturate around ~hundred
agents), because agents already carry foundation-model knowledge and coordinate existing
reasoning rather than learning from scratch.

## Working code (the layered edge generator)

The field-appropriate final artifact is the deterministic edge generator that fills the
runtime's one open slot. Grounded in the canonical `generate_mlp`:

```python
import math


def generate_topology(node_num: int) -> list[tuple[int, int]]:
    """Layered (MLP-shaped) collaboration DAG over agents 0 .. node_num-1.

    Partition agents into ~log2(node_num) ordered layers of roughly equal size and
    fully connect adjacent layers: parallel within a layer (width), stacked across
    layers (depth), every non-source node convergent over its whole previous layer
    (synthesis). Wide-and-shallow so the artifact-only memory mechanism does not
    lose distant agents.
    """
    layer_num = int(math.log(node_num, 2))           # depth ~ log2(N): few, wide layers

    layers = [node_num // layer_num for _ in range(layer_num)]
    layers[0] += node_num % layer_num                # remainder fattens the source layer

    # cumulative [start, end) index per layer; layer-by-layer numbering makes every
    # edge low-index -> high-index (a valid topological order; DAG is automatic).
    end_ids, start_ids = [layers[0]], [0]
    for i in range(1, len(layers)):
        start_ids.append(end_ids[-1])
        end_ids.append(end_ids[-1] + layers[i])

    # full bipartite connection between adjacent layers
    edges = []
    for i in range(len(layers) - 1):
        for u in range(start_ids[i], end_ids[i]):
            for v in range(start_ids[i + 1], end_ids[i + 1]):
                edges.append((u, v))
    return edges
```

This plugs into a runtime that attaches an input sentinel feeding the source layer and an
output sentinel collecting from the sink layer, schedules agents in topological order,
runs each edge as a critique-then-refine interaction propagating only the artifact, and
aggregates incoming artifacts at convergent nodes.
