**Problem (from step 1).** The chain landed as a diversity failure: 0.6667 / 0.6061 on HumanEval and
0.05 on `srdd_exec_rate` with a bloated 150-line `mean_loc`. The agents refine fine but all refine one
single draft, so a bad seed at node 0 cannot be recovered, and on a long project a forgetful late agent
rolls the artifact into something that no longer runs. The fix is to turn on the two dials the chain
left dormant — diversity and synthesis — without re-adding the depth that caused the rollback.

**Key idea (the layered / MLP-shaped DAG).** Partition the nodes into `~log2(node_num)` ordered layers
and fully connect adjacent layers. Within a layer the agents act in parallel (independent diverse
refinements → width); between layers artifacts flow forward stage by stage (depth); every non-source
node is convergent over its *entire* previous layer, so the runtime aggregates that layer's spread
before it refines (synthesis at every stage). Wide-and-shallow by design — depth `~log N`, width
`~N/log N` — so the runtime's artifact-only memory does not lose distant agents and the rollback that
wrecked the chain on SRDD stays small.

**Why it works.** The chain had a width-one source and no convergent node; the layered shape makes the
source the fattest layer (the remainder goes to layer 0, the broadest parallel-exploration stage) and
makes every downstream node a full aggregator. Where the chain's node 1 saw one draft, the layered
node 2 sees both upstream drafts and fuses them — bad-seed recovery the chain lacked, and on SRDD a
synthesis of two parallel project drafts that keeps the runnable parts of each rather than perturbing
one long thread. The `max(2, …)` floor on the layer count is load-bearing at small `node_num`: without
it `log2` floors to 1, there are no adjacent layer pairs, and the topology silently collapses to a
depthless ensemble.

**Hyperparameters.** `node_num = 4` → `layer_num = max(2, int(log2(4))) = 2`; layers `[2, 2]`; edges
`{0→2, 0→3, 1→2, 1→3}` (depth 2, width 2, every downstream node in-degree 2). Remainder added to the
source layer; index order is already a valid topological order.

**What to watch.** A modest HumanEval lift over the chain (a small spec'd function has little breadth,
but the second independent draft + downstream aggregation give bad-seed recovery, with qwen — the
chain's weakest — gaining most), and the clearest improvement on SRDD: `srdd_exec_rate` above 0.05 and
`mean_loc` falling sharply, because fusing two drafts should yield a tighter, more runnable project.
The open question for the next rung: is the win from diversity or from synthesis, and has this shape
over-invested in forced convergence?

```python
# EDITABLE region of custom_topology.py -- step 2: layered (MLP-like) topology
def generate_topology(node_num: int) -> list[tuple[int, int]]:
    """Design the multi-agent collaboration topology.

    Given N agent nodes (numbered 0 to node_num-1), return a list of
    directed edges (source, target) forming a DAG. The graph will
    automatically get input/output sentinel nodes (-1, -2) added.
    """
    # Layered (MLP-like) topology
    # Split nodes into layers, fully connect adjacent layers.
    # Balances depth (iterative refinement) and width (diversity).
    import math
    layer_num = max(2, int(math.log(node_num, 2)))
    layers = [node_num // layer_num] * layer_num
    # Distribute remainder to first layer
    layers[0] += node_num % layer_num

    # Compute start/end indices for each layer
    start_ids = []
    end_ids = []
    cur = 0
    for size in layers:
        start_ids.append(cur)
        end_ids.append(cur + size)
        cur += size

    # Fully connect adjacent layers
    edges = []
    for i in range(len(layers) - 1):
        for u in range(start_ids[i], end_ids[i]):
            for v in range(start_ids[i + 1], end_ids[i + 1]):
                edges.append((u, v))
    return edges
```
