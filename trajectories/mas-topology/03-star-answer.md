**Problem (from step 2).** The layered shape lifted HumanEval to 0.697 / 0.697 (diversity helped the
closed-domain function task) but stayed flat at 0.05 on `srdd_exec_rate` while `mean_loc` collapsed
from 150.7 to 12.5 — the forced per-stage synthesis fused two project drafts into a hollow 12-line
intersection that runs but does nothing. Read with the chain (150 lines, broken by rollback), both
SRDD failures come from strangling exploration: too much undiversified depth, then too much forced
merge. Diversity is the useful dial; forced interior synthesis is the one that hurts on breadth-hungry
tasks.

**Key idea (the star / hub-and-spoke).** Push diversity to its maximum and forced synthesis to its
minimum. One seed node broadcasts to every other node — edges `0→i` for `i = 1 .. node_num-1`. Same
edge count as the chain, but arranged as pure fan-out: depth 1, width `node_num-1` (the widest single-
layer divergence), and **zero convergent interior nodes** (each spoke has one incoming edge, so the
runtime's aggregation branch never fires mid-flight). The only synthesis is one shallow merge at the
output sentinel over the full spread of complete refinements. The divergent extreme of the DAG family.

**Why it works.** It maximizes the dial that helped (diversity: `node_num-1` independent parallel takes
on the seed) and removes the two operations that hurt — deep refinement (longest path is one hop, so
under artifact-only memory there is nothing distant to forget or roll back) and forced interior
aggregation (no spoke ever merges, so diversity is never compressed into a stub). Divergence is smooth;
convergence is hard and lossy — the star leans entirely on the smooth operation and defers the one hard
merge to the end, over solutions that are each already complete. What it gives up is depth: every spoke
is refined once and never again, a liability on tasks that need iterative deepening — so the star is the
right setting for open-ended, breadth-hungry tasks, not a universal winner.

**Hyperparameters.** `node_num = 4` → edges `{0→1, 0→2, 0→3}`; node 0 the single source/seed, nodes
1–3 parallel sinks, one synthesis at the output sentinel. Every edge low-index → high-index, so index
order is a valid topological order; guaranteed DAG.

**What to watch.** The clearest SRDD win of the ladder — `srdd_exec_rate` above 0.05 for the first
time, with `mean_loc` between the chain's bloat and the layered stub (real projects, chosen among
complete attempts rather than intersected from partial ones) — and HumanEval matching or modestly
exceeding 0.697 / 0.697 (breadth of independent attempts plus a less-lossy final merge, with at most a
small price for the lost depth at four nodes).

```python
# EDITABLE region of custom_topology.py -- step 3: star (hub-and-spoke)
def generate_topology(node_num: int) -> list[tuple[int, int]]:
    """Design the multi-agent collaboration topology.

    Given N agent nodes (numbered 0 to node_num-1), return a list of
    directed edges (source, target) forming a DAG. The graph will
    automatically get input/output sentinel nodes (-1, -2) added.
    """
    # Star topology: hub-and-spoke
    # Node 0 broadcasts to all other nodes in parallel.
    # Good for generating diverse solutions simultaneously.
    edges = []
    for i in range(1, node_num):
        edges.append((0, i))
    return edges
```
