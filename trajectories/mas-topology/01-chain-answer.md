**Problem.** Given a fixed actor/critic collaboration runtime (artifact-only memory, topological-order
execution, aggregation at convergent nodes), find the least-structured edge set that still makes the
agents collaborate — the floor every richer topology must beat. The runtime's only levers are depth,
diversity, and synthesis; the floor should have depth and neither of the other two.

**Key idea (the chain).** The path `0 → 1 → 2 → … → (node_num-1)` is the thinnest connecting DAG:
`node_num-1` edges, a single source, a single sink, every edge increasing the index (so index order is
already a valid topological order). Every node has exactly one incoming edge, so the runtime's
aggregation branch never fires — the artifact is refined once per hop by a single thread. Pure
sequential refinement: maximum depth for the agent count, zero diversity, zero synthesis.

**Why start here.** It exercises the actor-critic edge interaction (the core unit of useful
collaboration) without touching either dial I want to study, so it isolates "does sequential refinement
alone carry the task." It is also the multi-agent rendering of the prior art — Self-Refine's
generate-then-revise loop, ChatDev's design→code→test waterfall were all paths — making the ladder's
first comparison honest.

**What to watch.** Depth is the chain's strength: three rounds of review-and-refine on four nodes
should beat one-shot where the artifact improves monotonically under polishing. Its two structural
weaknesses come from the runtime: no diversity (every node sees one upstream draft, so a bad seed at
node 0 cannot be recovered) and, because only the artifact propagates, deep-path rollback (a late agent
can undo good work it can no longer see the justification for). Expect respectable HumanEval (sequential
bug-fixing on one approach suits a small spec'd function) and the weakest result on `srdd_exec_rate`,
where building a whole runnable project with no alternative approach and the most rollback surface hurts
most. The failure is a *diversity* failure, which is what forces the fan-out at step 2.

```python
# EDITABLE region of custom_topology.py -- step 1: chain (sequential pipeline)
def generate_topology(node_num: int) -> list[tuple[int, int]]:
    """Design the multi-agent collaboration topology.

    Given N agent nodes (numbered 0 to node_num-1), return a list of
    directed edges (source, target) forming a DAG. The graph will
    automatically get input/output sentinel nodes (-1, -2) added.
    """
    # Chain topology: sequential pipeline
    # Each agent improves upon the previous agent's solution.
    # 0 -> 1 -> 2 -> ... -> (node_num - 1)
    edges = []
    for i in range(node_num - 1):
        edges.append((i, i + 1))
    return edges
```
