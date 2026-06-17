## Problem

You are given a connected undirected graph on $V$ vertices and $E$ edges ($V \le 200$, $E \le 10000$). Every edge $e$ carries two positive integer weights: a *time* $t_e$ and a *cost* $c_e$, with $1 \le t_e, c_e < 256$.

For any spanning tree $T$ of the graph, define its value to be the product of its total time and its total cost:

$$\Big(\sum_{e \in T} t_e\Big)\cdot\Big(\sum_{e \in T} c_e\Big).$$

Find a spanning tree $T$ that minimizes this value, and output its edges.

## Code framework

```python
def mst(n, edges, weight_fn):
    """Minimize sum over a spanning tree of a LINEAR per-edge weight.
    n: number of vertices (0-indexed); edges: list of (u, v, t, c).
    weight_fn(t, c) -> number: the per-edge weight to minimize over spanning trees.
    Returns (sum_t, sum_c, chosen_edge_indices) for the resulting tree."""
    # TODO
    pass


def solve(n, edges):
    """edges: list of (u, v, t, c), 0-indexed vertices, graph connected.
    Returns (best_product, sum_t, sum_c, tree_edge_indices)."""
    # TODO
    pass
```
