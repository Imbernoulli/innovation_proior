The core answer of the zig-zag product is this: rather than explicitly writing down a single large graph that “looks random” all at once, split expansion into two composable sources.

Given a large `D`-regular graph `G` and a small `d`-regular graph `H`, where the number of vertices of `H` equals the degree `D` of `G`, the zig-zag product replaces each vertex of `G` with a small cloud that is a copy of `H`. A vertex of the new graph is `(v,a)`: `v` is a vertex of the large graph, and `a` is an edge label of the large graph. One step of the walk is:

1. Take one step along `H` inside the current cloud, scrambling the edge label.
2. Use the scrambled label to take one step in `G`, completing the global move.
3. Take one more step along `H` inside the destination cloud, repairing the local distribution after arrival.

The new graph therefore inherits roughly the size of `G`, the low degree controlled by `H`, and the combined expansion of both. The distinctive insight of this construction is: the large graph is responsible for global expansion, the small graph for local expansion; local mixing keeps the global edge choice from being dominated by a handful of labels, while the global move carries mass far away. The two are not simply stacked together — they complement each other.

This breaks through the obstacle that “random graphs exist, but explicit construction is hard.” Random regular graphs show that constant-degree expanders exist in abundance, but the randomness argument gives no adjacency rule that is recursively generatable, locally addressable, and verifiable step by step. The zig-zag product rewrites the problem as a stable construction loop:

```text
current constant-degree expander
  -> square the graph: strengthens global expansion, but raises the degree
  -> zig-zag: use a fixed small expander to restore low degree, while retaining enough expansion
  -> a larger constant-degree expander
```

As long as a fixed-size small expander is hardwired at the start, every round can deterministically produce a larger graph. A neighbor query is only a handful of rotation-map calls, so it is explicit; the expansion is recursively guaranteed by the product theorem, so there is no need to search a candidate space of exponential size for a random graph.

In one line: the zig-zag product decomposes “random global connectivity” into “iterable global amplification + reusable local mixing,” constructing arbitrarily large explicit expanders while keeping the degree low.

## Code illustration

```python
def rotation_map_G(v, a):
    """Return (w, b): G-edge (v,a) lands at w with back-label b."""
    # Example: explicit D-regular rule on Z_N (D must match |H|)
    w = (v + (2 * a + 1)) % G_N
    b = a
    return w, b

def rotation_map_H(a, i):
    """Return (a_prime, i_back): H-edge (a,i) lands at a_prime with back-label i_back."""
    # Example: explicit d-regular rule on Z_D
    a_prime = (a + i) % H_D
    i_back = (-i) % H_D
    return a_prime, i_back

def zig_zag_rotation_map(rot_g, rot_h, vertex, edge_label_pair):
    v, a = vertex
    i, j = edge_label_pair

    a_prime, i_back = rot_h(a, i)   # zig inside source cloud
    w, b_prime = rot_g(v, a_prime)  # global step in G
    b, j_back = rot_h(b_prime, j)   # zag inside destination cloud

    return (w, b), (j_back, i_back)
```
