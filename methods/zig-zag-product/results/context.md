## Research question

How can we build an explicit infinite family of constant-degree expander graphs, rather than merely know from random-graph arguments that such graphs exist?

A useful explicit expander family needs a deterministic neighbor rule, bounded degree independent of the number of vertices, and a proof that the spectral/edge expansion stays bounded away from zero at every size.

## Background

For a regular graph, the normalized adjacency matrix describes one step of the random walk. The top eigenvalue is `1`; expansion is controlled by the largest absolute nontrivial eigenvalue `lambda`. Smaller `lambda` means faster mixing and stronger expansion. A constant-degree expander family is a sequence of graphs with unbounded size, constant degree, and `lambda <= c < 1`.

A graph power improves expansion but increases degree: if `G` has normalized eigenvalue bound `lambda`, then `G^2` has roughly `lambda^2`, but its degree squares.

A zig-zag product combines a large graph `G` and a small graph `H` using rotation maps. If `G` is a `D`-regular graph on `N` vertices and `H` is a `d`-regular graph on `D` vertices, then `G zigzag H` has vertices `(v,a)` with `v` a vertex of `G` and `a` an edge-label/state in `[D]`. A step does three moves: a small step in `H`, one large step in `G` using the new label, and another small step in `H` at the destination cloud. The resulting graph has `N*D` vertices and degree `d^2`.

The large step can move mass across the global graph using an edge label. The first small step spreads labels before the global move; the second small step repairs the label distribution after arrival.

## Baselines

**Random regular graphs.** A random bounded-degree regular graph is an expander with high probability. This establishes existence and gives the right mental model of sparse pseudorandom connectivity.

**Complete or high-degree graphs.** Dense graphs mix immediately and have excellent expansion.

**Graph powering alone.** Squaring an expander improves the spectral gap. Every power multiplies the degree.

**Replacement-style products.** Replacing each vertex by a small cloud can reduce degree and preserve some local structure.

**Algebraic constructions.** Cayley/Ramanujan-style expanders give strong explicit families, often with excellent parameters. They rely on algebraic machinery such as representation theory or properties of finite groups to define the neighbor rule.

## Evaluation settings

The construction should be judged by four linked requirements:

- **Degree control:** the degree remains a fixed constant, independent of `N`.
- **Expansion control:** the nontrivial normalized eigenvalue remains bounded below `1` after every iteration.
- **Size growth:** each iteration increases the number of vertices by a constant factor, yielding arbitrarily large graphs.
- **Explicitness:** given a vertex and an edge label, the neighbor can be computed by a small deterministic composition of rotation maps.

The evaluation question is whether the product's expansion can be recycled across iterations. A construction is successful only if the proof is stable under iteration.

## Code framework

A rotation-map implementation exposes the constructive content directly.

```python
def zig_zag_neighbor(rot_g, rot_h, vertex, edge_label_pair):
    v, a = vertex
    i, j = edge_label_pair

    a_prime, i_back = rot_h(a, i)      # zig inside the source cloud
    w, b_prime = rot_g(v, a_prime)     # global step in the large graph
    b, j_back = rot_h(b_prime, j)      # zag inside the destination cloud

    return (w, b), (j_back, i_back)

```

The rotation map encodes the three-step walk: one local step in `H` at the source cloud, one global step in `G`, and one local step in `H` at the destination cloud. The product's neighbor rule is a small, deterministic composition of these maps.
