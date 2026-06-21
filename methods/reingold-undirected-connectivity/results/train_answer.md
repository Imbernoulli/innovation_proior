I am presenting Reingold's deterministic log-space algorithm for undirected s-t connectivity, the result that collapses the complexity class SL into L. The problem is deceptively simple: given an undirected graph and two vertices s and t, decide whether a path connects them. A standard breadth-first search solves this easily, but it must remember a visited set or a frontier, which consumes far more than logarithmic memory. A random walk decides connectivity in logarithmic space because only the current vertex and a step counter need to be stored, but that algorithm is randomized and may err with small probability. Reingold's contribution is to remove the randomness without paying the quadratic logarithmic space cost of Savitch's theorem.

My starting point is the observation that randomness was doing only one thing for us: it supplied a short sequence of edge labels that, with high probability, leads from s to t. Instead of trying to derandomize the random walk directly, which leads to the difficult problem of universal traversal sequences, I transform the graph itself into one where deterministic exhaustive search is good enough. The key idea is expansion. A constant-degree expander has logarithmic diameter, so if I can convert each connected component of the input graph into a constant-degree expander while preserving components and without leaving log space, then I can simply enumerate all logarithmically long walks from s and check whether t appears.

The first step is regularization. I replace each original vertex v by a cloud of N vertices, written v cross [N]. Label 1 walks forward along the cycle inside the cloud, label 2 walks backward, label 3 connects (v,w) to (w,v) whenever the original graph contains the undirected edge {v,w}, and all remaining labels are self-loops. The degree becomes a fixed constant D to the sixteenth power, every connected component is preserved exactly as S cross [N], and every component is non-bipartite because of the cycle edges. This regularization is local, which means I can answer rotation-map queries without building the whole regularized graph.

The second step is expanderization through repeated zig-zag products and powers. I use the rotation-map representation of a graph, where Rot_G(v,i) returns both the neighbor w and the return label j. This extra return label is what makes graph products purely local. Given a D to the sixteenth-regular graph G and a fixed D-regular expander H on [D to the sixteenth] with second eigenvalue at most one half, the zig-zag product G zigzag H has degree D squared. I then raise it to the eighth power to restore degree D to the sixteenth. The zig-zag product does not require storing the product graph; its rotation map is computed by a constant-size script of H rotations and recursive G rotations.

The spectral analysis is the heart of the construction. If lambda is the second eigenvalue of the current graph, the zig-zag bound and the eighth power give lambda_next strictly below [1 minus (1 minus lambda) divided by 3] to the eighth. When lambda is below one half, this expression stays below one half. When lambda is at least one half, the elementary inequality [1 minus (1 minus lambda) divided by 3] to the fourth is at most lambda implies that lambda_next is below lambda squared. Therefore each phase either already gives an expander or squares the eigenvalue. Starting from a connected non-bipartite D to the sixteenth-regular graph on M vertices, the initial eigenvalue is at most 1 minus 1 divided by (D M squared). Setting ell to 2 times the ceiling of log(D M squared) gives enough phases to force the final eigenvalue below one half, so the resulting graph is a constant-degree expander on each original component.

Both the zig-zag product and graph powering act separately on connected components, which is why distinct components never merge during the transformation. If a set S is separated from its complement in G_i, then in the zig-zag product the lifted set S cross [D to the sixteenth] has all incident edges inside the corresponding product, and powering cannot create new edges across a cut that did not exist. Inductively, the transform applied to the restriction of G to S is exactly the restriction of the full transform to S cross the appropriate label space. This component preservation is essential: without it, the final enumeration could falsely report a connection between originally disconnected vertices.

The final decision procedure is then straightforward. I lift s and t to canonical vertices (s, 1 to the ell+1) and (t, 1 to the ell+1) in the implicitly constructed expander, where the first 1 is the cloud coordinate and the remaining ell coordinates come from the transformation phases. Because each transformed component is a constant-degree expander, its diameter is logarithmic, so enumerating all labelled walks of logarithmic length from the lift of s will reach the lift of t if and only if s and t were connected in the original graph. The enumeration is deterministic and uses only logarithmic space because the walk length, the vertex names, and the labels are all logarithmically bounded.

What makes the whole algorithm stay in log space is the recursive rotation-map evaluator. Rather than materializing G_ell, I answer Rot_{G_ell}(v,a_0) on demand. The state consists of the shared vertex variable v, the label tuple a_0 through a_ell, the current recursion height I, per-level counters j_1 through j_ell, and a small reusable workspace. Since D is constant and ell is O(log N), the total state is logarithmic. Each phase adds only constant control over the previous phase, so I do not fall into the Savitch-style O(log squared N) recursion depth penalty. This is why the number of phases, which is logarithmic, does not multiply the per-phase space.

In summary, the algorithm derandomizes connectivity not by simulating the random walk with pseudorandom bits, but by constructing an explicit graph in which deterministic exhaustive exploration suffices. It regularizes the input, repeatedly applies the zig-zag product followed by powering to turn each component into an expander, and then performs a log-space deterministic search on the implicit transformed graph. The canonical name for this method is Reingold's undirected connectivity algorithm, and the theorem it proves is that undirected s-t connectivity is in deterministic log space, equivalently SL equals L.

```python
"""
Illustration of the core operations in Reingold's log-space undirected connectivity algorithm.
This script does not implement the full log-space evaluator; instead it demonstrates the
regularization step, the rotation-map view, the zig-zag product, and graph powering on
small explicit graphs so that the construction can be run and inspected.
"""
from collections import defaultdict
from itertools import product


def make_rotation_map(adj):
    """Given an undirected adjacency list with labelled ports, return Rot_G(v,i)."""
    rot = {}
    for v, nb in adj.items():
        for i, (w, j) in enumerate(nb):
            rot[(v, i)] = (w, j)
    return rot


def regularize_undirected_graph(original_adj, D):
    """
    Replace each vertex v by a cloud v x [N]. Labels 0,1 are cycle edges inside the cloud.
    Label 2 encodes original edges (v,w) <-> (w,v); remaining labels are self-loops.
    D must be at least 3 and is the final regular degree.
    """
    vertices = list(original_adj.keys())
    N = len(vertices)
    index = {v: idx for idx, v in enumerate(vertices)}
    reg_adj = defaultdict(list)
    for v in vertices:
        for k in range(N):
            me = (v, k)
            for label in range(D):
                if label == 0:
                    nbr = (v, (k + 1) % N)
                    ret = 1
                elif label == 1:
                    nbr = (v, (k - 1) % N)
                    ret = 0
                elif label == 2:
                    w = vertices[k]
                    if w in original_adj[v] or v in original_adj.get(w, set()):
                        nbr = (w, index[v])
                        ret = 2
                    else:
                        nbr = me
                        ret = label
                else:
                    nbr = me
                    ret = label
                reg_adj[me].append((nbr, ret))
    return dict(reg_adj)


def zigzag_product(G_rot, H_rot):
    """
    G is D^2-regular; its vertices are written (v,a) with a in [D^2].
    H is D-regular on the label set [D^2].
    Returns rotation map of G zigzag H, whose vertices are (v,a) and whose
    labels are ordered pairs (h1,h2) of H-edge labels; degree is D^2.
    Walk: (v,a) --H h1--> (v,b) --G b--> (w,c) --H h2--> (w,d).
    """
    H_labels = sorted({a for (a, _) in H_rot.keys()})
    D = max(k for (_, k) in H_rot.keys()) + 1
    vertices = [(v, a) for v in set(v for (v, _) in G_rot.keys()) for a in H_labels]
    rot = {}
    for v, a in vertices:
        for h1 in range(D):
            for h2 in range(D):
                b, h1_ret = H_rot[(a, h1)]
                w, c = G_rot[(v, b)]
                d, h2_ret = H_rot[(c, h2)]
                rot[((v, a), (h1, h2))] = ((w, d), (h2_ret, h1_ret))
    return rot


def power_graph(G_rot, power):
    """Return rotation map of G^power by composing power rotation-map queries."""
    if power == 1:
        return G_rot
    vertices = set(v for (v, _) in G_rot.keys())
    labels = list({i for (_, i) in G_rot.keys()})
    rot = {}
    for v in vertices:
        for label_seq in product(labels, repeat=power):
            cur_v, cur_label = v, label_seq[0]
            for nxt_label in label_seq[1:]:
                cur_v, cur_label = G_rot[(cur_v, cur_label)]
            final_v, final_ret = G_rot[(cur_v, cur_label)]
            rot[(v, label_seq)] = (final_v, final_ret)
    return rot


def adjacency_from_rot(rot):
    """Build adjacency set from a rotation map for inspection."""
    adj = defaultdict(set)
    for (v, i), (w, j) in rot.items():
        adj[v].add(w)
    return adj


def reachable_bfs(adj, start, target):
    """Decide connectivity by ordinary BFS, used here only as a ground-truth check."""
    seen = {start}
    frontier = {start}
    while frontier:
        nxt = set()
        for u in frontier:
            if u == target:
                return True
            nxt.update(adj[u] - seen)
        seen.update(nxt)
        frontier = nxt
    return False


if __name__ == "__main__":
    # Tiny input graph: a path 0-1-2-3 plus an isolated vertex 4.
    original = {
        0: {1},
        1: {0, 2},
        2: {1, 3},
        3: {2},
        4: set(),
    }
    D = 2                      # small-graph degree; G will be D^2 = 4-regular
    G_degree = D * D           # 4
    reg = regularize_undirected_graph(original, G_degree)
    G_rot = make_rotation_map(reg)
    print(f"Regularized graph: {len(reg)} vertices, degree {G_degree}.")

    # A toy 2-regular graph H on the 4 labels {0,1,2,3}.  It is chosen so that
    # label 0 is adjacent to label 2, the edge label used for original graph
    # edges; this makes the local composition visible in the tiny example.
    H_adj = {
        0: [(2, 0), (2, 1)],
        1: [(3, 0), (3, 1)],
        2: [(0, 0), (0, 1)],
        3: [(1, 0), (1, 1)],
    }
    H_rot = make_rotation_map(H_adj)

    # One zig-zag product (degree D^2 = 4) followed by squaring (degree 16).
    zz_rot = zigzag_product(G_rot, H_rot)
    squared_rot = power_graph(zz_rot, 2)
    product_degree = len(set(i for (_, i) in squared_rot.keys()))
    print(f"After zig-zag then square: {len(set(v for (v, _) in squared_rot.keys()))} vertices, "
          f"degree {product_degree}.")

    # Sanity checks: original-edge label 2 connects (0,1) to (1,0), and the
    # isolated vertex 4 stays disconnected from the rest of the graph.
    print(f"Label-2 edge from cloud vertex (0,1): {G_rot[((0, 1), 2)]}")
    print(f"Label-2 edge from cloud vertex (4,0): {G_rot[((4, 0), 2)]}")
    sq_adj = adjacency_from_rot(squared_rot)
    lift0 = ((0, 1), 0)
    lift4 = ((4, 0), 0)
    print(f"Lift of 0 reaches lift of 4 in transformed graph: "
          f"{reachable_bfs(sq_adj, lift0, lift4)}")
```
