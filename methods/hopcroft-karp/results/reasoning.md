I have a bipartite graph, a set of left vertices, a set of right vertices, and edges only across the split. I need as many disjoint pairs as possible. Trying edge subsets is out immediately; even the representation of a matching is already telling me the useful local state: for each left vertex, who is its current right partner, and for each right vertex, who is its current left partner. The question is how to move from a valid matching to a larger valid matching without breaking the disjointness condition.

If I look at a path that alternates between edges outside the current matching and edges inside it, there is one special case that changes the size. The path starts at a free vertex and ends at a free vertex. Since the first and last edges are outside the matching, the path contains one more outside edge than inside edge. Flipping membership along the path removes the matched edges on it and adds the unmatched edges on it. Every internal vertex still has exactly one matched incident edge after the flip, and the two endpoints become matched, so the matching grows by one.

I also need to know when I am allowed to stop. Suppose a matching `M` has no such augmenting path, but there is a larger matching `M*`. In the symmetric difference `M xor M*`, every vertex has degree at most two, so the components are alternating cycles and alternating paths. Cycles have the same number of edges from each matching. Since `M*` has more edges than `M`, at least one path component has more `M*` edges than `M` edges. Such a path starts and ends with `M*` edges, so both endpoints are free under `M`, and it is exactly an augmenting path. That contradicts the assumption. So a matching is maximum exactly when no augmenting path remains.

The direct algorithm is to find one augmenting path, flip it, and repeat. In a bipartite graph I can orient the search from free left vertices: unmatched edges go left to right, and a matched right vertex can continue only through its matched edge back to the left. A depth-first search can find one augmenting path this way. But if each improvement costs a scan of the graph and there can be a linear number of improvements, the worst-case time is still too high. The path criterion is correct, but it is not organized enough.

The waste is that a single search commits to one path while many independent improvements may be available at the same distance. I want to measure the alternating distance from all free left vertices at once, then confine the next searches to the nearest free right vertices. I only need to store layers on the left side. Free left vertices have `level = 0`. From a left vertex `u`, an edge `u-v` either reaches a free right vertex or reaches a matched right vertex whose partner `w = match_r[v]` becomes a left vertex in the next layer. If a free right is first seen from a left vertex at layer `k`, the shortest augmenting path has `k + 1` unmatched edges and `2(k + 1) - 1` total edges. I record that unmatched-edge count as `free_level`; once the next step from a left vertex would be at or beyond it, expanding through a matched right vertex could only build a longer augmenting path, so this phase should not go farther.

Once those layers are known, the second pass should search only forward through them. From left vertex `u`, an edge to a free right vertex is usable only when `level[u] + 1 == free_level`. An edge to a matched right vertex `v` is usable only when its partner `w = match_r[v]` has `level[w] == level[u] + 1`, and only while `level[u] + 1 < free_level`. A successful recursive search flips the path as it unwinds. A failed left vertex can be marked dead for this phase, because all routes from it inside the current layered graph have been exhausted. A successful left vertex can also be marked dead for the phase, because it has just been consumed by one of the paths in the current batch. Every accepted path reaches the first free-right layer, every recursive step advances one layer, and consumed vertices are removed from later searches, so the phase returns a maximal vertex-disjoint set of shortest augmenting paths.

Now I need the number of phases to be small, not merely the work inside one phase. Let the shortest augmenting path before a phase have edge length `ell`, and let the phase choose a maximal set `A = {P_1, ..., P_t}` of vertex-disjoint paths of that length. After flipping all paths in `A`, suppose a new augmenting path `P` remains. If `P` is vertex-disjoint from every path in `A`, then none of its edges changed status, so it was already an augmenting path before the phase. If its length were `ell`, I could have added it to `A`, contradicting maximality, and if its length were shorter than `ell`, that would contradict the definition of `ell`. So the only dangerous case is that `P` touches a path in `A`. At a shared vertex, `P` cannot use an endpoint of `P` as free under the new matching while the chosen path has just matched all of its vertices; the shared vertex is internal to `P`. The unique new matching edge incident to that shared vertex must lie on both `P` and the chosen path, because a matching gives at most one such edge. Therefore `P` shares at least one edge with the union of `A`. Now compare matchings: `M' = M xor (P_1 union ... union P_t)` has size `|M| + t`, and `M' xor P` has size `|M| + t + 1`. The symmetric difference `M xor (M' xor P) = (P_1 union ... union P_t) xor P` must contain at least `t + 1` old augmenting paths, each of length at least `ell`, so it has at least `(t + 1)ell` edges in those augmenting components. But the same symmetric difference has at most `t ell + |P| - 1` edges, because the chosen paths have `t ell` edges and `P` shares at least one edge with them. Hence `t ell + |P| - 1 >= (t + 1)ell`, so `|P| >= ell + 1`; in a bipartite graph augmenting paths have odd length, so it is really at least `ell + 2`. Every remaining augmenting path is longer than the old shortest path, which is exactly the phase progress I need.

Strict increase alone could still allow many phases, so I split the run by path length. Let `s` be the maximum matching size and take `p = max(1, floor(sqrt(s)))`. After `p` phases, the shortest remaining augmenting path, if any, has length at least `2p + 1`, because augmenting paths in a bipartite graph have odd length and the shortest length rises by at least two each phase. Compare the current matching `M` with a maximum matching `M*`. The symmetric difference contains exactly `s - |M|` vertex-disjoint augmenting paths with respect to `M`. Each of those paths has at least `2p + 1` edges, so each contains at least `p` edges of the current matching. Those current-matching edges are disjoint across the paths, and there are at most `s` of them. Thus `s - |M| <= s / p`, which is `O(sqrt(s))`. After the first `p` phases, only `O(sqrt(s))` augmentations remain, and each later phase adds at least one edge. Since `s <= (n + m) / 2`, the total number of phases is `O(sqrt(n + m))`.

Each phase scans the edges once in the breadth-first layering and once in the layered search, with dead left vertices preventing repeated failed exploration inside the same phase. So the total time is `O(E sqrt V)` for `V = n + m`, and the stopping condition is still Berge's criterion: when the breadth-first pass cannot reach any free right vertex, no augmenting path exists, so the matching is maximum.

```python
import sys


def read_bipartite(data):
    """Parse n (left size), m (right size), e (edge count), then e edges
    (u in [0, n), v in [0, m)) into a 0-based adjacency list adj[u] -> [v, ...]."""
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    e = int(next(it))
    adj = [[] for _ in range(n)]
    for _ in range(e):
        u = int(next(it))
        v = int(next(it))
        adj[u].append(v)
    return n, m, adj


def max_matching(n, m, adj):
    """Return maximum matching size and matched (left, right) pairs."""
    sys.setrecursionlimit(max(sys.getrecursionlimit(), n + m + 10))
    match_l = [-1] * n
    match_r = [-1] * m
    INF = n + m + 1
    level = [INF] * n
    free_level = INF

    def build_layers():
        nonlocal free_level
        queue = []
        head = 0
        free_level = INF
        for u in range(n):
            if match_l[u] == -1:
                level[u] = 0
                queue.append(u)
            else:
                level[u] = INF
        while head < len(queue):
            u = queue[head]
            head += 1
            next_level = level[u] + 1
            if next_level >= free_level:
                continue
            for v in adj[u]:
                w = match_r[v]
                if w == -1:
                    free_level = next_level
                elif level[w] == INF:
                    level[w] = next_level
                    queue.append(w)
        return free_level != INF

    def find_path(u):
        next_level = level[u] + 1
        for v in adj[u]:
            w = match_r[v]
            if w == -1:
                if next_level == free_level:
                    match_l[u] = v
                    match_r[v] = u
                    level[u] = INF
                    return True
            elif next_level < free_level and level[w] == next_level and find_path(w):
                match_l[u] = v
                match_r[v] = u
                level[u] = INF
                return True
        level[u] = INF
        return False

    while build_layers():
        for u in range(n):
            if match_l[u] == -1 and level[u] == 0:
                find_path(u)
    pairs = [(u, match_l[u]) for u in range(n) if match_l[u] != -1]
    return len(pairs), pairs


def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    n, m, adj = read_bipartite(data)
    size, pairs = max_matching(n, m, adj)
    out = [str(size)]
    for u, v in pairs:
        out.append(f"{u} {v}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
```
