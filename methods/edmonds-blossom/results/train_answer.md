The problem is to find a maximum-cardinality matching in a general undirected graph. A matching is a set of edges without shared endpoints, and the goal is to include as many edges as possible. The natural local improvement is to look for an augmenting path: an alternating path that starts and ends at exposed vertices, so flipping which edges are matched along it increases the matching size by one. Berge's theorem says a matching is maximum exactly when no augmenting path exists, so the whole problem reduces to searching for such a path or certifying that none remains.

The difficulty is that the obvious alternating search works cleanly only in bipartite graphs. There, every cycle is even, so vertices can be labeled by parity as the forest grows, and an edge between two outer vertices always connects different trees and immediately yields an augmenting path. In a general graph, an edge can join two outer vertices of the same tree. Those two tree paths plus the new edge form an odd alternating circuit, or blossom. This blossom is neither a direct augmentation nor something that can be safely ignored, because an augmenting path may enter the blossom from outside and exit through the right boundary vertex. Branching over every possibility inside the blossom would defeat the purpose of a polynomial search, so the algorithm needs a way to treat the blossom as a single structural unit without losing any valid augmenting path.

The method is Edmonds' blossom algorithm. It grows an alternating forest from every exposed vertex, scanning edges incident to outer vertices, and whenever it finds a blossom it shrinks the odd circuit into a single pseudovertex. The search then continues in the reduced graph. The key invariant is that the original graph has an augmenting path with respect to the current matching if and only if the reduced graph has an augmenting path with respect to the contracted matching. If the search in the reduced graph finds an augmenting path that avoids all pseudovertices, the same path works in the original graph. If the path uses a pseudovertex, the stored blossom is expanded and the unique internal alternating route that makes the lifted path valid is chosen. Each successful augmentation increases the matching size by one, and when the search exhausts all possibilities without finding an augmenting path, Berge's theorem certifies optimality.

This contraction-and-expansion machinery is what makes the algorithm polynomial. Shrinking collapses the local ambiguity of an odd cycle into a single vertex, preserving the global search structure. Expanding restores the original vertices only when an augmenting path actually passes through the blossom, using the fact that an odd cycle has a unique near-perfect matching compatible with whichever boundary vertex the outside path needs exposed. The dual certificate, the matching polytope odd-set constraints, and the blossoms handled during search are all the same odd-set obstructions in different guises.

```python
from collections import deque

def edmonds_blossom(n, edges):
    """Maximum cardinality matching in a general undirected graph.
    n: number of vertices labeled 0..n-1
    edges: list of unordered pairs (u, v)
    Returns match[v] = partner of v or -1 if v is exposed.
    """
    g = [[] for _ in range(n)]
    for u, v in edges:
        g[u].append(v)
        g[v].append(u)

    match = [-1] * n
    base = list(range(n))
    parent = [-1] * n
    used = [False] * n
    blossom = [False] * n

    def lca(a, b):
        used_path = [False] * n
        while True:
            a = base[a]
            used_path[a] = True
            if match[a] == -1:
                break
            a = parent[match[a]]
        while True:
            b = base[b]
            if used_path[b]:
                return b
            b = parent[match[b]]

    def mark_path(v, b, children):
        while base[v] != b:
            blossom[base[v]] = blossom[base[match[v]]] = True
            parent[v] = children
            children = match[v]
            v = parent[match[v]]

    def find_path(root):
        nonlocal base, parent, used, blossom
        used = [False] * n
        parent = [-1] * n
        base = list(range(n))
        q = deque()
        q.append(root)
        used[root] = True
        while q:
            v = q.popleft()
            for to in g[v]:
                if base[v] == base[to] or match[v] == to:
                    continue
                if to == root or (match[to] != -1 and parent[match[to]] != -1):
                    cur_base = lca(v, to)
                    blossom = [False] * n
                    mark_path(v, cur_base, to)
                    mark_path(to, cur_base, v)
                    for i in range(n):
                        if blossom[base[i]]:
                            base[i] = cur_base
                            if not used[i]:
                                used[i] = True
                                q.append(i)
                elif parent[to] == -1:
                    parent[to] = v
                    if match[to] == -1:
                        # augmenting path found, lift it
                        cur = to
                        while cur != -1:
                            prev = parent[cur]
                            nxt = match[prev] if prev != -1 else -1
                            match[cur] = prev
                            match[prev] = cur
                            cur = nxt
                        return True
                    else:
                        used[match[to]] = True
                        q.append(match[to])
        return False

    for v in range(n):
        if match[v] == -1:
            find_path(v)
    return match
```