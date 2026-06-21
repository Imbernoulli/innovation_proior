The question is whether every graph property preserved under vertex deletion, edge deletion, and edge contraction must have a finite list of minimal forbidden minors. Before Robertson and Seymour, the only way to answer this was family by family: planarity had K_5 and K_{3,3}, outerplanarity had its own small list, and forests forbade cycles. That cataloguing style works when geometry or simple structure gives a visible obstruction theory, but it gives no reason why an arbitrary minor-closed class should be finitely describable. The bad graphs could in principle form an infinite antichain, with each new graph avoiding all previous ones as minors, and nothing in the per-family approach prevents that. Even worse, some natural classes have obstruction sets that are too large to write down explicitly, so a method that depends on enumeration cannot be the general answer.

The new method is the Graph Minor Theorem, proved by Robertson and Seymour. It says that finite graphs are well-quasi-ordered by the minor relation: in every infinite sequence of finite graphs, some earlier graph is a minor of a later one. Equivalently, there is no infinite antichain under minors. For any minor-closed class F, the minimal graphs not in F are automatically pairwise incomparable under minors, so if there were infinitely many of them they would be an infinite antichain. The theorem rules this out, so every minor-closed class has a finite excluded-minor basis. This is a decisive reframing: instead of asking what the forbidden minors are for one particular class, the theorem asks why the minor order itself cannot support an infinite collection of mutually unrelated graphs.

The theorem is not just an order-theoretic compactness statement; its proof supplies a structural grammar for graph families. Tree decompositions split graphs into pieces arranged along a tree-like skeleton with bounded overlap. For bounded-treewidth graphs, this reduction to trees makes well-quasi-ordering arguments tractable, because trees and graphs of bounded treewidth behave enough like strings or finite trees for Kruskal-style arguments to apply. The deep case is unbounded treewidth. The excluded-minor structure theorem shows that graphs avoiding a fixed minor can be built by clique-sums from pieces that are nearly embedded on bounded-genus surfaces, with only bounded exceptional features such as apices and vortices. This decomposition isolates the reasons a large graph can avoid a forbidden minor and lets the well-quasi-ordering propagate through the pieces rather than fighting arbitrary graphs directly.

The conceptual shift is from classifying individual graphs to classifying whole families through decomposition and global order. Instead of trying to enumerate obstructions, which is hopeless for many natural classes, the method proves that the minor relation itself has enough regularity to force finiteness. Tree decompositions organize the graph, the excluded-minor structure theorem controls the pieces, and well-quasi-ordering turns that control into a finite forbidden-minor description. The lasting insight is that the apparent infinity of graph pathologies is not arbitrary: under minor operations, every minor-closed world is governed by finitely many forbidden witnesses. This does not mean the obstruction set is always easy to find, only that it must exist, which is exactly the kind of structural guarantee that makes further algorithmic and combinatorial work possible.

```python
from collections import deque
from itertools import combinations


def is_minor(G, H):
    """Return True if H is a minor of G.

    The test looks for disjoint non-empty connected subsets of G's vertices,
    one for each vertex of H, such that adjacent vertices of H correspond to
    subsets joined by at least one edge of G.  This is exponential and is
    meant only as a small-graph illustration of the theorem.
    """
    nG, nH = len(G), len(H)
    if nH > nG:
        return False

    adj_G = {v: set(G[v]) for v in range(nG)}
    adj_H = {v: set(H[v]) for v in range(nH)}

    def connected(subset):
        subset = set(subset)
        if not subset:
            return False
        start = next(iter(subset))
        seen = {start}
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for w in adj_G[u] & subset:
                if w not in seen:
                    seen.add(w)
                    queue.append(w)
        return seen == subset

    def edge_between(A, B):
        for u in A:
            if adj_G[u] & set(B):
                return True
        return False

    vertices = list(range(nG))
    assignment = {}

    def backtrack(i, used):
        if i == nH:
            for u in range(nH):
                for v in adj_H[u]:
                    if v <= u:
                        continue
                    if not edge_between(assignment[u], assignment[v]):
                        return False
            return True

        remaining = [v for v in vertices if v not in used]
        for size in range(1, len(remaining) + 1):
            for subset in combinations(remaining, size):
                if connected(subset):
                    assignment[i] = subset
                    if backtrack(i + 1, used | set(subset)):
                        return True
        return False

    return backtrack(0, set())


def in_minor_closed_class(G, obstructions):
    """Membership test for a minor-closed class with a known finite
    obstruction set.  By the Graph Minor Theorem, such a finite set exists
    for every minor-closed class, even if it is not always easy to write
    down explicitly.
    """
    for H in obstructions:
        if is_minor(G, H):
            return False
    return True


# Wagner's theorem: a graph is planar iff it excludes K_5 and K_{3,3}.
K5 = {
    0: {1, 2, 3, 4},
    1: {0, 2, 3, 4},
    2: {0, 1, 3, 4},
    3: {0, 1, 2, 4},
    4: {0, 1, 2, 3},
}

K33 = {
    0: {3, 4, 5},
    1: {3, 4, 5},
    2: {3, 4, 5},
    3: {0, 1, 2},
    4: {0, 1, 2},
    5: {0, 1, 2},
}

planar_obstructions = [K5, K33]

# The 3-cube is planar.
cube = {
    0: {1, 2, 4},
    1: {0, 3, 5},
    2: {0, 3, 6},
    3: {1, 2, 7},
    4: {0, 5, 6},
    5: {1, 4, 7},
    6: {2, 4, 7},
    7: {3, 5, 6},
}

print(in_minor_closed_class(cube, planar_obstructions))  # True
print(in_minor_closed_class(K5, planar_obstructions))    # False
print(in_minor_closed_class(K33, planar_obstructions))   # False
```
