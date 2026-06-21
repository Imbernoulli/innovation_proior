I solve the IOI 2014 Game problem by acting as a lazy adversary that keeps the final connectivity of the graph ambiguous until the very last edge query. The canonical name I propose for this approach is the lazy spanning-tree adversary, or equivalently the last-query tree strategy. The idea is to force the questioner to wait until all pairs have been asked before the connectivity of the graph becomes certain, no matter what order the questioner chooses.

The problem gives n cities and asks, for each unordered pair exactly once, whether a direct flight exists. After each answer, the questioner would like to declare that every graph consistent with the answers so far is connected, or that every such graph is disconnected. The responder wins by delaying that declaration until after the final answer. The strategy must therefore maintain, after every prefix of fewer than r = n(n-1)/2 answers, at least one completion that is connected and at least one completion that is disconnected.

Let G be the graph formed by all pairs answered yes so far, and let H be the graph formed by those yes edges together with every pair that has not yet been asked. The responder loses the ambiguity at a prefix exactly when G is already connected or H is already disconnected. So the goal is to keep G disconnected and H connected throughout the interaction.

A clean way to guarantee this is to decide in advance on a spanning tree T that will be the final yes graph, and then answer yes to an edge only when it is the last unasked edge of T that the questioner has not yet requested. If T is fixed, then before the last query some edge of T is still missing, so G has at most n-2 edges and cannot be connected. At the same time, every edge of T is either already a yes or still unasked, so H contains T and is connected. This invariant holds until the very last query, at which point the missing tree edge is finally answered yes and G becomes exactly T, a connected spanning tree.

The remaining difficulty is that the questioner chooses the order adaptively, so the responder cannot know which edge of T will be asked last. However, the responder can choose T to have a structure that makes the last edge of T easy to recognize online. The natural choice is a rooted tree in which every vertex w > 0 has exactly one neighbor with a smaller label. Such a tree has no cycles because following parent pointers strictly decreases the vertex label until reaching the root 0, and it spans all vertices because every vertex has a parent. The edge set can be partitioned by the larger endpoint: for each w, the owned pairs are E_w = {(w, 0), (w, 1), ..., (w, w-1)}. The tree will contain exactly one edge from each E_w.

With this partition, the online rule becomes extremely simple. For each vertex w, maintain a counter c[w] of how many pairs from E_w have already been queried. When a new pair (u, v) arrives, let w = max(u, v), increment c[w], and answer yes exactly when c[w] reaches w, that is, when this is the last queried pair owned by w. All earlier pairs owned by w are answered no. Because every w > 0 owns exactly w pairs, each such vertex contributes exactly one yes edge, and that edge connects w to some smaller-labeled vertex. Thus the final yes graph is a spanning tree rooted at 0.

The correctness follows directly from the spanning-tree invariant. After all queries, each vertex w > 0 has one yes edge to a smaller vertex, so the graph is connected. It is also acyclic: in any hypothetical cycle, consider the largest vertex w on that cycle; both cycle neighbors of w would have smaller labels, giving two yes edges from E_w, contradicting the fact that exactly one edge from E_w is answered yes. Therefore the final yes graph T is a tree.

For any non-final prefix, let the last query of the entire interaction be the one that completes some E_w. Before that moment, at least one edge of T is still missing from G, namely the final yes edge of that E_w. Since G is a subgraph of T with at most n-2 edges, it cannot be connected. On the other hand, every edge of T is either already in G or still unasked, so T is contained in H, and H is connected. Hence the responder preserves both a connected and a disconnected completion until the final answer.

The implementation is tiny and efficient. Only a counter per vertex is needed, initialized to zero. Each query takes constant time: compare u and v to find the larger endpoint, increment the counter, and test equality with the vertex label. Initialization is O(n), memory is O(n), and over all n choose 2 queries the total time is O(n^2). This easily satisfies the required constraints up to n = 1500.

The Python snippet below implements the lazy spanning-tree adversary and verifies it on random query orders. It defines the responder, simulates an arbitrary ordering of all pairs, checks the spanning-tree structure of the final yes graph, and confirms that the ambiguity invariant holds for every non-final prefix.

```python
import itertools
import random

def lazy_tree_adversary(n, queries):
    """Simulate the lazy spanning-tree adversary.
    queries is a list of unordered pairs (u, v) with u != v,
    containing each unordered pair exactly once.
    Returns (answers, final_edges) where answers[i] is 0/1.
    """
    c = [0] * n
    answers = []
    final_edges = []
    for (u, v) in queries:
        w = u if u > v else v
        c[w] += 1
        ans = 1 if c[w] == w else 0
        answers.append(ans)
        if ans == 1:
            final_edges.append((u, v))
    return answers, final_edges

def is_connected(edges, n):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    seen = [False] * n
    stack = [0]
    seen[0] = True
    count = 1
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                count += 1
                stack.append(v)
    return count == n

def check_invariant(n, queries, answers):
    """Check that before every non-final prefix G is disconnected
    and H (yes edges plus all unasked pairs) is connected."""
    asked = set()
    yes_edges = []
    total_pairs = n * (n - 1) // 2
    for idx, ((u, v), ans) in enumerate(zip(queries, answers)):
        asked.add((min(u, v), max(u, v)))
        if ans == 1:
            yes_edges.append((u, v))
        if idx + 1 == total_pairs:
            break
        # G is the yes graph so far.
        assert not is_connected(yes_edges, n), "G connected too early"
        # H is yes_edges plus every unasked pair.
        unasked = [(i, j) for i in range(n) for j in range(i + 1, n)
                   if (i, j) not in asked]
        assert is_connected(yes_edges + unasked, n), "H disconnected"

for n in [4, 7, 12]:
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    for trial in range(20):
        queries = all_pairs[:]
        random.shuffle(queries)
        answers, final_edges = lazy_tree_adversary(n, queries)
        assert len(final_edges) == n - 1
        assert is_connected(final_edges, n)
        # Tree has exactly n-1 edges and is connected, so it is acyclic.
        check_invariant(n, queries, answers)
print("All random tests passed.")
```
