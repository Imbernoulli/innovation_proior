I solve the IOI 2014 Game problem by acting as a lazy adversary that keeps the final connectivity of the graph ambiguous until the very last edge query. The canonical name I propose for this approach is the lazy spanning-tree adversary, or equivalently the last-query tree strategy. The idea is to force the questioner to wait until all pairs have been asked before the connectivity of the graph becomes certain, no matter what order the questioner chooses.

The problem gives n cities and asks, for each unordered pair exactly once, whether a direct flight exists. After each answer, the questioner would like to declare that every graph consistent with the answers so far is connected, or that every such graph is disconnected. The responder wins by delaying that declaration until after the final answer. The strategy must therefore maintain, after every prefix of fewer than r = n(n-1)/2 answers, at least one completion that is connected and at least one completion that is disconnected.

Let G be the graph formed by all pairs answered yes so far, and let H be the graph formed by those yes edges together with every pair that has not yet been asked. The responder loses the ambiguity at a prefix exactly when G is already connected or H is already disconnected. So the goal is to keep G disconnected and H connected throughout the interaction.

A clean way to guarantee this is to decide in advance on a spanning tree T that will be the final yes graph, and then answer yes to an edge only when it is the last unasked edge of T that the questioner has not yet requested. If T is fixed, then before the last query some edge of T is still missing, so G has at most n-2 edges and cannot be connected. At the same time, every edge of T is either already a yes or still unasked, so H contains T and is connected. This invariant holds until the very last query, at which point the missing tree edge is finally answered yes and G becomes exactly T, a connected spanning tree.

The remaining difficulty is that the questioner chooses the order adaptively, so the responder cannot know which edge of T will be asked last. However, the responder can choose T to have a structure that makes the last edge of T easy to recognize online. The natural choice is a rooted tree in which every vertex w > 0 has exactly one neighbor with a smaller label. Such a tree has no cycles because following parent pointers strictly decreases the vertex label until reaching the root 0, and it spans all vertices because every vertex has a parent. The edge set can be partitioned by the larger endpoint: for each w, the owned pairs are E_w = {(w, 0), (w, 1), ..., (w, w-1)}. The tree will contain exactly one edge from each E_w.

With this partition, the online rule becomes extremely simple. For each vertex w, maintain a counter c[w] of how many pairs from E_w have already been queried. When a new pair (u, v) arrives, let w = max(u, v), increment c[w], and answer yes exactly when c[w] reaches w, that is, when this is the last queried pair owned by w. All earlier pairs owned by w are answered no. Because every w > 0 owns exactly w pairs, each such vertex contributes exactly one yes edge, and that edge connects w to some smaller-labeled vertex. Thus the final yes graph is a spanning tree rooted at 0.

The correctness follows directly from the spanning-tree invariant. After all queries, each vertex w > 0 has one yes edge to a smaller vertex, so the graph is connected. It is also acyclic: in any hypothetical cycle, consider the largest vertex w on that cycle; both cycle neighbors of w would have smaller labels, giving two yes edges from E_w, contradicting the fact that exactly one edge from E_w is answered yes. Therefore the final yes graph T is a tree.

For any non-final prefix, let the last query of the entire interaction be the one that completes some E_w. Before that moment, at least one edge of T is still missing from G, namely the final yes edge of that E_w. Since G is a subgraph of T with at most n-2 edges, it cannot be connected. On the other hand, every edge of T is either already in G or still unasked, so T is contained in H, and H is connected. Hence the responder preserves both a connected and a disconnected completion until the final answer.

The implementation is tiny and efficient. Only a counter per vertex is needed, initialized to zero. Each query takes constant time: compare u and v to find the larger endpoint, increment the counter, and test equality with the vertex label. Initialization is O(n), memory is O(n), and over all n choose 2 queries the total time is O(n^2). This easily satisfies the required constraints up to n = 1500.

The program below implements the lazy spanning-tree adversary as a single self-contained C++17 file driven by the sample grader's stdin format: the first line is `n`, then the next `r = n(n-1)/2` lines each hold a queried pair `u v`, and for each query it prints `1` if a direct flight is claimed and `0` otherwise. The only state is one counter per vertex; each query compares the two endpoints, increments the owner's counter, and answers yes exactly when that counter reaches the owner's label, so the yes edges accumulate into the spanning tree only at the very last query of each owned set.

```cpp
// IOI 2014 "Game": lazy spanning-tree adversary.
// Reads from stdin: line 1 is n; then r = n(n-1)/2 lines, each "u v".
// For each query prints one line, 1 if a direct flight is claimed, else 0.
#include <cstdio>

static int c[1500];

int main() {
    int n;
    if (scanf("%d", &n) != 1) return 0;
    for (int i = 0; i < n; ++i) c[i] = 0;
    long long r = (long long)n * (n - 1) / 2;
    for (long long q = 0; q < r; ++q) {
        int u, v;
        if (scanf("%d %d", &u, &v) != 2) break;
        int w = u > v ? u : v;          // owner = larger endpoint
        int ans = (++c[w] == w) ? 1 : 0; // yes only on the last query owned by w
        printf("%d\n", ans);
    }
    return 0;
}
```
