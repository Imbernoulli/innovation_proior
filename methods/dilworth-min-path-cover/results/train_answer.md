I am handed a directed acyclic graph on $n$ vertices, and I want to cover all of it with as few vertex-disjoint directed paths as possible: every vertex must lie on exactly one path, where a lone vertex counts as a path of length zero, and I want the number of paths to be minimal. The naive route is to enumerate every way of partitioning the vertices into directed paths and keep the partition with the fewest pieces. On a handful of vertices that is fine, but the number of partitions explodes, and at the scale of interest — $n$ up to a few thousand, $m$ up to tens of thousands — brute force tells me what the answer is but gives me no way to compute it. I need to expose the combinatorial structure of a path cover rather than search over covers blindly.

The structure is this. Stare at a single path $v_1 \to v_2 \to \dots \to v_k$ in some cover. Every vertex except the last has exactly one vertex immediately after it — its successor — and every vertex except the first has exactly one immediately before it — its predecessor. A path is nothing more than the chain of these successor links, so a whole disjoint-path cover is nothing more than a choice, vertex by vertex, of "the next vertex on my path," with the path tails choosing nobody. I propose to solve the problem by making that observation literal: model the cover as a successor choice, recognize the constraints on it as two independent one-time budgets per vertex, and reduce the optimization to a maximum bipartite matching. The method is the split-vertex bipartite-matching reduction, and its answer is

$$\textbf{minimum path cover} = n - M,$$

where $M$ is the size of a maximum matching in a graph built by splitting each vertex in two.

Here is the mechanism in full. Write the cover as a partial function: for each vertex $v$, either pick one out-edge $v \to w$ and declare $w$ to be $v$'s successor, or pick nothing. A set of such choices is a legal disjoint-path cover exactly when three things hold. First, every chosen link is a real edge of the graph — I can only walk where the arrows point. Second, each vertex is the source of at most one chosen link (out-degree $\le 1$): a vertex names at most one successor. Third, each vertex is the target of at most one chosen link (in-degree $\le 1$): if two different vertices both named $w$ as their successor, then $w$ would have two predecessors and two paths would crash into it, violating disjointness. A fourth worry — that the successor links might close into a cycle $v_1 \to \dots \to v_1$, which is not a path — simply evaporates, because the graph is acyclic and following edges can never return to its start. That escape is not decoration; it is the hinge of the whole reduction, and I will return to it.

The decisive observation is that the source-role and the target-role of a vertex are *independent* budgets. The constraint never ties $v$'s out-choice to $v$'s in-choice; "$v$ used once as a source" and "$v$ used once as a target" are two separate one-time allowances. That independence is the tell: I should give each vertex two distinct identities, one spendable as a source and one spendable as a target, and the constraint becomes simply "use each identity at most once." So I make the two copies literal. For every vertex $v$, create a left copy $v_{\text{out}}$ ("$v$ acting as a source, having a successor") and a right copy $v_{\text{in}}$ ("$v$ acting as a target, having a predecessor"). For every directed edge $v \to w$, draw one undirected edge between copies,

$$v_{\text{out}} \;-\; w_{\text{in}}.$$

Now "out-degree $\le 1$ at $v$" reads as "$v_{\text{out}}$ touches at most one chosen edge," and "in-degree $\le 1$ at $w$" reads as "$w_{\text{in}}$ touches at most one chosen edge." A legal successor choice is therefore exactly a set of edges in this new graph where every copy touches at most one chosen edge — that is precisely a matching. And the new graph is bipartite by construction: every edge runs from a left ($\cdot_{\text{out}}$) copy to a right ($\cdot_{\text{in}}$) copy, never left-to-left or right-to-right. Maximizing the number of links over all legal successor choices is identical to finding a maximum-cardinality matching in this bipartite split graph.

Why the answer is $n - M$ and not something subtler is the one place to be careful, and it follows from a gluing count. Start with no links chosen: nobody has a successor, every vertex is its own isolated path, that is $n$ paths. Now switch on the chosen links one at a time. Each link $v \to w$ glues the path ending at $v$ to the path starting at $w$. Because the choice is injective on both sides — each $v$ has at most one out-link, each $w$ at most one in-link — turning on a link never creates a branch or a three-way merge; it always joins exactly two distinct fragments into one, dropping the path count by exactly one. So choosing $M$ links leaves $n - M$ paths, and minimizing paths is maximizing $M$.

That the correspondence is an exact bijection, in both directions, is what makes the formula trustworthy. Take any matching in the split graph and map each matched edge $v_{\text{out}} - w_{\text{in}}$ back to the original edge $v \to w$, collecting these into a subgraph $F$ of the DAG. Because it is a matching, $v_{\text{out}}$ is matched at most once, so $v$ has out-degree at most one in $F$, and $w_{\text{in}}$ is matched at most once, so $w$ has in-degree at most one. A directed subgraph in which every vertex has in-degree $\le 1$ and out-degree $\le 1$ is a disjoint union of simple paths and simple cycles — follow out-edges from any vertex and you can neither branch (out-degree $\le 1$) nor merge (in-degree $\le 1$), so each connected piece is a single chain, open or closed. Here the acyclicity does its work: $G$ has no cycles, so $F$ has none either, hence $F$ is a set of vertex-disjoint directed paths covering all $n$ vertices (vertices with no chosen link are length-zero paths), with $n - |F|$ paths. Conversely, any disjoint-path cover hands back its successor links, each a real edge, injective on both sides, hence a matching of size $n - (\text{number of paths})$. Covers and matchings are in bijection with sizes tied by paths $= n - |\text{matching}|$, so the minimum over covers is $n - M$. On a general digraph this would break: $F$ could contain a cycle, which is not a path, and $n - M$ could be too small — indeed minimum path cover on a general digraph is NP-hard. The clean reduction is a DAG-only phenomenon.

This same quantity has a second face. If the DAG is the strict comparability relation of a finite poset — an edge $v \to w$ whenever $v \prec w$ — then a directed path is a *chain* of pairwise-comparable elements, a disjoint-path cover is a partition into chains, and the minimum path cover is the minimum number of chains covering the poset. By Dilworth's theorem that number equals the maximum size of an *antichain*, a set of pairwise-incomparable elements, which is the *width* of the poset. Two incomparable elements can never share a chain, so the largest antichain is a lower bound on the chains needed, and Dilworth's theorem makes it tight, with the antichain serving as a certificate of optimality. So $n - M$ is also a constructive way to compute poset width, and the bipartite reduction is the algorithmic face of the min-chain-cover $=$ max-antichain identity. One care: for the poset reading I must feed in the transitive strict comparability edges — an edge for every $v \prec w$, not only cover-relation-adjacent pairs — because a chain is any increasing subset; given a sparser edge set the algorithm still correctly solves the path-cover problem for that edge set, but the number need not be the width.

The only remaining piece is computing $M$, and Kuhn's augmenting-path search supplies it in $O(V \cdot E) = O(n \cdot m)$, comfortable at this scale. For each right vertex I keep the left vertex it is currently matched to, or none. Processing left vertices one at a time, I run a DFS that tries to place the current vertex: walk to a neighbor $w$; if $w$ is free, claim it; if $w$ is taken by some $u'$, recursively try to re-place $u'$ on a different right vertex, freeing $w$ if that succeeds. A `used` flag per right vertex prevents revisiting a right vertex within one augmentation. Berge's lemma guarantees correctness: a matching is maximum exactly when no augmenting path exists, and the DFS finds one whenever one exists, so once every left vertex has been tried the matching is maximum and the count of successful placements is $M$. Wiring this to the split graph is direct: each original vertex $v$ is left index $v$ (its $v_{\text{out}}$) and right index $v$ (its $v_{\text{in}}$), both sides have $n$ indices, and each DAG edge $v \to w$ adds $w$ to the neighbor list of left vertex $v$. Then `min_path_cover` calls the matching routine and returns $n - M$.

```python
import sys

sys.setrecursionlimit(1_000_000)


def read_dag(data):
    """Parse n, m, and m directed edges (1-based in input) into a 0-based
    adjacency list of a DAG. Returns (n, adj)."""
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    adj = [[] for _ in range(n)]
    for _ in range(m):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        adj[u].append(v)
    return n, adj


def bipartite_matching(n_left, n_right, adj):
    """Maximum-cardinality matching of a bipartite graph by Kuhn's
    augmenting-path search. adj[u] lists right vertices joined to left vertex u.
    Returns the size of a maximum matching. O(V * E)."""
    match_right = [-1] * n_right       # right vertex -> its matched left vertex

    def try_kuhn(u, used):
        for w in adj[u]:
            if not used[w]:
                used[w] = True
                if match_right[w] == -1 or try_kuhn(match_right[w], used):
                    match_right[w] = u
                    return True
        return False

    size = 0
    for u in range(n_left):
        used = [False] * n_right
        if try_kuhn(u, used):
            size += 1
    return size


def min_path_cover(n, adj):
    bip = [[] for _ in range(n)]
    for v in range(n):
        for w in adj[v]:
            bip[v].append(w)
    matching_size = bipartite_matching(n, n, bip)
    return n - matching_size


def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    n, adj = read_dag(data)
    print(min_path_cover(n, adj))


if __name__ == "__main__":
    main()
```
