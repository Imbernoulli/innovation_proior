# Minimum Path Cover of a DAG via Bipartite Matching

## Problem

Given a directed acyclic graph $G = (V, E)$ on $n = |V|$ vertices, find the minimum
number of vertex-disjoint directed paths that cover every vertex exactly once (the
**minimum path cover**). A single vertex is a path of length zero, so the chosen
paths form a partition of $V$.

## Key idea — split each vertex, then match

A disjoint-path cover is exactly a choice of a *successor* for some vertices: for
each vertex $v$, either pick one out-edge $v \to w$ and call $w$ the next vertex on
$v$'s path, or pick nothing (a path tail). The choice is legal iff every link is a
real edge, each vertex is the **source** of at most one chosen link (out-degree
$\le 1$), and each vertex is the **target** of at most one chosen link (in-degree
$\le 1$). The source-role and target-role of a vertex are independent one-time
budgets, so **split each vertex $v$ into a left copy $v_{\text{out}}$ and a right
copy $v_{\text{in}}$**, and turn every DAG edge $v \to w$ into the bipartite edge

$$v_{\text{out}} \;-\; w_{\text{in}}.$$

A legal successor choice is then exactly a **matching** of this bipartite graph:
$v_{\text{out}}$ touched at most once $=$ at most one successor; $w_{\text{in}}$
touched at most once $=$ at most one predecessor. Let $M$ be the size of a maximum
matching. Then

$$\textbf{minimum path cover} = n - M.$$

## Why $n - M$ (correctness)

Start from $n$ singleton paths (no links chosen). Each chosen link $v \to w$ glues
the fragment ending at $v$ to the fragment starting at $w$; because the choice is
injective on both sides, switching on a link always joins exactly two distinct
fragments into one and drops the path count by one. So choosing $M$ links yields
$n - M$ paths, and minimizing paths means maximizing $M$.

The bijection is exact in both directions. Given a matching, map each matched edge
$v_{\text{out}} - w_{\text{in}}$ back to $v \to w$, forming a subgraph $F \subseteq E$.
Being a matching forces in-degree $\le 1$ and out-degree $\le 1$ at every vertex of
$F$, so $F$ is a disjoint union of simple paths and simple cycles. Since $G$ is
**acyclic**, $F$ has no cycles, hence is a set of vertex-disjoint paths covering all
$n$ vertices, with $n - |F| = n - |M|$ paths. Conversely a disjoint-path cover gives
its successor links as a matching of size $n - (\text{paths})$. Therefore the minimum
over covers equals $n - M$.

Acyclicity is essential: on a general digraph $F$ could contain a cycle (which is not
a path), the bijection breaks, and minimum path cover is in fact NP-hard. The
reduction to bipartite matching is a DAG-only phenomenon.

## Connection to Dilworth's theorem

If the DAG is the strict comparability relation of a finite poset (edge $v \to w$
whenever $v \prec w$), a directed path is a **chain** and a disjoint-path cover is a
partition into chains. The minimum path cover is then the minimum chain cover, which
by **Dilworth's theorem** equals the maximum size of an **antichain** (a set of
pairwise incomparable elements) — the **width** of the poset. Two incomparable
elements can never lie on one chain, so the largest antichain is a lower bound on
the number of chains, and Dilworth's theorem says it is tight. Thus $n - M$ is also
a constructive way to compute the width of a poset, and the bipartite reduction is
the algorithmic face of the min-chain-cover $=$ max-antichain identity.

The comparability assumption matters. If the input contains only cover-relation
edges or another sparse DAG representation of the same poset, a directed path may
miss comparable jumps that should be allowed inside a chain. For Dilworth's theorem
the graph should contain the transitive strict comparability edge $v \to w$ for
every $v \prec w$; otherwise the algorithm still correctly solves the path-cover
problem for the given edge set, but that number need not be the poset width.

## Algorithm

1. Build the split bipartite graph: left vertex $v$ (for $v_{\text{out}}$), right
   vertex $w$ (for $w_{\text{in}}$); add edge $v - w$ for each DAG edge $v \to w$.
2. Compute a maximum bipartite matching $M$ (Kuhn's augmenting-path search,
   $O(V \cdot E)$; Hopcroft–Karp $O(E\sqrt{V})$ if more speed is needed).
3. Return $n - M$.

## Code

Single self-contained C++17 program. It reads `n m` then `m` lines of 1-based
edges `u v` from stdin, and prints the single integer minimum path cover ($n - M$)
to stdout.

```cpp
// Minimum path cover of a DAG via the split-vertex bipartite-matching reduction.
// Reads from stdin: "n m" then m lines each "u v" (1-based) for a directed edge
// u -> v of an acyclic graph. Writes to stdout the single integer minimum number
// of vertex-disjoint paths covering every vertex exactly once, which equals
// n - (maximum bipartite matching of the split graph).
#include <bits/stdc++.h>
using namespace std;

int n;
vector<vector<int>> adj;   // adj[v] = right copies w_in joined to left copy v_out
vector<int> match_right;   // right vertex -> its matched left vertex, or -1
vector<char> used;         // per-augmentation visited flag on right vertices

// Kuhn's augmenting-path DFS: try to place left vertex u on some free/freeable
// right vertex. Returns true on success.
bool try_kuhn(int u) {
    for (int w : adj[u]) {
        if (!used[w]) {
            used[w] = 1;
            if (match_right[w] == -1 || try_kuhn(match_right[w])) {
                match_right[w] = u;
                return true;
            }
        }
    }
    return false;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    long long m_ll;
    if (!(cin >> n >> m_ll)) return 0;
    long long m = m_ll;

    adj.assign(n, {});
    for (long long i = 0; i < m; ++i) {
        int u, v;
        cin >> u >> v;
        --u; --v;                 // to 0-based
        adj[u].push_back(v);      // bipartite edge v_out - w_in
    }

    match_right.assign(n, -1);
    long long matching_size = 0;
    for (int u = 0; u < n; ++u) {
        used.assign(n, 0);
        if (try_kuhn(u)) ++matching_size;
    }

    cout << (n - matching_size) << "\n";
    return 0;
}
```

## Complexity

- Building the split graph: $O(n + m)$.
- Maximum matching with Kuhn: $O(V \cdot E) = O(n \cdot m)$ (Hopcroft–Karp
  $O(m\sqrt{n})$ if needed).
- Total $O(n \cdot m)$ time, $O(n + m)$ memory. The answer is the integer $n - M$.
