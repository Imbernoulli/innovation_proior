# Lazy adversary for connectivity

## Method

Partition the unordered pairs by their larger endpoint. For each $w=1,\dots,n-1$, define

$$
E_w = \{(w,0),(w,1),\dots,(w,w-1)\}.
$$

The set $E_w$ has exactly $w$ pairs. Answer yes to exactly the last queried pair in each $E_w$, and answer no to all earlier pairs in that set. A counter `c[w]` detects that last query online: after incrementing it, the pair is last exactly when `c[w] == w`.

```c
#include "game.h"

int c[1500];

void initialize(int n) {
    int i;
    for (i = 0; i < n; ++i) c[i] = 0;
}

int hasEdge(int u, int v) {
    int w = u > v ? u : v;
    return ++c[w] == w;
}
```

## Correctness

Across the whole run, every vertex $w>0$ gets exactly one yes edge to a smaller-labelled vertex. Following these edges strictly decreases labels until reaching $0$, so the final yes graph is connected. It is also acyclic: if a cycle existed, its largest-labelled vertex $w$ would have two incident cycle edges to smaller vertices, meaning two yes edges from $E_w$, but the strategy chooses only one edge from $E_w$. Thus the final yes graph $T$ is a spanning tree.

For any prefix before the last query, let $G$ be the confirmed yes graph and let $H$ be the graph containing confirmed yes edges plus all unasked pairs. The last query of the whole run is answered yes, so every earlier prefix is missing at least that final edge of the final tree $T$. All confirmed yes edges belong to $T$, hence $G$ has at most $n-2$ edges on the same $n$ vertices. A connected graph on $n$ vertices needs at least $n-1$ edges, so $G$ is disconnected. Meanwhile every edge of $T$ is present in $H$: already-asked tree edges are confirmed yes, and not-yet-asked tree edges are still possible. Therefore $H$ contains a spanning tree and is connected.

So before every non-final prefix, both a disconnected completion and a connected completion remain consistent with the answers. The final query adds the last missing tree edge, making the graph connected only then. The strategy uses $O(n)$ memory, $O(n)$ initialization time, and $O(1)$ time per query, hence $O(n^2)$ total time over all $\binom{n}{2}$ queries.
