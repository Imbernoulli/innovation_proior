# The Max-Flow Min-Cut Theorem

## Statement

In a directed network `G=(V,E)` with source `s`, sink `t`, and nonnegative capacities `c(e)`, the maximum value of a feasible `s-t` flow equals the minimum capacity of an `s-t` cut:

```text
max_f |f| = min_{S: s in S, t not in S} sum_{u in S, v not in S} c(u,v).
```

If all capacities are integers, there is an integer-valued maximum flow.

## Certificate Lemma

For any feasible flow `f` and cut `(S,T)`,

```text
|f| = sum_{S->T} f - sum_{T->S} f
     <= sum_{S->T} f
     <= sum_{S->T} c
     = cap(S,T).
```

Equality holds exactly when every arc from `S` to `T` is saturated and every arc from `T` to `S` carries zero flow. Thus one matching flow/cut pair certifies both optima.

## Residual Augmentation

For a current flow `f`, define residual capacity

```text
c_f(u,v) = c(u,v) - f(u,v) + f(v,u),
```

where missing arcs have capacity and flow zero. The first term is unused forward capacity; the second is flow on the reverse arc that can be canceled.

If the residual graph has an `s-t` path `P`, push

```text
F = min_{(u,v) in P} c_f(u,v)
```

along it. Each residual step cancels reverse flow first and then adds any remaining amount forward. This keeps all arc flows in `[0,c]`, preserves conservation at internal vertices, and increases the flow value by `F`.

If no residual `s-t` path exists, let `S` be the vertices reachable from `s` in the residual graph. Then `t` is outside `S`. No residual edge leaves `S`, so every original arc from `S` to `T` is saturated and every original arc from `T` to `S` carries zero flow. By the certificate lemma, `|f|=cap(S,T)`, so `f` is maximum and `(S,T)` is minimum.

## Algorithm

The landing is a single self-contained C++17 program. It reads from stdin `n m s t` followed by `m` lines `u v c` (1-indexed vertices, arc `u->v` with capacity `c`), and prints the maximum `s-t` flow value followed by the source side `S` of a minimum cut. Capacities accumulate in `long long` to avoid overflow.

```cpp
// Max-flow / min-cut via Edmonds-Karp (shortest residual augmenting paths).
// Reads from stdin: "n m s t" then m lines "u v c" (1-indexed vertices, arc u->v
// with capacity c). Prints the maximum s-t flow value, then the source side S of
// a minimum cut as a sorted vertex list. Use long long: total flow can exceed int.
#include <bits/stdc++.h>
using namespace std;

const long long INF = (long long)4e18;

int n, m, s, t;
vector<int> head;        // head[u] = first edge index out of u, or -1
vector<int> to_, nxt;    // adjacency: endpoint and next-edge pointers
vector<long long> cap;   // residual capacity of each directed edge

void add_edge(int u, int v, long long c) {
    // Forward edge with capacity c, reverse edge with capacity 0; they are
    // stored adjacently so the reverse of edge e is e^1.
    to_.push_back(v); cap.push_back(c); nxt.push_back(head[u]); head[u] = (int)to_.size() - 1;
    to_.push_back(u); cap.push_back(0); nxt.push_back(head[v]); head[v] = (int)to_.size() - 1;
}

// One BFS for a shortest residual s-t path; returns the bottleneck pushed, or 0.
long long bfs_augment() {
    vector<int> parent_edge(n + 1, -1);
    vector<char> seen(n + 1, 0);
    queue<int> q;
    q.push(s); seen[s] = 1;
    while (!q.empty()) {
        int u = q.front(); q.pop();
        if (u == t) break;
        for (int e = head[u]; e != -1; e = nxt[e]) {
            int v = to_[e];
            if (!seen[v] && cap[e] > 0) {
                seen[v] = 1;
                parent_edge[v] = e;
                q.push(v);
            }
        }
    }
    if (!seen[t]) return 0;
    long long f = INF;
    for (int v = t; v != s; ) {
        int e = parent_edge[v];
        f = min(f, cap[e]);
        v = to_[e ^ 1];
    }
    for (int v = t; v != s; ) {
        int e = parent_edge[v];
        cap[e] -= f;
        cap[e ^ 1] += f;
        v = to_[e ^ 1];
    }
    return f;
}

int main() {
    if (scanf("%d %d %d %d", &n, &m, &s, &t) != 4) return 0;
    head.assign(n + 1, -1);
    for (int i = 0; i < m; ++i) {
        int u, v; long long c;
        scanf("%d %d %lld", &u, &v, &c);
        add_edge(u, v, c);
    }

    long long value = 0;
    while (true) {
        long long f = bfs_augment();
        if (f == 0) break;
        value += f;
    }

    // Source side S of a minimum cut: vertices still reachable from s in the
    // final residual graph. By max-flow min-cut its forward-crossing capacity
    // equals the flow value.
    vector<char> inS(n + 1, 0);
    queue<int> q;
    q.push(s); inS[s] = 1;
    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int e = head[u]; e != -1; e = nxt[e]) {
            int v = to_[e];
            if (!inS[v] && cap[e] > 0) { inS[v] = 1; q.push(v); }
        }
    }

    printf("%lld\n", value);
    bool first = true;
    for (int v = 1; v <= n; ++v) {
        if (inS[v]) { printf(first ? "%d" : " %d", v); first = false; }
    }
    printf("\n");
    return 0;
}
```

With integer capacities, each augmentation raises the value by at least `1`, so the basic method terminates and returns an integral optimum. With breadth-first shortest augmenting paths, residual distances never decrease and each edge can be a bottleneck only `O(V)` times; hence there are `O(VE)` augmentations and total running time `O(VE^2)`.
