# Maximum-Weight Closure / Project Selection via Minimum Cut

## Problem

A directed graph $G=(V,E)$ has an integer weight $w_v$ on each vertex (possibly negative). A
**closure** is a vertex set $V_1 \subseteq V$ with no edge leaving it: for every edge
$u \to v$, if $u \in V_1$ then $v \in V_1$. The **maximum-weight closure** maximizes
$\sum_{v \in V_1} w_v$.

This models *project selection*: vertex = project, $w_v = p_v$ = its profit, edge $u \to v$
= "project $u$ requires project $v$." A feasible selection is exactly a closure, and the
maximum-weight closure is the most profitable feasible set of projects.

## Key idea

Reduce to a minimum $s$–$t$ cut. Build a network $N$:

- Add a source $s$ and sink $t$.
- Every positive vertex $v$ ($w_v>0$): edge $s \to v$ with capacity $w_v$.
- Every negative vertex $v$ ($w_v<0$): edge $v \to t$ with capacity $-w_v$.
- Every original edge $u \to v$: capacity `INF`, any integer strictly larger than
  $\sum_v |w_v|$.

A cut $[S,T]$ is **simple** if it cuts only $s$- and $t$-edges (no original edge).

**Lemma 1 (min cut is simple).** Cutting $s$ alone from the rest gives a finite cut of
capacity $\sum_{v:w_v>0} w_v \le \sum_v |w_v|$. Any cut using an `INF` edge has capacity
strictly larger. So the minimum cut never cuts an original edge — it is simple.

**Lemma 2 (bijection).** Simple cuts and closures correspond via $S = V_1 \cup \{s\}$. If
$V_1$ is a closure, no edge leaves $V_1$, so $[V_1\cup\{s\},\,V_2\cup\{t\}]$ cuts no original
edge (simple). Conversely if $[S,T]$ is simple and $V_1 = S\setminus\{s\}$, then for any
$u\in V_1$ and edge $u\to v$, that edge is uncut, so $v\in S$, i.e. $v\in V_1$ — $V_1$ is a
closure.

**Quantification.** For a simple cut, the only cut edges out of $s$ go to positive vertices
that were not selected, and the only cut edges into $t$ come from negative vertices that were
selected. Writing those sets as $V_2^+$ and $V_1^-$,

$$c[S,T] = \sum_{v\in V_2^+} w_v + \sum_{v\in V_1^-}(-w_v).$$

The closure's weight is $w(V_1) = \sum_{v\in V_1^+} w_v - \sum_{v\in V_1^-}(-w_v)$. Adding,
the $V_1^-$ terms cancel and the positive terms combine over all positive vertices:

$$w(V_1) + c[S,T] = \sum_{v\in V^+} w_v \quad(\text{constant}),
\qquad\Longrightarrow\qquad
w(V_1) = \sum_{v\in V^+} w_v - c[S,T].$$

**Optimality.** $\sum_{v\in V^+} w_v$ is fixed, so maximizing $w(V_1)$ equals minimizing the
simple-cut capacity; by Lemma 1 that is the global minimum cut. By max-flow–min-cut,

$$\text{max-weight closure} = \Big(\sum_{v:w_v>0} w_v\Big) - (\text{min cut})
= \Big(\sum_{v:w_v>0} w_v\Big) - (\text{max flow}).$$

The selected set is recovered as the residual-reachable vertices from $s$ (the source side
$S$); the `INF` edges guarantee this set is a valid closure.

## Algorithm

1. Build $N$ as above; let $P = \sum_{v:w_v>0} w_v$.
2. Compute a maximum $s$–$t$ flow (Dinic). Its value is the minimum cut $c$.
3. Answer profit $= P - c$.
4. (To recover the selection: BFS from $s$ over residual-positive edges; the reached original
   vertices are the optimal closure.)

Complexity: $O(\mathrm{MaxFlow}(N))$ — one max-flow on $|V|+2$ vertices and
$|E| + |V|$ edges.

## Code

The deliverable is a single self-contained C++17 program. It instantiates the reduction on the
relay-station selection problem (below): it reads the instance from stdin and prints the
maximum net profit to stdout. Stations are the negative-weight projects (a build cost),
user groups are the positive-weight projects (a revenue) with two prerequisite edges to the
stations they need. All capacities and the profit total use `long long` to avoid overflow.

```cpp
// Relay-station selection (maximum net profit) via maximum-weight closure / min-cut.
// Reads from stdin:
//   line 1: n m            (n stations, m user groups)
//   line 2: p_1 ... p_n    (cost to build station i)
//   next m lines: a_i b_i c_i  (group i uses stations a_i,b_i (1-indexed), revenue c_i)
// Writes to stdout: the maximum achievable net profit (total revenue - total build cost).
#include <bits/stdc++.h>
using namespace std;

struct Dinic {
    int n, s, t;
    vector<int> to, nxt, head;
    vector<long long> cap;
    vector<int> dep, it;
    Dinic(int n) : n(n), head(n, -1), dep(n), it(n) {}
    void add_edge(int u, int v, long long c) {
        to.push_back(v); cap.push_back(c); nxt.push_back(head[u]); head[u] = (int)to.size() - 1;
        to.push_back(u); cap.push_back(0); nxt.push_back(head[v]); head[v] = (int)to.size() - 1;
    }
    bool bfs() {
        fill(dep.begin(), dep.end(), -1);
        queue<int> q; q.push(s); dep[s] = 0;
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int e = head[u]; e != -1; e = nxt[e]) {
                int v = to[e];
                if (cap[e] > 0 && dep[v] == -1) { dep[v] = dep[u] + 1; q.push(v); }
            }
        }
        return dep[t] != -1;
    }
    long long dfs(int u, long long f) {
        if (u == t) return f;
        long long pushed = 0;
        for (int &e = it[u]; e != -1; e = nxt[e]) {
            int v = to[e];
            if (cap[e] > 0 && dep[v] == dep[u] + 1) {
                long long d = dfs(v, min(f - pushed, cap[e]));
                if (d > 0) {
                    cap[e] -= d; cap[e ^ 1] += d; pushed += d;
                    if (pushed == f) return pushed;
                }
            }
        }
        return pushed;
    }
    long long max_flow(int s_, int t_) {
        s = s_; t = t_;
        long long flow = 0;
        while (bfs()) {
            for (int i = 0; i < n; i++) it[i] = head[i];
            flow += dfs(s, LLONG_MAX);
        }
        return flow;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    // Node weight of each project (closure vertex):
    //   station i -> weight -p_i (a cost)
    //   user group j -> weight +c_j (a revenue), requires its two stations.
    vector<long long> cost(n);          // build cost of each station
    for (int i = 0; i < n; i++) cin >> cost[i];

    vector<int> ga(m), gb(m);
    vector<long long> gc(m);
    long long sumAbs = 0;
    for (int i = 0; i < n; i++) sumAbs += cost[i];
    for (int j = 0; j < m; j++) {
        cin >> ga[j] >> gb[j] >> gc[j];
        ga[j]--; gb[j]--;                // to 0-indexed stations
        sumAbs += gc[j];
    }

    // Vertices: 0..n-1 stations, n..n+m-1 user groups, then s, t.
    int S = n + m, T = n + m + 1;
    Dinic g(n + m + 2);
    long long INF = sumAbs + 1;          // strictly larger than any finite cut

    long long posSum = 0;                // sum of all positive node weights = sum of revenues
    // negative-weight stations: edge station -> t with cap = cost
    for (int i = 0; i < n; i++)
        if (cost[i] > 0) g.add_edge(i, T, cost[i]);
    // positive-weight user groups: edge s -> group with cap = revenue, plus prerequisite edges
    for (int j = 0; j < m; j++) {
        int gv = n + j;
        if (gc[j] > 0) { g.add_edge(S, gv, gc[j]); posSum += gc[j]; }
        g.add_edge(gv, ga[j], INF);      // group requires station a_j
        g.add_edge(gv, gb[j], INF);      // group requires station b_j
    }

    long long cut = g.max_flow(S, T);    // max-flow value = min-cut capacity
    cout << (posSum - cut) << "\n";
    return 0;
}
```

## Why max-flow = min-cut gives the optimum

By the max-flow–min-cut theorem the value of a maximum $s$–$t$ flow equals the capacity of a
minimum $s$–$t$ cut. The minimum cut is simple (Lemma 1), hence corresponds bijectively to a
feasible selection (Lemma 2), and selection profit $= P - c[S,T]$ with $P$ constant. So
minimizing the cut maximizes the profit, and a single max-flow computation yields both the
optimal value ($P - \text{flow}$) and — via residual reachability from $s$ — the set of
projects to undertake. A prerequisite edge has capacity larger than the whole possible finite
flow, so any such edge leaving a residual-reachable vertex still has residual capacity; the
recovered set is always a valid closure (every selected project's prerequisites are selected
too).

## A concrete instance: relay-station selection (the program's I/O)

$n$ candidate relay stations; building station $i$ costs $p_i$. There are $m$ user groups;
group $j$ uses stations $a_j,b_j$ and, if both are built, yields revenue $c_j$. Net profit =
revenue − cost. Model each user group $j$ as a vertex of weight $+c_j$ with two prerequisite
edges $j \to a_j$ and $j \to b_j$; model each station $i$ as a vertex of weight $-p_i$. The
maximum-weight closure of this graph is the maximum net profit — selecting a user group forces
building both of its stations, exactly the prerequisite structure.

This is exactly what the program above reads and solves. Input: the first line holds `n m`;
the second line the $n$ station costs $p_1\ldots p_n$; each of the next $m$ lines holds
`a_j b_j c_j` (stations 1-indexed). Output: a single integer, the maximum net profit. For
example, `2 1 / 5 4 / 1 2 10` (two stations costing $5,4$; one group using both for revenue
$10$) prints `1`, since $10-5-4=1>0$ so both stations are worth building; lowering the revenue
to $8$ makes the group unprofitable and the answer becomes `0`.
