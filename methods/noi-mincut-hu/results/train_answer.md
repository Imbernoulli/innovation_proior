We have $n$ projects, each with an integer profit $p_i$ that may be negative — a "project" that requires buying a machine is really a cost — and some projects depend on others: a project may be chosen only if every one of its prerequisites is chosen too. Prerequisites chain, and one project can be a prerequisite of many. We want the feasible subset that maximizes $\sum_{i \in \text{chosen}} p_i$. The obvious greedy — keep every $p_i > 0$, drop every $p_i < 0$ — collapses the moment prerequisites enter. A project earning $+100$ that depends on a $-30$ chain that depends on a $-40$ chain is worth $+30$ as a bundle and should be taken whole; nudge the middle cost to $-80$ and the same bundle is worth $-20$, so all three should be dropped despite the $+100$ looking irresistible. The coupling runs the other way too: a $-50$ project that unlocks three separate $+30$ projects sitting on top of it is worth keeping even though in isolation it is a pure loss. No project can be decided locally, because its value depends on the whole chain hanging off it and on who sits above it. Brute force over all $2^n$ subsets is correct but hopeless. What is needed is structure that captures the two-directional coupling exactly.

The structure becomes visible once a feasible subset is named for what it is. Draw each prerequisite as a directed edge $u \to v$ meaning "$u$ requires $v$." A feasible subset is then one with no edge leaving it: every out-neighbor of a chosen vertex is itself chosen. That is precisely a *closure* of the directed graph (the constraint is the "implies" relation $u$ chosen $\Rightarrow v$ chosen, and it handles cycles fine — mutually dependent projects are just taken whole). So the project problem is exactly the maximum-weight closure problem: among all closures of a vertex-weighted directed graph, find the one of largest total weight. I propose to solve it by reduction to a minimum $s$–$t$ cut, computed by a single max-flow — the construction I will call the **min-cut maximum-weight closure** reduction (Hu's project-selection reduction). The reason a cut is the right object is that a closure is a partition of the vertices into chosen $V_1$ and unchosen $V_2$, and the closure condition is a statement purely about which edges cross between the sides: no edge may run from $V_1$ to $V_2$. Edges inside either side, and even edges from $V_2$ to $V_1$, are all permitted. "Partition into two sides and forbid edges crossing in one direction" is exactly the shape of an $s$–$t$ cut.

Build a network by adding a source $s$ and a sink $t$, intending the source side $S$ of the cut to be $V_1 \cup \{s\}$ (chosen projects plus source) and $T = V_2 \cup \{t\}$. Two pressures must be priced. A positive vertex $v$ ($w_v = p_v > 0$) I want on the $S$ side; if it ends up on $T$, I forfeit its profit — so add an edge $s \to v$ with capacity $w_v$, which crosses from $S$ to $T$ exactly when $v$ is not selected and pays exactly the forgone profit. A negative vertex $v$ ($w_v < 0$) I want on the $T$ side; if I am forced to keep it, I pay $|w_v|$ — so add an edge $v \to t$ with capacity $-w_v$, which crosses $S \to T$ exactly when $v$ is selected and pays exactly the incurred cost. The closure constraint itself must be *enforced*, not merely priced: a violated prerequisite $u \to v$ means $u \in S$ but $v \in T$, which is again an $S \to T$ crossing. So keep every original prerequisite edge $u \to v$ in the network with capacity `INF` chosen larger than any finite cut. Concretely, the source-only cut (just $s$ on the source side) costs $\sum_{v:w_v>0} w_v \le \sum_v |w_v|$, so taking $\texttt{INF} = \sum_v |w_v| + 1$ makes any cut that severs even one prerequisite edge strictly more expensive than this finite cut — the obvious alternative of a merely "large" constant risks being beaten, whereas this bound provably rules illegal cuts out of the minimum.

That bound is the first load-bearing step. Call a cut *simple* if every edge it cuts is incident to $s$ or $t$ (it cuts no prerequisite edge). The network's edges come in exactly two flavors — finite ones touching $s$ or $t$, and the huge prerequisite edges touching neither — and since a finite cut of capacity $\le \sum_v |w_v|$ exists while any prerequisite-cutting cut costs `INF`, the minimum cut is simple. The second step is a genuine bijection between simple cuts and closures via $S = V_1 \cup \{s\}$: if $V_1$ is a closure, no original edge leaves it, so the cut severs no original edge and is simple; conversely, in a simple cut no original edge $u \to v$ with $u \in S$ is cut, forcing $v \in S$, so $V_1 = S \setminus \{s\}$ is a closure. The third step is the arithmetic that makes the cut track profit. For a simple cut the only cut $s$-edges go to positive vertices left on $T$ (call them $V_2^+$) and the only cut $t$-edges come from negative vertices pulled onto $S$ ($V_1^-$), so

$$c[S,T] = \sum_{v \in V_2^+} w_v + \sum_{v \in V_1^-} (-w_v)$$

— it charges the profit of every positive project not selected plus the cost of every negative project forced in, exactly the two ways a selection can be suboptimal. The closure's own weight is $w(V_1) = \sum_{v \in V_1^+} w_v - \sum_{v \in V_1^-}(-w_v)$, and adding the two expressions the $V_1^-$ terms cancel while the positive terms combine over *all* positive vertices:

$$w(V_1) + c[S,T] = \sum_{v \in V^+} w_v \quad (\text{constant}), \qquad\Longrightarrow\qquad w(V_1) = \sum_{v \in V^+} w_v - c[S,T].$$

Since $\sum_{v \in V^+} w_v$ does not depend on the selection, maximizing the profit $w(V_1)$ is identical to minimizing the simple-cut capacity, and because the global minimum cut is already simple, this equals minimizing over all cuts. Invoking max-flow–min-cut, the minimum cut value equals the maximum $s$–$t$ flow, so

$$\text{max-weight closure} = \Big(\sum_{v : p_v > 0} p_v\Big) - (\text{min cut}) = \Big(\sum_{v : p_v > 0} p_v\Big) - (\text{max flow}).$$

The picture is: take all the positive profit for free, then pay the minimum cut to back out of the impossible parts, where the cut measures precisely the unavoidable forgone profits and forced costs that prerequisites impose. To recover *which* projects to take I never search over cuts — I run one max-flow, then BFS from $s$ over residual-positive edges; the reached set is the source side $S$ of a minimum cut, and the chosen projects are $S \setminus \{s\}$. The `INF` edges make this recovered set automatically a legal closure: if $u$ is residual-reachable and $u \to v$ has capacity `INF`, then since the whole flow is at most $\sum_v |w_v| < \texttt{INF}$ that edge still has residual capacity, so $v$ is reachable too — every selected project's prerequisites are selected. A quick check on the framework instance with weights $5, -6, 7, 0, -3$ and edges $1\to2, 1\to4, 3\to4, 2\to5, 4\to5$: positive total $12$, best closure $\{3,4,5\}$ of weight $7+0-3=4$, and $S=\{s,3,4,5\}$ cuts $s\to1$ (cap $5$) and $5\to t$ (cap $3$) for a cut of $8$, giving $12-8=4$. The arithmetic closes. For the flow I use Dinic's algorithm — BFS labels each vertex with its distance from $s$ to build a level graph, then a DFS pushes a blocking flow stepping only from level $k$ to $k+1$ with a current-arc pointer so exhausted arcs are skipped, repeating until $t$ is unreachable; edges live in a flat array with forward edge $i$ paired to reverse $i \oplus 1$, so cancelling flow on the reverse arc is a single index flip. The deliverable is a single self-contained C++17 program that instantiates this reduction on the relay-station selection instance: $n$ stations (station $i$ costs $p_i$, a negative-weight project) and $m$ user groups (group $j$ uses stations $a_j,b_j$ and yields revenue $c_j$, a positive-weight project with two prerequisite edges to its stations). It reads `n m`, then the $n$ station costs, then $m$ lines `a_j b_j c_j` (stations 1-indexed) from stdin, and prints the maximum net profit (total revenue minus total build cost) to stdout. All capacities and the running total are `long long` so the positive-sum and flow never overflow.

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
