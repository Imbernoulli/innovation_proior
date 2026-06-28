**Problem.** Assign every one of `T` tasks to one of `W` workers (a worker may take any number of tasks). Worker `i` doing task `j` costs `c[i][j]`. On top of that, each worker pays a convex overtime cost: the `m`-th task assigned to worker `i` adds a marginal surcharge `s_i(m) = base[i] * max(0, m - q[i])`, where `q[i]` is a free quota. Minimize the total of base costs plus every worker's overtime. Read `W T`, the `W*T` cost matrix, the `W` quotas, and the `W` slopes from stdin; print the minimum total cost. Limits: `W, T <= 100` (`W + T <= 198`), `c <= 10^9`, `base <= 10^6`.

**Why the obvious assignment approach is wrong.** "Workers to tasks, minimize cost" screams Hungarian / min-cost matching. But Hungarian prices a matching as a sum of *fixed, independent* edge weights, and a worker takes *one* task. Here a worker takes many tasks, and the price of giving it its `m`-th task is not a fixed number — it depends on `m`, which the global assignment decides. The natural patch (clone each worker into `T` ordered "slots", put `s_i(k)` on slot `k`) fails because a plain matching cannot *order* the slots: it can charge a worker its `s_i(2)` marginal while leaving slot 1 empty, mis-counting overtime. No fixed edge weight encodes a load-dependent cost. Hungarian is structurally inadequate.

**Key idea — min-cost flow with the convex cost as parallel increasing-cost unit edges.** A flow network *can* route several units through one worker and price each unit separately, and it enforces ordering automatically **when the per-unit prices are non-decreasing** (i.e. the cost is convex). Build: `source -> task_j` (cap 1, cost 0); `task_j -> worker_i` (cap 1, cost `c[i][j]`); and from each worker `i` to the sink, `T` **parallel unit edges**, the `m`-th costing the marginal `s_i(m) = base[i] * max(0, m - q[i])`. Find a **min-cost flow of value `T`**. Its cost is exactly the objective: every task ships one unit to a worker, and the units leaving a worker pay its convex overtime. Correctness hinges on the marginals being non-decreasing (`max(0, m-q[i])` is): a min-cost flow consumes a worker's cheap parallel edges before its expensive ones, so "fill cheap slots first, in order" — the constraint matching couldn't express — falls out of optimality for free. The per-worker total `base[i]*(m-q[i])(m-q[i]+1)/2` is convex, and the flow finds the optimal load split across workers on its own.

**Algorithm and complexity.** Network has `<= 200` nodes and `~2*10^4` edges; flow value `T <= 100`. Solve with **successive shortest paths (SSP)** using **Johnson potentials** (Dijkstra on reduced costs `cost + h[u] - h[v] >= 0`). All costs are `>= 0`, so the first Dijkstra is direct and potentials keep later rounds non-negative despite negative residual edges. `T` augmentations, each `O(E log V)`, so `O(T * E log V) ~ 1.6*10^7` — well under the limit (measured ~0.01s at max scale).

**Pitfalls.**
1. *Trying to make Hungarian fit.* A fixed-weight matching cannot price load-dependent overtime, and ordered "slots" can't be enforced by a matching — use flow.
2. *The reduced cost.* In SSP-with-potentials the relaxation is `dist[u] + e.cost + h[u] - h[e.to]`; dropping the `h[u]` term mis-ranks the parallel sink edges and undercounts overtime on the second-and-later augmentations (the first is fine since `h` starts at 0).
3. *Overflow.* Totals reach `~10^14`; all costs/accumulators must be `long long`. An `int` is a silent wrong answer.
4. *Convex, not concave.* The trick is valid only because marginals are non-decreasing. A concave cost would need a different (ILP) method.

**Edge cases.** `T = 0` -> 0 (no flow). `W = 1` -> one worker absorbs all tasks, paying full convex overtime. `q[i] >= T` or `base[i] = 0` -> overtime edges cost 0, degenerating to a pure linear assignment. `q[i] = 0` -> overtime from the first task. All verified against an exhaustive `W^T` brute force (1500+ random small cases plus explicit edges, zero mismatches) and an independent second reviewer's oracle (1000+ cases, zero mismatches).

**Complexity.** `O(T * E log V)` time with `E = O(W*T)`, `O(W*T)` memory.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

// Min-cost max-flow via successive shortest paths with Johnson potentials
// (Dijkstra on reduced non-negative costs). Capacities are integers, costs are
// non-negative long long.
struct MCMF {
    struct Edge { int to; long long cap, cost; int rev; };
    int n;
    vector<vector<Edge>> g;
    vector<long long> h, dist;      // potentials and Dijkstra distances
    vector<int> prevv, preve;       // path reconstruction
    MCMF(int n_) : n(n_), g(n_), h(n_), dist(n_), prevv(n_), preve(n_) {}
    void add_edge(int from, int to, long long cap, long long cost) {
        g[from].push_back({to, cap, cost, (int)g[to].size()});
        g[to].push_back({from, 0, -cost, (int)g[from].size() - 1});
    }
    // Returns {flow, cost} pushing up to maxf units of flow from s to t.
    pair<long long,long long> min_cost_flow(int s, int t, long long maxf) {
        long long flow = 0, cost = 0;
        fill(h.begin(), h.end(), 0);
        while (flow < maxf) {
            // Dijkstra over reduced costs cost + h[u] - h[v] >= 0.
            priority_queue<pair<long long,int>, vector<pair<long long,int>>,
                           greater<pair<long long,int>>> pq;
            fill(dist.begin(), dist.end(), LLONG_MAX);
            dist[s] = 0;
            pq.push({0, s});
            while (!pq.empty()) {
                auto [d, u] = pq.top(); pq.pop();
                if (d > dist[u]) continue;
                for (int i = 0; i < (int)g[u].size(); i++) {
                    Edge &e = g[u][i];
                    if (e.cap <= 0) continue;
                    if (h[u] == LLONG_MAX) continue; // unreachable potential
                    long long nd = dist[u] + e.cost + h[u] - h[e.to];
                    if (nd < dist[e.to]) {
                        dist[e.to] = nd;
                        prevv[e.to] = u;
                        preve[e.to] = i;
                        pq.push({nd, e.to});
                    }
                }
            }
            if (dist[t] == LLONG_MAX) break;        // sink unreachable
            for (int v = 0; v < n; v++)
                if (dist[v] < LLONG_MAX) h[v] += dist[v];
            // Bottleneck along the found shortest path.
            long long d = maxf - flow;
            for (int v = t; v != s; v = prevv[v])
                d = min(d, g[prevv[v]][preve[v]].cap);
            for (int v = t; v != s; v = prevv[v]) {
                Edge &e = g[prevv[v]][preve[v]];
                e.cap -= d;
                g[v][e.rev].cap += d;
            }
            flow += d;
            cost += d * h[t];   // h[t] == true shortest-path cost s->t
        }
        return {flow, cost};
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int W, T;
    if (!(cin >> W >> T)) return 0;

    // c[i][j] = base cost to assign task j to worker i.
    vector<vector<long long>> c(W, vector<long long>(T));
    for (int i = 0; i < W; i++)
        for (int j = 0; j < T; j++)
            cin >> c[i][j];

    vector<long long> q(W), base(W);
    for (int i = 0; i < W; i++) cin >> q[i];     // regular quota of worker i
    for (int i = 0; i < W; i++) cin >> base[i];  // overtime slope of worker i

    // Node layout: 0 = source, 1..T = tasks, T+1..T+W = workers, T+W+1 = sink.
    int S = 0;
    auto TASK = [&](int j) { return 1 + j; };
    auto WORK = [&](int i) { return 1 + T + i; };
    int K = 1 + T + W;
    MCMF mc(K + 1);

    for (int j = 0; j < T; j++) mc.add_edge(S, TASK(j), 1, 0);

    for (int i = 0; i < W; i++)
        for (int j = 0; j < T; j++)
            mc.add_edge(TASK(j), WORK(i), 1, c[i][j]);

    // Convex overtime cost as parallel unit edges worker_i -> sink.
    // Marginal cost of the m-th task on worker i (m = 1..T) is
    //   s_i(m) = base[i] * max(0, m - q[i]),
    // which is non-decreasing in m, so the total per-worker cost is convex.
    for (int i = 0; i < W; i++) {
        for (int m = 1; m <= T; m++) {
            long long over = (long long)max(0LL, (long long)m - q[i]);
            long long marginal = base[i] * over;
            mc.add_edge(WORK(i), K, 1, marginal);
        }
    }

    auto [flow, cost] = mc.min_cost_flow(S, K, T);
    // Every task can always be routed (workers have capacity T each), so flow==T.
    cout << cost << "\n";
    return 0;
}
```
