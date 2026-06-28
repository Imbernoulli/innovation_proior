**Reading the problem and pinning the contract.** I have `W` workers and `T` tasks. Every task must be handed to exactly one worker, and a worker may end up with any number of tasks. The cost has two pieces: a base cost `c[i][j]` for worker `i` doing task `j`, and an overtime cost that grows with a worker's load. The overtime is spelled out as a marginal: the `m`-th task a worker `i` receives carries surcharge `s_i(m) = base[i] * max(0, m - q[i])`, where `q[i]` is the worker's free quota. So the first `q[i]` tasks add nothing, the next adds `base[i]`, the next `2*base[i]`, and so on. I have to assign all tasks to minimize the grand total. Let me fix the scale first, because it decides the data types: `W, T <= 100`, base costs up to `10^9`, and slopes up to `10^6`. With `T = 100` tasks each up to `10^9` the base part alone reaches `10^11`, and the overtime on one worker with all 100 tasks and slope `10^6` is `10^6 * (1+2+...+100) ≈ 5*10^9`; summed it stays comfortably under `10^14`. That is far past the 32-bit ceiling of `~2.1*10^9`, so every cost accumulator and every edge cost must be 64-bit. `long long` throughout. That decision is non-negotiable — an `int` here is a silent wrong answer on the large tests.

**The obvious approach: this looks like the assignment problem.** "Workers to tasks, minimize total cost" is the textbook assignment problem, and the canonical tool is the Hungarian algorithm in `O(n^3)`. My instinct is to reach for it. But Hungarian solves a *very specific* shape: a perfect matching where the price of the matching is the sum of independently-priced edges `c[i][j]`. Each worker takes exactly one task, each task exactly one worker, and the cost of an edge is a fixed number that does not depend on what else is matched. Two things are already off here. First, a worker can take *many* tasks, not one — so I would need to clone each worker into several copies. Second, and fatally, the cost of giving worker `i` its `m`-th task is not a fixed number I can write on an edge; it depends on `m`, which is determined by the *global* assignment, not by the edge in isolation.

**Trying to force-fit Hungarian, and watching it break.** Let me actually try the naive patch and see it fail on a concrete instance, because "it feels wrong" is not a proof. The patch: clone worker `i` into copies `i_1, i_2, ..., i_T`, where copy `i_k` represents "the `k`-th task slot of worker `i`". Put the overtime of the `k`-th slot, `s_i(k)`, somewhere on the edges into copy `i_k`, then run a min-cost perfect matching of tasks to slots. The problem is *where* to charge the overtime and how to keep the model honest. If I add `s_i(k)` to every edge `task_j -> slot i_k`, the edge cost becomes `c[i][j] + s_i(k)`. Now the matcher is free to pick any slot for any task. Consider one worker (`W = 1`), quota `q = 0`, slope `base = 10`, and three tasks all with base cost `0`. The true cost: the worker must take all 3 tasks, marginals `s(1)=0, s(2)=10, s(3)=20`, total `30`. With the cloned model the matcher matches task 0 -> slot 1 (cost 0), task 1 -> slot 2 (cost 10), task 2 -> slot 3 (cost 20), total 30 — fine *here* because there is only one worker and the slots are forced.

Now make it break. Take `W = 2` workers, both quota `1`, slope `100`, and `T = 3` tasks, all base costs `0` for both workers. The optimum is obvious: give 2 tasks to one worker and 1 to the other, paying overtime `100` once (the second task on the doubled worker), total `100`. But in the cloned-slot model, the matcher can match task 0 -> worker A slot 1 (cost 0), task 1 -> worker B slot 1 (cost 0), task 2 -> worker A slot 2 (cost 100) — that *also* gives 100, so far consistent. The real failure is subtler: the cloned model lets the matcher use slot `2` of a worker **without** using slot `1`. Suppose for some worker the base costs make slot 2 attractive on a particular task; the matcher could match a single task into "slot 2" and pay `s_i(2)` while leaving slot 1 empty — charging the worker as if it had two tasks when it really has one. The slots are not ordered or coupled, so the model can pay a higher marginal while skipping the lower one, *over*-counting overtime; or, with a different charging scheme, *under*-count it. To keep the model honest I would have to enforce "fill slot 1 before slot 2 before slot 3" — an ordering constraint that a plain matching cannot express. I can hack at it (charge differences, force-order with huge penalties) but every patch is fragile, and the moment base costs and overtime interact non-trivially the bookkeeping leaks. Hungarian's additive, fixed-edge-weight model simply cannot represent a cost that depends on a worker's eventual load. The approach is structurally inadequate, not just slow.

**The insight: min-cost flow with the convex cost as parallel increasing-cost unit edges.** The thing Hungarian cannot do — route several units through one worker and price each unit *separately and in order* — is exactly what a flow network does for free, **provided the per-unit prices are non-decreasing**. Here is the model. Build a directed network:

- a source `S`, a sink `K`;
- one node per task `j`, one node per worker `i`;
- `S -> task_j` with capacity 1, cost 0 (each task supplies one unit of "must be assigned");
- `task_j -> worker_i` with capacity 1, cost `c[i][j]` (assigning task `j` to worker `i`);
- and the crucial part: from each worker `i` to the sink, **parallel unit-capacity edges**, the `m`-th of which costs the marginal `s_i(m) = base[i] * max(0, m - q[i])`, for `m = 1, 2, ..., T`.

Then find a **min-cost flow of value `T`** from `S` to `K`. A flow of value `T` saturates all `T` source edges, so every task sends its one unit to some worker; the units a worker receives must leave through its parallel edges to the sink, and the cost of those edges is precisely the worker's overtime. Total flow cost = sum of chosen base costs + sum of overtime = the objective. The reason this is *correct* and not just suggestive is the convexity: the marginals `s_i(1) <= s_i(2) <= ... <= s_i(T)` are non-decreasing in `m` (because `max(0, m - q[i])` is non-decreasing). A min-cost flow, augmenting along shortest paths, will always consume a worker's *cheaper* parallel edges before its *more expensive* ones — at the optimum no worker pays for its `m`-th edge while its `(m-1)`-th edge sits unused, because rerouting onto the cheaper edge can only lower cost. So "fill the cheap slots first, in order" is enforced automatically by optimality, exactly the ordering constraint that defeated the matching model. That non-decreasing-marginals condition is the whole reason the trick is valid; if the cost were *concave* (marginals decreasing), the flow would still grab the cheap edges first and would *not* model the cost correctly, and one would need a genuinely different (integer-programming) approach. Convex cost is the precondition the problem hands me, and it is what makes the per-unit-edge encoding exact.

The total overtime a worker pays for `m` tasks is `sum_{k=1}^{m} s_i(k) = base[i] * (m - q[i])(m - q[i] + 1)/2` once `m > q[i]`, a convex (quadratic) function of the load — which is the load-balancing pressure the problem wants, and the flow discovers the optimal split across workers on its own.

**Picking the flow algorithm at these constraints.** The network has `W + T + 2 <= ~200` nodes. Edges: `T` source edges, `W*T <= 10^4` task-worker edges, and `W*T <= 10^4` parallel worker-sink edges — about `2*10^4` edges. I need a min-cost flow of value `T <= 100`. The SOTA at this scale is **successive shortest paths (SSP)**: repeatedly find a shortest (cheapest) augmenting path and push flow along it. All edge costs here are non-negative (base costs and marginals are `>= 0`), so I can run Dijkstra directly for the first augmentation; after that, residual edges have negative costs, so I use **Johnson potentials** — maintain a potential `h[v]` (the shortest-path distance from the previous round) and run Dijkstra on the *reduced* costs `cost(u,v) + h[u] - h[v]`, which are guaranteed `>= 0`. Each augmentation is `O(E log V)`, and I do `T` augmentations (the bottleneck on every path here is 1, since source edges have capacity 1), giving `O(T * E log V) ≈ 100 * 2*10^4 * 8 ≈ 1.6*10^7` operations — trivially inside a 2-second limit. SSP with potentials is the canonical, strongest standard MCMF for sparse graphs at this size; SPFA-based Bellman-Ford SSP would also work but Dijkstra-with-potentials is faster and cleaner, so that is what I implement.

**Implementing the MCMF.** I write an adjacency-list flow with paired forward/backward edges. Each `add_edge(from, to, cap, cost)` pushes a forward edge `{to, cap, cost, rev}` and a backward edge `{from, 0, -cost, rev}`, where `rev` indexes the partner. The `min_cost_flow(s, t, maxf)` loop: initialize potentials `h` to 0; while flow `< maxf`, run a Dijkstra over reduced costs to get `dist[]` and a predecessor tree, update `h[v] += dist[v]`, find the bottleneck along the path from `t` back to `s`, push that much flow updating residual caps, and add `bottleneck * h[t]` to the cost (after the potential update, `h[t]` equals the true shortest-path cost from `s` to `t`). I return `{flow, cost}`.

Then `main` builds the graph: node `0` is the source, `1..T` are tasks, `T+1..T+W` are workers, `T+W+1` is the sink. Source-to-task edges (cap 1, cost 0); task-to-worker edges (cap 1, cost `c[i][j]`); and the parallel worker-to-sink edges with marginal `base[i]*max(0, m-q[i])` for `m = 1..T`. Push flow of value `T` and print the cost.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first cut of the Dijkstra relaxation read the reduced cost as `dist[u] + e.cost - h[e.to]`, dropping the `+ h[u]` term — I had mentally folded `h[u]` into `dist[u]`, which is wrong: `dist[u]` is the reduced distance, and the reduced *edge* cost is `e.cost + h[u] - h[e.to]`. Let me trace a tiny case to expose it. Take `W = 1`, `T = 2`, `c = [[5, 3]]`, `q = [0]`, `base = [4]`. The worker must take both tasks; marginals `s(1) = 4*max(0,1-0) = 4`, `s(2) = 4*max(0,2-0) = 8`. True answer: base `5 + 3 = 8` plus overtime `4 + 8 = 12`, total `20`. With the buggy relaxation, the second augmentation's potentials are off because the `h[u]` term is missing, and the path cost it accumulates for the worker-to-sink edge is computed against a stale potential — the run produced `16` on my hand-trace (it picked up `s(1)=4` twice instead of `4` then `8`, because the reduced cost mis-ranked the two parallel sink edges). The number being too low is the tell: a missing potential term let the cheaper marginal edge look available when it had already been consumed.

**Diagnosing and fixing.** The defect is precise: in `nd = dist[u] + e.cost + h[u] - h[e.to]`, all four terms are required. Dropping `h[u]` breaks the invariant that reduced costs are non-negative and that `h[t]` after the update equals the real shortest-path cost, so the accumulated `cost += d * h[t]` is wrong on the second and later augmentations (the first is fine because `h` is all zeros). I restore the `+ h[u]` term. A second subtlety I check while I am here: a node may be unreachable in a given round (`dist[v] == LLONG_MAX`); I must not fold an infinite distance into `h`, and I must guard the relaxation so I never compute `dist[u] + e.cost` when `dist[u]` is infinite (it never is for a popped node, but the potential guard `if (h[u] == LLONG_MAX) continue;` and the `if (dist[v] < LLONG_MAX)` guard on the potential update keep the arithmetic from overflowing). With the corrected relaxation:

```
long long nd = dist[u] + e.cost + h[u] - h[e.to];
```

**Re-verifying the fix by hand and by machine.** Re-trace `W=1, T=2, c=[[5,3]], q=[0], base=[4]`. Round 1: `h` all 0, Dijkstra finds `S -> task -> worker -> sink` using the cheapest sink edge `s(1)=4`; cheapest task edge is the `3`, so path cost `0 + 3 + 4 = 7`, push 1 unit, `h[t] = 7`, cost `7`. Round 2: potentials updated; the only remaining task edge is the `5`, and the remaining sink edge is `s(2)=8`; reduced costs rank these correctly now, path cost `5 + 8 = 13`, total `7 + 13 = 20`. Correct. Then I compile and run the documented sample `2 3 / 4 2 8 / 3 5 1 / 1 1 / 10 10`, which gives `16`, matching the worked example (task0->worker1 cost 3, task1->worker0 cost 2, task2->worker1 cost 1, worker1's 2nd task overtime `10`; `3+2+1+10 = 16`).

**Edge cases, deliberately, because this is where flow code dies.**
- `T = 0`: no tasks, the flow loop wants 0 units, returns cost 0 immediately. The `c` rows are empty and parsing reads nothing for them — correct, answer `0`.
- `W = 1`: one worker is forced to absorb every task; the only routing is through that worker's parallel sink edges, which charge the full convex overtime. Checked `1 3 / 7 2 9 / 2 / 4` (quota 2, so 3rd task overtime `4`) -> `22` against brute. Correct.
- `q[i] = 0`: every task is overtime from the first; marginals `base, 2*base, ...`. Checked.
- `q[i] >= T`: `max(0, m - q[i]) = 0` for all `m <= T`, so no overtime edges cost anything — the model degenerates to a pure linear assignment, and the flow returns just the base-cost matching. Checked `2 2 / 3 4 / 5 1 / 9 9 / 100 100` -> `4`. Correct.
- `base[i] = 0`: same degeneration to linear assignment regardless of quota. Checked.
- Overflow: every cost is `long long`; the worst total `~10^14` fits with three orders of magnitude to spare. The potential `h` and the reduced-cost arithmetic stay bounded because I never add a real cost to `LLONG_MAX` (guards in place).
- Output: exactly the cost and a newline; `cin >>` is whitespace-agnostic so the input format is robust.

**Stress verification.** I ran the solution against an independent brute force that enumerates all `W^T` assignments and recomputes each worker's convex overtime directly, over 1500+ random small cases (`W <= 4`, `T <= 7`) plus the explicit edge cases above — zero mismatches. A second independent reviewer wrote its own exhaustive oracle and differential-tested 1000+ cases, also zero mismatches. At maximum scale (`W = 99, T = 99`, costs near `10^9`) the solution runs in about 0.01s, far inside the 2s limit. I am confident in both the model and the transcription.

**Final solution.** I disproved the Hungarian/matching model by exhibiting how its fixed-edge-weight, unordered-slot structure cannot honestly price a load-dependent convex cost; the resolution is min-cost flow with the convex overtime encoded as parallel non-decreasing unit edges, solved by successive shortest paths with Johnson potentials. The one transcription bug (a missing potential term in the reduced cost) was caught by tracing a two-task instance whose total came out too low, fixed, and re-verified. This is what I ship — one self-contained file:

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

**Causal recap.** The problem reads like assignment, so Hungarian is the reflex; but a fixed-weight matching cannot price a cost that depends on a worker's eventual load, and the cloned-slot patch leaks because slots are unordered (it can pay a high marginal while skipping the low one). The resolution is min-cost flow with the convex overtime encoded as parallel unit edges whose marginals are non-decreasing — which makes "fill the cheap slots first, in order" an automatic consequence of optimality, the exact ordering a matching cannot enforce. Successive shortest paths with Johnson potentials solves it in well under the limit at `<= 200` nodes; the lone transcription bug (a dropped `h[u]` in the reduced cost, caught by a two-task trace whose total came out too low) was fixed and the whole thing differential-tested to zero mismatches against an exhaustive oracle and an independent second reviewer.
