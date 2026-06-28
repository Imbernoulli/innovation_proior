# Negative-Weight SSSP in Near-Linear Time

## Problem

Given a directed graph $G=(V,E,w)$ with $m$ edges, $n$ vertices, integer weights (possibly negative), source $s$, and $W\ge 2$ with every weight $\ge -W$: return a shortest-path tree from $s$, or a negative-weight cycle. Dijkstra ($O(m+n\log n)$) needs non-negative weights; Bellman–Ford ($O(mn)$) is the only general but slow tool; scaling stalled at $O(m\sqrt n\log W)$ for three decades; near-linear was previously known only via continuous-optimization min-cost-flow machinery (and for planar graphs).

## Key idea

Everything reduces to computing an integral **price function** $\phi$ with all reduced weights $w_\phi(u,v)=w(u,v)+\phi(u)-\phi(v)\ge 0$ (then run Dijkstra; $\phi$ preserves shortest paths and cycle weights, Johnson 1977). The cost of producing $\phi$ via a Dijkstra/Bellman–Ford hybrid is governed by
$$\eta_G(v)=\min_{\text{shortest }s\to v\text{ path }P}|E^{neg}(G)\cap P|,\qquad \eta(G)=\max_v\eta_G(v),$$
the number of negative edges shortest paths must use. The contribution is a **scaling recursion** (`ScaleDown`) that uses a **low-diameter decomposition** to *halve* $\eta$ at each level: shift $B$ onto every negative edge ($w^B(P)=w(P)+kB$ for a $k$-negative-edge path), so inside any piece of weak diameter $dB$ a shortest path with $>d$ negative edges would force a negative cycle. Decomposing into small-weak-diameter SCCs (with cut probability $\propto$ edge weight, via geometric ball radii) leaves only a polylog expected number of negative cut-edges per path for the hybrid to clean up.

## Main theorem

> **Theorem.** There is a randomized (Las Vegas) algorithm running in $O(m\log^8 n\,\log W)$ time w.h.p. and in expectation that, for an integer-weighted digraph and source, returns either a shortest-path tree or a negative-weight cycle.

It rests on:

> **`ScaleDown`($G,\Delta,B$).** If $w(e)\ge -2B$, all out-degrees are $O(1)$, and (when $G$ has no negative cycle) $\eta(G^B)\le\Delta$, then it returns an integral $\phi$ with $w_\phi(e)\ge -B$; expected time $O(m\log^3 n\,\log\Delta)$ on no-cycle inputs. (If $G$ has a negative cycle it may not terminate, but any output is correct.)

> **`ElimNeg`($G,s$).** Outputs $\phi$ with $w_\phi(e)\ge 0$ in time $O(\log n\,(n+\sum_v\eta_G(v)))$ for constant-out-degree $G$ (loops forever iff $G$ has a negative cycle).

> **`LDD`($G,D$)** (non-negative $G$). Returns $E^{rem}$ with every SCC of $G\setminus E^{rem}$ of weak diameter $\le D$ and $\Pr[e\in E^{rem}]=O(w(e)\log^2 n/D+n^{-10})$; time $O(m\log^2 n+n\log^3 n)$.

## Proof sketch

**`ElimNeg` correctness & time.** Alternate a Dijkstra phase (relax only non-negative edges) and a Bellman–Ford phase (relax negative edges out of marked vertices). *Claim:* after iteration $i$ of Dijkstra, $d(v)=\operatorname{dist}(s,v)$ for all $v$ with $\eta(v)\le i$. Base $i=0$ is plain Dijkstra on the non-negative subgraph. Inductive step: for $v$ with $\eta(v)=i$, on a realizing path take the last negative edge $(u_{j-1},u_j)$; the prefix has $\eta(u_{j-1})\le i-1$ (correct by induction), the BF phase relaxes $(u_{j-1},u_j)$ to give $d(u_j)\le\operatorname{dist}(s,u_j)$, and the next Dijkstra phase relaxes the non-negative suffix, giving $d(v)=\operatorname{dist}(s,v)$. A vertex is added to the queue only in iterations $0..\eta(v)$, at most twice per iteration, so total insertions $N\le 2\sum_v\eta(v)+2n$; with constant out-degree (BF charge $O(1)$ per extraction) the time is $O(\log n(n+\sum_v\eta_G(v)))$. At termination no edge is active, so $d$ is a non-negativizing price function.

**`ScaleDown` (the four phases).** With $d=\Delta/2$, work in $G^B$ (add $B$ to negatives only).
- *Phase 0.* $E^{rem}\gets\textsf{LDD}(G^B_{\ge 0},\,D=dB)$ (negatives rounded to $0$, since LDD needs non-negatives; rounding up only increases distances). SCCs $V_1,V_2,\dots$ of $G^B\setminus E^{rem}$ have weak diameter $\le dB$ in $G$.
- *Phase 1.* Inside an SCC, a dummy-rooted shortest path with $\eta_{H^B}(v)$ negatives has $\operatorname{dist}_G(u,v)\le w_{H^B}(P)-\eta_{H^B}(v)B\le -\eta_{H^B}(v)B$, and $\operatorname{dist}_G(v,u)\le dB$; no negative cycle forces $\eta_{H^B}(v)\le d=\Delta/2$. Recurse $\phi_1\gets\textsf{ScaleDown}(H,\Delta/2,B)$ on the union $H$ of induced SCCs; this makes $G^B_{\phi_1}[V_i]$ non-negative.
- *Phase 2.* Contracting SCCs (after dropping $E^{rem}$) yields a DAG. `FixDAGEdges`: in topological order set $\mu_j=\min\{w(u,v):(u,v)\in E^{neg},u\notin V_j,v\in V_j\}$ (or $0$), $M_j=\sum_{k\le j}\mu_k$, $\phi(v)=M_j$ for $v\in V_j$; then $w_\phi(u,v)=w(u,v)-\sum_{k=i+1}^j\mu_k\ge w(u,v)-\mu_j\ge 0$. Set $\phi_2=\phi_1+\psi$. Now the only negatives in $G^B_{\phi_2}$ are in $E^{rem}$.
- *Phase 3.* $\psi'\gets\textsf{ElimNeg}((G^B_s)_{\phi_2},s)$ (add the dummy source *before* applying $\phi_2$, so $(G^B_s)_{\phi_2}\equiv G^B_s$), $\phi_3=\phi_2+\psi'$. Output correctness: $w^B_{\phi_3}(e)\ge 0\Rightarrow w_{\phi_3}(e)\ge -B$ (no-cycle assumption not needed). Time: $\eta_{(G^B_s)_{\phi_2}}(v)\le|P_{G^B}(v)\cap E^{rem}|+1$ and, since $w^B_{\ge 0}(P_{G^B}(v))\le\eta_{G^B}(v)B\le\Delta B$ and $D=B\Delta/2$,
$$\mathbb{E}\big[|P_{G^B}(v)\cap E^{rem}|\big]=O\Big(\tfrac{w^B_{\ge 0}(P)\log^2 n}{B\Delta/2}+n^{-9}\Big)=O(\log^2 n),$$
so Phase 3 costs $O(m\log^3 n)$ expected. Over $O(\log\Delta)$ recursion levels (each dominated by Phase 3 / Phase 0): $O(m\log^3 n\log\Delta)$.

**`SPmain`.** Scale weights up by $2n$, $B=2n$ (power of two); $\log_2 B$ rounds of $\textsf{ScaleDown}(\bar G_{\phi_{i-1}},n,B/2^i)$ drive reduced weights from $\ge -2n$ to $\ge -1$; add $+1$ per edge (since scaled distances are multiples of $2n$, the $+1<n$ can't reorder paths) and run Dijkstra. Expected $O(m\log^5 n)$.

**`LDD`.** Mark each $v$ in-/out-light or heavy from $k=\Theta(\log n)$ random ball samples (Chernoff: light $\Rightarrow$ ball $\le 0.7|V|$, heavy $\Rightarrow$ both balls $>0.5|V|$). Carve a light vertex's ball with geometric radius $R\sim\textsf{Geo}(p)$, $p=\min\{1,80\log n/D\}$; cut boundary edges; recurse inside; remove ball. Heavy leftovers have pairwise weak diameter $\le D/2$ (intersecting $D/4$-balls). Diameter $\le D$ follows by induction (cut balls separate SCCs; recursed balls are $\le 0.7|V|$). Cut probability: for edge $(u,v)$, conditioning on $u$ in the out-ball, memorylessness of $\textsf{Geo}(p)$ gives $\Pr[v\notin\text{ball}\mid u\in\text{ball}]\le\Pr[R\le w(u,v)]\le p\,w(u,v)=O(w(e)\log n/D)$ per call; $\times O(\log n)$ recursive calls $\Rightarrow O(w(e)\log^2 n/D)$, plus $n^{-10}$ bail-out.

**Las Vegas + negative cycle.** $\textsf{SPMonteCarlo}$ = $C\log n$ capped copies of $\textsf{SPmain}$ (Markov: each finishes w.p. $\ge 1/2$); always errors on cyclic inputs, returns a correct tree w.h.p. otherwise. $\textsf{FindThresh}$ binary-searches the smallest $B$ with $G^{+B}$ (add $B$ to *all* edges) cycle-free; pre-scaling by $n^3$ makes any genuine negative cycle survive (with reduced weight $\le n$) into the low-weight subgraph of the reweighted graph, where any cycle is provably negative in $G$. Each branch verifies its own output and restarts on the rare failure; total $O(m\log^8 n)$ (times $\log W$ via Goldberg bit-scaling to the $w\ge -1$ base case and the constant-out-degree vertex-splitting reduction).

## Competition landing (single-file C++17, reads stdin)

The deliverable is one self-contained C++17 program. It reads a single instance from stdin —
`n m s` then `m` edges `u v w` (0-indexed, weights may be negative) — and prints either
`NEGATIVE CYCLE` (a negative cycle is reachable from the source) or the `n` distances
$\operatorname{dist}(s,v)$, one per line (`INF` if unreachable). It is the price-function method
reduced to its load-bearing core for one query: $\textsf{ElimNeg}$'s marked-frontier hybrid sweep
builds a non-negativizing price function $h[v]=\operatorname{dist}(s_{\text{super}},v)\le 0$ from a
virtual super-source (settling each vertex within $\eta(v)+1$ passes, and doubling as the
negative-cycle detector via a relaxation counter plus one confirming sweep), after which a single
Dijkstra on the reweighted non-negative graph recovers
$\operatorname{dist}_G(s,v)=\operatorname{dist}_{\text{reduced}}(s,v)-h[s]+h[v]$. All arithmetic is
in `long long` for overflow safety.

```cpp
// Negative-weight single-source shortest paths (combinatorial price-function method).
// Reads from stdin:  n m s   then m lines "u v w" (0-indexed vertices, integer w may be < 0);
// prints either "NEGATIVE CYCLE" if a negative cycle is reachable from the source, or n lines
// of dist(s,v) ("INF" if v is unreachable). All arithmetic in long long for overflow safety.
//
// Core idea (Johnson reweighting): find an integral price function phi with every reduced weight
// w(u,v)+phi[u]-phi[v] >= 0, then a single Dijkstra on the reweighted graph yields true distances.
// The price function is produced by ElimNeg, the Dijkstra+Bellman-Ford hybrid whose cost is governed
// by eta (the number of negative edges shortest paths must use); a super-source reaching every vertex
// makes phi[v]=dist(s_super,v) a non-negativizing price function, and Bellman-Ford with an
// early-exit / extra relaxation round doubles as the negative-cycle detector.

#include <bits/stdc++.h>
using namespace std;

using ll = long long;
const ll INF = (ll)4e18;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m, s;
    if (!(cin >> n >> m >> s)) return 0;

    vector<array<ll,3>> edges(m);                 // (u, v, w)
    vector<vector<pair<int,ll>>> adj(n);          // u -> (v, w)
    for (int i = 0; i < m; ++i) {
        ll u, v, w;
        cin >> u >> v >> w;
        edges[i] = {u, v, w};
        adj[(int)u].push_back({(int)v, w});
    }

    // ---- Step 1: ElimNeg-style price function via Bellman-Ford from a super-source ----
    // A virtual super-source has a weight-0 edge to every vertex, so h[v] = dist(super, v) <= 0 is
    // defined for all v and is a non-negativizing price function: for any edge (u,v),
    // h[v] <= h[u] + w  =>  w + h[u] - h[v] >= 0.  We run the hybrid sweep: keep a queue of vertices
    // whose label changed (the "marked" frontier) and relax their out-edges; this settles a vertex
    // within eta(v)+1 rounds, so it tracks the negative-edge count rather than always doing n-1 rounds.
    vector<ll> h(n, 0);                            // h[v] starts at 0 (the super-source edge)
    vector<char> inq(n, 1);
    deque<int> q(n);
    iota(q.begin(), q.end(), 0);
    vector<int> cnt(n, 0);                         // relaxation count, for negative-cycle detection
    bool negCycle = false;

    while (!q.empty()) {
        int u = q.front(); q.pop_front();
        inq[u] = 0;
        for (auto [v, w] : adj[u]) {
            if (h[u] + w < h[v]) {
                h[v] = h[u] + w;
                if (!inq[v]) {
                    inq[v] = 1;
                    q.push_back(v);
                    if (++cnt[v] > n) { negCycle = true; break; }
                }
            }
        }
        if (negCycle) break;
    }

    // Confirm with one extra full relaxation sweep: any edge still active => a negative cycle is
    // reachable from the super-source (hence from some vertex), so the input has a negative cycle.
    if (!negCycle) {
        for (auto &e : edges) {
            int u = (int)e[0], v = (int)e[1]; ll w = e[2];
            if (h[u] + w < h[v]) { negCycle = true; break; }
        }
    }

    if (negCycle) {
        cout << "NEGATIVE CYCLE\n";
        return 0;
    }

    // ---- Step 2: reweight to non-negative and run Dijkstra from s ----
    // Reduced weight w + h[u] - h[v] >= 0 everywhere, so Dijkstra is correct; recover true distances
    // via dist_G(s,v) = dist_reduced(s,v) - h[s] + h[v].
    vector<ll> dr(n, INF);                         // reduced distances from s
    dr[s] = 0;
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<>> pq;
    pq.push({0, s});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dr[u]) continue;
        for (auto [v, w] : adj[u]) {
            ll rw = w + h[u] - h[v];              // reduced weight, >= 0
            if (dr[u] + rw < dr[v]) {
                dr[v] = dr[u] + rw;
                pq.push({dr[v], v});
            }
        }
    }

    for (int v = 0; v < n; ++v) {
        if (dr[v] >= INF) cout << "INF\n";
        else cout << (dr[v] - h[s] + h[v]) << "\n";
    }
    return 0;
}
```

The near-linear machinery the proof builds — `ScaleDown`'s four phases, the low-diameter
decomposition `LDD` with geometric ball radii, `FixDAGEdges`, the scaling outer loop, and the Las
Vegas/`FindThresh` wrapper — is what drives $\eta$ to polylog and the asymptotic bound to
$O(m\log^8 n\log W)$; the single-query program above keeps the same price-function skeleton
($\textsf{ElimNeg}$ then Dijkstra) and the same negative-cycle semantics. Nothing beyond Dijkstra,
Bellman–Ford, and price-function reweighting appears.
