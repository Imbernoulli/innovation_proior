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

## Algorithm in pseudocode

```python
import math, heapq, random

C_CONST = 8  # marking-sample constant (k = C_CONST * ln n)

# ---- ElimNeg: non-negativizing price function via Dijkstra+Bellman-Ford ----
def elim_neg(out_edges, vertices, s):
    """O(log n * (n + sum_v eta_G(v))); loops forever iff a negative cycle exists.
       out_edges[u] -> list of (v, w); requires constant out-degree."""
    INF = math.inf
    d = {v: INF for v in vertices}; d[s] = 0
    while True:
        pq = [(d[v], v) for v in vertices if d[v] < INF]
        heapq.heapify(pq); marked = []
        while pq:                                   # Dijkstra phase (non-negative edges)
            dv, v = heapq.heappop(pq)
            if dv > d[v]: continue
            marked.append(v)
            for (x, w) in out_edges[v]:
                if w >= 0 and d[v] + w < d[x]:
                    d[x] = d[v] + w; heapq.heappush(pq, (d[x], x))
        changed = False                             # Bellman-Ford phase (negative edges)
        for v in marked:
            for (x, w) in out_edges[v]:
                if w < 0 and d[v] + w < d[x]:
                    d[x] = d[v] + w; changed = True
        if not changed:
            return {v: (0 if d[v] == INF else d[v]) for v in vertices}

# ---- FixDAGEdges: price the SCC-DAG so inter-SCC edges become non-negative --
def fix_dag_edges(out_edges, vertices, sccs, scc_of, topo):  # O(m + n)
    mu = [0] * len(sccs)
    for u in vertices:
        for (v, w) in out_edges[u]:
            if scc_of[u] != scc_of[v] and w < mu[scc_of[v]]:
                mu[scc_of[v]] = w                   # most negative edge entering scc_of[v]
    phi = {}; M = 0
    for j in topo:                                  # topological order of SCCs
        M += mu[j]
        for v in sccs[j]: phi[v] = M
    return phi

# ---- Low-Diameter Decomposition on a NON-NEGATIVE graph ---------------------
def ldd(out_edges, verts, D, n):
    verts = list(verts)
    if len(verts) <= 1: return set()
    k = max(1, int(C_CONST * math.log(max(n, 2))))
    S = [random.choice(verts) for _ in range(k)]
    inb, outb = ball_sample_counts(out_edges, verts, S, D / 4)   # via k Dijkstras
    mark = {}
    for v in verts:
        if inb[v] <= 0.6 * k:    mark[v] = "in"
        elif outb[v] <= 0.6 * k: mark[v] = "out"
        else:                    mark[v] = "heavy"
    p = min(1.0, 80 * math.log(max(n, 2)) / D)
    E_rem, alive = set(), set(verts)
    light = [v for v in verts if mark[v] != "heavy"]
    for v in light:
        if v not in alive: continue
        R = sample_geometric(p)                                  # geometric radius
        ball = grow_ball(out_edges, v, R, mark[v], alive)        # Dijkstra to radius R
        E_rem |= boundary_edges(out_edges, ball, mark[v], alive)
        if R > D / 4 or len(ball) > 0.7 * len(alive):
            return all_edges(out_edges, verts)                  # bail (prob <= n^-20)
        sub = induced(out_edges, ball)
        E_rem |= ldd(sub, ball, D, n)                           # recurse inside the ball
        alive -= ball
    if not heavy_block_diameter_ok(out_edges, alive, D):
        return all_edges(out_edges, verts)                      # bail (prob <= n^-20)
    return E_rem

# ---- ScaleDown: halve B using the decomposition ----------------------------
def scale_down(G, Delta, B, n):
    """INPUT  w(e) >= -2B; const out-degree; if no neg cycle eta(G^B) <= Delta.
       OUTPUT integral phi with w_phi(e) >= -B."""
    phi2 = {v: 0 for v in G.vertices()}
    if Delta > 2:
        d = Delta / 2.0
        GB_nneg = shift_negatives(G, B, clamp_to_zero=True)     # G^B_{>=0}
        E_rem   = ldd(GB_nneg.adj, GB_nneg.vertices(), int(d * B), n)   # D = dB
        GB      = shift_negatives(G, B, clamp_to_zero=False)
        sccs, scc_of, topo = sccs_after_removing(GB, E_rem)
        H       = union_of_induced_sccs(G, sccs)                # Phase 1
        phi1    = scale_down(H, Delta / 2, B, n)                #   recurse
        GBm     = reweight(GB, phi1, drop=E_rem)                # Phase 2
        psi     = fix_dag_edges(GBm.adj, GBm.vertices(), sccs, scc_of, topo)
        phi2    = add_phi(phi1, psi)
    GB      = shift_negatives(G, B, clamp_to_zero=False)        # Phase 3
    GBs     = add_dummy_source(GB)                              #   add s, THEN price
    GBs_phi = reweight(GBs, {**phi2, GBs.dummy: 0})
    psi_p   = elim_neg(GBs_phi.adj, GBs_phi.vertices(), GBs.dummy)
    return add_phi(phi2, psi_p)

# ---- SPmain: scaling outer loop -> shortest-path tree ----------------------
def sp_main(G_in, s_in):
    n = G_in.num_vertices()
    G = scale_weights(G_in, 2 * n)                              # keep integral
    B = round_up_pow2(2 * n)
    phi = {v: 0 for v in G.vertices()}
    for i in range(1, int(math.log2(B)) + 1):
        psi = scale_down(reweight(G, phi), n, B // (2 ** i), n)
        phi = add_phi(phi, psi)                                 # w_phi(e) >= -B/2^i
    Gstar = add_one_to_each_edge(reweight(G, phi))              # clear the last -1's
    return dijkstra(Gstar, s_in)                                # true shortest-path tree
```

The helpers (`grow_ball`, `boundary_edges`, `shift_negatives`, `reweight`, `add_dummy_source`,
`sccs_after_removing`, `dijkstra`, `sample_geometric`) are elementary graph operations:
Dijkstra-based ball growth, price-function reweighting, the $B$-shift on negative edges, SCC and
topological-order computation, and geometric sampling. The Las Vegas wrapper
(`SPMonteCarlo` = $C\log n$ time-capped copies of `sp_main`; `FindThresh` binary search returning a
negative cycle) sits on top unchanged. Nothing beyond Dijkstra, Bellman–Ford, SCCs, topological
order, and geometric coin flips is used — a purely combinatorial near-linear algorithm.
