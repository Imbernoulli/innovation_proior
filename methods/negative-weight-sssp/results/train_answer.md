The thing that actually hurts here is simple to state. I have a directed graph $G=(V,E,w)$ with integer weights, some of them negative, and I want the shortest-path distances from a source $s$ (or a witness that some reachable cycle is negative). If every weight were non-negative I would run Dijkstra and finish in $O(m+n\log n)$ — essentially linear. But Dijkstra's correctness rests entirely on the greedy invariant "extract the smallest tentative distance, declare it final," and a single negative edge downstream can later undercut a vertex I have already finalized. So I am thrown back on Bellman–Ford, which survives negatives and detects negative cycles but costs $O(mn)$ — quadratic on a sparse graph. Three decades of bit-scaling work (Gabow, Gabow–Tarjan, Goldberg) pushed the *combinatorial* bound to $O(m\sqrt n\log W)$ and then it simply stalled; the $\sqrt n$ was a wall. The only people who reached near-linear time recently did it by reformulating the whole thing as a min-cost flow and throwing interior-point methods plus a tower of dynamic data structures at it — heavy continuous machinery for a problem this elementary, and machinery not even specific to shortest paths. What I want is near-linear time for *general* graphs using nothing but Dijkstra, Bellman–Ford, and graph surgery.

The right object to chase is the price function. Following Johnson, pick $\phi:V\to\mathbb{Z}$ and reweight every edge as $w_\phi(u,v)=w(u,v)+\phi(u)-\phi(v)$. Along any path the interior $\phi$ values telescope, so reduced path weights differ from real ones only by the endpoint prices, and around any *cycle* the prices cancel exactly: $w_\phi(C)=w(C)$. Hence $\phi$ changes neither which paths are shortest nor which cycles are negative — $G$ and $G_\phi$ are equivalent — and if I can find an integral $\phi$ that makes *every* reduced weight $\ge 0$, then $G_\phi$ is non-negative and one Dijkstra finishes everything. So the entire problem collapses to: find a non-negativizing integral price function (assuming no negative cycle). The obvious choice $\phi(v)=\operatorname{dist}(s,v)$ works by the triangle inequality, but computing those distances *is* the problem — a circle. I need to manufacture such a $\phi$ without already knowing the distances.

The key to escaping the circle is to look at *why* Bellman–Ford is slow. Its cost is not really $n$; it is the number of negative edges a shortest path is forced to thread, because each relaxation round only advances the correct-distance frontier past one negative hop. So for each vertex define
$$\eta_G(v)=\min_{\text{shortest }s\to v\text{ path }P}\,\big|E^{neg}(G)\cap P\big|,\qquad \eta(G)=\max_v\eta_G(v),$$
the fewest negative edges any shortest path to $v$ must use. If I can drive $\eta$ down to polylog, a hybrid of Dijkstra and Bellman–Ford produces the price function in near-linear time, and the whole game becomes a fight to make $\eta$ small.

The method I propose has four interlocking pieces: a hybrid price-finder $\textsf{ElimNeg}$, a scaling recursion $\textsf{ScaleDown}$ that halves $\eta$ using a low-diameter decomposition $\textsf{LDD}$, an outer scaling loop $\textsf{SPmain}$, and a Las Vegas wrapper that delivers a negative-cycle witness. Together they solve negative-weight SSSP in $O(m\log^8 n\,\log W)$ time, Las Vegas, by purely combinatorial means.

Start with $\textsf{ElimNeg}(G,s)$, the engine whose cost is governed by $\eta$. Keep estimates $d(v)$ with $d(s)=0$, the rest $\infty$, and alternate two phases. The Dijkstra phase runs an ordinary priority-queue Dijkstra but relaxes *only the non-negative edges*. The Bellman–Ford phase makes one sweep over the negative edges leaving the "marked" vertices (those whose label just settled) and relaxes them, pushing any improved vertex back into the queue. Then repeat. The correctness is an induction on $\eta$: after the iteration-$0$ Dijkstra phase — plain Dijkstra on the non-negative subgraph — $d(v)$ is correct for every $v$ with $\eta(v)=0$. Suppose after iteration $i-1$ every $v$ with $\eta(v)\le i-1$ is correct. For a $v$ with $\eta(v)=i$, take a realizing shortest path and walk to its *last* negative edge $(u_{j-1},u_j)$; the prefix uses $i-1$ negative edges, so $d(u_{j-1})$ is already correct, the Bellman–Ford phase relaxes $(u_{j-1},u_j)$ to give $d(u_j)\le\operatorname{dist}(s,u_j)$, and the iteration-$i$ Dijkstra phase relaxes the all-non-negative suffix in order, giving $d(v)=\operatorname{dist}(s,v)$. So $v$ settles within $\eta(v)+1$ iterations. A vertex can therefore only be queued in iterations $0,\dots,\eta(v)$, at most twice per iteration, so total insertions are $N\le 2\sum_v\eta_G(v)+2n$. The Bellman–Ford work charged to extracting $v$ is $O(\text{out-degree}(v))$, which is why I *require constant out-degree* — not cosmetic, but exactly what makes that charge $O(1)$; any high-degree vertex is split into a small $0$-weight cycle of copies, blowing up the graph by only $O(m)$ and changing no distances. The time is then
$$O\big(\log n\cdot(n+\textstyle\sum_v\eta_G(v))\big).$$
At termination no edge is *active* (no $(v,x)$ with $d(v)+w(v,x)<d(x)$), which is precisely the statement that $d$ is a valid non-negativizing price function. And if $G$ has a negative cycle, some edge stays active forever and $\textsf{ElimNeg}$ never halts — an acceptable behavior I convert to a clean error later. Throughout I attach a *dummy source* $s$ with a weight-$0$ edge to every vertex, so $\operatorname{dist}(s,v)\le 0$ for all $v$, everyone is reachable, $\eta$ is defined uniformly, and $G_s$ has a negative cycle iff $G$ does.

Now the fight to make $\eta$ small, which is the heart of the contribution. First, scaling: by Goldberg's bit-scaling it suffices to handle weights $\ge -1$, because a $1$-feasible *integral* price function then has all reduced weights $\ge 0$, and a general graph walks down to that case in $O(\log W)$ doublings. So I build $\textsf{ScaleDown}(G,\Delta,B)$ with the contract: if $w(e)\ge -2B$, out-degrees are $O(1)$, and (when there is no negative cycle) $\eta(G^B)\le\Delta$, then it returns an integral $\phi$ with $w_\phi(e)\ge -B$. Inside it I work with $G^B$: **add $B$ to every negative edge, leave non-negative edges alone.** The reason for shifting only the negatives is one identity — a path using $k$ negative edges satisfies
$$w^B(P)=w(P)+k\cdot B,$$
so the shifted weight *counts* negative edges (scaled by $B$). This lever converts "how negative a path is" into "how many negative edges it has." It pays off immediately: inside a strongly connected piece of weak diameter $D$ (every pair $u,v$ with $\operatorname{dist}_G(u,v)\le D$ and $\operatorname{dist}_G(v,u)\le D$), a dummy-rooted shortest path $P$ to $v$ with $k$ negative edges has $w^B(P)\le 0$, hence $w(P)\le -kB$; concatenating with $\operatorname{dist}_G(v,u)\le D$ gives a closed walk of weight $\le -kB+D$. If $kB>D$ that is a negative cycle — so with no negative cycle, $\eta_{G^B}(v)\le D/B$ inside the piece. You cannot pile up negativity in a region that is tightly reachable in the positive sense without closing a loop.

This dictates the target diameter exactly. To *halve* $\eta$ — from $\eta\le\Delta$ to $\eta\le\Delta/2$ — set $d=\Delta/2$ and demand pieces of weak diameter $D=dB=B\Delta/2$; then inside each piece $\eta_{G^B}(v)\le D/B=d=\Delta/2$, and recursion depth is $O(\log\Delta)$ down to the base case $\Delta\le 2$. The same $D$ also controls the time, which is why the choice is forced and not tuned: I need the decomposition to cut each edge with probability proportional to its weight, $\Pr[e\in E^{rem}]=O(w(e)\log^2 n/D+n^{-10})$. Since the rounded shifted weight of a shortest path satisfies $w^B_{\ge 0}(P_{G^B}(v))\le\eta_{G^B}(v)\,B\le\Delta B$, summing edge cut probabilities along the path gives
$$\mathbb{E}\big[|P_{G^B}(v)\cap E^{rem}|\big]=O\Big(\frac{w^B_{\ge 0}(P)\,\log^2 n}{B\Delta/2}+n^{-9}\Big)=O(\log^2 n),$$
exactly because $D=B\Delta/2$. So $\textsf{ScaleDown}$ runs in four phases. Phase 0 decomposes: $E^{rem}\gets\textsf{LDD}(G^B_{\ge 0},D=dB)$, where the still-negative shifted weights are rounded up to $0$ so the decomposition (which needs non-negatives) applies, and rounding up only increases distances, so the weak-diameter bound carries back to $G$; the SCCs $V_1,V_2,\dots$ of $G^B\setminus E^{rem}$ then each have weak diameter $\le dB$. Phase 1 makes the inside-SCC edges non-negative: on $H=\bigcup_i G[V_i]$ the small-diameter argument gives $\eta(H^B)\le d=\Delta/2$, so $H$ meets the input contract with $\Delta/2$ and I recurse $\phi_1\gets\textsf{ScaleDown}(H,\Delta/2,B)$; what I rely on is the *strong* internal guarantee that the recursion's final $\textsf{ElimNeg}$ drives $H^B_{\phi_1}$ to non-negative ($w^B_{\phi_1}(e)\ge 0$ on inside edges), not the weaker external "$w_{\phi_1}(e)\ge -B$." Phase 2 makes the between-SCC edges non-negative: contracting the SCCs (after dropping $E^{rem}$) leaves a DAG, and $\textsf{FixDAGEdges}$ prices it in topological order — for each SCC $V_j$ let $\mu_j=\min\{w(u,v):(u,v)\in E^{neg},\,u\notin V_j,\,v\in V_j\}$ (or $0$), set $M_j=\sum_{k\le j}\mu_k$, and $\phi(v)=M_j$ for $v\in V_j$; then for $u\in V_i\to v\in V_j$ with $i<j$, $w_\phi(u,v)=w(u,v)-\sum_{k=i+1}^{j}\mu_k\ge w(u,v)-\mu_j\ge 0$. A shared per-SCC price never disturbs the already-fixed inside edges, so after $\phi_2=\phi_1+\psi$ the only negatives left in $G^B_{\phi_2}$ lie in the cut set $E^{rem}$. Phase 3 cleans those up: $\psi'\gets\textsf{ElimNeg}((G^B_s)_{\phi_2},s)$, with the dummy source added *before* applying $\phi_2$ (defining $\phi_2(s)=0$) so that $(G^B_s)_{\phi_2}$ stays equivalent to $G^B_s$ — pricing first would silently change which graph I solve. Since the only negatives the hybrid sees are cut edges, $\eta_{(G^B_s)_{\phi_2}}(v)\le|P_{G^B}(v)\cap E^{rem}|+1$, so $\mathbb{E}[\eta]=O(\log^2 n)$ and Phase 3 costs $O(m\log^3 n)$ expected; the output $w^B_{\phi_3}(e)\ge 0\Rightarrow w_{\phi_3}(e)\ge -B$ meets the contract, and this correctness never used the no-cycle assumption. Over $O(\log\Delta)$ levels, $\textsf{ScaleDown}=O(m\log^3 n\log\Delta)$.

The decomposition $\textsf{LDD}(G,D)$ itself is built on ball-carving with a *geometrically random* radius. For a vertex $v$, $\operatorname{Ball}^{out}(v,R)=\{u:\operatorname{dist}(v,u)\le R\}$ with boundary edges $\partial$, and symmetrically $\operatorname{Ball}^{in}$. The random radius is what makes the cut probability proportional to edge weight: condition on $u$ already in an out-ball; then $(u,v)$ is cut iff the ball stops before reaching $v$, i.e. $R<\operatorname{dist}(v,u)+w(u,v)$, and memorylessness of $\textsf{Geo}(p)$ collapses this to $\Pr[R\le w(u,v)]\le p\,w(u,v)$. With $p=\min\{1,80\log n/D\}$ the per-edge, per-call cut probability is $O(w(e)\log n/D)$; a fixed radius would cut a deterministic annulus and lose all proportionality, so the geometric law is forced. The decomposition first marks vertices using $k=\Theta(\log n)$ random sample vertices and their radius-$D/4$ balls: a vertex is in-light if its sampled in-ball count is $\le 0.6k$, else out-light if its out-ball count is $\le 0.6k$, else heavy; by Chernoff, w.h.p. a light vertex's real ball is $\le 0.7|V|$ and a heavy vertex has both balls $>0.5|V|$. Then it carves: while a light vertex $v$ remains, draw $R_v\sim\textsf{Geo}(p)$, grow the appropriate ball, put its boundary into $E^{rem}$, recurse inside the ball, and delete it from the working graph; if ever $R_v>D/4$ or the ball exceeds $0.7|V|$, bail out to singletons (probability $\le n^{-20}$). Recursion inside the ball is needed because cutting the boundary only separates the ball from the rest; it says nothing about diameter *within* the ball. Each light ball is $\le 0.7|V|$, so the depth is $O(\log n)$ and every vertex/edge participates in $O(\log n)$ calls — which supplies the second $\log n$ in the cut bound, $O(w(e)\log^2 n/D)$. When only heavy vertices remain, any two have intersecting $D/4$-balls, so the leftover block has weak diameter $\le D/2$, checked by one Dijkstra (else bail). The whole decomposition runs in $O(m\log^2 n+n\log^3 n)$.

The outer loop $\textsf{SPmain}$ assembles the scaling: scale all weights up by $2n$ to keep everything integral, set $B$ to a power of two near $2n$, and run $\log_2 B=O(\log n)$ rounds of $\textsf{ScaleDown}(\bar G_{\phi_{i-1}},\Delta=n,B/2^i)$, marching reduced weights from $\ge -2n$ down to $\ge -1$. Because the scaling kept all distances multiples of $2n$, adding $+1$ to every reduced weight to clear the last $-1$'s cannot reorder any shortest path (distinct path weights differ by more than $n\ge|P|$), so a final Dijkstra returns a true shortest-path tree; the expected time is $O(m\log^5 n)$. To turn this expected-time, no-cycle engine into a Las Vegas algorithm that also returns a witness, I run $\textsf{SPMonteCarlo}$ = $C\log n$ independent copies of $\textsf{SPmain}$ each capped at $2\mathcal T$ steps (by Markov each finishes with probability $\ge 1/2$, so all fail with probability $\le n^{-C}$): on no-cycle inputs it returns a correct tree w.h.p., and on cyclic inputs every copy runs out of time and it correctly errors. For the negative cycle, $\textsf{FindThresh}$ binary-searches the smallest $B\ge 0$ for which $G^{+B}$ (add $B$ to *every* edge, including positives — a different operator from $G^B$) has no negative cycle, using $\textsf{SPMonteCarlo}$ as the detector. If the threshold is $0$ the graph had no negative cycle and I return the tree; otherwise, after pre-scaling weights by $n^3$ (so any genuine negative cycle has threshold $B\ge n^2$ and survives), I reweight $G^{+B}$ non-negative by its price function, keep only edges of reduced weight $\le n$, and read off any cycle in that low-weight subgraph — it is provably negative in the original, since its reweighted weight is $\le n^2$ while $B|C|\ge 2n^2$. Every branch verifies its own output and restarts on the rare failure, so the algorithm is Las Vegas: always correct, w.h.p. fast, total $O(m\log^8 n\log W)$, using only Dijkstra, Bellman–Ford, SCCs, topological order, and geometric coin flips.

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
