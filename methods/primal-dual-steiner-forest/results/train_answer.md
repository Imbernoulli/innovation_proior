We are given an undirected graph $G = (V, E)$ with nonnegative edge costs $c_e$ and a list of client requests, each a terminal pair $(s_j, t_j)$, and we must buy a cheapest edge set $F \subseteq E$ so that every $s_j$ is connected to its $t_j$ in $(V, F)$. The trap that has to be resisted is treating this as a Steiner tree: throwing all the terminals into one bag and connecting them up. The requested pairs are independent — one client wants Boston–Providence, another wants Dallas–Austin, and there is no reason on earth for those two little networks to touch. The cheapest legal answer is a forest of several disjoint trees, and forcing one connected network over all terminals can cost arbitrarily more than the true optimum. That is exactly why the connected Steiner-tree heuristics with their factor $2(1 - 1/|T|)$ are answering a different question; and Hochbaum's solve-the-LP-then-take-tight-edges and the textbook single-violation primal-dual both deliver only an $f$-approximation, where $f$ is the size of the largest cut hit, which for connection problems is as bad as the number of pairs $k$. The problem is NP-hard — it contains Steiner tree — so the exact optimum is out of reach in polynomial time; the realistic goal is a polynomial algorithm whose cost is provably within a small constant of optimal, and the target is $2$.

The first move is to find a handle on feasibility that works edge by edge rather than through the global "is everything connected." Connectivity is a statement about cuts: for any vertex set $S$, the edges leaving it are $\delta(S)$, and if $S$ contains $s_j$ but not $t_j$ then every $s_j$–$t_j$ path must cross out of $S$, so it must use an edge of $\delta(S)$; conversely, hitting every such crossing set leaves no way to wall the pair apart. So $F$ is feasible exactly when, for every set $S$ that separates some requested pair (exactly one of $s_j, t_j$ inside), $|F \cap \delta(S)| \ge 1$. That is a covering condition with one requirement per separating set — exponentially many, but with a clean shape — and it leads to the LP that minimizes $\sum_e c_e x_e$ subject to $\sum_{e \in \delta(S)} x_e \ge 1$ for every separating $S$, with dual

$$\max \sum_S y_S \quad \text{s.t.} \quad \sum_{S:\, e \in \delta(S)} y_S \le c_e \ \ \forall e, \quad y_S \ge 0.$$

I propose the primal–dual moat-growing algorithm for Steiner forest. Read the dual variable $y_S$ physically as the width of a moat — a ring — drawn around the set $S$; an edge crosses that moat exactly when $e \in \delta(S)$, and the constraint says the total width of all moats crossing edge $e$ cannot exceed $c_e$. The lever that makes the whole method possible is weak duality: *any* feasible $y$, optimal or not, satisfies $\sum_S y_S \le \mathrm{OPT}$. So the dual is not something to compute and round — it is something to grow alongside the primal. If I can grow a feasible $y$ and simultaneously buy a network $F$ with $\mathrm{cost}(F) \le 2 \sum_S y_S$, then $\mathrm{cost}(F) \le 2\,\mathrm{OPT}$ and I am done, having solved no LP at all.

The defining decision is *which* moats to grow, and the star instance dictates the answer. Take one common source $s = s_1 = \dots = s_k$ and $k$ distinct sinks $t_1, \dots, t_k$, each joined to $s$ by a unit-cost edge. If I grow only the single moat around $\{s\}$, it crosses all $k$ star edges at once; they all go tight together, I buy all $k$, and my one moat of width $1$ is charged for $k$ edges — a $k$-approximation, the single-violation wall. But the same picture, looked at across the *whole* family of violated sets $\{s\}, \{t_1\}, \dots, \{t_k\}$, tells the cure: the edge into $t_j$ crosses $\delta(\{s\})$ once and $\delta(\{t_j\})$ once, so every star edge crosses exactly two of these moats, $2k$ crossings spread over $k+1$ moats, an average of $2k/(k+1) < 2$ bought edges per moat. The edges are expensive only when the whole bill is dumped on one moat. So the rule is: grow *all active moats at once, uniformly, at the same rate*, where a component $C$ of the bought-so-far forest is **active** iff it still separates some unconnected pair (exactly one endpoint of some $(s_j, t_j)$ inside it). Inactive moats must never be grown — growing one is pure waste, and the analysis depends on never charging it.

The bookkeeping that survives merges is per vertex, not per component, because once an active component swallows an inactive Steiner-only piece the two sides have different growth histories. I store $d(v) = \sum_{S:\, v \in S} y_S$, the total moat width over grown sets containing $v$; growing an active component by $\varepsilon$ adds $\varepsilon$ to $d(v)$ for every $v$ in it. For an edge $e=(u,v)$ straddling two current components, the dual already packed onto it is $d(u)+d(v)$ — every grown set containing exactly one endpoint contributes to exactly one of the two labels — so its slack is $c_e - d(u) - d(v)$, eaten at a rate equal to the number of active endpoint components ($1$ or $2$). The first edge to go tight minimizes

$$\varepsilon = \frac{c_e - d(u) - d(v)}{\text{rate}}.$$

Grow every active component by that $\varepsilon$, buy the tight edge, merge its two components, recompute who is active, and stop when nothing is active — at which point every pair is connected. The $d$ labels are exactly what keep these reduced costs faithful across merges.

Buying every tight edge can leave redundant ones behind: later mergers can tie two components together through a longer route, leaving a cycle's worth of edges all crossing some moat, which reinflates the per-moat charge and kills the bound. The fix is a **reverse-delete** cleanup — after the growth phase connects everything, scan the bought edges in the *reverse* of the order they were bought, latest first, and drop any whose removal still leaves every pair connected. Reverse order is the load-bearing choice: when an early edge is considered, every later edge has already been examined and kept only if necessary, so the surviving forest is a *minimal* feasible one, and that minimality is the hook the counting needs.

The $2$ lives or dies in the count. Let $F'$ be the cleaned forest. Every kept edge is tight, $c_e = \sum_{S:\, e \in \delta(S)} y_S$, so swapping summation order,

$$\mathrm{cost}(F') = \sum_{e \in F'} \sum_{S:\, e \in \delta(S)} y_S = \sum_S y_S \,|F' \cap \delta(S)|.$$

Writing each moat width as the sum of its increments over the iterations in which $S$ was active, $y_S = \sum_{t:\, S \text{ active at } t} \varepsilon_t$, and regrouping by iteration with $A_t$ the components active at iteration $t$,

$$\mathrm{cost}(F') = \sum_t \varepsilon_t \!\!\sum_{S \in A_t}\! |F' \cap \delta(S)|, \qquad \sum_S y_S = \sum_t \varepsilon_t\, |A_t|,$$

so $\mathrm{cost}(F') \le 2\sum_S y_S$ follows from the per-iteration claim $\sum_{S \in A_t} |F' \cap \delta(S)| \le 2|A_t|$ — the star's "$2$ on average" demanded at every step. To prove it, fix $t$, contract each current component to a node, and put an edge between two nodes for each kept $F'$-edge joining those components, giving a graph $H$. The growth phase only buys edges between distinct components so the bought set is a forest; reverse-delete only removes; contracting connected parts of a forest makes no cycle; hence $H$ is a forest with at most $\#\text{nodes} - 1$ edges. Color a node red if active at $t$, blue if inactive; then $|F' \cap \delta(C)|$ is the degree of $C$'s node, and the quantity to bound is $\sum_{\text{red}} \deg_H$. A blue node cannot be a leaf: an inactive component with a single incident kept edge separates no still-unconnected pair, so removing that edge keeps every pair connected and reverse-delete would already have deleted it. With every (non-isolated) blue node of degree $\ge 2$, and $R, B$ the red and degree-$\ge 2$ blue nodes,

$$\sum_{\text{red}} \deg = \sum_{\text{all}} \deg - \sum_{\text{blue}} \deg \le 2(|R| + |B| - 1) - 2|B| = 2|R| - 2 \le 2|R| = 2|A_t|.$$

Chaining everything, $\mathrm{cost}(F') \le 2\sum_S y_S \le 2\,\mathrm{OPT}$. The grown dual $\sum_S y_S$ is a per-instance lower-bound certificate, often far better than $2$; and keeping the $-2$ with $K$ the initial number of active components — which never increases — sharpens the bound to $\mathrm{cost}(F') \le (2 - 2/K)\sum_S y_S$. Every piece is forced by a failure: growing only active moats is what lets the no-blue-leaf claim avoid charging inactive nodes; growing all active moats uniformly is what makes each step an average over $|A_t|$ rather than a dump on one moat; reverse-delete is what enforces the minimality that forbids inactive leaves; and the cut LP, never solved, is what supplies both the moats to grow and the lower bound to charge against. As a sanity check the rule reduces to the right textbook algorithms: a single pair $(s,t)$ grows two moats around $s$ and $t$ until a shortest-path edge goes tight, then reverse-delete strips it to the optimal $s$–$t$ path (bidirectional Dijkstra); and when every pair of vertices must connect, every component is always active, so each step buys the cheapest cross-component edge — Kruskal's exact minimum spanning tree.

```python
def steiner_forest(n, edges, pairs):
    """Min-cost forest connecting each (s, t) in `pairs`.
    n: vertices 0..n-1; edges: (u, v, cost) with cost >= 0; pairs: (s, t)."""
    parent = list(range(n))                       # union-find over vertices

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    d = [0.0] * n                                  # d[v] = sum of moats containing v

    def active(root):                              # separates some unconnected pair?
        for s, t in pairs:
            if (find(s) == root) != (find(t) == root):
                return True
        return False

    chosen = []                                    # edges bought, in order
    while True:
        roots = {find(v) for v in range(n)}
        act = {r for r in roots if active(r)}
        if not act:                                # all pairs connected -> feasible
            break
        best_eps, best_edge = None, None           # earliest edge to go tight
        for (u, v, c) in edges:
            ru, rv = find(u), find(v)
            if ru == rv:
                continue
            rate = (ru in act) + (rv in act)       # active endpoints: 1 or 2
            if rate == 0:
                continue
            eps = (c - d[u] - d[v]) / rate
            if best_eps is None or eps < best_eps:
                best_eps, best_edge = eps, (u, v, c)
        if best_edge is None:
            raise ValueError("no feasible edge can connect the remaining terminal pairs")
        for x in range(n):                         # grow every active moat by eps
            if find(x) in act:
                d[x] += best_eps
        u, v, c = best_edge
        chosen.append((u, v, c))
        parent[find(u)] = find(v)                  # buy tight edge, merge

    forest = list(chosen)                          # reverse-delete cleanup
    for e in reversed(chosen):
        removed = False
        trial = []
        for x in forest:
            if not removed and x is e:
                removed = True
                continue
            trial.append(x)
        if not removed:
            continue
        if _all_connected(n, trial, pairs):
            forest = trial
    return forest


def _all_connected(n, forest, pairs):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (u, v, _c) in forest:
        parent[find(u)] = find(v)
    return all(find(s) == find(t) for s, t in pairs)
```
