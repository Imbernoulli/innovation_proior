We are given an undirected graph $G = (V, E)$ with a nonnegative cost $c_e$ on every edge and, for each unordered pair of vertices, an integer connectivity requirement $r(uv) \ge 0$. We want the cheapest subgraph $H$ that contains at least $r(uv)$ edge-disjoint paths between $u$ and $v$ for every pair — a network that survives up to $r(uv)-1$ edge failures. The problem is NP-hard and APX-hard, so an exact polynomial algorithm is out of reach; what I am after is a polynomial-time algorithm that returns a feasible subgraph of cost at most a fixed constant times $\mathrm{OPT}$, and — this is the whole point — a constant that does not degrade as the requirements grow. The best general approach available, the primal-dual augmentation of Goemans, Goldberg, Plotkin, Shmoys, Tardos, and Williamson (1994), raises connectivity one unit at a time: at layer $k$ it holds a subgraph that is $(k-1)$-connected where required and runs a $0/1$ cut-covering primal-dual step to lift deficient pairs to $k$. Each layer is a clean $2$-approximation against its residual, but the layers stack, and the bound comes out around $2H(r_{\max}) = 2\bigl(1 + \tfrac12 + \cdots + \tfrac1{r_{\max}}\bigr)$. That logarithmic dependence on $r_{\max}$ is exactly what I want to kill, and its cause is structural: the analysis charges connectivity layer by layer and never looks at the whole problem at once. Doubling-based bounds give small constants only for uniform $k$-edge-connectivity and do not survive Steiner vertices and heterogeneous requirements; threshold rounding at a fixed cutoff stalls because an arbitrary optimal point of the relaxation can be $\tfrac13$-ish on every coordinate (a Petersen graph with unit requirements admits $x_e = \tfrac13$ everywhere), leaving any threshold above one third nothing to commit to. The lesson is to stop decomposing by connectivity level and instead reason about the entire LP at one shot.

The first move is to turn the routing condition into a counting condition. By Menger's theorem the maximum number of edge-disjoint $u$-$v$ paths in $H$ equals the minimum number of edges whose removal separates $u$ from $v$, so requiring $r(uv)$ edge-disjoint paths is the same as requiring that every cut separating $u$ from $v$ keeps at least $r(uv)$ edges of $H$. Aggregating over all pairs, define for a vertex set $S$
$$f(S) = \max_{u \in S,\ v \notin S} r(uv), \qquad f(\emptyset) = f(V) = 0,$$
and then $H$ is feasible exactly when $|\delta_H(S)| \ge f(S)$ for every $S \subseteq V$, where $\delta_H(S)$ is the set of edges of $H$ with exactly one endpoint in $S$. The whole problem is now: choose a minimum-cost edge set whose cut-degree dominates $f$, and both ends of the problem live in the language of cuts, computable by max-flow. The cut function $f$ has the structural property that makes everything downstream work — it is *weakly supermodular*: for all $A, B$ at least one of
$$f(A)+f(B) \le f(A\cup B)+f(A\cap B) \quad\text{or}\quad f(A)+f(B) \le f(A\setminus B)+f(B\setminus A)$$
holds. To see it, split $V$ into the atoms $P = A\cap B$, $Q = A\setminus B$, $R = B\setminus A$, $W = V\setminus(A\cup B)$. A pair witnessing $f(A)$ is of type $P\!-\!R$, $P\!-\!W$, $Q\!-\!R$, or $Q\!-\!W$; a pair witnessing $f(B)$ is of type $P\!-\!Q$, $P\!-\!W$, $R\!-\!Q$, or $R\!-\!W$. Checking the four-by-four, every combination lets me place the two witnessed requirements on distinct members of $\{A\cap B,\, A\cup B\}$ (giving the union/intersection inequality) or on distinct members of $\{A\setminus B,\, B\setminus A\}$ (giving the difference inequality). The only combinations not handled immediately are $P\!-\!W$ against $R\!-\!Q$ and $Q\!-\!R$ against $P\!-\!W$; there the $f(A)$ witness also crosses $B$ and the $f(B)$ witness also crosses $A$, so maximality forces $f(A) \le f(B) \le f(A)$, the two values coincide, and either pair carries that common value twice. With $f$ in hand the relaxation writes itself: put $x_e \in [0,1]$ on each edge and impose
$$\min \sum_e c_e x_e \quad\text{s.t.}\quad x(\delta(S)) := \sum_{e\in\delta(S)} x_e \ge f(S)\ \ \forall S,\qquad 0 \le x_e \le 1.$$
Its optimum lower-bounds $\mathrm{OPT}$, so a constant-factor rounding of a fractional solution finishes the job.

I propose to solve it by **iterative LP rounding**: repeatedly solve the covering LP to a *vertex*, round up every edge with $x_e \ge \tfrac12$, fix those edges, replace $f$ by the residual requirement on the remaining graph, and recurse until nothing is left to cover. The threshold of $\tfrac12$ is what delivers the factor: if every committed edge has $x_e \ge \tfrac12$ then $c_e \le 2 c_e x_e$, so the cost I commit in a round is at most twice the fractional cost of the edges I fixed, and that overpay-by-two never compounds across rounds. This works only because of two facts that I have to establish. The first is a structural theorem — in every basic feasible (vertex) solution of a nonzero residual LP there is an edge with $x_e \ge \tfrac12$ — without which a round could find no edge to commit and make no progress, exactly the Petersen-style stall. The second is closure of the problem class under fixing edges, without which the structural theorem would not apply at the next round.

Why must a vertex expose a half-paid edge? An arbitrary optimal point can be $\tfrac13$ everywhere, but a vertex is the unique solution of $|E|$ linearly independent tight constraints, and that rigidity is the lever. Work on the positive support; edges with $x_e = 0$ are deleted, and an edge with $x_e \ge \tfrac12$ already proves the theorem, so the dangerous case has $0 < x_e < \tfrac12$ on every edge and no active variable bound. The tight sets are those with $x(\delta(S)) = f(S)$, and to understand which can be tight together I uncross them. The cut function $x(\delta(\cdot))$ is submodular, and being symmetric it is also posimodular:
$$x(\delta(S)) + x(\delta(T)) \ge x(\delta(S\cup T)) + x(\delta(S\cap T)), \qquad x(\delta(S)) + x(\delta(T)) \ge x(\delta(S\setminus T)) + x(\delta(T\setminus S)).$$
If two tight sets $S, T$ cross, take whichever weak-super inequality $f$ provides — say the union/intersection one — and chain
$$f(S)+f(T) = x(\delta(S)) + x(\delta(T)) \ge x(\delta(S\cup T)) + x(\delta(S\cap T)) \ge f(S\cup T) + f(S\cap T) \ge f(S) + f(T),$$
where the first equality is tightness, the second is cut submodularity, the third is LP feasibility, and the last is weak supermodularity of $f$. The chain returns to its start, so every step is equality: $S\cup T$ and $S\cap T$ are also tight, and submodularity holding with equality means the cross-edge term in the identity $\chi(\delta(S)) + \chi(\delta(T)) = \chi(\delta(S\cup T)) + \chi(\delta(S\cap T)) + 2\,\chi(E(S\setminus T, T\setminus S))$ carries zero $x$-mass; since support values are positive there are no support edges between $S\setminus T$ and $T\setminus S$, the identity collapses, and the crossing row is spanned by the uncrossed tight rows plus the row it crossed. (The difference inequality instead replaces $\{S,T\}$ by $\{S\setminus T, T\setminus S\}$ via posimodularity.) Now take a maximal laminar family $M$ of tight sets. If $M$ did not span all tight rows, pick an outside tight set $S$ crossing as few members of $M$ as possible; it must cross some $L \in M$, and uncrossing produces an outside tight row that crosses strictly fewer members — every member crossing the new candidate already crossed $S$, while $L$ crossed $S$ but not the candidate — contradicting the choice of $S$. So a laminar family of tight cuts spans all tight rows, and in the dangerous case the vertex is pinned by an independent laminar family $L$ with $|L| = |E|$.

The contradiction comes from a token count over that laminar family. Give each edge one token. Edge $e = (u,v)$ sends $x_e$ to the smallest set of $L$ containing $u$, another $x_e$ to the smallest set containing $v$, and $1 - 2x_e$ to the smallest set containing both endpoints, whenever such sets exist; because $0 < x_e < \tfrac12$ all three pieces are positive and they sum to at most $x_e + x_e + (1-2x_e) = 1$, and missing recipients simply leave a piece unassigned. For $S \in L$ with children $R_1, \dots, R_k$, subtracting the child tight equations from the parent's gives $x(\delta(S)) - \sum_i x(\delta(R_i)) = f(S) - \sum_i f(R_i)$. Writing $O = S \setminus (R_1 \cup \cdots \cup R_k)$ for the part $S$ owns directly, the edges with nonzero coefficient fall into class $A$ (from $O$ to $V\setminus S$), class $B$ (from $O$ to one child), and class $C$ (between two different children), so the subtraction is $x(A) - x(B) - 2x(C) = f(S) - \sum_i f(R_i)$. An $A$ edge gives $S$ exactly $x_e$; a $B$ edge gives $x_e + (1-2x_e) = 1 - x_e$; a $C$ edge gives $1 - 2x_e$; and a fourth class $D$ of edges with both endpoints in $O$ cancels from the subtraction yet sends its whole token to $S$, contributing $1$ each. The mass $S$ collects is therefore
$$\sum_{e\in A} x_e + \sum_{e\in B}(1-x_e) + \sum_{e\in C}(1-2x_e) + |D| = |B| + |C| + |D| + f(S) - \sum_i f(R_i),$$
an integer. It is strictly positive: if it were zero, all of $A, B, C, D$ would be empty, so no edge has a nonzero coefficient in $\chi(\delta(S)) - \sum_i \chi(\delta(R_i))$, contradicting independence of the laminar basis. A positive integer is at least one, so every set of $L$ collects at least one token. Yet at least one positive piece is never collected: a maximal set $R \in L$ cannot be $V$ (since $f(V)=0$ makes $\chi(\delta(V))$ the zero vector, excluded from an independent basis) and cannot have empty support boundary (same reason), so some support edge crosses $R$, no laminar set contains both its endpoints, and that edge's $1 - 2x_e$ piece is positive and unclaimed. Hence strictly fewer than $|E|$ tokens are collected while all $|L|$ sets collect at least one, forcing $|L| < |E|$ and contradicting $|L| = |E|$. So some $x_e \ge \tfrac12$. The threshold survives precisely because $1 - 2x_e$ stays positive below one half; beyond one half the count has no slack, and that is why the constant is exactly $2$ and no smaller.

Closure of the class makes the recursion legitimate. After fixing an edge set $F$, the cut $S$ already has $|\delta_F(S)|$ committed edges, so the residual requirement is $g(S) = f(S) - |\delta_F(S)|$. The cut function $h(S) = |\delta_F(S)|$ is symmetric and submodular, hence posimodular. If $f$ gives the union/intersection inequality, submodularity gives $h(A)+h(B) \ge h(A\cup B)+h(A\cap B)$, and subtracting yields $g(A)+g(B) \le g(A\cup B)+g(A\cap B)$; if $f$ gives the difference inequality, posimodularity of $h$ gives the matching bound and subtracting yields the difference inequality for $g$. So $g$ is again weakly supermodular, the residual is another instance of covering a weakly-super function by a graph, and the structural theorem applies anew. The algorithm is then: start with $F = \emptyset$; while $g = f - |\delta_F|$ is not everywhere $\le 0$, solve the covering LP for $g$ to a vertex, round up every edge with $x_e \ge \tfrac12$ (the theorem guarantees at least one), add them to $F$, and repeat; when $g \le 0$ everywhere, $F$ is feasible. Each round fixes at least one new edge, so it terminates.

The factor follows by charging round by round and telescoping. In a round with vertex $x^*$ and rounded set $R = \{e : x^*_e \ge \tfrac12\}$ the committed cost is $\sum_{e\in R} c_e \le 2\sum_{e\in R} c_e x^*_e$. Deleting the $R$-coordinates of $x^*$ leaves, for every cut $S$, a fractional mass $x^*(\delta(S)) - x^*_R(\delta(S)) \ge g_{\text{current}}(S) - |\delta_R(S)| = g_{\text{next}}(S)$ since each rounded coordinate is at most $1$, so the restriction is feasible for the next residual LP and $\mathrm{LP}(\text{next}) \le \mathrm{LP}(\text{current}) - \sum_{e\in R} c_e x^*_e$. Telescoping,
$$\mathrm{cost}(F) = \sum_{\text{rounds}} \sum_{e\in R} c_e \le 2\sum_{\text{rounds}} \bigl(\mathrm{LP}(\text{current}) - \mathrm{LP}(\text{next})\bigr) \le 2\,\mathrm{LP}(\text{initial}) \le 2\,\mathrm{OPT},$$
a factor $2$ with no dependence on $r_{\max}$, because I reasoned about the whole LP vertex at once rather than one connectivity layer at a time. Solving the LP itself, which has one constraint per cut, is done by separation: put capacity $x_e$ on each free edge and $1$ on each fixed edge, build a Gomory-Hu tree with $n-1$ max-flows, and for each demanded pair the minimum tree edge on its path gives the pair's min-cut value; when that value falls below the requirement, removing the tree edge yields a violated shore $S$, and I add $\sum_{\text{free } e \in \delta(S)} x_e \ge f(S) - |\delta_F(S)|$, iterating until no cut is violated. In the runnable code the active relaxation is solved by CBC/CLP and a progress guard raises if a round somehow fails to return a half-paid edge.

```python
from itertools import combinations

import networkx as nx
import pulp

def requirement_on_cut(S, r):
    # Menger turns pair requirements into one cut requirement.
    best = 0
    for (u, v), req in r.items():
        if (u in S) ^ (v in S):
            best = max(best, req)
    return best

def delta(S, edges):
    return [e for e in edges if (e[0] in S) ^ (e[1] in S)]

def add_capacity(H, e, cap):
    u, v = e
    if H.has_edge(u, v):
        H[u][v]["cap"] += cap
    else:
        H.add_edge(u, v, cap=cap)

def edge_cost(costs, e):
    if e in costs:
        return costs[e]
    rev = (e[1], e[0])
    if rev in costs:
        return costs[rev]
    raise KeyError(f"missing cost for edge {e}")

def separation_oracle(x, free_edges, fixed_edges, V, r, tol=1e-7):
    # The fixed edges count integrally; the remaining edges carry their LP values.
    H = nx.Graph()
    H.add_nodes_from(V)
    for e in free_edges:
        add_capacity(H, e, max(0.0, float(x.get(e, 0.0))))
    for e in fixed_edges:
        add_capacity(H, e, 1.0)

    tree = nx.gomory_hu_tree(H, capacity="cap")
    for (u, v), req in r.items():
        if req <= 0 or u == v:
            continue
        if u not in H or v not in H:
            return frozenset({u})
        path = nx.shortest_path(tree, u, v)
        cut_edge = min(zip(path, path[1:]), key=lambda ab: tree[ab[0]][ab[1]]["weight"])
        cut_value = tree[cut_edge[0]][cut_edge[1]]["weight"]
        if cut_value < req - tol:
            witness = tree.copy()
            witness.remove_edge(*cut_edge)
            return frozenset(nx.node_connected_component(witness, u))
    return None

def solve_covering_lp_to_vertex(edges, V, r, fixed_edges, costs):
    fixed_edges = set(fixed_edges)
    free = [e for e in edges if e not in fixed_edges]
    prob = pulp.LpProblem("sndp_cut_cover", pulp.LpMinimize)
    xv = {e: pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1) for i, e in enumerate(free)}
    prob += pulp.lpSum(edge_cost(costs, e) * xv[e] for e in free)
    solver = pulp.PULP_CBC_CMD(msg=False)

    while True:
        status = prob.solve(solver)
        if pulp.LpStatus[status] != "Optimal":
            raise RuntimeError(f"residual LP is {pulp.LpStatus[status]}")

        x = {e: (xv[e].value() or 0.0) for e in free}
        S = separation_oracle(x, free, fixed_edges, V, r)
        if S is None:
            return x

        cross = delta(S, free)
        rhs = requirement_on_cut(S, r) - len(delta(S, fixed_edges))
        if rhs > 1e-7 and not cross:
            raise RuntimeError("residual instance is infeasible with the remaining edges")
        prob += pulp.lpSum(xv[e] for e in cross) >= rhs

def all_satisfied(V, fixed_edges, r):
    return separation_oracle({}, [], fixed_edges, V, r) is None

def cover_cut_requirements(V, edges, costs, r):
    edges = [tuple(e) for e in edges]
    F = set()

    while not all_satisfied(V, F, r):
        x = solve_covering_lp_to_vertex(edges, V, r, F, costs)
        R = [e for e, val in x.items() if val >= 0.5 - 1e-9]
        if not R:
            raise RuntimeError("expected an edge with value at least 1/2 in the basic solution")
        F.update(R)

    return F

def solution_cost(F, costs):
    return sum(edge_cost(costs, e) for e in F)

def is_feasible(V, F, r):
    H = nx.Graph()
    H.add_nodes_from(V)
    H.add_edges_from(F)
    for (u, v), req in r.items():
        if req <= 0:
            continue
        if nx.edge_connectivity(H, u, v) < req:
            return False
    return True

def brute_force_optimum(V, E, costs, r):
    best_cost = float("inf")
    best_set = None
    for size in range(len(E) + 1):
        for chosen in combinations(E, size):
            F = set(chosen)
            cost = solution_cost(F, costs)
            if cost >= best_cost:
                continue
            if is_feasible(V, F, r):
                best_cost = cost
                best_set = F
    return best_cost, best_set

if __name__ == "__main__":
    V = [0, 1, 2, 3]
    E = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    costs = {(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 0): 1, (0, 2): 5}
    r = {(0, 2): 2}

    F = cover_cut_requirements(V, E, costs, r)
    alg_cost = solution_cost(F, costs)
    opt_cost, opt_set = brute_force_optimum(V, E, costs, r)

    print("chosen edges:", sorted(F))
    print("algorithm cost:", alg_cost)
    print("exact optimum:", opt_cost, sorted(opt_set))
    print("feasible:", is_feasible(V, F, r))
    print("within 2x:", alg_cost <= 2 * opt_cost)
```
