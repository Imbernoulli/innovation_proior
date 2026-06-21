We are given a directed network with a source $s$, a sink $t$, and a nonnegative capacity $c(u,v)$ on each arc, and we want two things at once: the largest steady traffic that can be pushed from $s$ to $t$ without overloading any link or piling up flow at intermediate vertices, and the cheapest set of links whose removal disconnects $s$ from $t$. A feasible flow assigns $0 \le f(u,v) \le c(u,v)$ to each arc, conserves flow at every internal vertex, and has value $|f| = \sum_v f(s,v) - \sum_v f(v,s)$. A separating cut is a partition $(S,T)$ with $s \in S$, $t \in T$, and capacity $\mathrm{cap}(S,T) = \sum_{u \in S,\, v \in T} c(u,v)$, counting only forward-crossing arcs. The trouble is that the maximizing question (pack as much flow as possible) and the minimizing question (find the smallest bottleneck) look related but not obviously equal, and a useful method must produce both a large flow and a small separating certificate so that the claimed optimum is checkable directly on the network.

The existing options each fall short of this. Casting the problem as a linear program and running simplex is legitimate but treats the graph as a faceless pile of equations and gives no network-level reason why the answer is final. Greedy flooding — push along available routes, return excess, repeat — is easy to perform on a map but cannot certify: a greedy early route can lock up an arc that a better global routing needs, and a procedure that only ever adds flow has no principled way to retract a bad placement. Route decomposition, viewing a flow as a sum of path flows, makes existence and convexity visible but never tells a planner which paths to pick, when to stop, or how to read off a minimal separating set. What none of these supply is the one thing the problem demands: a stopping rule that simultaneously proves the flow cannot grow and exhibits a cut that cannot shrink.

The lever is the only inequality I can trust. Summing the net outflow over all vertices of the source side $S$, conservation erases every internal vertex and every arc with both ends in $S$ cancels, leaving $|f| = \sum_{S \to T} f - \sum_{T \to S} f$; dropping the nonnegative backward term and bounding forward flow by capacity gives $|f| \le \mathrm{cap}(S,T)$. This weak duality is more than a bound — it pins down exactly what equality requires. The first inequality is tight only when every backward arc from $T$ into $S$ carries zero flow, and the second only when every forward arc from $S$ into $T$ is saturated. So if I can ever manufacture a cut that is full forward and idle backward, the flow value and cut capacity coincide and neither object can be improved. The whole task becomes: build a situation where some cut is saturated in one direction and empty in the other.

What I propose is the Max-Flow Min-Cut method, realized through residual augmentation: the maximum $s$–$t$ flow value equals the minimum $s$–$t$ cut capacity, and the algorithm that achieves it makes "undoing" a first-class move. The defining object is the residual capacity. The naive idea — find a route with unused forward capacity, push the tightest amount, repeat — fails because of irreversibility: a unit sent through a middle diagonal can block two clean parallel routes, and a method that cannot take that unit back will stop short of the optimum while believing it is done. So for a current flow $f$ I allow a step from $u$ to $v$ in two ways at once — push more along a real arc $u \to v$ that is not full, up to $c(u,v) - f(u,v)$, or cancel flow on a real arc $v \to u$ that currently carries some, up to $f(v,u)$ — because both have the same net effect of moving flow from $u$ toward $v$. This gives the residual capacity

$$c_f(u,v) = c(u,v) - f(u,v) + f(v,u),$$

with missing arcs treated as zero. The graph of positive $c_f$ is precisely the graph of changes the flow can still undergo, and a path in it is stronger than an ordinary route because each step may add forward, cancel reverse, or both.

The mechanism is then an alternation between exactly two states, with no uncertified third. If the residual graph has an $s$–$t$ path $P$, I push $F = \min_{(u,v) \in P} c_f(u,v)$ along it, cancelling reverse flow first on each step and then adding any remainder forward. Because $F$ is no larger than the room at any step, every arc flow stays in $[0,c]$; because every interior vertex receives $F$ of net change from its predecessor and passes $F$ to its successor, conservation holds; and the value rises by exactly $F$. So a residual $s$–$t$ path is not a hint — it is a direct certificate that $f$ is not maximum. The decisive case is when no such path exists. Let $S$ be the vertices reachable from $s$ in the residual graph; then $t \notin S$, so $(S,T)$ is a genuine cut, and no residual edge leaves $S$. A forward arc $u \to v$ with $u \in S$, $v \in T$ must be saturated, for otherwise its unused forward capacity would make $v$ reachable; a backward arc $v \to u$ across the same cut must carry zero flow, for otherwise the cancelling residual step $u \to v$ would make $v$ reachable. These are precisely the two equality conditions extracted from weak duality, now forced by the absence of a path, so $|f| = \mathrm{cap}(S,T)$ and the same failed search proves both that the flow cannot be augmented and that the cut cannot be cheaper. The certificate is not bolted on afterward; it is the exact shape left behind when augmentation becomes impossible.

Two further choices make the method finite and efficient. With integer capacities, starting from the zero flow keeps every residual capacity integral, each augmentation raises the value by at least one unit, and the value is bounded above by any source cut (for instance the capacity of all arcs leaving $s$), so the process terminates with an integral optimum; rational capacities reduce to this by scaling. But arbitrary augmenting paths can still be ruinous — a small middle edge used back and forth yields one-unit gains when a large flow is available, and with irrational capacities careless choices need not terminate at all. So I take shortest residual augmenting paths via breadth-first search, i.e. Edmonds–Karp. Then residual distances from the source never decrease, since any newly created residual edge is the reverse of a just-used shortest-path edge; once an edge is a bottleneck and vanishes, it can return as a bottleneck only after a reverse use has pushed its endpoint distance up by at least two; distances are bounded by $|V|$, so each edge is critical only $O(V)$ times, giving $O(VE)$ augmentations and $O(VE^2)$ total time. The residual graph thus serves as a live record of every local way the flow can still change, undoing included, and augmenting paths and cut certificates meet exactly at the stopping condition — and that meeting is optimality.

```python
from collections import deque


def max_flow_min_cut(cap, s, t):
    """Return (value, flow, S), where S is the source side of a minimum cut.

    cap[u][v] is the capacity of arc u->v. Missing arcs have capacity 0.
    Uses shortest residual augmenting paths, i.e. Edmonds-Karp.
    """
    if s == t:
        raise ValueError("source and sink must be distinct")

    vertices = {s, t}
    for u, nbrs in cap.items():
        vertices.add(u)
        vertices.update(nbrs)

    capacity = {u: dict(cap.get(u, {})) for u in vertices}
    neighbors = {u: set() for u in vertices}
    for u in vertices:
        for v, c_uv in capacity[u].items():
            if c_uv < 0:
                raise ValueError("capacities must be nonnegative")
            neighbors[u].add(v)
            neighbors[v].add(u)

    flow = {u: {v: 0 for v in capacity[u]} for u in vertices}

    def residual(u, v):
        return capacity[u].get(v, 0) - flow[u].get(v, 0) + flow[v].get(u, 0)

    def bfs():
        parent = {s: None}
        q = deque([s])
        while q:
            u = q.popleft()
            for v in neighbors[u]:
                if v not in parent and residual(u, v) > 0:
                    parent[v] = u
                    if v == t:
                        return parent
                    q.append(v)
        return None

    def augment(parent):
        amount = float("inf")
        v = t
        while parent[v] is not None:
            u = parent[v]
            amount = min(amount, residual(u, v))
            v = u

        v = t
        while parent[v] is not None:
            u = parent[v]
            cancel = min(amount, flow[v].get(u, 0))
            if cancel:
                flow[v][u] -= cancel
            forward = amount - cancel
            if forward:
                flow[u][v] += forward
            v = u
        return amount

    value = 0
    while True:
        parent = bfs()
        if parent is None:
            break
        value += augment(parent)

    S = {s}
    q = deque([s])
    while q:
        u = q.popleft()
        for v in neighbors[u]:
            if v not in S and residual(u, v) > 0:
                S.add(v)
                q.append(v)

    return value, {u: dict(flow[u]) for u in vertices if flow[u]}, S
```
