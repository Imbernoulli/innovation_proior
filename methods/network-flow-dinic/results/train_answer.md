I present Dinic's algorithm, the canonical strongly-polynomial method for computing a maximum s-t flow in a directed network with nonnegative capacities. The problem is to route as much material as possible from a source vertex s to a sink vertex t without exceeding any arc capacity and without letting material accumulate at intermediate vertices. A feasible flow assigns a nonnegative value to each directed arc, respects the capacity bound on every arc, and conserves flow at every vertex other than s and t. The value of the flow is the net amount leaving s, which by conservation equals the net amount entering t.

The central obstacle is that a greedy sequence of augmenting paths can be inefficient. Pushing one path at a time is correct, but it may repeatedly touch the same small bottleneck and take many iterations. Dinic's insight is to augment many paths simultaneously by working in a level graph and finding a blocking flow in each phase. A level graph is the subgraph of the residual graph that contains only edges that participate in shortest residual paths from s. Vertices are grouped by their BFS distance from s, and every remaining edge points from one level to the next. A blocking flow in this level graph is a flow that saturates at least one edge on every s-t path within the level graph, so after the phase no shortest augmenting path of the current length remains.

The residual graph is what makes undoing previous routing decisions possible. For an arc u -> v with capacity c and current flow f, the residual capacity from u to v is the unused forward capacity c - f(u,v) plus the amount of flow currently on the reverse arc v -> u that can be canceled. Equivalently, a residual edge from u to v exists whenever material can be moved from u toward v by either adding forward flow or removing backward flow. Any s-t path in the residual graph is an augmenting path, and pushing the minimum residual capacity along it strictly increases the flow value while preserving feasibility and conservation.

Dinic's algorithm repeats two steps until the sink is unreachable from the source in the residual graph. First, it builds the level graph by a breadth-first search from s in the residual graph. If t is not reached, the algorithm stops. The set of vertices reachable from s in the residual graph is then the source side of a minimum s-t cut; every forward arc across the cut is saturated and every backward arc across the cut carries zero flow, so the max-flow min-cut theorem certifies optimality. Second, if t is reached, the algorithm sends a blocking flow through the level graph. This is done by depth-first search with a current-edge pointer: each search tries to push flow from s to t along level-respecting residual edges, backtracks when it hits a dead end, and skips edges that have become saturated. The current-edge pointer ensures that an edge examined and found saturated is not reconsidered in the same phase, so the blocking flow step is efficient.

The reason this converges quickly is that each phase increases the shortest residual distance from s to t. After a blocking flow is found, every s-t path in the residual graph must use at least one edge that is not in the previous level graph, which means its length is strictly larger. Since the length of any simple path is at most the number of vertices, there are at most O(V) phases. With an adjacency-list representation and current-edge pointers, each phase runs in O(V E) time for general capacities, giving an overall bound of O(V^2 E). For unit capacities and bipartite matching instances the bound improves to O(E sqrt(V)), which is why Dinic's algorithm is the standard choice for maximum matching in bipartite graphs.

The method is constructive and self-certifying. It returns not only the maximum flow value but also the final residual reachability set, which defines a minimum cut. This pair of objects satisfies the tight case of the max-flow min-cut inequality: every forward arc across the cut is saturated, so the flow value equals the cut capacity, and no larger flow or smaller cut is possible. The algorithm also preserves integrality automatically when capacities are integers, because it only ever adds or subtracts integer amounts along augmenting paths.

```python
from collections import deque


class Edge:
    __slots__ = ("to", "cap", "rev", "orig")

    def __init__(self, to, cap, rev):
        self.to = to
        self.cap = cap
        self.rev = rev
        self.orig = cap


class Dinic:
    def __init__(self, n, source, sink):
        self.n = n
        self.s = source
        self.t = sink
        self.graph = [[] for _ in range(n)]

    def add_edge(self, fr, to, cap):
        fwd = Edge(to, cap, len(self.graph[to]))
        rev = Edge(fr, 0, len(self.graph[fr]))
        self.graph[fr].append(fwd)
        self.graph[to].append(rev)

    def bfs(self):
        self.level = [-1] * self.n
        q = deque([self.s])
        self.level[self.s] = 0
        while q:
            v = q.popleft()
            for e in self.graph[v]:
                if e.cap > 0 and self.level[e.to] < 0:
                    self.level[e.to] = self.level[v] + 1
                    q.append(e.to)
        return self.level[self.t] >= 0

    def dfs(self, v, f):
        if v == self.t:
            return f
        for i in range(self.it[v], len(self.graph[v])):
            self.it[v] = i
            e = self.graph[v][i]
            if e.cap > 0 and self.level[v] + 1 == self.level[e.to]:
                d = self.dfs(e.to, min(f, e.cap))
                if d > 0:
                    e.cap -= d
                    self.graph[e.to][e.rev].cap += d
                    return d
        return 0

    def max_flow(self):
        flow = 0
        INF = 10 ** 18
        while self.bfs():
            self.it = [0] * self.n
            while True:
                pushed = self.dfs(self.s, INF)
                if pushed == 0:
                    break
                flow += pushed
        return flow

    def min_cut_source_side(self):
        # Call only after max_flow has terminated.
        visited = [False] * self.n
        q = deque([self.s])
        visited[self.s] = True
        while q:
            v = q.popleft()
            for e in self.graph[v]:
                if e.cap > 0 and not visited[e.to]:
                    visited[e.to] = True
                    q.append(e.to)
        return {v for v in range(self.n) if visited[v]}


if __name__ == "__main__":
    # Network that needs rerouting: the middle edge 1->2 can trap flow
    # unless the algorithm can cancel earlier choices via residual edges.
    dinic = Dinic(6, 0, 5)
    edges = [
        (0, 1, 16),
        (0, 2, 13),
        (1, 2, 10),
        (2, 1, 4),
        (1, 3, 12),
        (3, 2, 9),
        (2, 4, 14),
        (4, 3, 7),
        (3, 5, 20),
        (4, 5, 4),
    ]
    for u, v, c in edges:
        dinic.add_edge(u, v, c)

    value = dinic.max_flow()
    source_side = dinic.min_cut_source_side()
    sink_side = set(range(6)) - source_side
    cut_capacity = sum(
        e.orig for u in source_side for e in dinic.graph[u]
        if e.to in sink_side
    )
    print("max flow:", value)
    print("min-cut source side:", source_side)
    print("min-cut capacity:", cut_capacity)
    assert value == 23
    assert value == cut_capacity
```

Dinic's algorithm is the method I would recommend whenever a maximum flow or a minimum s-t cut is needed and the network is not so large that a more specialized push-relabel implementation is required. Its combination of level graphs and blocking flows turns the one-path-at-a-time bottleneck into a phase-by-phase expansion of the shortest residual structure, giving both a fast practical algorithm and a transparent proof of correctness through the max-flow min-cut theorem.
