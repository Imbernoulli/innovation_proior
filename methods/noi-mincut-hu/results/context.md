## Problem

There are $n$ projects. Project $i$ yields integer profit $p_i$, which may be negative (for
example, a project might require purchasing a machine, so undertaking it costs money rather
than earning it). Some projects depend on others: a project may be selected only if **all**
of its prerequisites are also selected. (Prerequisites may chain, and a project may be a
prerequisite of several others.)

Choose a subset of the projects, respecting every prerequisite, so as to maximize the total
profit $\sum_{i \in \text{chosen}} p_i$.

## Code framework

A standard integer Dinic max-flow primitive is available off the shelf. It supports adding
directed capacitated edges, computing `max_flow(s, t)`, and reading vertices reachable through
positive residual capacity from any start vertex.

```python
from collections import deque


class Dinic:
    """Dinic max-flow on integer capacities. Edges live in a flat list; the
    forward edge 2k and its reverse 2k+1 are paired, so reverse(i) = i ^ 1."""

    def __init__(self, n):
        self.n = n
        self.to = []
        self.cap = []
        self.head = [[] for _ in range(n)]   # head[u] = edge indices out of u

    def add_edge(self, u, v, c):
        self.head[u].append(len(self.to)); self.to.append(v); self.cap.append(c)
        self.head[v].append(len(self.to)); self.to.append(u); self.cap.append(0)

    def _bfs(self, s, t):                     # build the level graph
        self.dep = [-1] * self.n
        self.dep[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for i in self.head[u]:
                v = self.to[i]
                if self.cap[i] > 0 and self.dep[v] == -1:
                    self.dep[v] = self.dep[u] + 1
                    q.append(v)
        return self.dep[t] != -1

    def _dfs(self, u, t, f):                  # push blocking flow, level k -> k+1
        if u == t:
            return f
        pushed = 0
        while self.it[u] < len(self.head[u]):
            i = self.head[u][self.it[u]]
            v = self.to[i]
            if self.cap[i] > 0 and self.dep[v] == self.dep[u] + 1:
                d = self._dfs(v, t, min(f - pushed, self.cap[i]))
                if d > 0:
                    self.cap[i] -= d
                    self.cap[i ^ 1] += d       # cancel on the paired reverse arc
                    pushed += d
                    if pushed == f:
                        return pushed
            self.it[u] += 1                    # current-arc: don't revisit dead arcs
        return pushed

    def max_flow(self, s, t):
        flow = 0
        while self._bfs(s, t):
            self.it = [0] * self.n
            flow += self._dfs(s, t, float("inf"))
        return flow

    def reachable_from(self, start):           # residual-positive reachability
        seen = [False] * self.n
        seen[start] = True
        q = deque([start])
        while q:
            u = q.popleft()
            for i in self.head[u]:
                v = self.to[i]
                if self.cap[i] > 0 and not seen[v]:
                    seen[v] = True
                    q.append(v)
        return seen


def max_weight_closure(profit, prereq):
    """profit[i] = profit of project i (may be negative).
       prereq = list of (u, v): selecting u requires selecting v.
       Returns (best_total_profit, list of selected projects)."""
    # TODO: complete this function body.
    raise NotImplementedError


if __name__ == "__main__":
    # weights 5, -6, 7, 0, -3; edges 1->2, 1->4, 3->4, 2->5, 4->5 (0-indexed).
    profit = [5, -6, 7, 0, -3]
    prereq = [(0, 1), (0, 3), (2, 3), (1, 4), (3, 4)]
    print(max_weight_closure(profit, prereq))
```
