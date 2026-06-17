# Maximum-Weight Closure / Project Selection via Minimum Cut

## Problem

A directed graph $G=(V,E)$ has an integer weight $w_v$ on each vertex (possibly negative). A
**closure** is a vertex set $V_1 \subseteq V$ with no edge leaving it: for every edge
$u \to v$, if $u \in V_1$ then $v \in V_1$. The **maximum-weight closure** maximizes
$\sum_{v \in V_1} w_v$.

This models *project selection*: vertex = project, $w_v = p_v$ = its profit, edge $u \to v$
= "project $u$ requires project $v$." A feasible selection is exactly a closure, and the
maximum-weight closure is the most profitable feasible set of projects.

## Key idea

Reduce to a minimum $s$–$t$ cut. Build a network $N$:

- Add a source $s$ and sink $t$.
- Every positive vertex $v$ ($w_v>0$): edge $s \to v$ with capacity $w_v$.
- Every negative vertex $v$ ($w_v<0$): edge $v \to t$ with capacity $-w_v$.
- Every original edge $u \to v$: capacity `INF`, any integer strictly larger than
  $\sum_v |w_v|$.

A cut $[S,T]$ is **simple** if it cuts only $s$- and $t$-edges (no original edge).

**Lemma 1 (min cut is simple).** Cutting $s$ alone from the rest gives a finite cut of
capacity $\sum_{v:w_v>0} w_v \le \sum_v |w_v|$. Any cut using an `INF` edge has capacity
strictly larger. So the minimum cut never cuts an original edge — it is simple.

**Lemma 2 (bijection).** Simple cuts and closures correspond via $S = V_1 \cup \{s\}$. If
$V_1$ is a closure, no edge leaves $V_1$, so $[V_1\cup\{s\},\,V_2\cup\{t\}]$ cuts no original
edge (simple). Conversely if $[S,T]$ is simple and $V_1 = S\setminus\{s\}$, then for any
$u\in V_1$ and edge $u\to v$, that edge is uncut, so $v\in S$, i.e. $v\in V_1$ — $V_1$ is a
closure.

**Quantification.** For a simple cut, the only cut edges out of $s$ go to positive vertices
that were not selected, and the only cut edges into $t$ come from negative vertices that were
selected. Writing those sets as $V_2^+$ and $V_1^-$,

$$c[S,T] = \sum_{v\in V_2^+} w_v + \sum_{v\in V_1^-}(-w_v).$$

The closure's weight is $w(V_1) = \sum_{v\in V_1^+} w_v - \sum_{v\in V_1^-}(-w_v)$. Adding,
the $V_1^-$ terms cancel and the positive terms combine over all positive vertices:

$$w(V_1) + c[S,T] = \sum_{v\in V^+} w_v \quad(\text{constant}),
\qquad\Longrightarrow\qquad
w(V_1) = \sum_{v\in V^+} w_v - c[S,T].$$

**Optimality.** $\sum_{v\in V^+} w_v$ is fixed, so maximizing $w(V_1)$ equals minimizing the
simple-cut capacity; by Lemma 1 that is the global minimum cut. By max-flow–min-cut,

$$\text{max-weight closure} = \Big(\sum_{v:w_v>0} w_v\Big) - (\text{min cut})
= \Big(\sum_{v:w_v>0} w_v\Big) - (\text{max flow}).$$

The selected set is recovered as the residual-reachable vertices from $s$ (the source side
$S$); the `INF` edges guarantee this set is a valid closure.

## Algorithm

1. Build $N$ as above; let $P = \sum_{v:w_v>0} w_v$.
2. Compute a maximum $s$–$t$ flow (Dinic). Its value is the minimum cut $c$.
3. Answer profit $= P - c$.
4. BFS from $s$ over residual-positive edges; the reached original vertices are the optimal
   selection.

Complexity: $O(\mathrm{MaxFlow}(N))$ — one max-flow on $|V|+2$ vertices and
$|E| + |V|$ edges.

## Code

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
    n = len(profit)
    s, t = n, n + 1
    g = Dinic(n + 2)
    INF = sum(abs(w) for w in profit) + 1      # strictly bigger than any finite cut

    pos_sum = 0
    for i, w in enumerate(profit):
        if w > 0:
            g.add_edge(s, i, w)                # s -> profitable project, cap = profit
            pos_sum += w
        elif w < 0:
            g.add_edge(i, t, -w)               # loss-making project -> t, cap = |profit|

    for u, v in prereq:
        g.add_edge(u, v, INF)                  # prerequisite: never worth cutting

    cut = g.max_flow(s, t)                     # max-flow value = min-cut capacity
    S = g.reachable_from(s)
    selected = [i for i in range(n) if S[i]]   # chosen = source side, minus s
    return pos_sum - cut, selected


if __name__ == "__main__":
    # weights 5, -6, 7, 0, -3; edges 1->2, 1->4, 3->4, 2->5, 4->5 (0-indexed).
    profit = [5, -6, 7, 0, -3]
    prereq = [(0, 1), (0, 3), (2, 3), (1, 4), (3, 4)]
    print(max_weight_closure(profit, prereq))  # (4, [2, 3, 4])
```

## Why max-flow = min-cut gives the optimum

By the max-flow–min-cut theorem the value of a maximum $s$–$t$ flow equals the capacity of a
minimum $s$–$t$ cut. The minimum cut is simple (Lemma 1), hence corresponds bijectively to a
feasible selection (Lemma 2), and selection profit $= P - c[S,T]$ with $P$ constant. So
minimizing the cut maximizes the profit, and a single max-flow computation yields both the
optimal value ($P - \text{flow}$) and — via residual reachability from $s$ — the set of
projects to undertake. A prerequisite edge has capacity larger than the whole possible finite
flow, so any such edge leaving a residual-reachable vertex still has residual capacity; the
recovered set is always a valid closure (every selected project's prerequisites are selected
too).

## A concrete instance: relay-station selection

$n$ candidate relay stations; building station $i$ costs $p_i$. There are $m$ user groups;
group $j$ uses stations $a_j,b_j$ and, if both are built, yields revenue $c_j$. Net profit =
revenue − cost. Model each user group $j$ as a vertex of weight $+c_j$ with two prerequisite
edges $j \to a_j$ and $j \to b_j$; model each station $i$ as a vertex of weight $-p_i$. The
maximum-weight closure of this graph is the maximum net profit — selecting a user group forces
building both of its stations, exactly the prerequisite structure.
