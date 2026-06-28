OK, let me think this through from scratch. I have $n$ projects. Each project $i$ comes with a number $p_i$, its profit, and that number can be negative — some "projects" are really costs, like a machine I have to buy. And there are prerequisites: I can take project $i$ only if I've also taken everything it depends on. I want a subset that respects all the prerequisites and maximizes the sum of the $p_i$ over the subset.

My first instinct is the obvious greedy: keep every project with $p_i > 0$, drop every project with $p_i < 0$. But the prerequisites wreck that immediately. Suppose project $A$ earns $+100$ but depends on $B$, which costs $-30$, and $B$ in turn depends on $C$, costing $-40$. If I want $A$ I'm forced to drag in $B$ and $C$, and the bundle is worth $100 - 30 - 40 = +30$ — still good, so I take all three. But change $B$ to $-80$: now the bundle is $100 - 80 - 40 = -20$, and I should take *none* of them, even though $A$ on its own looks fantastic. So whether I keep a positive project depends on the whole chain of costs hanging off it.

And it goes the other way too. A costly project might be worth keeping. Say $D$ costs $-50$, but three separate profitable projects $E, F, G$ (each $+30$) all require $D$. Individually $D$ is a loss, but it unlocks $90$ of profit, so I'd happily eat the $-50$. So I can't decide $D$ in isolation either — its value depends on who's sitting on top of it. The decisions are coupled through the prerequisite edges in both directions. There's no safe local rule.

Could I just enumerate? Every project is in or out, so $2^n$ subsets, check feasibility, take the best. Correct but hopeless for any real $n$. I need structure.

Let me look hard at what a feasible subset actually *is*. Draw the prerequisites as a directed graph: put an edge $u \to v$ meaning "$u$ requires $v$." Then "I picked a feasible subset" means: for every project $u$ I picked, every project $v$ that $u$ points to is also picked. In words, the chosen set has no edge leaving it — every out-neighbor of a chosen vertex is itself chosen. A set with that property has a name in graph theory: it's a *closure* of the directed graph. (Equivalently: $u$ chosen $\Rightarrow v$ chosen for every edge $u \to v$; it's exactly an "implies" relation.) Note the graph need not be acyclic — two projects could mutually depend on each other, and a closure handles that fine; cycles just have to be taken whole.

So I've turned the project problem into a clean combinatorial one: among all closures of this directed graph, find the one whose vertex weights sum to the maximum. Maximum-weight closure. The reframing didn't solve anything yet, but it stripped the problem down to its bones — a weighted graph and a "no edge may leave the chosen set" constraint.

Now, what makes this hard is the mix of signs. If all weights were positive, take everything (the whole vertex set is trivially a closure). If all negative, take nothing. The tension is entirely between positive vertices I'd love to grab and the negative vertices they force me to grab. So really I'm partitioning the vertices into two groups: $V_1$ = chosen, $V_2$ = not chosen. And the closure condition is a statement purely about *which edges cross between the two groups*: no edge may go from $V_1$ to $V_2$. Edges inside $V_1$, edges inside $V_2$, and even edges from $V_2$ to $V_1$ are all fine — only the $V_1 \to V_2$ direction is forbidden.

"Partition the vertices into two sides, and forbid (or penalize) edges crossing in a particular direction." Where have I seen that before? In a flow network you split the vertices into a source side $S$ and a sink side $T$, and the cut capacity is the total capacity of edges crossing from $S$ to $T$ — directional, exactly like my forbidden $V_1 \to V_2$ crossing. That's enough of a structural match to be worth chasing: if I can arrange a network where the source side plays the role of "chosen," and where the cut capacity measures exactly the profit I'm giving up by my choice, then the cheapest cut would be the best selection. I don't know yet whether the capacities can be made to line up that cleanly — that's the thing I have to actually build and check, not assume.

I'll add two new vertices: a source $s$ and a sink $t$. I want $S$ (the source side of the cut) to be $V_1 \cup \{s\}$ — the chosen projects plus the source — and $T = V_2 \cup \{t\}$. So $s$ should end up grouped with the chosen vertices and $t$ with the unchosen ones.

How do I make cutting an edge *cost* the right thing? Two pressures act on a vertex. A positive vertex $v$ ($w_v > 0$, writing $w_v = p_v$) I would like to keep — to put on the $S$ side. If I fail to, I lose its profit $w_v$. That says to draw an edge $s \to v$ with capacity $w_v$: when $v$ ends up on the $T$ side, this edge crosses from $S$ to $T$ and I pay exactly the profit forgone by not taking it. A negative vertex $v$ ($w_v < 0$) I would like to drop — to put on the $T$ side. If I'm forced to keep it, I pay $|w_v|$. That says to draw an edge $v \to t$ with capacity $-w_v = |w_v|$: when $v$ ends up on the $S$ side, this edge crosses from $S$ to $T$ and I pay exactly the cost incurred by taking it.

Those two edge types make the cut pay for "missed profit" and "incurred cost." Now I still have to *enforce* the closure constraint — the forbidden $V_1 \to V_2$ edges. What about the original prerequisite edges $u \to v$? In the cut picture, a prerequisite $u \to v$ being violated means $u$ is chosen ($u \in S$) but $v$ is not ($v \in T$) — that's precisely an edge crossing from $S$ to $T$. I must make such a crossing impossible, or at least never worth it. So I keep every original edge $u \to v$ in the network and give it a capacity `INF` that is larger than any finite cut I can construct. Then any cut that puts $u \in S$ and $v \in T$ would have to cut this $u \to v$ edge and pay too much — it could never be the minimum.

Let me pin down what `INF` needs to be in code. It just has to exceed any cut value I'd ever consider. The source-only cut has cost $\sum_{v:w_v>0} w_v$, and that is at most $\sum_v |w_v|$. So define `INF` as any integer strictly larger than $\sum_{v \in V} |w_v|$, for example `sum(abs(w) for w in profit) + 1`. Then no minimum cut will ever touch a prerequisite edge.

Now let me make sure this whole thing is airtight, because it's easy to wave hands here. Call a cut $[S,T]$ a *simple cut* if every edge it cuts is incident to either $s$ or $t$ — i.e. it doesn't cut any of the original prerequisite edges.

First, is the minimum cut always simple? The edges of the network come in exactly two flavors: the finite ones, which all touch $s$ or $t$ (capacities $w_v$ or $|w_v|$), and the huge prerequisite edges, which are the original edges and touch neither $s$ nor $t$. Here's a specific cut with finite cost: put $s$ alone on the source side, everything else on the sink side. Then the only cut edges are the $s \to v$ edges to positive vertices, total capacity $\sum_{v: w_v>0} w_v \le \sum_v |w_v|$. So a finite cut exists, and the minimum cut's capacity is at most $\sum_v |w_v|$. But any cut that includes even one prerequisite edge pays `INF`, strictly larger. Therefore the minimum cut includes no prerequisite edge — it is simple. That's what the `INF` choice buys: it doesn't just discourage illegal cuts, it rules them out of the minimum, and the inequality $\sum_v|w_v| < \text{INF}$ is exactly the slack that makes the "rules them out" step go through.

Second — and this is the part that has to be a genuine bijection, not a vibe — every simple cut corresponds to exactly one closure, and vice versa, via $V_1 \cup \{s\} = S$. Let me check both directions.

Take a closure $V_1$ and set $S = V_1 \cup \{s\}$, $T = V_2 \cup \{t\}$ (where $V_2$ is everything not in $V_1$). Is $[S,T]$ simple? Suppose not — suppose it cuts some original edge $u \to v$, meaning $u \in S \setminus \{s\} = V_1$ and $v \in T \setminus \{t\} = V_2$. But then $u$ is in the closure and points to $v$ outside it, contradicting that $V_1$ is a closure. So no original edge is cut: $[S,T]$ is simple.

Conversely, take any simple cut $[S,T]$ and set $V_1 = S \setminus \{s\}$. Is $V_1$ a closure? Take any $u \in V_1$ and any original edge $u \to v$. Because the cut is simple, it cuts no original edge, so this $u \to v$ does *not* cross from $S$ to $T$; since $u \in S$, that forces $v \in S$, hence $v \in S \setminus \{s\} = V_1$ (it can't be $s$ since $s$ has no incoming original edges). So every out-neighbor of $u$ is in $V_1$ — $V_1$ is a closure. The correspondence is a clean bijection.

Now the arithmetic, which is the whole point — I need the cut capacity to track the profit. Take a simple cut $[S,T]$ with its closure $V_1$, $S = V_1 \cup \{s\}$. Split the cut edges by what they touch. A simple cut only cuts $s$-edges and $t$-edges, so $[S,T] = [\{s\}, V_2] \cup [V_1, \{t\}]$ (there are no cut edges of the third, $V_1 \to V_2$, type — that's simplicity). The source $s$ only has edges to positive vertices, so the cut $s$-edges go exactly to the positive vertices on the $T$ side; call that set $V_2^+$ (positive vertices in $V_2$). Similarly $t$ only receives edges from negative vertices, so the cut $t$-edges come exactly from the negative vertices on the $S$ side, $V_1^-$. Therefore

$$c[S,T] = \sum_{v \in V_2^+} w_v + \sum_{v \in V_1^-} (-w_v).$$

In words: the cut pays for the profit of every positive project I *didn't* select, plus the cost of every negative project I *was forced to* select. That's a satisfying sentence — it's exactly the two ways my selection can be suboptimal.

And the profit of the selection itself? The closure $V_1$ has weight

$$w(V_1) = \sum_{v \in V_1^+} w_v - \sum_{v \in V_1^-} (-w_v)$$

— the positive weights I kept, minus the absolute cost of the negative ones I kept. Now add the two expressions:

$$w(V_1) + c[S,T] = \Big(\sum_{v \in V_1^+} w_v - \sum_{v \in V_1^-}(-w_v)\Big) + \Big(\sum_{v \in V_2^+} w_v + \sum_{v \in V_1^-}(-w_v)\Big).$$

The two $\sum_{V_1^-}(-w_v)$ terms cancel exactly, and $\sum_{v \in V_1^+} w_v + \sum_{v \in V_2^+} w_v$ is just the sum over *all* positive vertices (every positive vertex is in $V_1$ or $V_2$, nowhere else). So

$$w(V_1) + c[S,T] = \sum_{v \in V^+} w_v,$$

and rearranging,

$$w(V_1) = \sum_{v \in V^+} w_v - c[S,T].$$

There it is. The total of all positive profits, $\sum_{v \in V^+} w_v$, is a *constant* — it doesn't depend on which selection I make. So maximizing the selection profit $w(V_1)$ is identical to *minimizing* the simple-cut capacity $c[S,T]$. And by the first lemma the minimum cut is simple, so minimizing over simple cuts is the same as minimizing over all cuts. The best feasible selection corresponds to the minimum cut of the network, and

$$\boxed{\text{max profit} = \Big(\sum_{v : p_v > 0} p_v\Big) - (\text{min cut}).}$$

That's the model: take all the positive profit for free, then pay the minimum cut to back out of the impossible parts. And the cut measures exactly the unavoidable losses — forgone profits and forced costs — that the prerequisite structure imposes.

To actually compute it I lean on the max-flow–min-cut theorem: the value of a maximum $s$–$t$ flow equals the capacity of a minimum $s$–$t$ cut. So I don't search over cuts at all — I run a max-flow, and its value is my min cut. Then to recover *which* projects to take, I look at the residual network after the flow is maxed: the set of vertices still reachable from $s$ along edges with leftover capacity is exactly the source side $S$ of a minimum cut, and the chosen projects are $S \setminus \{s\}$. (The `INF` prerequisite edges guarantee this $S$ is a legal closure: if $u$ is reachable and $u \to v$ has capacity `INF`, the whole max flow is at most $\sum_v |w_v|$, so that edge still has residual capacity and $v$ is reachable too — the selection respects every prerequisite automatically.)

Before I trust any of this, let me run the whole pipeline on a small instance by hand and then against the brute force, because the derivation has enough moving parts that a sign error or a backwards arrow would slip right past me. Five projects with weights $5, -6, 7, 0, -3$ (I'll index them $0..4$) and prerequisites $0\to1$, $0\to3$, $2\to3$, $1\to4$, $3\to4$. The positive total is $P = 5 + 7 = 12$.

First, what *is* the right answer, computed independently of my construction? There are only $2^5 = 32$ subsets, so I can just enumerate the feasible ones (those with no prerequisite leaving the set) and take the max. Doing that, the best feasible set is $\{2,3,4\}$ with weight $7 + 0 + (-3) = 4$: project $2$ requires $3$, $3$ requires $4$, $4$ requires nothing — self-contained, no edge escapes. The runner-up is interesting and worth noting, because it's the trap: the *entire* vertex set $\{0,1,2,3,4\}$ is trivially a closure (nothing can leave a set that contains everything), and its weight is $5-6+7+0-3 = 3$. So "take all positive projects, swallow whatever they drag in" scores $3$, but the true optimum $4$ comes from *not* taking project $0$ at all — its $+5$ isn't worth the $-6$ that prerequisite $0\to1$ forces. That's precisely the coupling I argued can't be decided locally, now showing up as a concrete $4 > 3$ gap. Good: the brute-force optimum is $4$, on the set $\{2,3,4\}$.

Now does my network reproduce that? Predicted min cut is $P - 4 = 12 - 4 = 8$. Let me verify the cut by hand at the closure $V_1 = \{2,3,4\}$, so $S = \{s,2,3,4\}$ and $T = \{0,1,t\}$. Which finite edges cross $S \to T$? Source edges go to positive vertices $0$ and $2$; of those, $0 \in T$ so $s\to 0$ (capacity $5$) is cut, while $2 \in S$ so $s \to 2$ is not. Sink edges come from negative vertices $1$ and $4$; of those, $1 \in T$ so $1 \to t$ is *not* cut, while $4 \in S$ so $4 \to t$ (capacity $3$) is cut. No prerequisite edge crosses, since $V_1$ is a closure. Total cut $= 5 + 3 = 8$, matching $V_2^+ = \{0\}$ and $V_1^- = \{4\}$ in my formula, and $P - 8 = 4$ recovers the optimum. The arithmetic closes.

But the hand check and the brute force are both *me*, and the place I'm least sure of is the arrow direction: I wrote prerequisite "$u$ requires $v$" as the edge $u \to v$, and the entire cut argument hinges on a violated prerequisite being an $S \to T$ crossing. It would be easy to have this backwards. So let me actually run the code on this instance, and also run the variant where I orient prerequisites the *other* way, $v \to u$, to see which one the brute force blesses. With $u \to v$: the code returns INF $= \sum|w| + 1 = 21 + 1 = 22$, a max flow / min cut of $8$, residual-reachable source side $\{2,3,4\}$ (plus $s$), selection $[2,3,4]$, profit $12 - 8 = 4$ — exactly the brute-force optimum, and the returned set passes the closure check. With the reversed orientation $v \to u$, the same code returns profit $12$ on the set $[0,2]$. That's an immediate red flag: $12$ is the entire positive total, i.e. the network is claiming I can grab every profit at zero cost, and the set $\{0,2\}$ violates $0\to1$, $0\to3$, $2\to3$ — it isn't even feasible. So the reversed arrows quietly compute the wrong problem (it's effectively the closure of the *reverse* graph, "$u$ is required-by $v$"), and the only thing that catches it is checking against an independent answer. The orientation that agrees with brute force is $u \to v$, the one I have. With that pinned down I trust the construction.

For the flow algorithm itself I'll use Dinic's, which finds the max flow in phases: a BFS labels each vertex with its distance from $s$, building a level graph; then a DFS pushes a blocking flow that only ever steps from level $k$ to level $k+1$, with a current-arc pointer so each phase skips arcs it has already exhausted. Repeat until $t$ is unreachable. I'll store edges in a flat array with each forward edge $i$ paired to its reverse $i \oplus 1$, so cancelling flow on the reverse arc is a one-line index flip. After the flow saturates, one more BFS over residual-positive edges from $s$ gives me $S$, and the chosen projects fall out.

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
    return pos_sum - cut, selected             # take all positive profit, pay the min cut


if __name__ == "__main__":
    # weights 5, -6, 7, 0, -3; edges 1->2, 1->4, 3->4, 2->5, 4->5 (0-indexed).
    profit = [5, -6, 7, 0, -3]
    prereq = [(0, 1), (0, 3), (2, 3), (1, 4), (3, 4)]
    print(max_weight_closure(profit, prereq))  # (4, [2, 3, 4])
```

Stepping back over the path: greedy failed because prerequisites couple the keep-or-drop decisions both ways, which pushed me to think of a feasible selection as a two-way partition with no edge leaving the chosen side — a closure. The directional "no $V_1 \to V_2$ edge" matched the shape of an $s$–$t$ cut, and chasing that match, wiring each positive project to $s$ and each negative project to $t$ with prerequisite edges set above any finite cut, made the cut capacity come out as the constant positive total minus the selection profit. The min-cut-is-simple inequality and the cut/closure bijection are what license reading the answer off a single max-flow, and the small instance both confirmed the value ($4 = 12 - 8$) and, via the reversed-orientation experiment, caught the one step where I could most plausibly have gone wrong. The arrow direction is the load-bearing modeling choice; everything downstream is the standard max-flow machinery.
