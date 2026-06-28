We are given a directed network with a source $s$, a sink $t$, and a nonnegative capacity $c(u,v)$ on each arc, and we want two things at once: the largest steady traffic that can be pushed from $s$ to $t$ without overloading any link or piling up flow at intermediate vertices, and the cheapest set of links whose removal disconnects $s$ from $t$. A feasible flow assigns $0 \le f(u,v) \le c(u,v)$ to each arc, conserves flow at every internal vertex, and has value $|f| = \sum_v f(s,v) - \sum_v f(v,s)$. A separating cut is a partition $(S,T)$ with $s \in S$, $t \in T$, and capacity $\mathrm{cap}(S,T) = \sum_{u \in S,\, v \in T} c(u,v)$, counting only forward-crossing arcs. The trouble is that the maximizing question (pack as much flow as possible) and the minimizing question (find the smallest bottleneck) look related but not obviously equal, and a useful method must produce both a large flow and a small separating certificate so that the claimed optimum is checkable directly on the network.

The existing options each fall short of this. Casting the problem as a linear program and running simplex is legitimate but treats the graph as a faceless pile of equations and gives no network-level reason why the answer is final. Greedy flooding — push along available routes, return excess, repeat — is easy to perform on a map but cannot certify: a greedy early route can lock up an arc that a better global routing needs, and a procedure that only ever adds flow has no principled way to retract a bad placement. Route decomposition, viewing a flow as a sum of path flows, makes existence and convexity visible but never tells a planner which paths to pick, when to stop, or how to read off a minimal separating set. What none of these supply is the one thing the problem demands: a stopping rule that simultaneously proves the flow cannot grow and exhibits a cut that cannot shrink.

The lever is the only inequality I can trust. Summing the net outflow over all vertices of the source side $S$, conservation erases every internal vertex and every arc with both ends in $S$ cancels, leaving $|f| = \sum_{S \to T} f - \sum_{T \to S} f$; dropping the nonnegative backward term and bounding forward flow by capacity gives $|f| \le \mathrm{cap}(S,T)$. This weak duality is more than a bound — it pins down exactly what equality requires. The first inequality is tight only when every backward arc from $T$ into $S$ carries zero flow, and the second only when every forward arc from $S$ into $T$ is saturated. So if I can ever manufacture a cut that is full forward and idle backward, the flow value and cut capacity coincide and neither object can be improved. The whole task becomes: build a situation where some cut is saturated in one direction and empty in the other.

What I propose is the Max-Flow Min-Cut method, realized through residual augmentation: the maximum $s$–$t$ flow value equals the minimum $s$–$t$ cut capacity, and the algorithm that achieves it makes "undoing" a first-class move. The defining object is the residual capacity. The naive idea — find a route with unused forward capacity, push the tightest amount, repeat — fails because of irreversibility: a unit sent through a middle diagonal can block two clean parallel routes, and a method that cannot take that unit back will stop short of the optimum while believing it is done. So for a current flow $f$ I allow a step from $u$ to $v$ in two ways at once — push more along a real arc $u \to v$ that is not full, up to $c(u,v) - f(u,v)$, or cancel flow on a real arc $v \to u$ that currently carries some, up to $f(v,u)$ — because both have the same net effect of moving flow from $u$ toward $v$. This gives the residual capacity

$$c_f(u,v) = c(u,v) - f(u,v) + f(v,u),$$

with missing arcs treated as zero. The graph of positive $c_f$ is precisely the graph of changes the flow can still undergo, and a path in it is stronger than an ordinary route because each step may add forward, cancel reverse, or both.

The mechanism is then an alternation between exactly two states, with no uncertified third. If the residual graph has an $s$–$t$ path $P$, I push $F = \min_{(u,v) \in P} c_f(u,v)$ along it, cancelling reverse flow first on each step and then adding any remainder forward. Because $F$ is no larger than the room at any step, every arc flow stays in $[0,c]$; because every interior vertex receives $F$ of net change from its predecessor and passes $F$ to its successor, conservation holds; and the value rises by exactly $F$. So a residual $s$–$t$ path is not a hint — it is a direct certificate that $f$ is not maximum. The decisive case is when no such path exists. Let $S$ be the vertices reachable from $s$ in the residual graph; then $t \notin S$, so $(S,T)$ is a genuine cut, and no residual edge leaves $S$. A forward arc $u \to v$ with $u \in S$, $v \in T$ must be saturated, for otherwise its unused forward capacity would make $v$ reachable; a backward arc $v \to u$ across the same cut must carry zero flow, for otherwise the cancelling residual step $u \to v$ would make $v$ reachable. These are precisely the two equality conditions extracted from weak duality, now forced by the absence of a path, so $|f| = \mathrm{cap}(S,T)$ and the same failed search proves both that the flow cannot be augmented and that the cut cannot be cheaper. The certificate is not bolted on afterward; it is the exact shape left behind when augmentation becomes impossible.

Two further choices make the method finite and efficient. With integer capacities, starting from the zero flow keeps every residual capacity integral, each augmentation raises the value by at least one unit, and the value is bounded above by any source cut (for instance the capacity of all arcs leaving $s$), so the process terminates with an integral optimum; rational capacities reduce to this by scaling. But arbitrary augmenting paths can still be ruinous — a small middle edge used back and forth yields one-unit gains when a large flow is available, and with irrational capacities careless choices need not terminate at all. So I take shortest residual augmenting paths via breadth-first search, i.e. Edmonds–Karp. Then residual distances from the source never decrease, since any newly created residual edge is the reverse of a just-used shortest-path edge; once an edge is a bottleneck and vanishes, it can return as a bottleneck only after a reverse use has pushed its endpoint distance up by at least two; distances are bounded by $|V|$, so each edge is critical only $O(V)$ times, giving $O(VE)$ augmentations and $O(VE^2)$ total time. The residual graph thus serves as a live record of every local way the flow can still change, undoing included, and augmenting paths and cut certificates meet exactly at the stopping condition — and that meeting is optimality.

Concretely the landing is a single self-contained C++17 program. It reads from stdin `n m s t` followed by `m` lines `u v c` (1-indexed vertices, arc `u->v` with capacity `c`), and prints the maximum `s-t` flow value followed by the source side `S` of a minimum cut. The residual graph is stored as a flat edge list where each arc and its reverse occupy adjacent indices, so the reverse of edge `e` is `e^1`; flow values accumulate in `long long` to stay clear of overflow.

```cpp
// Max-flow / min-cut via Edmonds-Karp (shortest residual augmenting paths).
// Reads from stdin: "n m s t" then m lines "u v c" (1-indexed vertices, arc u->v
// with capacity c). Prints the maximum s-t flow value, then the source side S of
// a minimum cut as a sorted vertex list. Use long long: total flow can exceed int.
#include <bits/stdc++.h>
using namespace std;

const long long INF = (long long)4e18;

int n, m, s, t;
vector<int> head;        // head[u] = first edge index out of u, or -1
vector<int> to_, nxt;    // adjacency: endpoint and next-edge pointers
vector<long long> cap;   // residual capacity of each directed edge

void add_edge(int u, int v, long long c) {
    // Forward edge with capacity c, reverse edge with capacity 0; they are
    // stored adjacently so the reverse of edge e is e^1.
    to_.push_back(v); cap.push_back(c); nxt.push_back(head[u]); head[u] = (int)to_.size() - 1;
    to_.push_back(u); cap.push_back(0); nxt.push_back(head[v]); head[v] = (int)to_.size() - 1;
}

// One BFS for a shortest residual s-t path; returns the bottleneck pushed, or 0.
long long bfs_augment() {
    vector<int> parent_edge(n + 1, -1);
    vector<char> seen(n + 1, 0);
    queue<int> q;
    q.push(s); seen[s] = 1;
    while (!q.empty()) {
        int u = q.front(); q.pop();
        if (u == t) break;
        for (int e = head[u]; e != -1; e = nxt[e]) {
            int v = to_[e];
            if (!seen[v] && cap[e] > 0) {
                seen[v] = 1;
                parent_edge[v] = e;
                q.push(v);
            }
        }
    }
    if (!seen[t]) return 0;
    long long f = INF;
    for (int v = t; v != s; ) {
        int e = parent_edge[v];
        f = min(f, cap[e]);
        v = to_[e ^ 1];
    }
    for (int v = t; v != s; ) {
        int e = parent_edge[v];
        cap[e] -= f;
        cap[e ^ 1] += f;
        v = to_[e ^ 1];
    }
    return f;
}

int main() {
    if (scanf("%d %d %d %d", &n, &m, &s, &t) != 4) return 0;
    head.assign(n + 1, -1);
    for (int i = 0; i < m; ++i) {
        int u, v; long long c;
        scanf("%d %d %lld", &u, &v, &c);
        add_edge(u, v, c);
    }

    long long value = 0;
    while (true) {
        long long f = bfs_augment();
        if (f == 0) break;
        value += f;
    }

    // Source side S of a minimum cut: vertices still reachable from s in the
    // final residual graph. By max-flow min-cut its forward-crossing capacity
    // equals the flow value.
    vector<char> inS(n + 1, 0);
    queue<int> q;
    q.push(s); inS[s] = 1;
    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int e = head[u]; e != -1; e = nxt[e]) {
            int v = to_[e];
            if (!inS[v] && cap[e] > 0) { inS[v] = 1; q.push(v); }
        }
    }

    printf("%lld\n", value);
    bool first = true;
    for (int v = 1; v <= n; ++v) {
        if (inS[v]) { printf(first ? "%d" : " %d", v); first = false; }
    }
    printf("\n");
    return 0;
}
```
