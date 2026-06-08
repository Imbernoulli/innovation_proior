Let me start from the thing I actually have in front of me, which is a map. A rail network, links between cities, and a number written on each link saying how much traffic it can carry per day. Two cities matter: one where the traffic originates, one where it has to arrive. And the question that came down to me is blunt — how much can move from the one to the other, in steady state, if I'm allowed to split the traffic across every parallel route the map offers and let it recombine downstream?

There's a twin question riding alongside it, and honestly it's the one the people who handed me this care more about. They don't want to move the traffic; they want to *stop* it. They want to know the cheapest set of links to destroy so that nothing at all can get from origin to destination. So I'm holding two problems that point in opposite directions: maximize what crosses, versus block all of it at least cost. I have a feeling these aren't really two problems, but I can't say why yet, so let me build the objects carefully and see.

I need a precise meaning of flow on this thing. I'll make the links directed for now — traffic on a link goes one way at a time — so I have a directed graph, arcs with capacities. A flow is just an assignment of a number `f(a)` to each arc, how much I'm actually sending along it. Two rules. It can't exceed the link's capacity: `f(a) ≤ c(a)`. And at every city that's neither the source nor the sink, nothing piles up and nothing appears from nowhere — whatever comes in goes back out. Conservation: `Σ f(into v) = Σ f(out of v)`. The *value* of the flow is what leaves the source net of what comes back to it, `value(f) = Σ f(out of s) − Σ f(into s)`. And here's a small sanity check I want before I trust the definition: is the value also equal to what arrives at the sink? Sum the net outflow `∂f(v) = Σf(out) − Σf(in)` over *all* vertices. Every arc `u→w` contributes `+f` at `u` and `−f` at `w`, so the grand total is zero. But conservation makes `∂f(v) = 0` at every interior vertex, leaving `∂f(s) + ∂f(t) = 0`. So net out of the source equals net into the sink. Good — the definition is consistent, value is well-defined.

Now the other object, the cut. I want to wall off the source from the sink. Take any set of vertices `S` that contains the source but not the sink; call the rest `T`. The arcs that go *from* `S` *to* `T` are the wall. If I delete them, no path from source to sink survives, because any such path has to step from the `S`-side to the `T`-side somewhere and that step is gone. The cost of this wall is the sum of capacities of those forward-crossing arcs, `cap(S,T) = Σ_{u∈S, v∈T} c(u→v)`. And I notice — write this down — the definition only counts arcs going `S→T`. Arcs that run *back*, `T→S`, are not in the wall and cost nothing. That asymmetry feels arbitrary right now but I'll let it stand and watch it.

Here's the first thing worth staring at. Every unit of flow that gets from source to sink has to cross the wall, going forward, at some point. So intuitively the flow can't be bigger than the wall's forward capacity. Let me make that exact, because if it's exact for *every* wall, it's a strong statement. Take any flow `f` and any cut `(S,T)`. The value is `∂f(s)`. But `∂f(v) = 0` for every interior vertex, and in particular for every vertex of `S` except the source. So I lose nothing by adding all those zeros:
`value(f) = ∂f(s) = Σ_{v∈S} ∂f(v) = Σ_{v∈S} ( Σ f(v→w) − Σ f(u→v) )`.
Now look at an arc with both endpoints inside `S`, say `x→y` with `x,y ∈ S`. It shows up twice: once as outflow of `x` (with `+`), once as inflow of `y` (with `−`). They cancel. So every internal `S`–`S` arc drops out, and I'm left only with arcs that cross the boundary:
`value(f) = Σ_{u∈S, v∈T} f(u→v) − Σ_{u∈T, v∈S} f(u→v)`.
That's clean and it's exact: the value of *any* flow equals the flow crossing the cut forward minus the flow crossing it backward. Now bound it. The backward term is a sum of flows, all `≥ 0`, so dropping it only increases the right side: `value(f) ≤ Σ_{u∈S, v∈T} f(u→v)`. And each forward flow is at most its capacity: `Σ f(u→v) ≤ Σ c(u→v) = cap(S,T)`. So

`value(f) ≤ cap(S,T)`,

for *every* flow and *every* cut. There it is, and there's the answer to why the cut definition was asymmetric — I need an *upper* bound, and backward-crossing flow only lowers the net crossing, so to bound the value from above I throw that subtracted nonnegative term away rather than pay for the backward arcs. If I put `T→S` capacities into the wall I loosen the bound for no reason. Counting only `S→T` is exactly what makes the inequality tight where it can be tight.

And now the two problems collapse into one observation. Maximum flow `≤` minimum cut, immediately, because the max over flows of the left side is `≤` the min over cuts of the right side. So if I can ever exhibit a particular flow and a particular cut whose value and capacity are *equal*, I'm done twice over: that flow must be maximum (no flow can beat the cut it equals) and that cut must be minimum (no cut can undercut the flow it equals). They certify each other. The interdiction people get their bottleneck and the transport people get their maximum, from one equation. The entire game is now: can I always close the gap to zero? Is there always a flow and a cut that meet?

Let me re-read that bounding chain, because the *equality* conditions are sitting right there and they're the whole prize. The inequality `value ≤ cap` came from two slacks. The first slack was dropping the backward flow `Σ_{T→S} f`; that slack is zero exactly when every arc from `T` back to `S` carries *no* flow — is *avoided*. The second slack was `f ≤ c` on the forward arcs; that's tight exactly when every arc from `S` to `T` is *saturated*. So:

`value(f) = cap(S,T)` ⟺ every `S→T` arc is saturated and every `T→S` arc is avoided.

That's a precise target. I don't need to cleverly match some flow to some cut; I need a flow and a partition such that the wall is fully saturated forward and fully idle backward. If I can manufacture that, the gap is zero.

So how do I build the flow. The obvious thing, the thing the flooding people do, is greedy: find a route from source to sink with room to spare on every link, pour as much down it as the tightest link allows, repeat. Let me actually try that and feel where it breaks, because it's clearly not always optimal and I want to see *why* in a way I can fix.

Picture four cities. Source `s`, two middle cities `u` and `v`, sink `t`. Arcs `s→u`, `s→v`, `u→t`, `v→t` each of capacity 1, and one diagonal arc `u→v` also of capacity 1. The honest maximum is 2: send a unit along `s→u→t` and a unit along `s→v→t`, leaving the diagonal idle. But suppose greed grabs the diagonal first, routing `s→u→v→t`: that commits 1 unit onto `s→u`, onto `u→v`, onto `v→t`, and now `s→u`, `u→v`, `v→t` are all saturated. Greed looks for another source-to-sink route with room on every link and there isn't one — `s→u` is full, and the only way to leave `v` toward `t` is `v→t`, which is full. So greed stops at value 1, having committed flow onto `s→u` and `v→t` that "belongs" to a wasteful routing through the diagonal. The unit on `s→u` would be better spent going `s→u→t`, and the unit on `v→t` would be better fed by `s→v→t`, but greed has no operation that *takes back* the diagonal commitment. It only ever adds. That's the wall — the failure is irreversibility. A strictly better flow provably exists (value 2, right there on the map) and yet the procedure, having no way to undo a placement, can't reach it.

So the fix has to be: give myself an operation that *cancels* committed flow. When I'm looking for a way to increase the total, I should be allowed not only to push more onto an under-used arc, but also to *retract* flow from an arc that's currently carrying some, if retracting it frees up a better overall routing. Let me formalize "ways I can still improve" as a graph of its own — a graph that encodes both kinds of move for every ordered pair of vertices.

Take a possible step `x→y`. Two distinct opportunities can make that step useful. One: if the real arc `x→y` exists and is not full, I can push up to `c(x→y) − f(x→y)` more in the forward direction. Two: if the opposite real arc `y→x` currently carries flow, I can cancel up to `f(y→x)` of it, and that cancellation has the same net effect on the vertices as moving flow from `x` to `y`. If both real arcs exist, these opportunities add; a step from `x` to `y` can first undo flow going `y→x` and then, if there is still amount left to send, push new flow on `x→y`. So the "room to move from `x` to `y`" is

`c_f(x→y) = (c(x→y) − f(x→y))` [forward slack on a real arc `x→y`] `+ f(y→x)` [cancellable flow on the real arc `y→x`].

Call `c_f` the *residual capacity*, and let the *residual graph* `G_f` consist of every pair `x→y` with `c_f(x→y) > 0`. The forward terms are the obvious "push more"; the second terms are the back-edges, the undo channels. If `0 < f(u→v) < c(u→v)` then both `u→v` (push more) and `v→u` (cancel some) appear in `G_f`. This residual graph is the whole device — it turns "irreversible greed" into "search that can also retract."

Now the move. Suppose `G_f` has a directed path from source to sink — call it an *augmenting* path `P`, and take it simple. Every arc on it has positive residual room. Let `F = min over arcs of P of c_f` — the bottleneck residual along the path, the largest amount that every step can absorb. For each step `x→y` of `P`, I implement that net step in two pieces if necessary: cancel `α = min(F, f(y→x))` units on the opposite real arc `y→x`, then push the remaining `β = F − α` units on the real arc `x→y`. The choice of `F` makes the second piece feasible, because `β ≤ c(x→y) − f(x→y)`. It also keeps the cancellation feasible, because `α ≤ f(y→x)`. Every changed arc remains between `0` and its capacity.

The balance check is the point of packaging those two pieces as one residual step. A step `x→y` increases the net outflow of `x` and decreases the net outflow of `y` by a total of `F`, exactly as if `F` units moved from `x` to `y`; cancellation of `y→x` and pushing on `x→y` have the same signed effect on the two endpoint balances. At each interior vertex of the path, `F` units arrive through one residual step and `F` units leave through the next, so conservation survives. At the source, the first residual step increases net outflow by `F` — either by increasing an outgoing flow, decreasing an incoming flow, or a mix of both. At the sink, the last step increases net inflow by `F`. So `value(f') = value(f) + F > value(f)`. So:

if the residual graph has an `s`→`t` path, the current flow is **not** maximum — I can strictly increase it.

That handles one side. The other side is the one I actually want, the certificate. Suppose `G_f` has *no* path from source to sink. Then I'm stuck — but "stuck" is exactly the situation I should mine, because the equality conditions might be hiding in it. Let `S` be the set of vertices *reachable from the source in the residual graph* `G_f`, and `T` everything else. The source is in `S` (trivially reachable), the sink is in `T` (not reachable, by assumption), so `(S,T)` is a genuine cut. Now look at what "no residual edge leaves `S`" forces. Take any real arc `u→v` with `u∈S`, `v∈T`. If it had any forward slack, `c(u→v) − f(u→v) > 0`, then `c_f(u→v) > 0`, so `v` would be reachable from `u` and hence from the source — contradiction, `v∈T`. So `c(u→v) = f(u→v)`: every `S→T` arc is *saturated*. Now take any real arc `v→u` with `v∈T`, `u∈S`, i.e. an arc running backward across the cut. If it carried any flow, `f(v→u) > 0`, then the residual back edge `u→v` would have `c_f(u→v) ≥ f(v→u) > 0`, again making `v` reachable — contradiction. So `f(v→u) = 0`: every `T→S` arc is *avoided*.

But saturated-forward and avoided-backward is *exactly* the equality condition I extracted from the bounding chain. So for this flow and this cut,

`value(f) = cap(S,T)`.

The gap is closed. The flow I'm stuck at equals the capacity of the cut I read off the reachable set — so that flow is maximum, that cut is minimum, and each is the other's proof. And notice how the cut is *constructed*: it isn't guessed, it falls out of the residual graph for free the instant augmentation halts — `S` is just "what's still reachable from the source when no more flow can be pushed." The certificate is a by-product of getting stuck.

Putting the two cases together: a flow is maximum **iff** its residual graph has no augmenting path, and when that happens the reachable set hands me a minimum cut of equal capacity. The maximum flow value equals the minimum cut capacity — always, in every network. The gap I worried might be forced open is never open. That's the max-flow min-cut theorem, and it isn't just an existence statement: the proof is the algorithm. Start from the zero flow. While the residual graph has an `s`→`t` path, augment along it. When it doesn't, stop — the current flow is maximum and the reachable set is your minimum cut.

Let me make sure this actually halts. Take integer capacities first — tonnages, so this is the real case. At the start `f = 0` and every residual capacity is an original capacity, an integer. Each augmentation pushes the bottleneck `F`, which is a min of positive integers, hence `F ≥ 1`, and it changes every touched flow and residual by an integer, so integrality is preserved forever. So each round raises the value by at least 1, and the value is bounded above — by the capacity of *any* cut, e.g. the arcs leaving the source. A quantity that strictly increases by integer steps and is bounded must stop. So it terminates, and as a bonus, since the flow stays integer throughout, there's a maximum flow that is integer on every arc — the integrality theorem, free. Rational capacities reduce to the integer case by scaling.

But "terminates" is weaker than I'd like, and there's a real trap. Nothing in the argument said *which* augmenting path to take, and a careless choice is genuinely bad. Two middle vertices `u`, `v`; arcs `s→u`, `s→v`, `u→t`, `v→t` each of capacity `X` (some huge integer), and one middle arc `u→v` of capacity 1. Suppose I keep alternating: augment along `s→u→v→t`, pushing 1 unit across the middle edge; that creates a residual back edge `v→u`; so next I augment along `s→v→u→t`, pushing 1 unit back across it; and repeat. Each augmentation moves only 1 unit because the middle edge (capacity 1) is the bottleneck every time. I'd need about `2X` augmentations to reach the true maximum `2X`, when a smarter pair of paths would have finished in two. `X` can be written in about `log X` bits, so this is exponential in the size of the input. Worse, if I allow *irrational* capacities, a malicious choice of paths can augment forever by smaller and smaller amounts and converge to a flow that isn't even maximum. So the procedure is correct but its running time, left to arbitrary path choices, is not under control.

The trap is that I let the bottleneck stay tiny by repeatedly threading paths through the same near-empty middle edge. What stops that? Pick the *shortest* augmenting path each time — fewest arcs, which I get for free by searching the residual graph breadth-first instead of depth-first. Let me see why that tames it. Let `δ(v)` be the breadth-first distance from the source to `v` in the current residual graph, with `δ(v)=∞` if `v` is unreachable. I need these distances not to slide backward.

Suppose after one shortest-path augmentation some vertex gets closer to the source, and take the first such vertex `v` on a new shortest path. Let `u` be its predecessor on that new shortest path. Then `u` did not get closer, so `δ_new(u) ≥ δ_old(u)`, while the new path gives `δ_new(v)=δ_new(u)+1`. The edge `u→v` cannot have already existed before the augmentation, because then the old distance would have satisfied `δ_old(v) ≤ δ_old(u)+1 ≤ δ_new(u)+1 = δ_new(v)`, contradicting that `v` got closer. So `u→v` is a newly created residual edge. New residual edges are reverses of edges on the just-used shortest path, meaning the old path used `v→u`; along that old shortest path, `δ_old(u)=δ_old(v)+1`. Put the inequalities together: `δ_new(v)=δ_new(u)+1 ≥ δ_old(u)+1 = δ_old(v)+2`, which says `v` got farther, not closer. Contradiction. So BFS distances are monotone non-decreasing.

Now watch a single directed residual edge `u→v` when it is a bottleneck on a shortest augmenting path. At that moment it lies from level `δ(u)` to level `δ(u)+1`, and the augmentation removes it. Before the same directed edge can be a bottleneck again, some later augmentation has to use the reverse step `v→u` to recreate room in the `u→v` direction. On that later shortest path, `v` is one level before `u`, so the distance to `u` at that later time is at least two larger than it was when `u→v` was last saturated. Distances never exceed `n-1` while reachable, so any directed edge can be a bottleneck only `O(n)` times. With `m` original arcs there are `O(m)` directed residual edge types to count, hence `O(nm) = O(VE)` augmentations. Each augmentation is one breadth-first search plus one push, `O(E)`. So the whole computation is `O(VE²)` — polynomial, and crucially *independent of the capacity magnitudes*, which kills both the exponential alternation and the irrational non-termination. Always taking the shortest augmenting path is the refinement that makes the method genuinely efficient.

So let me state what I've actually got, end to end. Define flow and cut. Prove weak duality — any flow `≤` any cut — by pushing the value through the cut and reading off that the value is the forward-minus-backward crossing flow, with the two slacks identifying *saturated forward* and *avoided backward* as the equality condition. Introduce the residual graph with its forward "push" edges and reverse "cancel" edges, the one device that makes flow placement reversible. Augment along residual `s`→`t` paths; each augmentation strictly raises the value, so a flow with an augmenting path isn't maximum. When no augmenting path remains, the set of vertices still reachable from the source is a cut whose every forward edge is saturated and every backward edge avoided — so its capacity equals the flow value, certifying both as optima. That's strong duality, max-flow `=` min-cut, with the cut constructed for free at the moment of getting stuck. Integer capacities make it terminate and force an integral optimum; shortest (breadth-first) augmenting paths make it run in `O(VE²)` regardless of the capacities. Concretely:

```python
from collections import deque

def max_flow(cap, s, t):
    """cap[u][v] = capacity of arc u->v. Returns (value, flow, min_cut_S)."""
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
            neighbors[v].add(u)          # the opposite direction may cancel flow

    flow = {u: {v: 0 for v in capacity[u]} for u in vertices}

    def available_room(u, v):
        # residual capacity: unused u->v capacity plus cancellable v->u flow
        return capacity[u].get(v, 0) - flow[u].get(v, 0) + flow[v].get(u, 0)

    def find_source_sink_route():
        # Edmonds-Karp: BFS gives a shortest augmenting path in the residual graph.
        parent = {s: None}
        q = deque([s])
        while q:
            u = q.popleft()
            for v in neighbors[u]:
                if v not in parent and available_room(u, v) > 0:
                    parent[v] = u
                    if v == t:
                        return parent
                    q.append(v)
        return None

    def push(parent):
        # Bottleneck residual capacity along the path.
        F, v = float("inf"), t
        while parent[v] is not None:
            u = parent[v]
            F = min(F, available_room(u, v))
            v = u

        # Implement each residual step by canceling reverse flow first,
        # then pushing any remaining amount on the forward arc.
        v = t
        while parent[v] is not None:
            u = parent[v]
            cancel = min(F, flow[v].get(u, 0))
            if cancel:
                flow[v][u] -= cancel
            forward = F - cancel
            if forward:
                flow[u][v] += forward
            v = u
        return F

    def certifying_cut():
        # Vertices still reachable from s in the residual graph form the cut.
        S, q = {s}, deque([s])
        while q:
            u = q.popleft()
            for v in neighbors[u]:
                if v not in S and available_room(u, v) > 0:
                    S.add(v)
                    q.append(v)
        return S

    value = 0
    while True:
        parent = find_source_sink_route()
        if parent is None:
            break
        value += push(parent)

    return value, {u: dict(flow[u]) for u in vertices if flow[u]}, certifying_cut()
```

The causal chain in one breath: the value of any flow equals the flow crossing any cut forward minus backward, hence is at most the cut's forward capacity, so max-flow `≤` min-cut with equality exactly when the cut is saturated forward and idle backward; greedy pushing can't reach that because it can't undo a placement, so I build a residual graph whose reverse edges *are* the undo, and I augment along residual source-to-sink paths; while such a path exists the flow isn't maximal, and the moment none exists the vertices still reachable from the source form a cut that is forced saturated-forward and avoided-backward — so its capacity equals the flow value, proving both optimal at once; integer capacities make this halt with an integral optimum, and choosing shortest augmenting paths makes the distances monotone, bounding the work at `O(VE²)`.
