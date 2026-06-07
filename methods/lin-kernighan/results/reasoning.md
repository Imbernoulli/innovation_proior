OK, let me think this through from scratch. I've got a symmetric distance matrix on `n` cities and I
want the shortest tour — one cyclic visit through all of them. The search space is `(n-1)!/2` tours,
so I'm not going to enumerate, and exact methods choke past a few dozen cities. So I'm in the
iterative-improvement game: start from some random tour `T`, find a transformation that gives a
shorter tour `T'`, jump there, repeat until nothing improves — that's a local optimum — then restart
from another random tour and keep the best. The whole quality of this scheme lives in one place: the
transformation in the middle. If it's weak, I get stuck in lots of bad local optima; if it's strong,
the local optima are few and good and a random start often lands on the global one. So everything is
about designing that step.

The standard step is the `k`-opt interchange. Take the current tour, delete `k` of its links, and
reconnect the pieces with `k` new links so it's a tour again; keep it if it's shorter. Croes did this
with `k = 2`: pull two links, and there's exactly one other way to reconnect two paths into a tour —
it reverses a subsegment. Cheap, always feasible, but shallow: tons of 2-opt local optima are far
from optimal. Lin did `k = 3`: pull three links, and now there are several reconnection patterns; the
tours you reach are 3-opt, hence also 2-opt, and noticeably better. But the cost jumped — scanning the
3-opt neighborhood is on the order of `n^3` — and you could keep climbing, `k = 4`, `k = 5`, getting
better tours at worse and worse cost.

And here's what actually bugs me about all of it. I have to *fix `k` in advance*. The work grows like
`n^k`, and there's no bound on how many improving exchanges a tour even has, so the per-`k` cost is
already steep — but worse, I have no way to know the right `k` for a given instance before I run.
Pick `k` too small and I stall at a mediocre local optimum; pick it too big and I pay through the nose
on every move, much of it wasted. The right depth surely varies from instance to instance and even
from move to move within one run. Fixing `k` up front feels exactly backwards. I'd like the *data* to
tell me how deep to go.

So let me reframe what "`T` is not optimal" even means. It means there's some set of links `x_1, ...,
x_k` in `T` that are out of place, and they ought to be swapped for some links `y_1, ..., y_k` from
outside `T`. The whole game is identifying `k` and the `x`'s and `y`'s. But I just said I don't know
`k`. So why am I committing to a `k` and then examining every `k`-subset for that fixed `k`? That's
the artificial move. Let me instead build the two sets `X = {x_1,...}` and `Y = {y_1,...}` *element by
element*, choosing at each step the most out-of-place pair I can — `x_1, y_1` first, set them aside,
then the most out-of-place pair in what remains, `x_2, y_2`, and so on — and let some stopping rule
decide when to quit. Then `k` isn't an input; it's an *outcome*. The depth varies by itself.

For this to be more than a vague wish I need the bookkeeping to be clean. Define the gain of swapping
`x_i` out for `y_i` in as `g_i = |x_i| - |y_i|` — the length I remove minus the length I add. If I do
a whole group of swaps, I'd love the total profit to be just `g_1 + g_2 + ... + g_k`, the gains simply
adding up. If that holds, then for any tour `T'` I might reach, `f(T) - f(T') = sum of g_i`, and `T'`
is better exactly when that sum is positive. Now I can reason about *partial* progress, not just
finished exchanges. That additivity is the thing to lean on.

Now, how do I build the chain so it's always one step from being a valid tour? Let me lay down a city
chain `t_1, t_2, t_3, ...` and tie the broken and added links to it. Pick a starting city `t_1`. Let
`x_1 = (t_1, t_2)` be one of the two tour links at `t_1` — that's the first link I break. From `t_2`,
I add a new link `y_1 = (t_2, t_3)` going off to some other city `t_3`. Now I'm holding a path: I broke
the tour at `t_1`, and I've grown an edge out to `t_3`. The pattern I want to keep is: `x_i` and `y_i`
share the endpoint `t_{2i}`, and `y_i` and `x_{i+1}` share the endpoint `t_{2i+1}`. So in general
`x_i = (t_{2i-1}, t_{2i})` is a tour link I break, and `y_i = (t_{2i}, t_{2i+1})` is a new link I add,
and the sequence `x_1, y_1, x_2, y_2, ...` is a chain of adjoining links zig-zagging through the
cities. This is a *sequential* exchange.

Why insist on this adjoining structure? Because it's what lets me *close up to a tour at any moment*.
After I've added `y_{i-1}` reaching city `t_{2i-1}`, I'm sitting on a Hamiltonian path. From
`t_{2i-1}` there are the two tour links; I break one of them as `x_i`, reaching its other end
`t_{2i}`. Here's the constraint that makes the whole thing work: I must break the `x_i` such that
joining `t_{2i}` straight back to the start `t_1` gives a single closed tour, not two disconnected
loops. Given `y_{i-1}`, exactly one of the two choices of `x_i` keeps "close `t_{2i}` to `t_1`" a
valid tour; the other splits the cycle. So `x_i` is *forced* — uniquely determined — by the demand
that I can always close up. That's the feasibility criterion, and it's a gift: at every depth `i` I
have a legal tour I could snap to by adding the link `(t_{2i}, t_1)`. I never assemble a pile of swaps
only to find at the end they can't be made into a tour.

That immediately tells me how to evaluate "stop here." Before I commit to adding the open-ended `y_i`,
let me check the close-up. Tentatively join `t_{2i}` to `t_1` with `y_i* = (t_{2i}, t_1)`; the gain of
that closing link is `g_i* = |x_i| - |y_i*|`. Because the feasibility criterion held, this *is* a
tour, and its improvement over `T` is `G_{i-1} + g_i*`, where `G_{i-1} = g_1 + ... + g_{i-1}` is the
running gain of the open chain so far. So I keep a running best `G* = max over i of (G_{i-1} + g_i*)`,
remembering the depth `k` at which it was achieved; `G*` starts at 0 and only goes up. At the end, if
`G* > 0`, I do the `k`-exchange that realizes it. The depth that wins is whatever it is — variable.

Now the real question: how do I keep this from exploding? At each step I'm choosing `y_i` to go to
some city `t_{2i+1}`, and naively that's a choice over all cities. Two things rein it in. First, to
make `g_i = |x_i| - |y_i|` large I want `|y_i|` *small*, so I should try the nearest neighbors of
`t_{2i}` first — short new links are where the gain is. That bounds the branching to a few candidates.
Second, and this is the crucial one: when do I stop extending the chain?

Here's the tempting-but-wrong answer: stop as soon as a `g_i` goes negative. But that's too greedy —
a step that loses a little can set up a later step that wins a lot, the classic way past a local
minimum. So I shouldn't stop on a single negative `g_i`. The looser rule would be: keep going as long
as the *total* could still come out positive. But that's almost no constraint at all and lets the
search wander forever.

Let me look harder at the running sum `G_i = g_1 + g_2 + ... + g_i`. I want to claim: I only ever need
to extend the chain while `G_i > 0` — insist that *every partial sum* stays positive — and I lose no
improving exchange by doing so. That sounds far too restrictive. Why would demanding positivity at
*every* prefix not throw away good moves whose gain dips negative in the middle?

Because of a fact about the gains as a cyclic sequence. Suppose I have a sequence of numbers `g_1,
..., g_n` with positive total sum. Then there is a *cyclic permutation* of them — a choice of where to
start reading the cycle — such that every partial sum is positive. Let me actually prove it, because
the whole pruning argument rests on it. Write the prefix sums `S_j = g_1 + ... + g_j`, with `S_0 = 0`,
and `S_n > 0` by assumption. Let `k` be the *largest* index for which `S_{k-1}` is the minimum among
`S_0, S_1, ..., S_{n-1}`. Now read the sequence cyclically starting at `g_k`: that is `g_k, g_{k+1},
..., g_n, g_1, ..., g_{k-1}`. Take any partial sum of this reordered sequence.

If the partial sum ends at index `j` with `k ≤ j ≤ n`, it equals `g_k + ... + g_j = S_j - S_{k-1}`.
Since `S_{k-1}` is the minimum prefix and `k` is the *largest* index achieving it, `S_j > S_{k-1}` for
every `j ≥ k`, so this partial sum is strictly positive.

If instead the partial sum wraps around — ending at index `j` with `1 ≤ j < k` — it equals `g_k + ...
+ g_n + g_1 + ... + g_j = (S_n - S_{k-1}) + S_j`. Now `S_j ≥ S_{k-1}` because `S_{k-1}` is the minimum
prefix, so this is `≥ (S_n - S_{k-1}) + S_{k-1} = S_n > 0`. Positive again.

So every partial sum of the cyclically-shifted sequence is positive. That's the whole point: *any*
improving sequential exchange — any chain whose total gain is positive — can be re-read, starting from
the right city, so that every running partial sum is positive along the way. Which means: if I insist
that `G_i > 0` at every step and I'm willing to try every starting city `t_1`, I am not missing a
single improving move. I'm only requiring that I *start the chain at the right place*. And in exchange
for that mild requirement, the positivity-at-every-step rule lets me abandon a chain the instant its
running gain hits zero — pruning away an enormous mass of fruitless deep searches. This positive-gain
criterion is the heart of the stopping rule, and the reason a variable-depth search is even
affordable.

So now the basic step has a shape. Pick `t_1`, break `x_1 = (t_1, t_2)`. Add `y_1 = (t_2, t_3)` with
`g_1 = |x_1| - |y_1| > 0` — if no such positive first step exists from this `t_1`, this start is dead.
Then loop: `i ← i + 1`; the forced `x_i` is the tour link at `t_{2i-1}` that keeps close-up feasible;
check the close-up gain `G_{i-1} + g_i*` against `G*` and update if better; then pick `y_i = (t_{2i},
t_{2i+1})` among nearest neighbors with `G_i = G_{i-1} + g_i > 0`, subject to the bookkeeping
constraints; if such a `y_i` exists, keep going, else stop. Terminate the chain when no `x_i, y_i`
satisfy the constraints or when `G_i ≤ G*` — there's no point pushing a chain whose running gain has
already fallen at or below the best tour I can already lock in. If `G* > 0`, make the exchange,
`f(T') = f(T) - G*`, and restart the whole construction from the new tour.

I need a couple of housekeeping constraints to keep this honest. The sets `X` and `Y` must be
*disjoint*: a link I've broken can't be re-added in this same iteration, and a link I've added can't be
re-broken. Once an element moves one way, it stays put for this iteration (it may move back on the
next iteration — different story). This isn't just tidiness: it stops the chain from undoing its own
work, simplifies the gain function, keeps running time down, dodges subtle implementation bugs, and
gives me a clean stop — only finitely many disjoint links to use. I also need `y_i` chosen so that an
`x_{i+1}` *can* be broken next, i.e. the chain can continue to satisfy feasibility one more step;
otherwise I've added a link that paints me into a corner.

Let me sanity-check the depth in the example. Start with tour `T`, pick adjacent `t_1, t_2`, so `x_1 =
(t_1, t_2)`. Let `t_3` be the nearest city to `t_2`; `y_1 = (t_2, t_3)`, with the proviso that `y_1`
isn't already a tour link at `t_2` (disjointness). If `g_1 = |x_1| - |y_1|` isn't positive, back up and
take `t_2` to be `t_1`'s other tour neighbor. Set `i = 2`. `t_4` is the tour neighbor of `t_3` such
that breaking `x_2 = (t_3, t_4)` lets me close up; if I instead join `t_4` to `t_1` and `g_1 + g_2 >
0`, that's a candidate 2-opt improvement — remember it as `G*`, `k = 2`. Otherwise pick `t_5` near
`t_4`, `y_2 = (t_4, t_5)`. Now there's a unique `x_3 = (t_5, t_6)` that allows close-up; check joining
`t_6` to `t_1` against `G*`, update `k = 3` if it's better. If `g_1 + g_2 + g_3 ≤ G*`, the stopping
rule says cash in the `k = 2` exchange. Otherwise continue with `t_7`, and so on. The depth just keeps
going as long as the running gain stays above the best-so-far. Exactly the variable `k` I wanted —
and notice that any tour this produces is at least 3-opt, since the first few levels reproduce a
2- or 3-exchange, so I'm guaranteed to do no worse than the fixed-3-opt method, in much less time.

Now, the chain as described only ever produces *sequential* exchanges — chains that close up nicely by
a renumbering of the affected links. Most improving moves can be written that way. But not all: there
are exchanges (the smallest is a 4-link one) where no ordering of the affected links forms a single
adjoining closing chain — they're genuinely non-sequential, and a purely sequential search will miss
them. I can buy back some of that lost power cheaply at one spot. When I break `x_2 = (t_3, t_4)`,
the "feasible" choice keeps me able to close up — but let me also allow the *other*, infeasible choice
of `x_2`, the one that momentarily can't close up at `i = 2`. Breaking that alternate `x_2` introduces
a temporary feasibility violation, so I only permit it at level 2, never deeper. Concretely: `y_2`
must not join back to `t_1` (that would leave two disconnected halves). If the new endpoint `t_5`
lands between `t_2` and `t_3`, then `t_6` can sit on either side of `t_5`, and I try both. If `t_5`
lands between `t_1` and `t_4`, there's only one valid `t_6` (between `t_5` and `t_4`, to keep
feasibility) and then `t_7` is pinned between `t_2` and `t_3`, with `t_8` free on either side. After
that branch I rejoin the normal course of the algorithm at the next level. It adds some complexity,
but it substantially increases overall effectiveness, so it earns its keep — and only at `i = 2`.

What about getting unstuck when a given start yields nothing? If `G* = 0` after exhausting a chain — no
improvement from this `t_1, x_1, y_1` — I backtrack, but I keep it strictly limited. First, try
alternate `y_2`'s in increasing length, as long as `g_1 + g_2 > 0` (the gain criterion still gates
them). If those are exhausted without profit, try the alternate `x_2` (the infeasible-at-`i=2` branch
above). If still nothing, back up further: try alternate `y_1`'s in increasing length, then the
alternate `x_1`, then a fresh starting city `t_1`. The procedure ends when all `n` choices of `t_1`
have been tried without profit, at which point this local optimum is done and I can restart from
another random tour.

Why stop the backtracking at levels 1 and 2 — why not backtrack everywhere? Two reasons, and both
matter. In principle, full backtracking at all levels would find the optimum, but the running time
would be immense — it'd be exhaustive search wearing a disguise. And empirically, when a gain is going
to be found at a node, it's almost always the *first* candidate the procedure tries: the mean choice
number is about 1.2 at the first level and 1.8 at the second. So there's almost no payoff to deep
backtracking — the good move, if it exists, is right at the top of the candidate list. That also tells
me how many candidates to keep: capping `y_1` and `y_2` to about five contenders each loses essentially
no optima and cuts running time by nearly half versus considering all of them; limited experiments
with third-level backtracking show a considerable time penalty for little gain. Backtracking only at
the first level, on the other hand, weakens the procedure — good costs but the global optimum appears
less often. Levels 1 and 2 are the sweet spot.

Stepping back: the whole thing settles into a recognizable rhythm on a real instance. The first
several improvements found tend to have *large* `k` — deep chains, changes touching a good fraction of
the tour, often with zero overshoot (the chain stops right at its best depth). As the tour gets good,
improvements get scarcer and the moves shrink to small `k`, around 2 to 7, with a little overshoot.
The number of improvements to reach a local optimum runs between `n/4` and `n/3`, and the running time
per local optimum grows on average like `n^{2.2}` — gentle enough to take many random starts and pick
the best. The gain criterion is doing the heavy lifting: it's strong enough that the search does far
less work than its unbounded depth would suggest.

Now let me land it on code. I'll represent a tour as an ordered city list plus its set of edges (for
membership tests), with `around(node)` giving a city's two tour-neighbors, `contains(edge)` testing
membership, and `generate(broken, joined)` rebuilding the order from "current edges minus broken plus
joined" and reporting whether the result is a single tour — that's my feasibility/close-up oracle.

```python
from copy import deepcopy

def makePair(i, j):
    return (i, j) if i < j else (j, i)   # undirected edge as a sorted tuple


class Tour:
    def __init__(self, tour):
        self.tour = tour
        self.size = len(tour)
        self.edges = {makePair(tour[i - 1], tour[i]) for i in range(self.size)}

    def around(self, node):              # the two tour-neighbours of `node`
        idx = self.tour.index(node)
        succ = idx + 1 if idx + 1 < self.size else 0
        return (self.tour[idx - 1], self.tour[succ])

    def contains(self, edge):
        return edge in self.edges

    def generate(self, broken, joined):
        # new edge set = (tour edges - broken) + joined; rebuild and check it is ONE cycle.
        # Returns (is_tour, new_tour) -- this is the close-up / feasibility oracle.
        edges = (self.edges - broken) | joined
        if len(edges) < self.size:
            return False, []
        succ, node = {}, 0
        while edges:                     # chain the edges into a successor map
            for i, j in edges:
                if i == node: succ[node] = j; node = j; break
                if j == node: succ[node] = i; node = i; break
            edges.remove((i, j))
        if len(succ) < self.size:
            return False, []
        new_tour, cur, seen = [0], succ[0], {0}
        while cur not in seen:           # walk successors; a premature repeat = disjoint subtours
            seen.add(cur); new_tour.append(cur); cur = succ[cur]
        return len(new_tour) == self.size, new_tour
```

The driver restarts the search every time it finds an improving move, and gives up when a full pass
finds nothing — that's the local optimum:

```python
class LinKernighan(TSP):
    def _optimise(self):
        better = True
        self.solutions = set()
        self.neighbours = {i: [j for j, d in enumerate(TSP.edges[i]) if d > 0
                                and j in self.heuristic_path]
                           for i in self.heuristic_path}
        while better:
            better = self.improve()                       # one improving step, or False
            self.solutions.add(str(self.heuristic_path))  # remember tours seen, to break cycles
        self.save(self.heuristic_path, self.heuristic_cost)
```

`closest` is where the nearest-neighbour preference and the positive-gain gate live: from the current
chain end `t2i` with running gain `gain = G_{i-1} + |x_i|` already accumulated, it considers candidate
new links `y_i = (t2i, node)`, computes the would-be running gain `Gi = gain - |y_i|`, and *only* keeps
those with `Gi > 0` that aren't already broken and aren't tour edges — and orders them by how
promising the next omission looks, so the best `y` is tried first:

```python
    def closest(self, t2i, tour, gain, broken, joined):
        candidates = {}
        for node in self.neighbours[t2i]:
            yi = makePair(t2i, node)
            Gi = gain - TSP.dist(t2i, node)               # running gain if we ADD y_i = (t2i,node)
            if Gi <= 0 or yi in broken or tour.contains(yi):
                continue                                  # POSITIVE-GAIN CRITERION + disjointness
            for succ in tour.around(node):                # the x_{i+1} we could break next
                xi = makePair(node, succ)
                if xi not in broken and xi not in joined: # ensure an x_{i+1} can still be broken
                    diff = TSP.dist(node, succ) - TSP.dist(t2i, node)
                    if node not in candidates or diff > candidates[node][0]:
                        candidates[node] = [diff, Gi]
        return sorted(candidates.items(), key=lambda x: x[1][0], reverse=True)
```

`improve` runs the level-1 part: for each start city `t1`, both choices of `x1 = (t1, t2)` (its two
tour links), the initial gain is `|x1|`, and `closest` gives the `y1 = (t2, t3)` candidates with `g1 >
0`. Capping to five candidates is the limited level-1 backtracking; trying both `t2`'s is the alternate
`x1`; iterating over all `t1` is "new starting city":

```python
    def improve(self):
        tour = Tour(self.heuristic_path)
        for t1 in self.heuristic_path:
            around = tour.around(t1)
            for t2 in around:                             # the two choices of x1 (alternate x1)
                broken = {makePair(t1, t2)}
                gain = TSP.dist(t1, t2)                   # |x1|
                close = self.closest(t2, tour, gain, broken, set())  # y1 candidates, g1 > 0
                tries = 5                                 # limited level-1 backtracking (~5)
                for t3, (_, Gi) in close:
                    if t3 in around:                      # y1 must not already be a tour link at t2
                        continue
                    joined = {makePair(t2, t3)}
                    if self.chooseX(tour, t1, t3, Gi, broken, joined):
                        return True                       # improvement -> restart from Step 2
                    tries -= 1
                    if tries == 0:
                        break
        return False
```

`chooseX` chooses the link `x_i` to break from the last reached city. The forced-vs-alternate choice
shows up here: normally both tour links at the last city are tried, but once the chain is deep enough
(four broken links) it commits to the longer one. For each candidate `x_i` it forms the close-up edge
`(t2i, t1)`, asks `generate` whether that yields a tour, and if so with positive relink gain it takes
the improvement; otherwise it recurses into `chooseY` to extend the chain — and the rule that
backtracking is full only at level 2 but single-shot deeper is exactly the `len(broken) == 2` branch:

```python
    def chooseX(self, tour, t1, last, gain, broken, joined):
        if len(broken) == 4:                              # deep: commit to the longer x_i
            pred, succ = tour.around(last)
            around = [pred] if TSP.dist(pred, last) > TSP.dist(succ, last) else [succ]
        else:
            around = tour.around(last)                    # both tour links are candidate x_i
        for t2i in around:
            xi = makePair(last, t2i)
            Gi = gain + TSP.dist(last, t2i)               # add |x_i| to the running gain
            if xi in joined or xi in broken:              # disjointness: X and Y stay disjoint
                continue
            added   = deepcopy(joined); added.add(makePair(t2i, t1))   # close-up edge (t2i, t1)
            removed = deepcopy(broken); removed.add(xi)
            relink = Gi - TSP.dist(t2i, t1)               # gain G* if we close up here
            is_tour, new_tour = tour.generate(removed, added)
            if not is_tour and len(added) > 2:            # infeasible close-up allowed only at i = 2
                continue
            if str(new_tour) in self.solutions:           # already-seen tour -> stop, avoid cycling
                return False
            if is_tour and relink > 0:                    # a strictly better tour: take it
                self.heuristic_path = new_tour
                self.heuristic_cost -= relink
                return True
            choice = self.chooseY(tour, t1, t2i, Gi, removed, joined)  # else extend the chain
            if len(broken) == 2:
                return choice if choice else self._next_x2(...)        # full backtracking at level 2
            return choice                                              # single shot for i > 2
        return False
```

`chooseY` chooses the next added link `y_i` from the close-up candidates — five contenders at level 2,
just the single best deeper, which is the asymmetric backtracking budget made concrete:

```python
    def chooseY(self, tour, t1, t2i, gain, broken, joined):
        ordered = self.closest(t2i, tour, gain, broken, joined)
        top = 5 if len(broken) == 2 else 1                # 5 candidates at level 2, else nearest only
        for node, (_, Gi) in ordered:
            added = deepcopy(joined); added.add(makePair(t2i, node))   # y_i = (t2i, node)
            if self.chooseX(tour, t1, node, Gi, broken, added):
                return True
            top -= 1
            if top == 0:
                return False
        return False
```

So the causal chain: I refused to fix `k` because the right depth can't be known in advance, so I
build the exchange one link at a time and let the depth fall out. Tying the breaks and adds to a city
chain `t_1, t_2, t_3, ...` with `x_i = (t_{2i-1}, t_{2i})` and `y_i = (t_{2i}, t_{2i+1})` keeps me one
forced break away from a closeable tour at every depth, so I can stop anywhere and read off a valid
tour. Additive gains `g_i = |x_i| - |y_i|` let me track total profit as a running sum, and the
cyclic-permutation fact proves that insisting every partial sum `G_i` stay positive — provided I try
every start `t_1` — discards no improving move while pruning the search enormously. Nearest-neighbour
preference for `y_i` keeps `|y_i|` small and the branching tight; disjointness of `X` and `Y` keeps the
chain from undoing itself; the alternate infeasible `x_2` recovers some non-sequential power at one
cheap spot; and backtracking limited to levels 1 and 2 with about five candidates each captures
essentially all the gains because the right move is almost always the first one tried. The result is a
variable-depth local search that is at least as strong as fixed 3-opt, at a fraction of the cost, with
running time growing about as `n^{2.2}`.
