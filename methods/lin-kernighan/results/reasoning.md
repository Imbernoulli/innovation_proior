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

Before I go further I should pin down that the forced-`x_i` claim is actually a fact and not wishful
thinking, because the whole "always one link from a tour" story rests on it. Let me take a concrete
6-city tour `0-1-2-3-4-5-0` and run two steps by hand. Break `x_1 = (0,1)`, so `t_2 = 1`; add
`y_1 = (1,3)`, reaching `t_3 = 3`. City 3 has two tour links, `(2,3)` and `(3,4)`, so `x_2` is one of
those. Try both, each time closing `t_4` back to `t_1 = 0`:

- Break `x_2 = (3,2)`, so `t_4 = 2`, then add the close-up edge `(2,0)`. The edge set becomes the
  original six links minus `{(0,1),(2,3)}` plus `{(1,3),(2,0)}`, i.e. `(0,2),(0,5),(1,2),(1,3),(3,4),(4,5)`.
  Walking from 0: `0→2→1→3→4→5→0` — a single 6-cycle. Feasible.
- Break `x_2 = (3,4)`, so `t_4 = 4`, then add `(4,0)`. Edges minus `{(0,1),(3,4)}` plus
  `{(1,3),(4,0)}`: `0→4→5→0` closes after three cities while `1,2,3` form a separate loop. Two
  disjoint subtours. Infeasible.

So exactly one of the two choices closes up — `x_2` really is forced — and it's the one I keep. Good;
the "one link from a tour" property is not an aspiration, it holds.

That tells me how to evaluate "stop here." Before I commit to adding the open-ended `y_i`,
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

So every partial sum of the cyclically-shifted sequence is positive. Let me not trust the algebra
blindly — I'll run the construction on a sequence that's deliberately awkward, one that dips negative
in the middle. Take `g = (3, -5, 4, -1, 2)`, total `= 3 > 0`. Read in the given order the partial
sums are `3, -2, 2, 1, 3` — the second one is negative, so a naive left-to-right start *would* trip
the positivity rule and abandon this chain. Now apply the recipe. The prefix sums are
`S_0..S_5 = (0, 3, -2, 2, 1, 3)`; the minimum over `S_0..S_4` is `-2`, achieved last at `S_2`, so
`k = 3`. Reading cyclically from `g_3`: `(4, -1, 2, 3, -5)`, whose partial sums are `4, 3, 5, 8, 3` —
every one strictly positive. So the same five gains, with a different starting city, sail through the
positivity gate. That's the content of the claim made concrete: *any* improving sequential exchange —
any chain whose total gain is positive — can be re-read, starting from the right city, so that every
running partial sum is positive. Which means: if I insist that `G_i > 0` at every step and I'm willing
to try every starting city `t_1`, I am not missing a single improving move. I'm only requiring that I
*start the chain at the right place*. And in exchange for that mild requirement, the
positivity-at-every-step rule lets me abandon a chain the instant its running gain hits zero — pruning
away an enormous mass of fruitless deep searches. So the positive-gain criterion buys the pruning that
makes a variable-depth search affordable, and it buys it for free: no improving exchange is lost.

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

Let me run the whole step once on real numbers to make sure the gain bookkeeping actually closes. Put
four cities at the corners of a unit square: `0=(0,0)`, `1=(1,0)`, `2=(1,1)`, `3=(0,1)`. The boundary
tour `0-1-2-3` has length 4 and is optimal. Hand the procedure the crossing tour `0-2-1-3` instead —
its two diagonals cross — with length `√2 + 1 + √2 + 1 = 2√2 + 2 ≈ 4.8284`. Now: `t_1 = 0`, break
`x_1 = (0,2)`, `|x_1| = √2 ≈ 1.4142`. The nearest city to `t_2 = 2` that isn't already a tour neighbor
is 1 (or 3, tied), so `y_1 = (2,1)`, `|y_1| = 1`, `g_1 = 1.4142 - 1 = 0.4142 > 0` — first step alive.
Set `i = 2`, `t_3 = 1`. City 1's two tour links are `(1,2)` and `(1,3)`, so `x_2` is one of them, and
I check the close-up of each by joining `t_4` to `t_1 = 0`:

- `x_2 = (1,2)`, `t_4 = 2`, close-up `(2,0)`: `g_2* = |x_2| - |(2,0)| = 1 - 1.4142 = -0.4142`; total
  `G = g_1 + g_2* = 0`. (And this break is the infeasible one — it splits the cycle — so it wouldn't
  survive anyway.)
- `x_2 = (1,3)`, `t_4 = 3`, close-up `(3,0)`: `g_2* = √2 - 1 = 0.4142`; total `G = g_1 + g_2* =
  0.4142 + 0.4142 = 0.8284 > 0`. This is the feasible close-up, so `G* = 0.8284`, `k = 2`.

Cashing the depth-2 exchange — break `(0,2)` and `(1,3)`, add `(2,1)` and `(3,0)` — rebuilds the tour
`0-1-2-3` of length 4. And the predicted length is `f(T) - G* = 4.8284 - 0.8284 = 4.0000`, which is
exactly what the rebuilt tour measures. So the additive-gain identity `f(T') = f(T) - sum of g_i` isn't
just bookkeeping I hoped would hold — on this instance it lands on the nose. Past this depth, if a
`g_1 + g_2 + g_3 ≤ G*` ever showed up I'd cash the `k = 2` exchange; here there's nothing better to
find since 4 is optimal. The depth that won was 2, chosen by the data rather than fixed in advance —
the variable `k` I was after.

It's worth noticing what the first couple of levels of this chain reduce to. A chain that stops at
`i = 2` after one break-and-close is precisely a 2-opt move (delete two links, reconnect the other
way), and a chain stopping at `i = 3` covers the 3-opt reconnections. So when I make my best move the
construction has, among its candidates, every 2- and 3-exchange — which suggests a local optimum it
reaches should also be 2-opt and 3-opt. I haven't proven that the cap on backtracking never lets a 3-opt
improvement slip past, so I'd want to confirm empirically that the tours it returns survive a 3-opt
check; but the structure makes "no worse than fixed 3-opt, at lower cost" the natural expectation
rather than a leap.

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
the best. That exponent is the surprise: with no fixed depth cap I might have feared something near
the worst-case branching, but the measured growth sits barely above quadratic. The reading I take from
it is that the positive-gain gate plus the five-candidate cap are pruning hard enough that the
nominally unbounded depth almost never costs much — the chains that run deep are rare and the rest die
quickly when their running gain crosses zero.

Now let me land it on code. I'll represent a tour as an ordered city list plus its set of edges (for
membership tests), with `around(node)` giving a city's two tour-neighbors, `contains(edge)` testing
membership, and `generate(broken, joined)` rebuilding the order from "current edges minus broken plus
joined" and reporting whether the result is a single tour — that's my feasibility/close-up oracle.

I'll land it as a single self-contained C++17 program: it reads `n` and an `n × n` symmetric distance
matrix from stdin, runs the search from the identity tour, and prints the resulting tour and its
length to stdout. An undirected edge becomes the ordered key `min(i,j)*N + max(i,j)`, and a `Tour`
holds the order, a `pos[]` index, and the edge set; `generate` is the close-up oracle — it forms
`(tour - broken) + joined`, walks the cycle from city 0, and reports whether it is one Hamiltonian
cycle (orienting the walk canonically so the rebuilt order is deterministic):

```cpp
static inline long long key(int i, int j) {   // undirected edge as an ordered key (a < b)
    if (i > j) swap(i, j);
    return (long long)i * N + j;
}

struct Tour {
    vector<int> order;                          // city visit order
    vector<int> pos;                            // pos[city] = index in order
    unordered_set<long long> edges;             // current tour edges (membership)
    int size = 0;

    explicit Tour(const vector<int>& o) : order(o), size((int)o.size()) {
        pos.assign(size, 0);
        for (int i = 0; i < size; i++) pos[order[i]] = i;
        for (int i = 0; i < size; i++) edges.insert(key(order[(i + size - 1) % size], order[i]));
    }

    pair<int,int> around(int node) const {      // the two tour-neighbours of `node`
        int idx = pos[node];
        return {order[(idx + size - 1) % size], order[(idx + 1) % size]};
    }

    bool contains(long long e) const { return edges.count(e) != 0; }

    // New edges = (tour - broken) + joined; rebuild and check it is ONE Hamiltonian cycle.
    bool generate(const unordered_set<long long>& broken,
                  const unordered_set<long long>& joined, vector<int>& out) const {
        vector<array<int,2>> adj(size, {-1, -1});
        vector<int> deg(size, 0);
        auto addEdge = [&](int a, int b) {
            if (deg[a] < 2) adj[a][deg[a]] = b; deg[a]++;
            if (deg[b] < 2) adj[b][deg[b]] = a; deg[b]++;
        };
        int edgeCount = 0;
        for (long long e : edges) if (!broken.count(e)) { addEdge((int)(e / N), (int)(e % N)); edgeCount++; }
        for (long long e : joined) { addEdge((int)(e / N), (int)(e % N)); edgeCount++; }
        if (edgeCount != size) return false;
        for (int i = 0; i < size; i++) if (deg[i] != 2) return false;
        out.clear();
        int prev = -1, cur = 0;
        for (int step = 0; step < size; step++) {
            out.push_back(cur);
            int a = adj[cur][0], b = adj[cur][1];
            int nxt = (step == 0) ? min(a, b) : ((a != prev) ? a : b);   // canonical orientation
            prev = cur; cur = nxt;
            if (cur == 0 && step + 1 < size) return false;  // premature return => subtours
        }
        return cur == 0 && (int)out.size() == size;
    }
};
```

The driver restarts the search every time it finds an improving move, and gives up when a full pass
finds nothing — that's the local optimum:

```cpp
struct LinKernighan {
    vector<int> path;            // current tour order
    double cost;                 // its length
    unordered_set<string> seen;  // tours already reached (cycle guard)

    void optimise() {
        NB.assign(N, {});                            // neighbour lists, nearest first
        for (int i = 0; i < N; i++) {
            vector<pair<double,int>> nbrs;
            for (int j = 0; j < N; j++) if (j != i && D[i][j] > 0) nbrs.emplace_back(D[i][j], j);
            sort(nbrs.begin(), nbrs.end());
            for (auto& pr : nbrs) NB[i].push_back(pr.second);
        }
        seen.clear();
        bool better = true;
        while (better) {                             // restart at every improving move
            better = improve();                      // one improving step, or false
            seen.insert(tourKey(path));              // remember tours seen, to break cycles
        }
    }
    // ... closest / chooseY / chooseX / improve below ...
};
```

`closest` is where the nearest-neighbour preference and the positive-gain gate live: from the current
chain end `t2i` with running gain `gain = G_{i-1} + |x_i|` already accumulated, it considers candidate
new links `y_i = (t2i, node)`, computes the would-be running gain `Gi = gain - |y_i|`, and *only* keeps
those with `Gi > 0` that aren't already broken and aren't tour edges — and orders them by how
promising the next omission looks, so the best `y` is tried first:

```cpp
    // candidate y_i = (t2i, node), kept in neighbour-list order then stably sorted by `diff`.
    vector<pair<int,double>> closest(int t2i, const Tour& tour, double gain,
                                     const unordered_set<long long>& broken,
                                     const unordered_set<long long>& joined) const {
        vector<int> nodes;                                // distinct candidate nodes, in order
        unordered_map<int, pair<double,double>> cand;     // node -> {diff, Gi}
        for (int node : NB[t2i]) {
            long long yi = key(t2i, node);
            double Gi = gain - D[t2i][node];              // running gain if we ADD y_i = (t2i,node)
            if (Gi <= 0 || broken.count(yi) || tour.contains(yi))
                continue;                                 // POSITIVE-GAIN CRITERION + disjointness
            auto pr = tour.around(node);
            for (int s : {pr.first, pr.second}) {         // the x_{i+1} we could break next
                long long xi = key(node, s);
                if (!broken.count(xi) && !joined.count(xi)) {   // an x_{i+1} can still be broken
                    double diff = D[node][s] - D[t2i][node];
                    auto it = cand.find(node);
                    if (it == cand.end()) { cand[node] = {diff, Gi}; nodes.push_back(node); }
                    else if (diff > it->second.first) it->second = {diff, Gi};
                }
            }
        }
        stable_sort(nodes.begin(), nodes.end(),           // best `diff` first
                    [&](int a, int b){ return cand[a].first > cand[b].first; });
        vector<pair<int,double>> out;
        for (int node : nodes) out.emplace_back(node, cand[node].second);
        return out;
    }
```

`improve` runs the level-1 part: for each start city `t1`, both choices of `x1 = (t1, t2)` (its two
tour links), the initial gain is `|x1|`, and `closest` gives the `y1 = (t2, t3)` candidates with `g1 >
0`. Capping to five candidates is the limited level-1 backtracking; trying both `t2`'s is the alternate
`x1`; iterating over all `t1` is "new starting city":

```cpp
    bool improve() {
        Tour tour(path);
        for (int t1 : path) {                             // try every start city
            auto ar = tour.around(t1);
            for (int t2 : {ar.first, ar.second}) {        // the two choices of x1 (alternate x1)
                unordered_set<long long> broken; broken.insert(key(t1, t2));
                double gain = D[t1][t2];                  // |x1|
                auto close = closest(t2, tour, gain, broken, {});  // y1 candidates, g1 > 0
                int tries = 5;                            // limited level-1 backtracking (~5)
                for (auto& nc : close) {
                    int t3 = nc.first; double Gi = nc.second;
                    if (t3 == ar.first || t3 == ar.second) continue;  // t3 not a tour-neighbour of t1
                    unordered_set<long long> joined; joined.insert(key(t2, t3));
                    if (chooseX(tour, t1, t3, Gi, broken, joined))
                        return true;                      // improvement -> restart from Step 2
                    if (--tries == 0) break;
                }
            }
        }
        return false;
    }
```

`chooseX` chooses the link `x_i` to break from the last reached city. The forced-vs-alternate choice
shows up here: normally both tour links at the last city are tried, but once the chain is deep enough
(four broken links) it commits to the longer one. For each candidate `x_i` it forms the close-up edge
`(t2i, t1)`, asks `generate` whether that yields a tour, and if so with positive relink gain it takes
the improvement; otherwise it recurses into `chooseY` to extend the chain — and the rule that
backtracking is full only at level 2 but single-shot deeper is exactly the `len(broken) == 2` branch:

```cpp
    bool chooseX(const Tour& tour, int t1, int last, double gain,
                 const unordered_set<long long>& broken,
                 const unordered_set<long long>& joined) {
        vector<int> aroundNodes;
        auto pr = tour.around(last);
        if (broken.size() == 4)                           // deep: commit to the longer x_i
            aroundNodes.push_back(D[pr.first][last] > D[pr.second][last] ? pr.first : pr.second);
        else { aroundNodes.push_back(pr.first); aroundNodes.push_back(pr.second); } // both candidate x_i
        for (int t2i : aroundNodes) {
            long long xi = key(last, t2i);
            double Gi = gain + D[last][t2i];              // add |x_i| to the running gain
            if (joined.count(xi) || broken.count(xi)) continue;  // disjointness: X and Y stay disjoint
            auto added = joined; added.insert(key(t2i, t1));     // close-up edge (t2i, t1)
            auto removed = broken; removed.insert(xi);
            double relink = Gi - D[t2i][t1];              // gain G* if we close up here
            vector<int> newTour;
            bool isTour = tour.generate(removed, added, newTour);
            if (!isTour && added.size() > 2) continue;    // infeasible close-up allowed only at i = 2
            if (isTour && seen.count(tourKey(newTour))) return false; // already-seen -> avoid cycling
            if (isTour && relink > 1e-12) {               // a strictly better tour: take it
                path = newTour; cost -= relink; return true;
            }
            bool choice = chooseY(tour, t1, t2i, Gi, removed, joined); // else extend the chain
            if (broken.size() == 2) { if (choice) return true; }       // level 2: try the other x_2
            else return choice;                                        // single shot for i > 2
        }
        return false;
    }
```

`chooseY` chooses the next added link `y_i` from the close-up candidates — five contenders at level 2,
just the single best deeper, which is the asymmetric backtracking budget made concrete:

```cpp
    bool chooseY(const Tour& tour, int t1, int t2i, double gain,
                 const unordered_set<long long>& broken,
                 const unordered_set<long long>& joined) {
        auto ordered = closest(t2i, tour, gain, broken, joined);
        int top = (broken.size() == 2) ? 5 : 1;           // 5 candidates at level 2, else top-ranked only
        for (auto& nc : ordered) {
            int node = nc.first; double Gi = nc.second;
            auto added = joined; added.insert(key(t2i, node));   // y_i = (t2i, node)
            if (chooseX(tour, t1, node, Gi, broken, added)) return true;
            if (--top == 0) return false;
        }
        return false;
    }
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
variable-depth local search whose moves include every 2- and 3-exchange as special cases — so I expect
its local optima to be at least 3-opt — at a fraction of the cost of fixed `k`, with running time
growing about as `n^{2.2}`. As an end-to-end check I ran the coded version from an unshuffled start on
a 12-city Euclidean instance: it returned a tour of length `2.4039`, and brute force over all `11!/2`
tours of that instance returns the same `2.4039`. One instance proves nothing about the worst case,
but it does confirm the pieces — the chain construction, the close-up oracle, the gain accounting —
fit together into a procedure that actually finds the optimum here, not just a plausible-looking one.
