# Beam search

## Problem

Many problems build a solution as a sequence of decisions, forming a layered
decision graph: layer `t` holds the partial solutions that have made `t`
decisions, and each node fans out into a few children at layer `t+1`. We want
the sequence with the best final cumulative score. The exact frontier is
hopeless — with branching factor `b` and depth `D` it grows to `b^D` (millions
of decode paths, `2^n` knapsack subsets). We need a search that runs in time and
memory **linear in depth**, forward-only (no backtracking), and returns a
high-scoring solution even though we forgo a proof of optimality.

## Key idea

Sweep the graph forward layer by layer, but keep only a bounded *beam* of the
most promising partial solutions at each layer. Maintain a frontier of at most
`W` states (the **beam width**). To advance one layer: expand every state in the
frontier into its children, rank all the children by an evaluation that
estimates the **final** score of their best completion (`g + h`: score-so-far
plus an optimistic estimate of the rest), and **prune** the pool back to the top
`W`. Repeat to the last layer; the best feasible solution ever seen is the answer.

`W` is a dial. At `W = ∞` nothing is pruned: the frontier is the entire layer,
so the search becomes the exact full-breadth sweep — exhaustive (exponential
over a tree, or full-label dynamic programming once equal states are merged). At
`W = 1` it keeps the single best child each layer — greedy hill-climbing. In
between it trades a little optimality for a lot of speed: with a constant-time
evaluation and linear-time top-`W` selection the cost is `O(W · b · D)` — linear
in `W` and in depth, never exponential. Quality *tends* to improve as `W` grows
(and the exact optimum is recovered once the beam is wide enough never to prune
the eventual winner), but ordinary beam search carries **no monotonicity
guarantee** — a wider beam can occasionally return a worse solution; the
monotonic variants are separate algorithms. Beam search is deliberately
**incomplete and non-optimal** — a goal/optimum can be pruned — in exchange for a
predictable, tunable, linear-time forward search.

Two important refinements:

- **Filtered beam search.** When the accurate evaluation is expensive, prune in
  two stages: a cheap local score filters the children down to a *filter width*,
  then the expensive global evaluation ranks the survivors down to the *beam
  width* `W` — most of the ranking quality at a fraction of the evaluation cost.
- **The diversity trap and Chokudai search.** A fixed beam tends to fill all `W`
  slots with near-duplicates of one strong early lineage, so widening `W` gives
  diminishing returns. Fixes: dedup states (e.g. by hash), cap same-score
  states, or sample survivors stochastically (probability rising with score)
  instead of taking a hard top-`W`. The structural fix — **Chokudai search** —
  keeps a priority queue *per layer* of every state that ever reached it; one
  *pass* pops the single best un-expanded state at each layer in order and pushes
  its children forward (one pass ≈ a width-1 beam run to the end), then repeats,
  each pass tending to develop a different lineage. After `k` passes you have
  roughly `k` diverse solutions; it is anytime (loop until the time budget is
  spent, no `W` to tune) at the cost of retaining all states.

## Algorithm

```
beam_search(problem, W):
    frontier = [ initial_state ]
    best = initial_state
    repeat depth times:                         # forward, one layer per step
        pool = []
        for s in frontier:                      # expand every live state
            for c in children(s):               # one decision deeper
                pool.append(c)
        rank pool by  g(c) + h(c)               # score-so-far + optimistic est.
        frontier = top W of pool                # PRUNE -- the beam
        best = better of (best, best feasible state in frontier)
    return best
```

- `g(c)`: objective accumulated along the partial solution.
- `h(c)`: an *optimistic* estimate of the best remaining completion (for
  knapsack, the fractional-relaxation / LP bound). Ranking by `g + h` keeps a
  currently-behind-but-promising state alive instead of cutting it for an
  early-easy one.
- The single pruning line `frontier = top W of pool` is the whole method; no
  backtracking, memory bounded by `W`. Work is `O(W · b · D)` *given a
  constant-time evaluation and linear top-`W` selection*; an `O(n)`-per-state
  bound and a per-layer sort add their own factors (for this knapsack code,
  roughly `O(n · W · n)` from the bound plus `O(n · W log W)` from the sorts).

## Code

A single self-contained C++17 program that reads one knapsack instance from
stdin: the first line is `n W cap` (item count, beam width, capacity), then `n`
lines of `w v`; it prints the best value beam search finds and the chosen item
indices. Values accumulate in `long long` for overflow safety.

```cpp
// Beam search for 0/1 knapsack (single-file C++17, reads stdin).
// Input: first line "n W cap" (item count, beam width, capacity); then n lines
// "w v". Output line 1: the best value beam search finds; line 2: the chosen
// item indices (0-based, increasing), or blank if none.
#include <bits/stdc++.h>
using namespace std;

struct State {
    int idx;              // how many items decided (the layer)
    long long value;      // value packed so far (this is g)
    long long cap;        // remaining capacity
    double key;           // rank by value-so-far + optimistic bound (g + h)
    vector<int> chosen;   // item indices taken
};

// h: optimistic completion -- the LP (fractional-knapsack) relaxation of the
// items not yet decided: pack them in DESCENDING value/weight order, last item
// taken fractionally. The fractional optimum is the LP relaxation of the integer
// remainder, so it never underestimates the best integer completion -- exactly
// what we want to rank states by. (The descending-density order is what makes it
// an upper bound; input order is not.) 'rem' is items[idx..n-1] pre-sorted by
// descending density so this is a single scan.
static double bound(long long value, long long cap,
                    const vector<pair<long long, long long>>& rem) {
    double b = (double)value;
    long long c = cap;
    for (const auto& wv : rem) {
        long long w = wv.first, v = wv.second;
        if (w <= c) { c -= w; b += (double)v; }
        else { b += (double)v * ((double)c / (double)w); break; }
    }
    return b;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long W, capacity;
    if (!(cin >> n >> W >> capacity)) return 0;
    vector<pair<long long, long long>> items(n);  // (weight, value) in input order
    for (int i = 0; i < n; ++i) cin >> items[i].first >> items[i].second;

    // For each layer, pre-sort the remaining items[layer..n-1] by descending
    // value/weight so the bound is a simple scan.
    vector<vector<pair<long long, long long>>> remByDensity(n + 1);
    for (int layer = 0; layer <= n; ++layer) {
        vector<pair<long long, long long>> rem(items.begin() + layer, items.end());
        sort(rem.begin(), rem.end(),
             [](const pair<long long, long long>& a,
                const pair<long long, long long>& b) {       // descending density
                 return (long double)a.second * b.first > (long double)b.second * a.first;
             });
        remByDensity[layer] = move(rem);
    }

    vector<State> layer;
    layer.push_back(State{0, 0, capacity, bound(0, capacity, remByDensity[0]), {}});
    State best = layer[0];

    for (int step = 0; step < n; ++step) {           // forward, one layer per item
        vector<State> pool;
        pool.reserve(layer.size() * 2);
        for (const auto& s : layer) {                // expand every live state
            long long w = items[s.idx].first, v = items[s.idx].second;
            // skip item s.idx
            {
                State c = s;
                c.idx = s.idx + 1;
                c.key = bound(c.value, c.cap, remByDensity[c.idx]);
                pool.push_back(move(c));
            }
            // take item s.idx, if it fits
            if (w <= s.cap) {
                State c = s;
                c.idx = s.idx + 1;
                c.value = s.value + v;
                c.cap = s.cap - w;
                c.chosen.push_back(s.idx);
                c.key = bound(c.value, c.cap, remByDensity[c.idx]);
                pool.push_back(move(c));
            }
        }
        // rank by g + h, keep the top W -- the single pruning line, no backtracking
        sort(pool.begin(), pool.end(),
             [](const State& a, const State& b) { return a.key > b.key; });
        if ((long long)pool.size() > W) pool.resize((size_t)W);
        layer = move(pool);
        for (const auto& s : layer)                  // track best feasible value
            if (s.value > best.value) best = s;
    }

    cout << best.value << "\n";
    for (size_t i = 0; i < best.chosen.size(); ++i)
        cout << best.chosen[i] << (i + 1 == best.chosen.size() ? '\n' : ' ');
    if (best.chosen.empty()) cout << "\n";
    return 0;
}
```

Running it against a brute-force optimum shows the signature: on a 20-item
instance (capacity 60, optimum 185), feeding the same items at `W = 1, 2, 5, 20,
100` yields best values `179, 183, 185, 185, 185` — the achieved value climbs as
the beam widens and flattens once the beam is wide enough never to prune the
eventual winner —

```
brute-force optimum: 185
beam W=  1: value=179  gap= 6  items=12
beam W=  2: value=183  gap= 2  items=13
beam W=  5: value=185  gap= 0  items=13
beam W= 20: value=185  gap= 0  items=13
beam W=100: value=185  gap= 0  items=13
```
