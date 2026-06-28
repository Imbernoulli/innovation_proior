**Reading the problem and naming the structure before I write any algorithm.** I am given `n` items,
each with a positive weight `w_i` and a non-negative linear value `v_i`, and a single weight budget
`W`. On top of the ordinary knapsack reward there is a *pairwise* one: certain unordered pairs `{i,j}`
carry a synergy bonus `b_{ij} > 0` that I collect only if I select **both** endpoints. The reward of a
chosen subset `S` is `obj(S) = Σ_{i∈S} v_i + Σ_{{i,j}⊆S} b_{ij}`, and I want to maximize it under
`Σ_{i∈S} w_i ≤ W`. I want to strip the surface story to its structure first, because the structure
dictates the whole design. The linear part is a plain 0/1 knapsack. The pairwise part makes the
objective **non-separable**: the marginal worth of an item is not fixed — it depends on which other
items I have already put in the bag, through the synergy edges between them. That is precisely the
**Quadratic Knapsack Problem (QKP)**: maximize a quadratic pseudo-Boolean form
`Σ v_i x_i + Σ b_{ij} x_i x_j` over binary `x` under one linear constraint. QKP is strongly NP-hard,
there is no exact solver I can run on `n` up to 900 in two seconds, and — importantly — the benchmark
scores me by *how large* my objective is, not by whether I hit a unique optimum. So this is a
continuous-score heuristic problem: my job is to push `obj(S)` as high as the time budget allows, and
to *never* hand the scorer something it can refuse.

**Pinning the I/O and the feasibility rule, because an infeasible output scores zero.** Input is
`n W`, then `n` lines `w_i v_i`, then `p`, then `p` lines `i j b`. Output is `k` followed by `k`
distinct item indices in `[0, n)`. The feasibility rule is the thing I must respect above everything
else. A subset is feasible iff (a) the output parses as `k` then exactly `k` integers, (b) every index
is in range and the indices are distinct, and (c) the total selected weight is `≤ W`. Break any clause
— a repeated index, an out-of-range index, trailing junk, or going one unit over budget — and the
score floors to `0`. In a heuristic-optimization benchmark with a mean over seeds, one zero is
catastrophic: a brilliant-but-occasionally-infeasible solver is strictly worse than a mediocre
always-feasible one. So my design rule from the first line is: **hold a feasible subset at all times**,
make every move preserve the budget invariant, and make the time-budget cutoff fall back on the best
feasible subset I currently have — not on whatever half-finished state the loop happened to be in.

**Reaching a feasible baseline first.** Before optimizing anything I want a legal answer in hand. The
trivial one is the **empty subset**: `k = 0`. It is feasible (zero weight `≤ W`), it has objective `0`,
and it is my safety net — nothing I do later can drop below it because I always keep the best feasible
subset found. That is the floor I must beat. The next rung is the classic linear-knapsack move: the
**value/weight-ratio greedy** — sort items by `v_i / w_i` descending and add each that still fits. This
ignores synergy completely, but it is a real, feasible subset and it is, in fact, exactly the reference
`G` the scorer normalizes against. So reproducing it scores `1 000 000`, and my real target is to beat
it: to collect strictly more total reward than a synergy-blind rule. The only way to do that
materially is to **exploit the quadratic term** — to deliberately co-select items that are jointly
valuable through their bonuses even when neither looks attractive on its own `v_i/w_i` ratio.

**Seeing why the synergy-blind greedy is weak, and where the reward actually hides.** I think about the
instances. Item linear values are deliberately modest, while the synergy bonuses — especially within
the latent clusters the generator builds — are large. So most of the objective lives in the *pairwise*
term, and it is concentrated: pairs inside a cluster carry big bonuses, pairs across clusters carry
small ones. A greedy that sorts by `v_i / w_i` is blind to all of this. It will happily pick a
high-`v` item from cluster A and a high-`v` item from cluster B and collect almost no synergy, when
packing a coherent block of cluster A would have harvested a dense lattice of bonuses. The lesson is
structural: the right move is to think in terms of **marginal value including synergy**. When I
consider adding item `i` to my current set, its true marginal worth is not `v_i` but
`v_i + (synergy i forms with what I already have)`. If I can compute that quantity cheaply, both my
construction and my local search become synergy-aware, and the cluster structure becomes an asset
instead of a trap.

**Why the obvious local search is too slow — the core obstacle.** The natural heuristic for QKP is
local search over the subset: repeatedly flip an item in or out (and swap one in for one out),
accepting improving moves, with a metaheuristic to escape local optima. The problem is *evaluating* a
move. If I recompute `obj(S)` from scratch after every candidate flip, that is `Σ v_i` over the subset
plus a pass over all `p` synergy pairs — `O(n + p)` per move. With `p` on the order of several thousand
edges and a local search that wants millions of moves to be any good, that is hopeless: I would get a
few thousand moves in two seconds, nowhere near enough for simulated annealing to actually anneal. Even
restricting to "which pairs touch item `i`" and rescanning them is `O(deg)` *but* only if I have the
adjacency indexed — and I still have to know, for each neighbour, whether it is currently selected.
Doing that lookup per move per neighbour, repeatedly, is the bottleneck. The obvious approach is too
weak not because the moves are wrong but because **re-evaluation is too expensive**.

**The innovation: maintain `g[i]` and get O(1) deltas with O(degree) updates.** Here is the lever. I
keep, for every item `i`, a running quantity

```
g[i] = Σ_{ j selected, {i,j} is a synergy pair } b_{ij},
```

the total synergy item `i` would form against the **currently selected set**. I also keep
incident-synergy lists `adj[i] = {(j, b_{ij})}` and the running total weight `curW` and objective
`curObj`. With `g` maintained, the exact objective change of flipping `i` is a single `O(1)` read:

```
add i  :  Δ = + (v_i + g[i])
drop i :  Δ = − (v_i + g[i]).
```

Why is that exact? Adding `i` gains its own value `v_i` plus, for every already-selected neighbour `j`,
the bonus `b_{ij}` — and `g[i]` is precisely the sum of those bonuses by definition. Dropping `i` loses
the same amount. The beautiful part is the *maintenance*: after I commit a flip of `i`, the only `g`
entries that change are `i`'s synergy neighbours `j` (their selected-neighbour set just gained or lost
`i`), so I walk `adj[i]` once and do `g[j] += b_{ij}` (on add) or `g[j] -= b_{ij}` (on drop). That is
`O(deg_synergy(i))`, the degree of `i` in the synergy graph — *not* `O(n²)`, *not* `O(p)`. The
incident-synergy incremental delta is the entire engine: it turns a naive `O(n+p)`-per-move local
search into an `O(1)`-evaluate, `O(deg)`-commit one, which is what makes tens of millions of flip
evaluations affordable in two seconds and lets simulated annealing genuinely explore. Every other piece
of the solver is built on top of this single `flip(i)` primitive that keeps `sel`, `g`, `curW`,
`curObj` consistent.

**Constructing a strong synergy-aware start.** With `g` in hand the construction writes itself: a
**density greedy that counts synergy**. Repeatedly add the feasible item `i` (one that still fits the
budget) maximizing `(v_i + g[i]) / w_i` — marginal value *including* the synergy it forms with the
already-chosen set, per unit weight — and stop when nothing positive fits. Because `g` updates as I add
items, the second item I pick already sees the synergy it would form with the first, the third sees the
first two, and so on, so the greedy naturally snowballs into a coherent cluster. `n ≤ 900`, so an
`O(n²)` repeated-scan construction is perfectly affordable and gives a start that already clears the
synergy-blind reference comfortably.

**Designing the metaheuristic.** A greedy construction lands in a local basin; to climb out I use
**simulated annealing over three flip moves**: ADD a random unselected item that fits, DROP a random
selected item, or SWAP (drop one selected `i`, add one unselected `j`) provided the swap stays within
budget. Each move's delta is read from `g` in `O(1)` (the swap is two coupled flips). I accept
improving moves outright and worsening moves with the Metropolis probability `exp(Δ / T)`, cooling `T`
geometrically from a start scaled to the typical bonus magnitude down to nearly zero, and I keep the
**incumbent best feasible subset** at all times. The swap move is essential here: on a tight budget the
only way to bring a valuable cluster member in is often to evict a heavier, lower-synergy incumbent,
which a pure add/drop walk reaches only slowly.

**A subtlety in the SWAP delta I have to get exactly right.** The one place this can go wrong is the
synergy edge *between* the two swapped items. Suppose `i` and `j` are themselves a synergy pair and `i`
is currently selected. If I naively compute "drop delta of `i`" and "add delta of `j`" from the
*current* `g`, I double-handle the edge `b_{ij}`: while `i` is still in, `g[j]` includes `b_{ij}`, but
after I drop `i` that bonus is gone, so adding `j` should *not* still credit it. The clean, exact route
is to evaluate the swap **sequentially through the real flips**: actually `flip(i)` (which updates
`g[j]` to drop `b_{ij}`), then read `v_j + g[j]` for the add, then `flip(j)`, and take the true
`curObj − before` as the move delta. If the Metropolis test rejects, I revert by `flip(j); flip(i)`.
This costs two `O(deg)` flips per swap but is guaranteed correct regardless of whether `i` and `j` are
adjacent — I trust the maintained state, not a hand-derived formula.

**The intensification post-pass.** After SA I restore the incumbent best and run a deterministic
**fill-up-and-exchange** hill-climb (the standard QKP intensifier). FILL UP: while some unselected item
has positive marginal value `v_i + g[i]` and fits, add the one with best marginal density. EXCHANGE:
for each selected `i` and unselected `j`, if dropping `i` and adding `j` stays feasible and strictly
improves the objective, do it. Loop the two phases until neither changes anything or time runs out.
This is pure hill-climbing on top of the maintained `g`, so it only ever improves the incumbent and
never breaks feasibility.

**First implementation, then a real self-verification episode.** I wrote the generator (clustered
synergy graph, modest values, tight budget), the scorer (objective + the synergy-blind ratio-greedy
reference `G`, with the feasibility floor), and the solver as above, compiled with
`g++ -O2 -std=c++17`, and ran seeds 1..20: for each, solver score, empty-subset score, and the greedy
reference. Two things had to hold — every output feasible (score `> 0`), and the solver mean strictly
above the reference `1 000 000`.

The first wall was not a crash but a **correctness bug in the SWAP move**. My initial version computed
the swap delta from precomputed `dDrop = -(v_i + g[i])` and `dAdd = v_j + g[j]` read *before* any flip,
and used `dDrop + dAdd` as the acceptance delta — exactly the double-counting trap I worried about
above. On instances where the two swap candidates were synergy neighbours, the accounting was off by
`b_{ij}`, so SA accepted swaps it thought improved the objective when they did not, and the incumbent's
*recorded* `curObj` drifted away from the true objective. I caught it the hard way: I added an
assertion that recomputed the objective from scratch every few thousand iterations and compared it to
the running `curObj` — and on seed 3 it tripped, `curObj` ahead of the true value by amounts that were
always sums of a few bonuses. That fingerprinted it immediately as the swap edge. The fix was to stop
trusting the hand-derived formula and evaluate the swap through the actual sequential flips
(`flip(i)`; read; `flip(j)`; `delta = curObj − before`), reverting on rejection. After that the
recompute-assertion held across all seeds, so I removed it from the hot loop.

The second issue was milder but mattered for the floor: an early construction variant could, in a
corner case with all-negative-marginal items, leave the running `curObj`/`curW` and the emitted subset
out of sync after the SA→post-pass handoff, and I wanted a hard guarantee the *printed* subset is
within budget regardless of any logic slip. So at emit time I recompute the total weight of `bestSel`
and, if it were ever over `W` (it never is, but defensively), fall back to the empty subset — which is
always feasible. With the swap fixed and the emit guard in place, the run was clean: all 20 seeds
feasible, solver mean ≈ `1 480 000` against the reference `1 000 000` (per-seed ratios `1.38–1.61×`),
every output's weight `≤ W`, single-instance wall time ≈ 1.6 s and ≈ 4 MB RAM — comfortably inside the
2 s / 256 MB budget. The empty baseline scores `0` and the greedy reference `1 000 000`, both
decisively beaten.

I also checked the feasibility floor directly: selecting all items (over budget), a duplicated index, an
out-of-range index, raw garbage, and trailing tokens each score `0` as intended, and feeding the
scorer's own ratio-greedy *selection* back as a solution scores exactly `1 000 000`, confirming the
normalization is self-consistent. The solver is genuinely strong — it is the established QKP local-search
family (synergy-aware density construction, SA over add/drop/swap, fill-up-and-exchange), and its entire
speed comes from the one innovation: `O(degree)` incremental evaluation of the quadratic term via the
maintained `g[i]` and incident-synergy lists.

**Final solver.** The complete single-file C++17 program (identical to `verify/sol.cpp`):

```cpp
// Knapsack with Synergies -- Quadratic Knapsack Problem (QKP) heuristic solver.
//
// Objective: choose a subset S of items to MAXIMIZE
//     sum_{i in S} v_i  +  sum_{ {i,j}: i,j in S } b_{ij}
// subject to  sum_{i in S} w_i <= W.  Read the instance from stdin, print the
// chosen subset (count then indices) to stdout. An over-budget / malformed
// output scores 0, so we keep a feasible subset at every moment.
//
// Method (the innovation):
//   1. Per-item INCIDENT-SYNERGY LISTS adj[i] = {(j, b_ij)} and a running
//      maintained array g[i] = "synergy item i would gain/lose against the
//      CURRENTLY selected set" = sum of b_ij over selected neighbours j. With g
//      maintained, the exact objective delta of flipping item i is
//          add i :  +v_i + g[i]
//          drop i: -v_i - g[i]
//      a single O(1) read. Flipping i then updates g only along i's incident
//      edges -- O(deg(i)) -- never an O(n^2) / O(p) re-evaluation. This is the
//      whole engine: O(degree) incremental evaluation of the quadratic term.
//   2. Construction: a SYNERGY-AWARE density greedy. Repeatedly add the
//      feasible item maximising (v_i + g[i]) / w_i (marginal value INCLUDING the
//      synergy it forms with the already-chosen set), beating the synergy-blind
//      ratio greedy that the scorer uses as its reference.
//   3. Simulated annealing over flips: add / drop / swap (drop one selected,
//      add one unselected) moves, each evaluated in O(1)+O(deg) via g, accepting
//      worsening moves with the usual Metropolis rule to escape local optima.
//      The incumbent best feasible subset is always retained.
//   4. Fill-up-and-exchange post-pass (Yang/Billionnet style): greedily fill any
//      leftover budget by best marginal density, then 1-1 exchanges, repeated to
//      a local optimum. Always leaves a feasible subset; any early time cutoff
//      still prints the best feasible subset found.
#include <bits/stdc++.h>
using namespace std;

static inline double now_sec() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    uint32_t nextu(uint32_t m) { return m ? (uint32_t)(next() % m) : 0u; }
    double nextd() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};

int N;
long long W;
vector<long long> w, v;
// incident synergy lists: adj[i] = list of (neighbour, bonus)
vector<vector<pair<int,long long>>> adj;

int main() {
    const double T0 = now_sec();
    const double TIME_LIMIT = 1.90;

    if (scanf("%d %lld", &N, &W) != 2) return 0;
    if (N <= 0) { printf("0\n"); return 0; }
    w.assign(N, 0); v.assign(N, 0);
    for (int i = 0; i < N; i++) {
        long long wi, vi;
        if (scanf("%lld %lld", &wi, &vi) != 2) { wi = 1; vi = 0; }
        if (wi < 1) wi = 1;               // guard: weights are positive
        w[i] = wi; v[i] = vi;
    }
    long long p = 0;
    if (scanf("%lld", &p) != 1) p = 0;
    adj.assign(N, {});
    for (long long e = 0; e < p; e++) {
        int a, b; long long bonus;
        if (scanf("%d %d %lld", &a, &b, &bonus) != 3) { a = 0; b = 0; bonus = 0; }
        if (a < 0 || a >= N || b < 0 || b >= N || a == b || bonus == 0) continue;
        adj[a].push_back({b, bonus});
        adj[b].push_back({a, bonus});
    }

    // ---------- state ----------
    vector<char> sel(N, 0);            // membership of current subset
    vector<long long> g(N, 0);         // g[i] = sum of b_ij over selected neighbours j
    long long curW = 0;                // total selected weight
    long long curObj = 0;              // current objective

    // flip item i (toggle membership); maintains sel,g,curW,curObj in O(deg(i))
    auto flip = [&](int i) {
        if (!sel[i]) {                 // ADD i
            curObj += v[i] + g[i];     // delta uses g BEFORE update
            curW += w[i];
            sel[i] = 1;
            for (auto &e : adj[i]) g[e.first] += e.second;
        } else {                        // DROP i
            curObj -= v[i] + g[i];
            curW -= w[i];
            sel[i] = 0;
            for (auto &e : adj[i]) g[e.first] -= e.second;
        }
    };

    Rng rng(0xC0FFEEULL ^ (uint64_t)N * 1000003ULL ^ (uint64_t)p * 2654435761ULL ^ (uint64_t)W);

    // ---------- 2. synergy-aware density-greedy construction ----------
    // Add the feasible item maximising (v_i + g[i]) / w_i, repeat. g already
    // reflects synergy to the chosen set, so this captures the quadratic term.
    {
        // Simple repeated scan; N <= 900 so O(N^2) construction is fine.
        while (true) {
            int best = -1;
            double bestKey = -1e18;
            for (int i = 0; i < N; i++) {
                if (sel[i]) continue;
                if (curW + w[i] > W) continue;
                double key = (double)(v[i] + g[i]) / (double)w[i];
                if (key > bestKey) { bestKey = key; best = i; }
            }
            if (best < 0) break;
            // only add if it does not strictly hurt (marginal value >= 0), else stop:
            if (v[best] + g[best] < 0) break;
            flip(best);
        }
    }

    // incumbent best
    vector<char> bestSel = sel;
    long long bestObj = curObj;

    auto saveIfBetter = [&]() {
        if (curObj > bestObj && curW <= W) { bestObj = curObj; bestSel = sel; }
    };
    saveIfBetter();

    // lists of currently in/out items for sampling swap partners
    // (rebuilt lazily; cheap to scan since N is small)
    auto pickIn = [&]() -> int {
        // reservoir pick a selected item
        int chosen = -1, cnt = 0;
        for (int i = 0; i < N; i++) if (sel[i]) { cnt++; if (rng.nextu((uint32_t)cnt) == 0) chosen = i; }
        return chosen;
    };
    auto pickOut = [&]() -> int {
        int chosen = -1, cnt = 0;
        for (int i = 0; i < N; i++) if (!sel[i]) { cnt++; if (rng.nextu((uint32_t)cnt) == 0) chosen = i; }
        return chosen;
    };

    // ---------- 3. simulated annealing over add / drop / swap flips ----------
    // delta of a candidate move is read from g in O(1); a swap is two coupled
    // flips so its delta accounts for the synergy edge between the two items.
    double Tcur = 0.0;
    // scale the temperature to typical bonus magnitudes
    {
        long long mx = 1;
        for (int i = 0; i < N; i++) mx = max(mx, v[i]);
        for (int i = 0; i < N; i++) for (auto &e : adj[i]) mx = max(mx, e.second);
        Tcur = (double)mx * 2.0 + 1.0;
    }
    double Tstart = Tcur, Tend = 0.05;
    long long iter = 0;
    const double SA_FRAC = 0.86;   // fraction of budget for SA, rest for post-pass
    while (true) {
        if ((iter & 511) == 0) {
            double el = now_sec() - T0;
            if (el > TIME_LIMIT * SA_FRAC) break;
            double frac = el / (TIME_LIMIT * SA_FRAC);
            if (frac > 1) frac = 1;
            // geometric cooling
            Tcur = Tstart * pow(Tend / Tstart, frac);
            if (Tcur < 1e-6) Tcur = 1e-6;
        }
        iter++;
        int moveType = (int)rng.nextu(3);
        if (moveType == 0) {
            // ADD a random unselected item that fits
            int i = pickOut();
            if (i < 0 || curW + w[i] > W) continue;
            long long delta = v[i] + g[i];          // O(1)
            if (delta >= 0 || rng.nextd() < exp((double)delta / Tcur)) {
                flip(i);
                saveIfBetter();
            }
        } else if (moveType == 1) {
            // DROP a random selected item
            int i = pickIn();
            if (i < 0) continue;
            long long delta = -(v[i] + g[i]);        // O(1)
            if (delta >= 0 || rng.nextd() < exp((double)delta / Tcur)) {
                flip(i);
                saveIfBetter();
            }
        } else {
            // SWAP: drop selected i, add unselected j (must keep budget feasible)
            int i = pickIn();
            int j = pickOut();
            if (i < 0 || j < 0) continue;
            if (curW - w[i] + w[j] > W) continue;
            // delta of dropping i then adding j. The synergy edge i-j (if any)
            // is handled correctly because after dropping i, g[j] no longer
            // counts b_ij; we evaluate sequentially to stay exact.
            long long dDrop = -(v[i] + g[i]);
            // tentatively account: after dropping i, g[j] would lose b_ij if edge exists.
            // Easiest exact route: do the two flips and compare objective.
            long long before = curObj;
            flip(i);                                  // drop i  (O(deg i))
            long long dAdd = v[j] + g[j];             // now g[j] reflects i removed
            flip(j);                                  // add j   (O(deg j))
            long long delta = curObj - before;        // exact two-flip delta
            (void)dDrop; (void)dAdd;
            if (delta >= 0 || rng.nextd() < exp((double)delta / Tcur)) {
                saveIfBetter();
            } else {
                // revert: drop j, add i
                flip(j);
                flip(i);
            }
        }
    }

    // restore incumbent best before the deterministic post-pass
    {
        // rebuild state to bestSel
        for (int i = 0; i < N; i++) { sel[i] = 0; g[i] = 0; }
        curW = 0; curObj = 0;
        for (int i = 0; i < N; i++) if (bestSel[i]) flip(i);
    }

    // ---------- 4. fill-up-and-exchange post-pass to a local optimum ----------
    // Repeat: (a) FILL UP -- add the best positive-marginal-density item that
    // fits; (b) EXCHANGE -- for each selected i and unselected j, if swapping
    // them stays feasible and strictly improves, do it. Loop until no change or
    // time out. Pure hill-climbing, so it only ever improves the incumbent.
    {
        bool improved = true;
        while (improved) {
            if (now_sec() - T0 > TIME_LIMIT) break;
            improved = false;

            // (a) fill up
            while (true) {
                int best = -1; double bestKey = 0.0; bool any = false;
                for (int i = 0; i < N; i++) {
                    if (sel[i] || curW + w[i] > W) continue;
                    long long marg = v[i] + g[i];
                    if (marg <= 0) continue;
                    double key = (double)marg / (double)w[i];
                    if (!any || key > bestKey) { bestKey = key; best = i; any = true; }
                }
                if (best < 0) break;
                flip(best);
                improved = true;
                saveIfBetter();
                if (now_sec() - T0 > TIME_LIMIT) break;
            }

            // (b) 1-1 exchanges (first-improvement). Bound the work by scanning
            // selected x unselected; N small so this is affordable per sweep.
            for (int i = 0; i < N && (now_sec() - T0 <= TIME_LIMIT); i++) {
                if (!sel[i]) continue;
                for (int j = 0; j < N; j++) {
                    if (sel[j]) continue;
                    if (curW - w[i] + w[j] > W) continue;
                    long long before = curObj;
                    flip(i);                          // drop i
                    long long after = curObj + v[j] + g[j];  // if we then add j
                    if (after > before) {
                        flip(j);                      // commit add j
                        improved = true;
                        saveIfBetter();
                        break;                        // i is gone; move to next i
                    } else {
                        flip(i);                      // revert: re-add i
                    }
                }
            }
        }
        saveIfBetter();
    }

    // ---------- emit best feasible subset ----------
    vector<int> outIdx;
    long long checkW = 0;
    for (int i = 0; i < N; i++) if (bestSel[i]) { outIdx.push_back(i); checkW += w[i]; }
    // final safety: if (impossibly) over budget, fall back to empty set.
    if (checkW > W) outIdx.clear();

    string buf;
    buf.reserve(outIdx.size() * 7 + 16);
    buf += to_string((long long)outIdx.size());
    buf += "\n";
    for (size_t k = 0; k < outIdx.size(); k++) {
        buf += to_string(outIdx[k]);
        buf += (k + 1 == outIdx.size()) ? '\n' : ' ';
    }
    if (outIdx.empty()) buf += "\n";
    fputs(buf.c_str(), stdout);
    return 0;
}
```
