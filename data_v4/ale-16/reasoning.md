**Reading the problem and fixing the structure before any algorithm.** A patrol vehicle sits at a
fixed depot and may drive out to collect prizes from `n` candidate sites, returning to the depot at
the end. Each site has a coordinate and a positive prize; I pay Euclidean distance for the route and I
earn the prizes of whatever I visit. The first thing I want to settle is what is actually being
optimized, because the surface story hides a second decision that the obvious framings miss. The cost
is the closed-route length depot → … → depot, the reward is the sum of collected prizes, and the
objective is `profit = Σ prizes − route length`. The non-obvious part is the word *may*: I am not
forced to visit everything. A remote site with a tiny prize can cost more in detour than it is worth,
and skipping it is the right call. So the decision is twofold and coupled: **which** sites to visit and
**in what order**. That is the Prize-Collecting TSP. It generalizes metric TSP — make every prize huge
and I must visit all sites, recovering plain TSP — but the extra selection layer is what makes it its
own beast. It is NP-hard, there is no exact answer at `n` up to 1200 in two seconds, and the benchmark
scores me by *how much profit* I earn, not by hitting a unique optimum. So my job is to push profit up
as far as the time budget allows, and — above everything — to never emit an output the scorer can
refuse.

**Pinning the I/O and the feasibility rule, because an infeasible output scores zero.** Input is `n`,
then the depot `dx dy`, then `n` lines `x y prize`, all integers, coordinates distinct. Output is `k`
followed by `k` site ids in visiting order, read as the closed loop `depot → p[0] → … → p[k−1] →
depot`. The feasibility rule is the thing I must respect above all: the output must be a single `k`
followed by exactly `k` integer ids, all in range and **pairwise distinct**, with the header `k`
matching the count. A wrong count, a repeat, an out-of-range id, a header that disagrees with the list,
a stray token — any of those floors the score to `0`. In a heuristic optimization problem one zero in
the mean is catastrophic: a brilliant-but-occasionally-invalid solver is strictly worse than a mediocre
always-valid one. So my design rule from the first line is: *hold a valid tour at all times*, make the
time cutoff fall back on whatever valid tour I currently have, and treat permutation-distinctness as an
invariant maintained by construction, not as something I check at the end and hope.

**Reaching a feasible baseline first.** Before I optimize anything I want a legal answer in hand. The
most trivial one is `k = 0`: visit nothing, profit `0`. It is always feasible and it is my ultimate
safety net — whatever else happens, I can always print `0` and score a clean, non-crashing result. It
is also a meaningful semantic floor: a real answer must do strictly better than collecting nothing. The
first *non-trivial* baseline is **visit-all nearest-neighbour**: ignore selection entirely, start at
the depot, repeatedly drive to the nearest unvisited site, close the loop. It is feasible by
construction (I only ever pick unvisited sites). I deliberately note that this is exactly the scorer's
reference — the score is normalized so that this visit-all NN tour earns exactly `1 000 000`, a better
solution earns more, a worse one earns less but never below `0`. So my real target is sharp: beat the
visit-all NN tour, which means I must *both* drop the sites whose prize doesn't cover their detour
*and* find a shorter order for the ones I keep.

**Understanding why the obvious decompositions fail — this is where the lever lives.** The tempting
move is to split the coupled problem into two phases I already know how to do. Two natural splits, and
both are traps. First split: **threshold the prizes, then run TSP**. Decide the visited set by a static
rule — keep site `i` if `prize_i > 2·dist(depot, i)`, say — then solve TSP on the survivors. The
failure is that a site's *true* cost is not its distance to the depot; it is its **insertion detour
into the eventual route** — how much longer the tour gets when I splice it between two consecutive
stops. A far site can be nearly free if the route already passes close by, and a near site can be
expensive if it forces a back-and-forth. A static depot-distance threshold therefore keeps losers and
drops winners; it is optimizing the wrong quantity. Second split: **alternate** — fix the set, run TSP
to get a good order, then re-evaluate the set against *that* order, drop/add, re-run TSP, repeat. This
is better because it at least uses insertion cost, but the two phases fight: improving the order changes
which sites are worth keeping, and changing the set changes the best order. Alternation stalls in a
poor joint local optimum, and each TSP re-solve is expensive. The realization is that selection and
ordering are not two problems; they are one, and I should search them in **one** neighbourhood where a
single move can simultaneously change membership and order. The PCTSP literature is clear on this: the
strong approach is a **fused local search** with add/drop/relocate/2-opt moves, and the engineering
that makes it fly is an **O(1) incremental gain test** per move.

**Deriving the incremental gain formulas.** Every move I care about touches only a constant number of
edges, so its effect on profit is a constant-time delta. Suppose the route, as a doubly linked list,
has `prev[v]` and `next[v]` for each visited site (the depot is a virtual node in the list). Then:

- **ADD `v` after `u`** (where `w = next[u]`): I remove edge `(u, w)` and add edges `(u, v)`, `(v, w)`,
  and I gain `prize_v`. The profit delta is
  `prize_v − (d(u,v) + d(v,w) − d(u,w))`.
  The parenthesized term is exactly the **insertion detour** — the right cost, not the depot distance.
- **DROP `v`** (with `u = prev[v]`, `w = next[v]`): I add back edge `(u, w)`, remove `(u, v)`, `(v, w)`,
  and lose `prize_v`. Delta `= −prize_v + (d(u,v) + d(v,w) − d(u,w))`. The parenthesized term is the
  detour I *save* by skipping `v` — a loser is precisely a site whose saved detour exceeds its prize.
- **RELOCATE `v`** (Or-opt of segment length 1): conceptually DROP `v` then ADD it at its best slot
  elsewhere. The combined delta is `dropGain(v) + addGain(best u, v)`. If that sum is positive, moving
  `v` improves the route *order* while keeping it in the set — and because it is literally a drop
  followed by an add, the same machinery also lets the in/out toggle happen as a side effect.
- **2-opt**: pick two route edges `(a,b)` and `(c,d)` with `c` after `b`, replace them by `(a,c)`,
  `(b,d)` and reverse the segment between. Delta `= (d(a,c) + d(b,d)) − (d(a,b) + d(c,d))`. This
  un-crosses crossings and is the only move here that re-orders a whole segment.

The thing competitors miss is the **fusion**: if I fix the visited set first and only then optimize the
order, I never see that dropping `v` *and* re-ordering its neighbours is a single profitable step. By
keeping drop, add, relocate and 2-opt in one search over a linked-list tour, the selection and the
ordering co-evolve. The O(1) deltas are what make thousands of these moves per millisecond feasible,
which is what a continuous-score heuristic needs.

**Choosing the data structures.** Two accelerators. (1) A **doubly linked list** over visited sites
plus a virtual depot node, so `prev`/`next` of any site is O(1); add/drop/relocate are then constant
time. (2) **Candidate lists**: for each site, its `K` nearest other sites, built once from a uniform
spatial grid. Add and relocate only try slots adjacent to a few near neighbours (and the depot), and
2-opt only considers reconnecting `b` to one of its near neighbours `c` — so each move costs O(K)
instead of O(k). Restricting to near neighbours is standard for Euclidean TSP and loses almost nothing
in quality because good edges are short edges.

**The construction and the search skeleton.** Start from the empty tour (the depot looping to itself).
Greedily cheapest-insert: repeatedly find the (site, slot) with the largest positive add-gain and
splice it in, until no positive-gain insertion remains. That already implements selection-by-insertion-
cost. Then the inner loop is a **deterministic descent** that sweeps ADD every profitable not-in-tour
site, DROP every loser, RELOCATE every site whose drop+best-readd is a net gain, then a 2-opt pass to
un-cross edges — repeating until a whole sweep makes no improvement. Every accepted move is a strict
improvement, so profit climbs monotonically to a joint local optimum. To escape that local optimum, the
outer loop is **iterated local search**: kick a handful of sites in/out at random, re-run the descent,
and accept the new local optimum by a simulated-annealing rule (so I can cross small valleys), always
remembering and finally printing the best tour seen. The empty tour is always available as a fallback,
and if my best profit ever went negative I print `k = 0` instead.

**First implementation, then a real debug episode.** I wrote the linked-list tour, the four gain
formulas, the grid candidate lists, the greedy construction, and — in my first cut — a single
**simulated-annealing loop** that picks a random move type (add/drop/relocate) each iteration and
accepts by the Metropolis rule, with an occasional 2-opt pass every few thousand iterations. I compiled
it, generated seed 1, ran it, and scored it. The score came back **`0`**. That is the worst possible
result and it sent me straight to the scorer to see *which* failure it was — infeasible output or a
clamped-to-zero profit. I printed the components: the output parsed fine as a permutation (so it was
*feasible*), but the solver's profit was `159.7M` while the visit-all NN baseline's profit was
`194.7M`. My solver was producing a **worse** tour than the trivial baseline, so the normalized score
`1e6 + 1e6·(P − P_base)/D` went negative and clamped to `0`. Two distinct bugs were hiding here.

The first bug was in the **instance design**, and the scorer diagnostics exposed it: I had set prizes
on the same scale as the whole coordinate grid, so total prizes (`221M`) dwarfed total route length
(`26M`) by more than eight to one. In that regime visiting *everything* is overwhelmingly correct,
"skip the loser" is nearly irrelevant, and the visit-all baseline is almost optimal — there was no
headroom for selection to matter, and my churn-y SA was strictly hurting the order. I re-derived the
prize scale from the geometry: the marginal detour to insert a site is roughly twice its distance to
the nearest route site, and with `n` sites in a side-`S` box the typical nearest-neighbour spacing is
about `S/√n`. So I retied prizes to `unit = S/√n`: cluster sites get a few `unit`s (worth a short
detour), background sites get below one-to-two `unit`s (usually *not* worth their long detour). After
the fix the prize-to-distance ratio dropped to about `2.4–2.9`, and a quick Python "drop every loser
from the NN tour" reference confirmed real headroom — dropping `5–8 %` of sites alone gained tens of
thousands of normalized points, before any re-ordering.

The second bug was in the **solver itself**, and it was a control-flow tangle in the relocate move. My
SA relocate did `drop v; add v at best slot`, but the revert path when the move was *rejected* was
muddled: I had written a branch that "kept the drop if dropping alone helped, else re-added at the best
slot", which is **not** a clean revert — it silently mutated the tour on a rejected move and could leave
a profitable site permanently displaced, slowly degrading the route. Worse, the SA ran add/drop with
almost no 2-opt (every 4000 iterations), so the *order* of the kept set never got cleaned, and the tour
length ballooned. I rebuilt the optimization core around the cleaner structure above: a deterministic
descent (monotone, every move a strict improvement, 2-opt run *every* sweep) wrapped in ILS, with an
**exact** relocate revert — I remember `u0 = prev[v]` before dropping and, if the relocate isn't a net
gain, re-link `v` after `u0`, restoring the tour bit-for-bit. I also added a `restoreFromVec` helper so
the ILS kick always perturbs from a known-good accepted state and recomputes profit exactly, never
drifting.

**Re-verifying.** I recompiled and re-ran seed 1: score `1 185 790` — comfortably above the `1 000 000`
visit-all anchor, with `k = 968` of `1032` sites kept (so ~64 losers correctly skipped) and the order
2-opted down. I then ran the full seed set 1–20 against three baselines: the empty tour, the
input-order visit-all tour, and the NN visit-all reference. Every one of the twenty outputs was
feasible (`score > 0`), the per-seed scores ranged `1.09–1.24 M`, the mean was `1 171 587`, and the
solver strictly beat all three baselines on every single seed (empty and input-order both score `0`
here; the NN reference is `1 000 000`). I also checked the edge cases that usually break a tour solver:
`n = 0` prints `0`; `n = 1` with a close high-prize site visits it; two far low-prize sites are both
*skipped* (output `0`, and the scorer rewards skipping because visiting them would be a big loss);
three sites with two cheap winners and one far loser visits exactly the two winners. The wall-clock time
sits at the `1.85 s` budget with `~4 MB` of memory, inside the two-second / 256 MB limits. The tour is a
valid permutation at every instant, so even a mid-iteration timeout prints a feasible best-so-far.

**What the final solver is.** A doubly-linked-list PCTSP tour with grid-built candidate lists; greedy
cheapest-insertion construction; one **fused** local-search neighbourhood — ADD, DROP, RELOCATE
(Or-opt-1, the in/out toggle), and candidate-restricted 2-opt — every move scored by an O(1) profit
delta over only the edges it touches; a deterministic monotone descent as the inner loop; and iterated
local search with random in/out kicks and SA acceptance as the outer loop, always printing the best
feasible tour seen and falling back to the empty tour if profit is ever negative. The non-obvious idea
that earns the score over the obvious phase-split approaches is the fusion of the in/out toggle with
re-ordering inside a single O(1)-delta neighbourhood. Here is the final program.

```cpp
// Prize-Collecting Patrol -- heuristic solver.
//
// Objective: starting and ending at a fixed DEPOT, choose a SUBSET of the n
// optional nodes and a visiting order for them, forming a closed tour
//     depot -> p[0] -> ... -> p[k-1] -> depot,
// to MAXIMIZE   profit = (sum of prizes of visited nodes) - (Euclidean travel).
// Read the instance from stdin; write k then the k chosen node ids (visiting
// order), one per line, to stdout. k = 0 (visit nothing, profit 0) is a legal,
// always-feasible answer and is our safety net.
//
// Method (the innovation):
//   * Tour kept as a doubly linked list over node ids plus a virtual DEPOT, so
//     prev/next of every visited node is O(1). `inTour[v]` flags membership.
//   * Construction: greedy cheapest-insertion of profitable nodes (insert the
//     node whose prize minus insertion-detour is largest, while positive).
//   * ONE fused local-search neighbourhood, all moves with an O(1) gain test from
//     cached prev/next:
//       - ADD v after u:   gain = prize[v] - (d(u,v)+d(v,next[u]) - d(u,next[u]))
//       - DROP v:          gain = -prize[v] + (d(prev[v],v)+d(v,next[v])
//                                              - d(prev[v],next[v]))
//       - RELOCATE v (Or-opt-1): drop v then re-add it at its best candidate slot
//         -- this simultaneously reorders the visited set AND can be combined with
//         toggles; the in/out toggle is the move competitors miss when they fix
//         the visited set first.
//       - 2-opt (un-cross two tour edges) restricted to candidate neighbours,
//         done on an array snapshot, to clean up route crossings.
//   * Candidate lists: each node's K nearest other nodes (and the depot), built
//     once from a uniform spatial grid, so ADD/RELOCATE only try good slots.
//   * The inner loop is a deterministic descent that sweeps ADD/DROP/RELOCATE +
//     2-opt to a local optimum. The outer loop is iterated local search: kick a
//     few nodes in/out, re-descend, and accept the new local optimum by an
//     SA-style rule (so small valleys can be crossed). We always remember the
//     best feasible tour seen and print THAT.
// The linked list is a valid tour at all times, so any early stop (time limit)
// still yields a feasible solution.
#include <bits/stdc++.h>
using namespace std;

static inline double now_sec() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    uint64_t next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return s;
    }
    uint32_t nextu(uint32_t m) { return (uint32_t)(next() % m); }  // [0, m)
    double nextd() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};

int N;                 // number of optional nodes
int DEPOT;             // virtual id == N
vector<double> X, Y;   // coords, size N+1 (index N = depot)
vector<double> P;      // prize, size N+1 (depot prize = 0)

static inline double dist(int a, int b) {
    double dx = X[a] - X[b];
    double dy = Y[a] - Y[b];
    return sqrt(dx * dx + dy * dy);
}

// ---- tour state (doubly linked list over visited ids + depot) ----
vector<int> nxt, prv;     // size N+1
vector<char> inTour;      // size N+1 (depot always "in")
double curProfit = 0.0;   // profit of the current tour

// candidate neighbours (nearest nodes) per node
vector<vector<int>> cand;

// insert v immediately after u in the linked list; updates curProfit by delta
static inline void linkAfter(int u, int v) {
    int w = nxt[u];
    nxt[u] = v; prv[v] = u;
    nxt[v] = w; prv[w] = v;
    inTour[v] = 1;
    curProfit += P[v] - (dist(u, v) + dist(v, w) - dist(u, w));
}

// remove v from the linked list; updates curProfit by delta
static inline void unlink(int v) {
    int u = prv[v], w = nxt[v];
    nxt[u] = w; prv[w] = u;
    inTour[v] = 0;
    curProfit += -P[v] + (dist(u, v) + dist(v, w) - dist(u, w));
}

// gain of ADDING v after u (without doing it)
static inline double addGain(int u, int v) {
    int w = nxt[u];
    return P[v] - (dist(u, v) + dist(v, w) - dist(u, w));
}
// gain of DROPPING v (without doing it)
static inline double dropGain(int v) {
    int u = prv[v], w = nxt[v];
    return -P[v] + (dist(u, v) + dist(v, w) - dist(u, w));
}

int main() {
    const double T0 = now_sec();
    const double TIME_LIMIT = 1.85;  // wall-clock budget (seconds)

    // ---- read instance ----
    if (scanf("%d", &N) != 1) return 0;
    if (N < 0) N = 0;
    DEPOT = N;
    X.assign(N + 1, 0.0); Y.assign(N + 1, 0.0); P.assign(N + 1, 0.0);
    {
        double dx = 0, dy = 0;
        if (scanf("%lf %lf", &dx, &dy) != 2) { dx = 0; dy = 0; }
        X[DEPOT] = dx; Y[DEPOT] = dy; P[DEPOT] = 0.0;
    }
    for (int i = 0; i < N; i++) {
        double xi, yi, pi;
        if (scanf("%lf %lf %lf", &xi, &yi, &pi) != 3) { xi = 0; yi = 0; pi = 0; }
        X[i] = xi; Y[i] = yi; P[i] = pi;
    }

    // Degenerate: nothing to choose.
    if (N == 0) { printf("0\n"); return 0; }

    Rng rng(0xC0FFEEULL ^ (uint64_t)N * 0x9E3779B97F4A7C15ULL);

    // ---- candidate lists: K nearest nodes per node, via a uniform grid ----
    const int K = min(N - 1 >= 0 ? N - 1 : 0, N <= 600 ? 12 : 8);
    cand.assign(N, {});
    if (K > 0) {
        // grid over the bounding box of nodes (depot included for safety)
        double minx = X[DEPOT], maxx = X[DEPOT], miny = Y[DEPOT], maxy = Y[DEPOT];
        for (int i = 0; i < N; i++) {
            minx = min(minx, X[i]); maxx = max(maxx, X[i]);
            miny = min(miny, Y[i]); maxy = max(maxy, Y[i]);
        }
        double w = max(1.0, maxx - minx), h = max(1.0, maxy - miny);
        int G = max(1, (int)floor(sqrt((double)N / 2.0)));
        double cw = w / G, ch = h / G;
        auto cellOf = [&](int i, int &cx, int &cy) {
            cx = (int)((X[i] - minx) / cw); if (cx < 0) cx = 0; if (cx >= G) cx = G - 1;
            cy = (int)((Y[i] - miny) / ch); if (cy < 0) cy = 0; if (cy >= G) cy = G - 1;
        };
        vector<vector<int>> grid(G * G);
        for (int i = 0; i < N; i++) {
            int cx, cy; cellOf(i, cx, cy);
            grid[cy * G + cx].push_back(i);
        }
        vector<pair<double,int>> buf;
        for (int i = 0; i < N; i++) {
            int cx, cy; cellOf(i, cx, cy);
            buf.clear();
            // expand rings until we have comfortably more than K candidates
            for (int r = 0; ; r++) {
                int x0 = max(0, cx - r), x1 = min(G - 1, cx + r);
                int y0 = max(0, cy - r), y1 = min(G - 1, cy + r);
                buf.clear();
                for (int gy = y0; gy <= y1; gy++)
                    for (int gx = x0; gx <= x1; gx++)
                        for (int j : grid[gy * G + gx])
                            if (j != i) buf.push_back({dist(i, j), j});
                if ((int)buf.size() >= K + 2 || (x0 == 0 && y0 == 0 && x1 == G - 1 && y1 == G - 1))
                    break;
            }
            int kk = min((int)buf.size(), K);
            nth_element(buf.begin(), buf.begin() + kk, buf.end());
            sort(buf.begin(), buf.begin() + kk);
            cand[i].reserve(kk);
            for (int t = 0; t < kk; t++) cand[i].push_back(buf[t].second);
        }
    }

    // ---- linked-list init: empty tour (just the depot looping to itself) ----
    nxt.assign(N + 1, -1); prv.assign(N + 1, -1); inTour.assign(N + 1, 0);
    nxt[DEPOT] = DEPOT; prv[DEPOT] = DEPOT; inTour[DEPOT] = 1;
    curProfit = 0.0;

    // ---- greedy cheapest-insertion construction ----
    // Repeatedly insert the (node, slot) with the largest positive add-gain.
    // To stay cheap, scan each not-in-tour node's best slot among: the depot,
    // and the tour-neighbours of its candidate nodes that are already in tour.
    {
        // seed with the single most profitable node placed at the depot, if any
        int bestSeed = -1; double bestSeedGain = 0.0;
        for (int v = 0; v < N; v++) {
            double g = P[v] - 2.0 * dist(DEPOT, v); // insert between depot and depot
            if (g > bestSeedGain) { bestSeedGain = g; bestSeed = v; }
        }
        if (bestSeed >= 0) linkAfter(DEPOT, bestSeed);

        // iterative greedy insertion
        bool progress = true;
        int guard = 0;
        while (progress && guard++ < N + 5) {
            progress = false;
            double bestG = 1e-9; int bestV = -1, bestU = -1;
            for (int v = 0; v < N; v++) {
                if (inTour[v]) continue;
                // try inserting after the depot and after each in-tour candidate
                double localBest = -1e18; int localU = -1;
                double g0 = addGain(DEPOT, v);
                if (g0 > localBest) { localBest = g0; localU = DEPOT; }
                for (int c : cand[v]) {
                    if (inTour[c]) {
                        double g = addGain(c, v);
                        if (g > localBest) { localBest = g; localU = c; }
                        int pc = prv[c];
                        double g2 = addGain(pc, v);
                        if (g2 > localBest) { localBest = g2; localU = pc; }
                    }
                }
                if (localBest > bestG) { bestG = localBest; bestV = v; bestU = localU; }
            }
            if (bestV >= 0) { linkAfter(bestU, bestV); progress = true; }
            if (now_sec() - T0 > TIME_LIMIT * 0.4) break; // leave time for local search
        }
    }

    // snapshot the best tour seen
    auto snapshot = [&](vector<int> &out) {
        out.clear();
        for (int v = nxt[DEPOT]; v != DEPOT; v = nxt[v]) out.push_back(v);
    };
    vector<int> bestTour; snapshot(bestTour);
    double bestProfit = curProfit;

    // ---- helper local moves ----
    // try the best ADD for node v (not in tour): scan candidate slots, return gain & u
    auto bestAddSlot = [&](int v, int &outU) -> double {
        double best = -1e18; int bu = -1;
        double g0 = addGain(DEPOT, v);
        best = g0; bu = DEPOT;
        for (int c : cand[v]) {
            if (!inTour[c]) continue;
            double g = addGain(c, v);
            if (g > best) { best = g; bu = c; }
            int pc = prv[c];
            double g2 = addGain(pc, v);
            if (g2 > best) { best = g2; bu = pc; }
        }
        outU = bu;
        return best;
    };

    // 2-opt pass on an array snapshot of the tour: un-cross edges using candidate
    // lists. Operates on the sequence depot, t[0..m-1], back to depot.
    auto twoOptPass = [&]() {
        vector<int> t; snapshot(t);
        int m = (int)t.size();
        if (m < 3) return;
        // sequence with depot at both ends: seq[0]=DEPOT, seq[1..m]=t, seq[m+1]=DEPOT
        vector<int> seq; seq.reserve(m + 2);
        seq.push_back(DEPOT);
        for (int v : t) seq.push_back(v);
        seq.push_back(DEPOT);
        vector<int> posInSeq(N + 1, -1);
        for (int i = 1; i <= m; i++) posInSeq[seq[i]] = i;
        bool improved = true; int rounds = 0;
        while (improved && rounds++ < 6) {
            improved = false;
            for (int i = 1; i <= m; i++) {
                int a = seq[i - 1], b = seq[i];
                double dab = dist(a, b);
                int base = (b == DEPOT) ? DEPOT : b;
                if (base == DEPOT) continue;
                for (int c : cand[base]) {
                    int j = posInSeq[c];
                    if (j <= i) continue;            // need j > i, c after b
                    int cc = seq[j], dd = seq[j + 1];
                    double dcd = dist(cc, dd);
                    double dac = dist(a, cc), dbd = dist(b, dd);
                    if (dac + dbd + 1e-9 < dab + dcd) {
                        // reverse seq[i..j]
                        int lo = i, hi = j;
                        while (lo < hi) {
                            swap(seq[lo], seq[hi]);
                            posInSeq[seq[lo]] = lo; posInSeq[seq[hi]] = hi;
                            lo++; hi--;
                        }
                        if (lo == hi) posInSeq[seq[lo]] = lo;
                        improved = true;
                        b = seq[i]; dab = dist(a, b);
                        if (b == DEPOT) break;
                    }
                }
                if (now_sec() - T0 > TIME_LIMIT) break;
            }
            if (now_sec() - T0 > TIME_LIMIT) break;
        }
        // rebuild linked list from seq and recompute profit exactly
        int prevNode = DEPOT;
        double dtot = 0.0, prizetot = 0.0;
        for (int i = 1; i <= m; i++) {
            int v = seq[i];
            nxt[prevNode] = v; prv[v] = prevNode;
            dtot += dist(prevNode, v);
            prizetot += P[v];
            prevNode = v;
        }
        nxt[prevNode] = DEPOT; prv[DEPOT] = prevNode;
        dtot += dist(prevNode, DEPOT);
        curProfit = prizetot - dtot;
    };

    // restore the linked list (and inTour/curProfit) from an ordered id vector
    auto restoreFromVec = [&](const vector<int> &t) {
        for (int v = 0; v < N; v++) inTour[v] = 0;
        inTour[DEPOT] = 1;
        int pn = DEPOT; double dtot = 0.0, prizetot = 0.0;
        for (int v : t) {
            nxt[pn] = v; prv[v] = pn; inTour[v] = 1;
            dtot += dist(pn, v); prizetot += P[v]; pn = v;
        }
        nxt[pn] = DEPOT; prv[DEPOT] = pn;
        dtot += dist(pn, DEPOT);
        curProfit = prizetot - dtot;
    };

    // One full deterministic local-search descent over the FUSED neighbourhood:
    //   ADD every profitable not-in-tour node at its best slot,
    //   DROP every loser (positive drop-gain),
    //   RELOCATE (Or-opt-1) every node whose drop+best-readd is a net gain,
    //   then a 2-opt pass to un-cross edges.
    // Repeat until a sweep makes no improvement or time runs out. Every accepted
    // move is a strict improvement, so curProfit climbs monotonically here.
    auto localDescent = [&]() {
        bool improvedAny = true;
        int sweep = 0;
        while (improvedAny && (now_sec() - T0) < TIME_LIMIT) {
            improvedAny = false;
            sweep++;
            // ADD pass
            for (int v = 0; v < N; v++) {
                if (inTour[v]) continue;
                int u; double g = bestAddSlot(v, u);
                if (g > 1e-7) { linkAfter(u, v); improvedAny = true; }
                if ((v & 1023) == 0 && (now_sec() - T0) > TIME_LIMIT) return;
            }
            // DROP pass
            for (int v = 0; v < N; v++) {
                if (!inTour[v]) continue;
                if (dropGain(v) > 1e-7) { unlink(v); improvedAny = true; }
            }
            // RELOCATE pass (drop then best re-add; commit only if net gain)
            for (int v = 0; v < N; v++) {
                if (!inTour[v]) continue;
                int u0 = prv[v];
                double dg = dropGain(v);
                unlink(v);
                int u; double ag = bestAddSlot(v, u);
                if (dg + ag > 1e-7 && u != u0) {
                    linkAfter(u, v); improvedAny = true;
                } else {
                    linkAfter(u0, v);  // exact revert to original slot
                }
                if ((v & 1023) == 0 && (now_sec() - T0) > TIME_LIMIT) return;
            }
            // 2-opt cleanup (reorders the kept set)
            double before = curProfit;
            twoOptPass();
            if (curProfit > before + 1e-7) improvedAny = true;
        }
    };

    // Initial descent from the greedy construction.
    localDescent();
    if (curProfit > bestProfit) { bestProfit = curProfit; snapshot(bestTour); }

    // ---- iterated local search with a perturbation kick ----
    // Kick = randomly toggle a handful of nodes (force a few in and a few out),
    // then re-descend; accept by an SA-style rule so we can cross small valleys,
    // always tracking the best tour seen.
    double T_start;
    {
        double s = 0; int cnt = 0;
        for (int i = 0; i < N && cnt < 2000; i++)
            for (int c : cand[i]) { s += dist(i, c); if (++cnt >= 2000) break; }
        T_start = (cnt ? s / cnt : 1000.0);
        if (T_start < 1.0) T_start = 1.0;
    }
    double accProfit = bestProfit;       // profit of the "current" ILS state
    vector<int> accTour = bestTour;      // current ILS state tour
    long long kicks = 0;
    while ((now_sec() - T0) < TIME_LIMIT) {
        kicks++;
        double prog = min(1.0, (now_sec() - T0) / TIME_LIMIT);
        double T = T_start * pow(1e-2, prog);  // cooling
        // perturb from the current accepted state
        restoreFromVec(accTour);
        int kickSize = 1 + (int)rng.nextu(4);  // 1..4 forced toggles
        for (int t = 0; t < kickSize; t++) {
            int v = rng.nextu(N);
            if (inTour[v]) {
                unlink(v);                       // force OUT
            } else {
                int u; bestAddSlot(v, u);        // pick a good slot
                linkAfter(u, v);                 // force IN (even if a small loss)
            }
        }
        localDescent();
        double newProfit = curProfit;
        if (newProfit > bestProfit) {
            bestProfit = newProfit; snapshot(bestTour);
        }
        // SA acceptance of the new local optimum as the next ILS state
        if (newProfit >= accProfit || rng.nextd() < exp((newProfit - accProfit) / T)) {
            accProfit = newProfit; snapshot(accTour);
        }
        if ((kicks & 63) == 0 && (now_sec() - T0) > TIME_LIMIT) break;
    }

    // final cleanup descent on the best tour
    restoreFromVec(bestTour);
    localDescent();
    if (curProfit > bestProfit) { bestProfit = curProfit; snapshot(bestTour); }

    // The empty tour (profit 0) is always available; if our best is negative,
    // emit the empty tour instead (never worse than 0).
    if (bestProfit < 0.0) bestTour.clear();

    // ---- output: k then the ids in visiting order ----
    string out;
    out += to_string((int)bestTour.size()); out += "\n";
    for (int v : bestTour) { out += to_string(v); out += "\n"; }
    fputs(out.c_str(), stdout);
    return 0;
}
```
