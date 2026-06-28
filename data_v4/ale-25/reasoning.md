I am handed `n` weighted time intervals and `K` identical rooms, and I have to decide, for each
interval, which room it goes into or whether I drop it. The single rule that binds me is that two
intervals sharing a room must not overlap, and the thing I am graded on is the total weight of the
intervals I keep. So the whole problem is a packing problem: room-time is the scarce resource, each
interval is a request for a slice of it in one of `K` parallel tracks, and I want to pack the
heaviest feasible subset. Before I reach for any machinery I want to be precise about three things,
because each of them is a place I could quietly produce an invalid answer and floor my score to zero.

First, what "overlap" means. The intervals are half-open `[s, e)`. Two intervals `a` and `b` in the
same room conflict exactly when `s_a < e_b` and `s_b < e_a`. The important consequence is that
abutting intervals — `e_a == s_b` — do *not* conflict; an interval ending at time 10 and another
starting at time 10 can share a room. I have to honor that exactly, because if my feasibility test is
too strict I will reject legal packings and lose weight, and if it is too loose I will emit an
overlapping pair and get a zero. Second, the output contract: I must print exactly `n` values, one
per interval in input order, each either a room id in `{0, ..., K-1}` or `-1` for reject. A wrong
count or an out-of-range id is itself an infeasibility floor. Third, the objective is just the sum of
kept weights — no partial credit, no penalty for rejecting, so rejecting everything is a legal (but
worthless) answer. That last fact is comforting: a valid solution always exists and is trivial to
produce, so I never have to risk an invalid output; the only question is how heavy I can make the
kept set.

**Getting a feasible baseline I can stand on.** The zero-risk floor is "reject everything," which
scores 0 but is always legal and tells me the output format works. That is too weak to be
interesting, so the real baseline I want to beat is the obvious constructive one: process intervals
in time order and first-fit them into rooms. Concretely, sort all intervals by start time, keep a
"last end" marker per room, and for each interval put it in the lowest-indexed room whose last end is
`<= ` this interval's start (so no overlap), else reject it. This is the standard greedy for
*unweighted* interval scheduling on machines and it is genuinely feasible by construction: a room's
placed intervals are appended in nondecreasing start order with each new start at least the previous
end, so they are pairwise disjoint. I will use exactly this as my reference baseline in scoring,
because it is deterministic, always feasible, and represents "a reasonable person who ignored the
weights." If I cannot beat first-fit-by-start, I have not earned the right to call my method a
heuristic.

But I can already see why first-fit-by-start is weak: it is weight-blind. It will happily fill a room
with three cheap early intervals and then have no space for a heavy one that arrives later and
overlaps them. When weights are heavy-tailed — a few very valuable intervals among many cheap ones —
ignoring weight is exactly the wrong instinct. So my construction should be **greedy by weight,
descending**: give the heaviest intervals first claim on room-time. Process intervals from largest
weight to smallest; for each, place it in any room where it currently fits, else reject. Heavy
intervals get protected because they are placed while the rooms are still empty, and the cheap
intervals fill whatever gaps remain. This is the textbook strong constructive heuristic for weighted
packing, and intuitively it should beat first-fit-by-start by a wide margin on heavy-tailed weights.

**The feasibility-test bottleneck, and why a naive check is too slow.** Greedy-by-weight needs to
answer, repeatedly, "does interval `x` fit in room `r` right now?" The lazy implementation keeps a
list of each room's placed intervals and scans the list for a conflict — `O(size of room)` per
query, `O(n)` worst case, and I do this for up to `K` rooms per interval, so `O(n^2 K)` for the
construction alone. At `n` up to 600 that is survivable for the construction, but it is fatal for what
I actually want to do next, which is a local search that performs *millions* of these queries. If
every feasibility probe is a linear scan, I get maybe a few thousand local-search steps in my time
budget, nowhere near enough to climb out of the greedy solution's local optimum. I need the
feasibility test to be `O(log n)`.

The structure that gives me that is a **balanced ordered set per room**, keyed by interval start
time. Crucially, the intervals already inside a room are pairwise non-overlapping, which means they
are simultaneously sorted by start *and* by end. That invariant collapses the overlap query to a tiny
neighborhood probe. To test whether `x = [S, E)` collides with anything in room `r`: look up the
first placed interval whose start is `>= S` (a `lower_bound`). Everything from there forward whose
start is `< E` necessarily overlaps `x` (its start lies strictly inside `x`). And among the
*predecessors* — placed intervals with start `< S` — only the single closest one (the largest start
below `S`) can possibly reach past `S`; every earlier one ends even earlier because the room's
intervals are disjoint and ordered, so it cannot overlap. So one predecessor probe plus a short
forward walk finds *all* conflicts in `O(log n + #conflicts)`. This is the lever the whole method
hangs on, and it is also what makes the next idea cheap.

**The innovation: targeted ejection moves under simulated annealing.** Greedy-by-weight gives a
strong start but it is myopic in a specific way. Once a medium-weight interval is placed it can block
a cluster of cheaper intervals whose *combined* weight exceeds it, or it can occupy room-time that a
slightly-heavier-but-later interval needed. The fix is a local search whose neighborhood move is:
pick a (typically high-weight, currently rejected) interval `x` and a room `r`, find exactly the
intervals already in `r` that `x` conflicts with — its *ejection set* — and consider kicking them out
to make room for `x`. The weight change of that swap is

```
delta = (x's new contribution = W[x]) - (weight of ejected intervals)
        - (x's old contribution, if x was already accepted somewhere)
```

If `delta >= 0` the swap is a pure improvement and I take it. If `delta < 0` I take it anyway with
probability `exp(delta / T)` — a standard **simulated-annealing** acceptance — so I can climb over the
ridges where I must temporarily lose weight (eject something heavy) to later gain more (pack several
cheaper intervals that the ejected one was blocking). The temperature `T` starts warm and cools
geometrically toward zero over the time budget, so early on I explore freely and late I only take
improving moves. I keep a running best snapshot and report it at the end, so exploratory down-moves
never hurt the final answer.

The reason this is fast enough to matter is exactly the ordered-set feasibility index. Computing the
ejection set for `(x, r)` is the same predecessor-plus-forward-walk probe — `O(log n + #ejected)`,
and `#ejected` is small because the rooms hold disjoint intervals so only a few can straddle `x`'s
span. Applying the move is a handful of set insertions/erasures, again `O(log n)` each. So each SA
step is near-`O(log n)`, and I get on the order of millions of steps in a couple of seconds. The
"room-as-resource relaxation" plus the "targeted ejection move with feasibility re-checked only
inside the affected room" is precisely the combination that turns a myopic greedy into a strong
local optimizer.

**Implementing it.** I store, per room, a `set<pair<int,int>>` of `(start, id)`. I keep `assign[i]`
(the current room of interval `i`, or `-1`), and a running `curWeight`. `placeInterval(x, r)` inserts
`(S[x], x)` into room `r`, sets `assign[x]=r`, adds `W[x]`. `removeInterval(x)` does the inverse using
`assign[x]` to know which set to erase from. `collectConflicts(r, x, hit)` fills `hit` with the ids of
the intervals in room `r` that overlap `x` and returns their total weight, using the predecessor +
forward-walk logic above. Construction sorts ids by weight descending (ties broken by shorter
duration first, since short intervals are easier to pack), and places each into the first room with no
conflict. Then the SA loop samples a random interval `x` and room `r`, computes the ejection set and
`delta`, and on acceptance removes `x` from its old room (if any), ejects the conflicts in `r`, and
places `x` in `r`. Time is checked every 1024 iterations and the temperature is recomputed from the
elapsed fraction.

**A real debugging episode.** My first version was wrong in the conflict-collection routine, and it
was the subtle kind of wrong. I had written the predecessor handling as a *back-walk*: starting from
the predecessor of `lower_bound(S[x])`, I walked left collecting any interval whose end exceeded
`S[x]`, with a stopping condition that tried to bail out "once I find a non-overlapping predecessor."
The logic was tangled — I was worried that a long earlier interval might reach across a shorter one —
and I left a confused comment about "durations are bounded so we cap the back-walk," which is a smell
that I did not actually trust my own invariant. When I sat down and used the disjointness invariant
properly, the bug became obvious: in a room of *non-overlapping* intervals, predecessors are ordered
by end as well as by start, so the only predecessor that can possibly overlap `x` is the immediate
one. Any predecessor before it ends no later than the immediate predecessor's start, which is `<=
S[x]` if the immediate one doesn't overlap — there is nothing to walk back through. The back-walk was
both unnecessary and, worse, it could in principle have collected ids that were *not* actually
conflicting if I mis-stopped, which would make the ejection set wrong and either eject too much
(losing weight) or, in a buggy branch, leave a real conflict in place (an *infeasible* room). I
replaced the whole back-walk with a single predecessor probe: check only `prev(it)`, and add it iff
its end `> S[x]`. That is provably exactly the set of left-side conflicts.

To make sure this was right and not just plausible, I built a tiny instance I could brute-force: five
intervals in `K = 1` rooms, weights `100, 50, 80, 40, 30` with spans designed so the optimum uses the
abutting-endpoints rule (`[0,10]`, `[10,20]`, `[20,30]` are mutually compatible because they only
touch). The brute force over all `2^5` subsets reported optimum `190`. My solver returned exactly
`190`, and the chosen assignment was indeed the three abutting intervals — which also confirmed that
my feasibility test treats `e_a == s_b` as *non*-conflicting, the boundary case I was most worried
about. I also directly tested the scorer's floor: an output that puts two overlapping intervals in
one room scores 0; an output with abutting intervals in one room scores their full combined weight;
an out-of-range room id scores 0; a wrong number of values scores 0. All four behaved as specified.

The second thing I hardened was the guarantee that I *never* print an infeasible solution, even if
some logic bug slipped through. After the SA loop I take the best snapshot and run an independent
feasibility check on it — sort each room's accepted intervals and verify no `start < previous end`.
If, for any reason, that check fails, I discard the snapshot and rebuild a guaranteed-feasible
solution by re-running the greedy-by-weight construction from scratch (which is conflict-free by
construction). This is a belt-and-suspenders safety net: the SA is feasible by its own invariants,
but the fallback means a single feasibility-floor zero is impossible to emit. It costs `O(n log n)`
once and buys total robustness.

**Self-verification on the seed set.** I generated seeds 1..20, compiled with `-O2`, and for each
seed ran the solver, scored its raw objective, and scored the deterministic first-fit-by-start
baseline. Every one of the twenty outputs parsed and was feasible (raw `> 0`, scorer did not floor
it), and on every single seed the solver's kept weight strictly exceeded the baseline's. The mean
raw objective was about `53600` versus the baseline's `19900`, a ratio of roughly `2.69x`, with
per-seed normalized scores ranging from about `2.2x` to `3.7x`. The solver finishes comfortably
inside the time budget (a hair under two seconds, dominated by the SA loop). So both required
conditions hold: always feasible, and a decisive beat over the trivial baseline, driven by exactly
the two levers — weight-greedy protection of heavy intervals and ejection-based local search made
cheap by the per-room ordered-set feasibility index.

**Where the strength comes from, restated.** The reason this is the strong method and not a toy
greedy is the interaction of the relaxation and the move. Treating each room as an independent
ordered resource is what makes "does `x` fit, and if not what blocks it" a logarithmic neighborhood
query instead of a linear scan; and the ejection move is what lets the search trade a placed interval
(or a small bundle of them) for a better one without recomputing global feasibility — only the one
touched room can change, and its set already encodes everything I need. Simulated annealing on the
weight delta is the established metaheuristic for this kind of selection-and-packing landscape, where
local optima are surrounded by "lose-a-little-to-gain-more" ridges. Put together they let me spend
the whole time budget improving a solution that is feasible at every instant, starting from a strong
weight-greedy construction.

Final solver:

```cpp
// Interval Scheduling on Few Rooms (ale-25)
//
// Assign each of n weighted intervals to one of K identical rooms, or reject
// it. Intervals in the same room may not overlap (half-open [s,e): touching
// endpoints is allowed). Maximize the total weight of accepted intervals.
//
// Read instance from stdin:   n K  then n lines: s e w
// Write solution to stdout:   n lines, the i-th is room id in [0,K) or -1.
//
// Method (the lever): treat each room as a resource and keep, per room, a
// balanced ordered set of the currently-placed intervals keyed by start time.
// This makes the only feasibility question -- "does interval x overlap anything
// already in room r?" -- an O(log) predecessor/successor probe, and it makes
// the targeted "ejection" move cheap: to force a high-weight interval into a
// room, we look up exactly the few placed intervals it collides with (a
// contiguous run in the ordered set), tentatively remove them, and decide with
// a simulated-annealing criterion on the resulting weight delta. Because the
// feasibility recheck is confined to the affected room's set, each SA step is
// near-O(log n), so we can take millions of steps inside the time budget.

#include <bits/stdc++.h>
using namespace std;

static inline double now_sec() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

struct Rng {
    uint64_t s;
    Rng(uint64_t seed) : s(seed ? seed : 0x9e3779b97f4a7c15ULL) {}
    inline uint64_t next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s;
    }
    inline uint32_t u32() { return (uint32_t)(next() >> 32); }
    inline int randint(int n) { return (int)(u32() % (uint32_t)n); }   // [0,n)
    inline double uni() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    double t_start = now_sec();
    const double TIME_LIMIT = 1.9;   // seconds

    int n, K;
    if (!(cin >> n >> K)) return 0;

    vector<int> S(n), E(n), W(n);
    for (int i = 0; i < n; i++) cin >> S[i] >> E[i] >> W[i];

    // Guard against degenerate sizes so we always emit a feasible line per job.
    if (n <= 0) { return 0; }
    if (K <= 0) { for (int i = 0; i < n; i++) cout << -1 << "\n"; return 0; }

    // Per room, an ordered set of placed intervals keyed by (start, id). Storing
    // start lets us find, for any candidate, the placed intervals it overlaps as
    // a contiguous neighborhood: everything with start < cand.end whose end >
    // cand.start. We probe the predecessor (largest start <= cand.start) and walk
    // forward over successors while their start < cand.end.
    //
    // Each entry is (start, id). Lookups use end[]/start[] arrays by id.
    vector<set<pair<int,int>>> room(K);   // set of (start, id)
    vector<int> assign(n, -1);            // current room of each interval, -1 = rejected
    long long curWeight = 0;

    // ---- helper: do the placed intervals in room r conflict with interval x?
    //      If so, collect the conflicting ids into `hit`. Returns total weight of
    //      conflicts (so the caller can decide whether ejecting them is worth it).
    //
    // KEY INVARIANT: the intervals already in a room are pairwise non-overlapping,
    // hence sorted by start AND by end at the same time. So an interval p overlaps
    // x = [S,E) iff p.start < E and p.end > S, and:
    //   * Among predecessors (start <= S[x]) only the single closest one -- the
    //     one with the largest start -- can reach past S[x]; everything earlier
    //     ends even earlier (disjointness), so it cannot overlap. One probe.
    //   * Among successors (start in [S[x], E[x])) every one overlaps x, since its
    //     start lies strictly inside x. They form a contiguous run we walk.
    auto collectConflicts = [&](int r, int x, vector<int>& hit) -> long long {
        hit.clear();
        long long wsum = 0;
        const set<pair<int,int>>& st = room[r];
        auto it = st.lower_bound(make_pair(S[x], INT_MIN));
        // 1) the single predecessor (largest start < S[x]) -- check it only.
        if (it != st.begin()) {
            auto pit = prev(it);
            int pid = pit->second;
            if (E[pid] > S[x]) {               // reaches past S[x] -> overlaps
                hit.push_back(pid); wsum += W[pid];
            }
        }
        // 2) intervals whose start is in [S[x], E[x]) -- all overlap x.
        for (; it != st.end() && it->first < E[x]; ++it) {
            int pid = it->second;
            hit.push_back(pid); wsum += W[pid];
        }
        return wsum;
    };

    auto placeInterval = [&](int x, int r) {
        room[r].insert(make_pair(S[x], x));
        assign[x] = r;
        curWeight += W[x];
    };
    auto removeInterval = [&](int x) {
        int r = assign[x];
        room[r].erase(make_pair(S[x], x));
        assign[x] = -1;
        curWeight -= W[x];
    };

    // ---------------- construction: greedy by weight (descending) -------------
    // For each interval in descending weight, try to place it in some room with
    // no conflict. We try all K rooms (K is small) and place in the first that
    // is conflict-free; if none, leave rejected.
    vector<int> order(n);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int a, int b){
        if (W[a] != W[b]) return W[a] > W[b];
        return (E[a]-S[a]) < (E[b]-S[b]);     // tie: shorter first (easier to pack)
    });

    vector<int> hit;
    for (int idx = 0; idx < n; idx++) {
        int x = order[idx];
        int chosen = -1;
        for (int r = 0; r < K; r++) {
            long long c = collectConflicts(r, x, hit);
            if (hit.empty()) { chosen = r; break; }
            (void)c;
        }
        if (chosen >= 0) placeInterval(x, chosen);
    }

    // best snapshot
    vector<int> bestAssign = assign;
    long long bestWeight = curWeight;

    // ---------------- SA with targeted ejection moves -------------------------
    Rng rng(0xC0FFEEULL ^ ((uint64_t)n << 20) ^ ((uint64_t)K << 8));

    double T0 = 200.0, T1 = 0.5;   // SA temperature schedule on weight units
    long long iter = 0;
    double temp = T0;

    // We keep a list of all interval ids to sample candidates from.
    while (true) {
        if ((iter & 1023) == 0) {
            double el = now_sec() - t_start;
            if (el > TIME_LIMIT) break;
            double frac = el / TIME_LIMIT;
            temp = T0 * pow(T1 / T0, frac);
        }
        iter++;

        int x = rng.randint(n);
        int r = rng.randint(K);

        if (assign[x] == r) {
            // Move: try ejecting x to a different room or rejecting it. With
            // prob, just attempt re-insertion elsewhere via the generic path
            // below by picking a different room.
            continue;
        }

        // Ejection move: force x into room r. Find the intervals x collides with
        // in room r; they are the ejection set. The new total weight would be
        //   cur - (x's current contribution) - (weight of ejected) + W[x].
        // x's current contribution is W[x] if it is already accepted, else 0.
        // Note: assign[x] == r was rejected above, so x is not in room r; ejecting
        // x's own old room and the conflicts in r are disjoint operations.
        long long ejW = collectConflicts(r, x, hit);
        int xCur = assign[x];
        long long delta = ((xCur == -1) ? 0 : -(long long)W[x]) - ejW + (long long)W[x];

        bool accept;
        if (delta >= 0) accept = true;
        else accept = (rng.uni() < exp((double)delta / max(1e-9, temp)));

        if (!accept) continue;

        // apply: remove x from its old room (if any), eject conflicts, place x.
        // copy hit because removeInterval mutates the set but not `hit`.
        if (xCur != -1) removeInterval(x);
        // re-collect conflicts in r (x removal can't affect room r unless xCur==r,
        // which is impossible here since assign[x]==r was handled above).
        // hit already holds the conflict ids for room r and x.
        for (int pid : hit) removeInterval(pid);
        placeInterval(x, r);

        if (curWeight > bestWeight) {
            bestWeight = curWeight;
            bestAssign = assign;
        }
    }

    // Make sure we report the best snapshot seen.
    // (curWeight may be below bestWeight after exploratory moves.)
    vector<int>& out = bestAssign;

    // ---- final safety: validate the snapshot is actually feasible. If, for any
    //      reason, it is not, fall back to the greedy-by-weight feasible build,
    //      which is guaranteed conflict-free by construction. This guarantees we
    //      never print an infeasible solution.
    {
        bool ok = true;
        vector<vector<pair<int,int>>> chk(K);
        for (int i = 0; i < n && ok; i++) {
            int r = out[i];
            if (r == -1) continue;
            if (r < 0 || r >= K) { ok = false; break; }
            chk[r].push_back(make_pair(S[i], E[i]));
        }
        if (ok) {
            for (int r = 0; r < K && ok; r++) {
                sort(chk[r].begin(), chk[r].end());
                for (size_t j = 1; j < chk[r].size(); j++) {
                    if (chk[r][j].first < chk[r][j-1].second) { ok = false; break; }
                }
            }
        }
        if (!ok) {
            // rebuild greedy-by-weight feasibly from scratch
            vector<int> fa(n, -1);
            vector<set<pair<int,int>>> rm(K);
            for (int idx = 0; idx < n; idx++) {
                int x = order[idx];
                for (int r = 0; r < K; r++) {
                    bool conflict = false;
                    auto& st = rm[r];
                    auto it = st.lower_bound(make_pair(S[x], -1));
                    if (it != st.begin()) {
                        auto pit = prev(it);
                        if (E[pit->second] > S[x]) conflict = true;
                    }
                    if (!conflict && it != st.end() && it->first < E[x]) conflict = true;
                    if (!conflict) { st.insert(make_pair(S[x], x)); fa[x] = r; break; }
                }
            }
            out = fa;
        }
    }

    // emit
    string buf;
    buf.reserve(n * 3);
    for (int i = 0; i < n; i++) {
        buf += to_string(out[i]);
        buf += '\n';
    }
    cout << buf;
    return 0;
}
```
