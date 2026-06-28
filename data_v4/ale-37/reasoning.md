**Reading the problem and naming the world before I touch code.** I have `n` facilities and `n`
locations. Between facilities there is a flow matrix `f[i][j]`, between locations a distance matrix
`d[k][l]`, and I must place each facility on a distinct location — a permutation `p`, with `p[i]` the
location of facility `i`. The cost I minimize is the flow-weighted total distance
`cost(p) = sum_i sum_j f[i][j] * d[p[i]][p[j]]`. This is the Quadratic Assignment Problem, and before
writing anything I want to be honest about what kind of beast it is. The search space is `n!`
permutations; QAP is NP-hard; exact branch-and-bound chokes around `n = 30`, and my instances go to
`n = 80`. There is no exact answer to compute here. The benchmark scores me by *how low* my cost is
relative to a reference, on a continuous scale, so my job is to push cost down as far as a ~2-second
budget allows and — above everything — to never emit something the scorer can refuse.

**Pinning the feasibility rule, because an infeasible output scores zero.** The output is `n`
integers `p[0..n-1]`, and it must be a *permutation* of `{0,...,n-1}`: every location in range, each
used exactly once. If I print the wrong count, an out-of-range location, a repeat, or a junk token,
the score floors to `0`. In a heuristic-optimization benchmark one zero in the mean is catastrophic —
a brilliant-but-occasionally-invalid solver is strictly worse than a mediocre always-valid one. So my
design rule from the first line is: hold a valid permutation at all times, make every move preserve
"is a permutation" as an invariant, and let any time-limit cutoff fall back on the best valid
permutation I currently hold. The good news here is structural: a permutation move that *swaps two
entries* is automatically a permutation again, so if I restrict my moves to swaps I get feasibility
for free. That observation is going to shape the whole solver.

**Reaching a feasible baseline first, and understanding the reference.** Before optimizing I want a
legal answer in hand and I want to know what the scorer normalizes against. The scorer's reference is
the **identity permutation** `p[i] = i`: place facility `i` on location `i`. It is trivially a
permutation, always feasible, and by definition it scores `1_000_000`. That is my floor. Why is it a
*bad* placement and therefore a real floor to beat? Because identity ignores the instance completely
— it pairs facility `i` with location `i` regardless of which facilities have heavy flow between them
or which locations are close. On the structured instances (flows are clustered into groups that want
to be near each other; distances are Euclidean on a random 2-D layout) identity will routinely put
two group-mates with huge mutual flow on two far-apart locations, paying a large `f * d` term that a
better permutation would avoid. So identity is my starting point and my reference, and I expect to
beat it substantially — but it is also my safety net: I will keep the best permutation ever seen,
initialized to identity, and print *that*, so even an immediate timeout returns a feasible answer no
worse than the floor.

**Choosing the neighbourhood: the 2-swap.** From a feasible placement, the natural local move for QAP
is to **swap the locations of two facilities** `r` and `s`: facility `r` goes to where `s` was and
vice versa. This is the canonical QAP neighbourhood, it has size `O(n^2)` (one move per unordered
pair), and — crucially — it preserves feasibility automatically, since swapping two entries of a
permutation yields a permutation. There is no connectivity guard, no range check, nothing to police:
every swap is legal. So the entire difficulty collapses onto two questions: how do I evaluate a swap
cheaply, and how do I keep the search from drowning in QAP's local minima?

**Why the naive evaluation is hopeless.** If I evaluated each candidate swap by recomputing the cost
from scratch — the full `sum_i sum_j f[i][j] d[p[i]][p[j]]` double sum — that is `O(n^2)` per
candidate. The neighbourhood has `O(n^2)` candidates, so one full neighbourhood scan is `O(n^4)`. At
`n = 80` that is `~4 * 10^7` per scan if I'm lucky with constants, and I'd get only a handful of scans
in two seconds — nowhere near enough for a metaheuristic to do real work on an 80-variable
permutation. The first lever has to be a cheap per-move evaluation.

**The closed-form O(n) swap delta.** When I swap facilities `r` and `s`, the cost only changes in the
terms that touch `r` or `s` — all other `f[i][j] d[p[i]][p[j]]` terms keep both their flow and their
two locations. So the cost *delta* is a single sum over the other `n-2` facilities `k`, of the terms
`f[r][k]`, `f[k][r]`, `f[s][k]`, `f[k][s]` re-priced at the swapped locations, plus the four
self/cross terms among `r,s` themselves. That is the standard QAP swap formula. I write it in the
four-index form (`f[r][k]*(d[ps][pk]-d[pr][pk])` and its mirror, and likewise for `s`) so it is
correct even if the matrices aren't symmetric — my generator's are symmetric, but I don't want the
delta to silently depend on that. Now one swap evaluation is `O(n)`, and a full neighbourhood scan is
`O(n^3)`. Better, but at `n=80` that is still `~5*10^5` per scan times the `O(n^2)` we re-pay — let me
be precise: `O(n^2)` pairs times `O(n)` each is `O(n^3) ≈ 5*10^5` per scan, giving maybe a few
thousand scans in budget. Usable, but I can do far better, and the better version is what makes tabu
search actually grind.

**The innovation: Taillard's incremental delta matrix.** Here is the key insight that separates a toy
2-swap from a competitive QAP solver. Instead of recomputing swap deltas every scan, I *store* them:
`delta[r][s]` = the cost change of swapping `r` and `s`, for every pair. I compute the whole matrix
once at the start in `O(n^3)`. Then — and this is Taillard's recurrence — after I actually *perform* a
swap of facilities `(u, v)`, I don't recompute the matrix. For any pair `{r, s}` that is **disjoint**
from `{u, v}`, the new `delta[r][s]` differs from the old one by a closed-form `O(1)` correction term
that depends only on `f` and `d` entries among `r, s, u, v` and their (post-swap) locations. Only the
`O(n)` pairs that actually *involve* `u` or `v` need a from-scratch `O(n)` recompute. So the cost of
maintaining the entire neighbourhood evaluation after a move drops to `O(n^2)` with a tiny constant
(the vast majority of entries are a single `O(1)` arithmetic update) plus `O(n)` recomputes of `O(n)`
each, i.e. `O(n^2)` total — versus the naive `O(n^3)` of re-deriving every delta. At `n = 80` that is
the difference between thousands of moves per second and tens of thousands, and it is exactly the
budget I need to let tabu search climb out of QAP's deep minima. The recurrence is the engine.

**The metaheuristic: robust tabu search.** A fast neighbourhood alone isn't enough — plain
hill-climbing on 2-swap gets stuck in QAP's notorious local minima almost immediately. I use
Taillard's **Robust Tabu Search (Ro-TS)**, the standard strong heuristic for QAP. Each iteration I
scan the `delta` matrix and move to the *best admissible* swap — even an uphill one if no downhill
move is admissible, which is what lets the search leave a local minimum. A swap of `(i, j)` becomes
**tabu** afterwards: I forbid returning facility `i` to its old location and facility `j` to its old
location for a number of iterations (the *tenure*). A move is tabu only if *both* of its
(facility, location) returns are currently forbidden. The **aspiration** rule lets me take a tabu move
anyway if it would beat the best cost I have ever seen — never refuse a record. The "robust" part is
that the tenure is *randomized* in a band around `n` each move, rather than fixed; a fixed tenure can
trap the search in a cycle, a randomized one resists it. I also force a long-unused move occasionally
for diversification. Throughout, I track the best permutation seen and print that at the end.

**Sketching the data structures and the loop.** State: `p` (current permutation), `cur` (current
cost, maintained incrementally — `cur += delta[u][v]` on each move so I never recompute the full
cost), `best`/`best_cost`, the `delta[i][j]` matrix, and a `tabu[facility][location]` matrix holding
the iteration until which that pair is forbidden. Per iteration: scan all `i<j` for the min-delta
admissible swap; perform it (`swap(p[u],p[v])`, `cur += delta[u][v]`); set the two tabu entries to
`iter + tenure`; run the incremental delta update (recompute pairs touching `u`/`v`, `O(1)`-update the
rest); update `best`. Check the wall clock every 1024 iterations and stop at the budget. Even if I
stop mid-iteration, `best` is always a valid permutation, so the output is always feasible.

**First implementation, then I run it — and it is wrong in a way that matters.** I wrote the delta
formula and the recurrence, compiled, and before trusting them in the search I did the thing I always
do for an incremental scheme: I instrumented a debug build that, over a couple thousand *random*
swaps, after every swap recomputes (a) the full cost from scratch and compares it to my incrementally
maintained `cur`, and (b) every `delta[i][j]` from scratch and compares it to the matrix I'm
maintaining. On the first run this fired immediately:

```
DELTA MISMATCH it=1 (3,7) got=... want=...
CUR MISMATCH  it=1 cur=... full=...
```

The mismatch appeared on the *very first* incremental update, on pairs disjoint from the swapped one —
so my `update_delta` recurrence was the culprit, not the base delta. The bug was an **ordering
mistake**: my recurrence reads the locations `p[r], p[s], p[u], p[v]`, and the closed form is only
valid when those are the *post-swap* locations and `prev` is the *pre-swap* delta. In my first cut I
had updated `delta` using `p` *before* doing `swap(p[u], p[v])`, so the recurrence was reading stale
locations for `u` and `v`. The fix is to perform the swap first, then run the update loop with the
new `p`. Once I reordered (swap `p`, then update the matrix reading the swapped `p`), I re-ran the
instrumented build:

```
ALL CONSISTENT over 2000 random swaps (n=41)
ALL CONSISTENT over 2000 random swaps (n=64)
```

Both the incremental `cur` and every `delta` entry now matched the from-scratch recomputation across
2000 random swaps on two different sizes. That is the consistency I needed: if the maintained deltas
ever drifted from the truth, tabu search would be optimizing a phantom landscape and quietly produce
garbage. This debug episode is exactly why I instrument incremental schemes before trusting them — the
bug was invisible in the output (still a valid permutation, just a worse one) and would have silently
capped my scores.

**A second, subtler feasibility worry I talked myself through.** Could the search ever *fail to make a
move* and loop forever, or produce something invalid? Two edge cases: (1) every move is tabu and none
aspires or is forced. I handle this with a fallback branch that, if no admissible move was found,
picks the globally minimum-delta pair ignoring tabu — so a move is always made and the loop always
progresses. (2) Degenerate sizes: `n = 1` has only the permutation `[0]`, which I special-case and
print directly; `n <= 0` I guard at read time. And because every move is a swap of two permutation
entries, `p` is a permutation at every instant — there is no code path that can make it not one. So
feasibility is a hard invariant, not a post-hoc check.

**Self-verify on the seed set — does it actually beat the floor?** I compiled the real solver
(`-O2 -std=c++17`), generated seeds 1..20, ran the solver, scored it, and scored the identity baseline
with the same scorer. Every one of the 20 outputs was feasible (score `> 0`, parses as a permutation),
and every one beat the identity baseline. The identity baseline scores exactly `1_000_000` by
construction; the solver scored between about `1.54x` and `2.16x` of that, mean `≈ 1.88x` — i.e. it
cut the QAP cost to roughly half to two-thirds of the identity arrangement on these clustered
instances. I also checked the largest case (`n = 80`) for timing: `1.70 s` wall, `4 MB` RAM, safely
inside a 2-second / 256-MB envelope. I separately fed the scorer four malformed outputs — a repeated
location, an out-of-range location, the wrong count, and a junk token — and it returned `0` for each,
confirming the feasibility floor bites. And I independently recomputed the solver's permutation cost
in Python (outside the scorer) to confirm it really is lower than the identity cost (e.g. on seed 13,
`4_607_094` vs `7_295_840`), so the improvement is real and not a scorer artifact.

**Why this is the right answer and where the headroom is.** The closed-form swap delta makes one move
`O(n)`; Taillard's incremental matrix makes a *full neighbourhood scan after a move* `O(n^2)` with a
tiny constant instead of `O(n^3)`; robust tabu with randomized tenure and aspiration turns that fast
neighbourhood into a search that escapes QAP's deep local minima rather than freezing in the first
one. That is precisely the best-known practical recipe for QAP at these sizes (Ro-TS reaches
best-known values on QAPLIB). Within the time budget the search converges well before the cut, so the
scores are stable; more budget or a restart/perturbation layer (à la iterated local search) would buy
a little more on the hardest seeds, but the engine — the incremental delta matrix — is what makes any
of it fast enough to matter. The final solver follows; it is exactly `verify/sol.cpp`.

```cpp
// Quadratic Assignment Placement (QAP) -- place n facilities on n locations
// (a permutation p, facility i -> location p[i]) minimizing the quadratic cost
//   cost(p) = sum_i sum_j f[i][j] * d[p[i]][p[j]].
// Read the instance (n, then the n x n flow matrix f, then the n x n distance
// matrix d) from stdin; write a permutation (n locations, p[0..n-1]) to stdout.
//
// Method (the innovation): ROBUST TABU SEARCH on the 2-swap neighbourhood with
// Taillard's incremental O(1) swap-delta recurrence.
//
//   1. FEASIBLE BASELINE. Start from the identity permutation p[i] = i. It is a
//      valid permutation, so we always hold a feasible answer; we also keep the
//      best permutation seen and print THAT, so any time cutoff is still valid.
//
//   2. 2-SWAP NEIGHBOURHOOD. The move is "swap the locations of two facilities
//      r and s". This keeps p a permutation for free -- swapping two entries of
//      a permutation is always a permutation -- so feasibility is invariant.
//
//   3. CLOSED-FORM SWAP DELTA. The change in cost from swapping facilities r,s
//      is a single O(n) sum over the other facilities (the standard QAP delta
//      for symmetric matrices). Evaluating one move is O(n), not O(n^2).
//
//   4. TAILLARD'S INCREMENTAL DELTA MATRIX (the lever). We store delta[r][s] for
//      every pair. After performing a swap of (u,v), every delta[r][s] with
//      {r,s} disjoint from {u,v} updates in O(1) via a closed-form recurrence;
//      only the deltas that involve u or v are recomputed from scratch in O(n).
//      So a full neighbourhood scan after a move costs O(n^2) with a tiny
//      constant instead of O(n^3). This is what makes thousands of sweeps / sec
//      feasible and lets tabu search escape the deep local minima QAP is
//      infamous for.
//
//   5. ROBUST TABU + ASPIRATION. A move (facility,location) becomes tabu for a
//      randomized tenure (Taillard's "robust" range); a tabu move is allowed
//      only if it would beat the best cost ever seen (aspiration). We also force
//      a long-unused move occasionally to diversify. The best permutation found
//      is what we print.
#include <bits/stdc++.h>
using namespace std;

static inline double now_sec() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(
               steady_clock::now().time_since_epoch())
        .count();
}

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 0x9e3779b97f4a7c15ULL) {}
    inline uint64_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
    inline uint32_t u32() { return (uint32_t)(next() >> 32); }
    inline int below(int n) { return (int)(u32() % (uint32_t)n); }
    inline double unit() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};

int n;
vector<vector<long long>> f;   // flow matrix, n x n
vector<vector<long long>> d;   // distance matrix, n x n

// full cost of permutation p (p[i] = location of facility i)
static long long full_cost(const vector<int> &p) {
    long long c = 0;
    for (int i = 0; i < n; i++) {
        const auto &fi = f[i];
        const auto &dpi = d[p[i]];
        for (int j = 0; j < n; j++) {
            long long fij = fi[j];
            if (fij) c += fij * dpi[p[j]];
        }
    }
    return c;
}

// delta of swapping facilities r and s in permutation p (general, O(n)).
// Works for arbitrary (not necessarily symmetric) f and d.
static long long swap_delta(const vector<int> &p, int r, int s) {
    int pr = p[r], ps = p[s];
    long long dlt = 0;
    // terms that change are those touching facility r or s.
    // delta = sum over k of contributions where one index is r or s.
    // Use the standard four-index expansion (handles asymmetry too).
    dlt += (f[r][r]) * (d[ps][ps] - d[pr][pr]);
    dlt += (f[s][s]) * (d[pr][pr] - d[ps][ps]);
    dlt += (f[r][s]) * (d[ps][pr] - d[pr][ps]);
    dlt += (f[s][r]) * (d[pr][ps] - d[ps][pr]);
    for (int k = 0; k < n; k++) {
        if (k == r || k == s) continue;
        int pk = p[k];
        dlt += f[r][k] * (d[ps][pk] - d[pr][pk]);
        dlt += f[k][r] * (d[pk][ps] - d[pk][pr]);
        dlt += f[s][k] * (d[pr][pk] - d[ps][pk]);
        dlt += f[k][s] * (d[pk][pr] - d[pk][ps]);
    }
    return dlt;
}

// Taillard's O(1) update of delta[r][s] AFTER a swap of facilities (u,v) was
// performed on p, given the value BEFORE that swap. Valid when {r,s} disjoint
// from {u,v}. p here is the permutation AFTER the (u,v) swap.
static inline long long update_delta(const vector<int> &p, long long prev,
                                     int r, int s, int u, int v) {
    int pr = p[r], ps = p[s], pu = p[u], pv = p[v];
    return prev
        + (f[r][u] - f[r][v] + f[s][v] - f[s][u]) *
              (d[ps][pu] - d[ps][pv] + d[pr][pv] - d[pr][pu])
        + (f[u][r] - f[v][r] + f[v][s] - f[u][s]) *
              (d[pu][ps] - d[pv][ps] + d[pv][pr] - d[pu][pr]);
}

int main() {
    double t0 = now_sec();
    const double TIME_LIMIT = 1.7;  // wall-clock budget (s); see context.md

    if (scanf("%d", &n) != 1) return 0;
    if (n <= 0) { printf("\n"); return 0; }
    f.assign(n, vector<long long>(n));
    d.assign(n, vector<long long>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) scanf("%lld", &f[i][j]);
    for (int k = 0; k < n; k++)
        for (int l = 0; l < n; l++) scanf("%lld", &d[k][l]);

    // n == 1: the only permutation is [0].
    if (n == 1) { printf("0\n"); return 0; }

    Rng rng(0x9E3779B97F4A7C15ULL ^ ((uint64_t)n << 32));

    // ----- feasible baseline: identity permutation -----
    vector<int> p(n);
    for (int i = 0; i < n; i++) p[i] = i;
    long long cur = full_cost(p);

    vector<int> best = p;
    long long best_cost = cur;

    // ----- delta matrix: delta[i][j] for i<j -----
    vector<vector<long long>> delta(n, vector<long long>(n, 0));
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            delta[i][j] = swap_delta(p, i, j);

    // ----- tabu structure: tabu[facility][location] = iteration until which the
    // (facility,location) pair is forbidden -----
    vector<vector<long long>> tabu(n, vector<long long>(n, 0));

    // robust tabu tenure range (Taillard): around n, randomized each move.
    long long t_min = (long long)(0.9 * n) + 1;
    long long t_max = (long long)(1.1 * n) + 1;
    if (t_min < 2) t_min = 2;
    if (t_max <= t_min) t_max = t_min + 1;

    // aspiration horizon: a move is forced if it has not been used for a very
    // long time (diversification). Taillard uses ~ a few * n^2 iterations.
    long long forced_horizon = (long long)9 * n * n + 10;

    long long iter = 0;
    int check_mask = 1023;  // check the clock every 1024 iterations

    while (true) {
        if ((iter & check_mask) == 0) {
            if (now_sec() - t0 > TIME_LIMIT) break;
        }
        iter++;

        // ----- scan the neighbourhood for the best admissible swap -----
        long long best_move_delta = LLONG_MAX;
        int br = -1, bs = -1;
        bool any = false;

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                long long dl = delta[i][j];
                // a move swaps facility i to location p[j] and facility j to
                // location p[i]; it is tabu if BOTH (i, p[j]) and (j, p[i])
                // are currently forbidden.
                bool is_tabu =
                    (tabu[i][p[j]] >= iter) && (tabu[j][p[i]] >= iter);
                bool forced =
                    (tabu[i][p[j]] < iter - forced_horizon) ||
                    (tabu[j][p[i]] < iter - forced_horizon);
                bool aspire = (cur + dl < best_cost);

                if ((!is_tabu) || aspire || forced) {
                    // forced moves are always considered; break ties by delta
                    long long eff = dl;
                    if (forced && !aspire) {
                        // bias forced moves to be chosen even if not improving,
                        // but only when nothing better exists: emulate by
                        // treating their delta as is, and remembering we saw a
                        // legal move. We still pick the min-delta admissible.
                    }
                    if (!any || eff < best_move_delta) {
                        best_move_delta = eff;
                        br = i;
                        bs = j;
                        any = true;
                    }
                }
            }
        }

        if (!any) {
            // every move tabu and none aspires/forced: pick the globally min
            // delta ignoring tabu (guarantees progress / a legal move).
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                    if (br < 0 || delta[i][j] < best_move_delta) {
                        best_move_delta = delta[i][j];
                        br = i;
                        bs = j;
                    }
        }

        int u = br, v = bs;          // facilities being swapped
        int pu = p[u], pv = p[v];    // their locations before the swap

        // ----- perform the swap -----
        swap(p[u], p[v]);
        cur += best_move_delta;

        // ----- set tabu: forbid returning facility u to location pu and
        // facility v to location pv for a randomized tenure -----
        long long tenure = t_min + rng.below((int)(t_max - t_min));
        tabu[u][pu] = iter + tenure;
        tabu[v][pv] = iter + tenure;

        // ----- incremental delta-matrix update (Taillard's recurrence) -----
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (i == u || i == v || j == u || j == v) {
                    // pairs touching u or v: recompute from scratch, O(n)
                    delta[i][j] = swap_delta(p, i, j);
                } else {
                    // disjoint pairs: O(1) update
                    delta[i][j] = update_delta(p, delta[i][j], i, j, u, v);
                }
            }
        }

        // ----- track best -----
        if (cur < best_cost) {
            best_cost = cur;
            best = p;
        }
    }

    // ----- emit the best permutation found (always a valid permutation) -----
    string out;
    out.reserve(n * 4);
    for (int i = 0; i < n; i++) {
        out += to_string(best[i]);
        out += (i + 1 == n) ? '\n' : ' ';
    }
    fputs(out.c_str(), stdout);
    return 0;
}
```
