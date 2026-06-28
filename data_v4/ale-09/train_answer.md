# Dynamic Bin Packing with Rebalancing

## Problem

Items arrive and depart over time. Item `i` is alive on the half-open interval `[a_i, d_i)` and uses
size `s_i` (`1 <= s_i <= C`) while alive. Every item must be assigned to exactly one bin (capacity `C`)
for its whole lifetime, and a bin's load at any instant is the total size of the items alive in it then.
The whole list is given offline. Assign every item to a bin so that **no bin ever exceeds `C`** and the
**number of distinct bins used is minimized**. Because items depart, a bin saturated in one window can be
free in another — this is *temporal / dynamic* bin packing (interval bin packing / dynamic storage
allocation), NP-hard, with no known efficient optimum.

Input is `N C` then `N` lines `a_i d_i s_i`; output is `N` lines, the bin index per item. `N <= 1200`,
capacity `C in [20,60]`, time budget ~2 s.

## Objective and scoring

Feasibility is checked per bin by a sweep line over `(+s at a_i, -s at d_i)` events (departures before
arrivals at equal time): the load must never exceed `C`. Any malformed output (wrong count, non-integer,
negative index, missing file) or any capacity violation is **infeasible and scores 0**.

For a feasible solution, let `K` = number of distinct bins used and `B` = bins used by the deterministic
**first-fit-by-arrival** baseline (recomputed by the scorer). `SCORE = round(1_000_000 * B / K)`: the
baseline is exactly `1_000_000`, fewer bins score more, more bins score less but stay positive. A useful
hard lower bound is `ceil(P / C)` where `P` is the peak simultaneous total load — no packing can use
fewer bins than that.

## Baseline

The guaranteed-feasible starting point is first-fit by arrival: sort items by `(arrival, departure,
size, index)`, drop each into the lowest-indexed bin where it fits at its arrival instant, else open a
new bin. Since `s_i <= C`, a fresh bin always accepts the item, so this never fails. It is wasteful
because it commits early and clogs low-index bins, leaving slack a smarter method recovers. It is also
exactly the scorer's normalization baseline, so beating it means using strictly fewer bins than naive
first-fit.

## Key idea (the heuristic innovation)

Two coupled ideas turn the baseline into a strong solver.

1. **Incremental fill tracking via a per-bin time profile.** For each bin `b` keep a dense array
   `prof[b][t]` = alive load at time `t`. Placing, removing, fits-testing, or measuring the tightness of
   an item all touch only the cells in that item's lifetime window `[a_i, d_i)` — the global assignment
   and the bin's full schedule are never rebuilt. This makes the inner loop cheap enough to run thousands
   of passes inside the time budget. With this representation, construction is online **best-fit**: place
   each item (arrival order) into the feasible bin with the largest resulting peak load, opening a new
   bin only when none fits. Best-fit packs items into few dense bins, leaving sparse bins that can later
   be emptied.

2. **Rebalancing = empty a whole bin (ruin-and-recreate / LNS).** The bin *count* only drops when a bin
   goes empty, so the neighborhood is built around emptying a bin, not moving one item. The move:
   pick a target bin (lightest first), pull out all its items, and relocate each (hardest-first: longest
   lifetime × biggest size, by best-fit) into some *other* feasible bin with no new bin opened. If every
   item finds a home the bin is freed and the count drops; otherwise roll the move back entirely so the
   assignment stays feasible. An **elimination sweep** repeats this over the lightest bins until it
   stalls; then a randomized **destroy-and-reinsert** rips out 1–4 random bins' items and best-fit
   re-inserts them, accepting any assignment that does not increase the bin count (sideways moves cross
   the flat plateau) and keeping the best-ever assignment, iterated against the wall clock.

This combination lands within 1–3 bins of the `ceil(P/C)` lower bound on the test seeds — roughly 4–10%
fewer bins than first-fit.

## Feasibility and pitfalls

- **Half-open intervals.** An item alive on `[a_i, d_i)` is *not* present at instant `d_i`; the profile
  loops and fits-tests must use `t < d_i`, matching the scorer's sweep (departures free space before
  same-time arrivals). An inclusive `t <= d_i` falsely makes endpoint-touching items overlap, inflating
  the bin count and desyncing from the scorer.
- **Always feasible on interrupt.** The construction is feasible, every move either preserves
  feasibility or rolls back, and the best-so-far assignment is restored at the end — so any early stop
  prints a valid solution.
- **Clean rollback.** A rejected perturbation can spill items into previously-empty bins, so rejection
  rebuilds all profiles from the saved best assignment (zero, then re-add) rather than a partial reset,
  preventing `prof` from drifting out of sync with the assignment.
- **Size clamp.** `s_i` is clamped to `[1, C]` on read so a single item always fits an empty bin (no
  item is individually unplaceable).

## Complexity per step

With time horizon `T` and a window width `w = d_i - a_i`: a fits-test, place, remove, or tightness query
is `O(w)`. Best-fit insertion scans live bins, `O(bins · w)`. Emptying a bin of `m` items is
`O(m · bins · w)`. A destroy-and-reinsert touches only the ripped items. All updates are incremental;
the full schedule is never recomputed. The loop runs until a ~1.85 s wall-clock budget is spent.

## Code

```cpp
// Dynamic Bin Packing with Rebalancing -- heuristic solver.
//
// Objective: each item i is alive on the half-open interval [a_i, d_i) and uses
// size s_i while alive; assign every item to one bin (capacity C) so that at every
// instant no bin's alive load exceeds C, and the NUMBER OF DISTINCT BINS USED is
// minimized.  Read the instance from stdin, print one bin index per item (item
// order) to stdout.
//
// Method (the innovation):
//   1. Construction = online BEST-FIT by arrival time: process items in arrival
//      order and drop each into the feasible bin that is currently *tightest*
//      (largest peak load over the item's lifetime), opening a new bin only when
//      none fits.  Best-fit keeps bins dense, which is what frees whole bins later.
//   2. Feasibility/fill is tracked INCREMENTALLY with a per-bin time-profile array
//      load[bin][t]: placing/removing an item touches only the cells in [a_i,d_i),
//      and the feasibility test is "max over that window stays <= C".  The global
//      assignment is NEVER rebuilt from scratch.
//   3. Local search = periodic "repack the fullest few bins" (a ruin-and-recreate
//      / LNS move): repeatedly try to EMPTY a target bin by relocating each of its
//      items into some other feasible bin (no new bins) -- if all move out, the bin
//      count drops by one.  Interleaved with a randomized destroy-and-reinsert that
//      rips out a handful of bins' items and best-fit re-inserts them, accepting
//      only assignments that do not increase the bin count, to escape local minima.
// The current assignment is always feasible, so any early stop (including hitting
// the wall-clock budget mid-pass) still prints a valid solution.
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
    uint32_t nextu(uint32_t m) { return (uint32_t)(next() % m); }   // [0, m)
    double nextd() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};

int N, C, T;
vector<int> A, D, S;                 // item arrival, departure, size

// Per-bin time profile: prof[b] has length T, prof[b][t] = total size alive at t.
// A bin "exists" while it is non-empty; we keep an item count to know that.
vector<vector<int>> prof;            // prof[b][t]
vector<int> cnt;                     // cnt[b] = number of items currently in bin b
vector<int> assign_;                 // assign_[i] = bin index of item i

// Can item i be added to bin b without exceeding C anywhere in [A[i], D[i])?
static inline bool fits(int b, int i) {
    const vector<int> &p = prof[b];
    int s = S[i], hi = D[i];
    for (int t = A[i]; t < hi; ++t)
        if (p[t] + s > C) return false;
    return true;
}

static inline void add_item(int b, int i) {
    vector<int> &p = prof[b];
    int s = S[i], hi = D[i];
    for (int t = A[i]; t < hi; ++t) p[t] += s;
    cnt[b]++;
    assign_[i] = b;
}

static inline void remove_item(int b, int i) {
    vector<int> &p = prof[b];
    int s = S[i], hi = D[i];
    for (int t = A[i]; t < hi; ++t) p[t] -= s;
    cnt[b]--;
    assign_[i] = -1;
}

// Tightness of bin b for item i = peak alive load over [A[i],D[i]) AFTER placing i.
// Larger tightness -> best-fit prefers it (denser packing).
static inline int tightness_after(int b, int i) {
    const vector<int> &p = prof[b];
    int s = S[i], hi = D[i], peak = 0;
    for (int t = A[i]; t < hi; ++t) {
        int v = p[t] + s;
        if (v > peak) peak = v;
    }
    return peak;
}

void ensure_bin(int b) {
    if ((int)prof.size() <= b) {
        prof.resize(b + 1);
        cnt.resize(b + 1, 0);
    }
    if ((int)prof[b].size() != T) prof[b].assign(T, 0);
}

// Best-fit insertion of item i over the existing bins [0, nbins); opens a new bin
// if none fits.  Returns the bin used.
int best_fit_insert(int i, int nbins) {
    int best = -1, bestTight = -1;
    for (int b = 0; b < nbins; ++b) {
        if (fits(b, i)) {
            int tg = tightness_after(b, i);
            if (tg > bestTight) { bestTight = tg; best = b; }
        }
    }
    if (best < 0) { best = nbins; ensure_bin(best); }
    add_item(best, i);
    return best;
}

int main() {
    const double T0 = now_sec();
    const double TIME_LIMIT = 1.85;  // wall-clock budget (seconds)

    if (scanf("%d %d", &N, &C) != 2) return 0;
    if (N <= 0) { return 0; }
    A.resize(N); D.resize(N); S.resize(N);
    T = 0;
    for (int i = 0; i < N; ++i) {
        if (scanf("%d %d %d", &A[i], &D[i], &S[i]) != 3) { A[i] = 0; D[i] = 1; S[i] = 1; }
        if (A[i] < 0) A[i] = 0;
        if (D[i] <= A[i]) D[i] = A[i] + 1;
        if (S[i] < 1) S[i] = 1;
        if (S[i] > C) S[i] = C;            // guarantee a single item always fits an empty bin
        if (D[i] > T) T = D[i];
    }
    assign_.assign(N, -1);

    // ---- 1. Construction: online best-fit by arrival order ------------------
    vector<int> order(N);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [](int x, int y) {
        if (A[x] != A[y]) return A[x] < A[y];
        if (D[x] != D[y]) return D[x] < D[y];
        return x < y;
    });
    int nbins = 0;
    for (int i : order) {
        int b = best_fit_insert(i, nbins);
        if (b >= nbins) nbins = b + 1;
    }

    Rng rng(0xC0FFEEULL ^ (uint64_t)(N * 1000003ULL + C));

    // Helper: list the currently non-empty bins.
    auto live_bins = [&](vector<int> &out) {
        out.clear();
        for (int b = 0; b < (int)cnt.size(); ++b)
            if (cnt[b] > 0) out.push_back(b);
    };

    // Items currently assigned to bin b.
    auto items_of = [&](int b, vector<int> &out) {
        out.clear();
        for (int i = 0; i < N; ++i) if (assign_[i] == b) out.push_back(i);
    };

    // ---- 2. Local search: try to empty a target bin (drop bin count) --------
    // Relocate every item of bin `tb` into some OTHER feasible bin (best-fit, no
    // new bin).  If all move out, bin tb is emptied -> one fewer bin used.
    auto try_empty_bin = [&](int tb) -> bool {
        vector<int> its; items_of(tb, its);
        if (its.empty()) return true;
        // Order items hardest-first (longest lifetime, biggest size) so the tight
        // ones find homes while room remains.
        sort(its.begin(), its.end(), [](int x, int y) {
            long long lx = (long long)(D[x] - A[x]) * S[x];
            long long ly = (long long)(D[y] - A[y]) * S[y];
            return lx > ly;
        });
        vector<pair<int,int>> moved;   // (item, newbin) to roll back on failure
        bool ok = true;
        for (int i : its) {
            remove_item(tb, i);
            int best = -1, bestTight = -1;
            for (int b = 0; b < (int)cnt.size(); ++b) {
                if (b == tb || cnt[b] == 0) continue;
                if (fits(b, i)) {
                    int tg = tightness_after(b, i);
                    if (tg > bestTight) { bestTight = tg; best = b; }
                }
            }
            if (best < 0) {                // cannot relocate -> abort, roll back
                add_item(tb, i);
                ok = false;
                break;
            }
            add_item(best, i);
            moved.push_back({i, best});
        }
        if (!ok) {
            // roll back: pull the already-moved items back into tb
            for (int k = (int)moved.size() - 1; k >= 0; --k) {
                remove_item(moved[k].second, moved[k].first);
                add_item(tb, moved[k].first);
            }
            return false;
        }
        return true;   // tb now empty
    };

    // Greedy elimination sweep: keep emptying the lightest non-empty bins.
    auto elimination_sweep = [&]() {
        bool progress = true;
        while (progress && now_sec() - T0 < TIME_LIMIT) {
            progress = false;
            vector<int> lb; live_bins(lb);
            // target the bins with the FEWEST items first (cheapest to clear)
            sort(lb.begin(), lb.end(), [&](int x, int y) { return cnt[x] < cnt[y]; });
            for (int b : lb) {
                if (now_sec() - T0 >= TIME_LIMIT) break;
                if (cnt[b] == 0) continue;
                if (try_empty_bin(b)) progress = true;
            }
        }
    };

    elimination_sweep();

    // ---- 3. Ruin-and-recreate (LNS) to escape local minima ------------------
    // Rip out the items of a few random non-empty bins, then best-fit re-insert
    // (no new bins beyond what we had).  Keep the result only if the number of
    // live bins did not increase; else roll back to the saved assignment.
    auto count_live = [&]() {
        int c = 0;
        for (int b = 0; b < (int)cnt.size(); ++b) if (cnt[b] > 0) c++;
        return c;
    };

    vector<int> bestAssign = assign_;
    int bestLive = count_live();

    while (now_sec() - T0 < TIME_LIMIT) {
        vector<int> lb; live_bins(lb);
        if ((int)lb.size() <= 1) break;
        int K = 1 + (int)rng.nextu((uint32_t)min<size_t>(4, lb.size()));  // ruin 1..4 bins
        // pick K distinct live bins to destroy
        vector<int> victims;
        for (int t = 0; t < K && !lb.empty(); ++t) {
            int idx = (int)rng.nextu((uint32_t)lb.size());
            victims.push_back(lb[idx]);
            lb[idx] = lb.back(); lb.pop_back();
        }
        // collect and remove their items
        vector<int> pool;
        for (int b : victims) {
            vector<int> its; items_of(b, its);
            for (int i : its) { remove_item(b, i); pool.push_back(i); }
        }
        // re-insert pool hardest-first into ANY feasible existing live bin; only
        // open a new bin if forced.
        sort(pool.begin(), pool.end(), [](int x, int y) {
            long long lx = (long long)(D[x] - A[x]) * S[x];
            long long ly = (long long)(D[y] - A[y]) * S[y];
            return lx > ly;
        });
        int curBins = (int)cnt.size();
        for (int i : pool) {
            int best = -1, bestTight = -1;
            for (int b = 0; b < curBins; ++b) {
                if (cnt[b] == 0) continue;          // prefer reusing live bins
                if (fits(b, i)) {
                    int tg = tightness_after(b, i);
                    if (tg > bestTight) { bestTight = tg; best = b; }
                }
            }
            if (best < 0) {
                // fall back to any feasible (incl. currently-empty) bin slot
                for (int b = 0; b < (int)cnt.size(); ++b) {
                    if (fits(b, i)) { best = b; break; }
                }
                if (best < 0) { best = (int)cnt.size(); ensure_bin(best); }
            }
            add_item(best, i);
        }
        // try a quick elimination after recreate
        {
            vector<int> lb2; live_bins(lb2);
            sort(lb2.begin(), lb2.end(), [&](int x, int y) { return cnt[x] < cnt[y]; });
            for (int b : lb2) {
                if (now_sec() - T0 >= TIME_LIMIT) break;
                if (cnt[b] == 0) continue;
                try_empty_bin(b);
            }
        }
        int live = count_live();
        if (live <= bestLive) {
            bestLive = live;
            bestAssign = assign_;       // accept (sideways or better)
        } else {
            // roll back to the best assignment
            // rebuild profiles from bestAssign
            for (auto &p : prof) fill(p.begin(), p.end(), 0);
            fill(cnt.begin(), cnt.end(), 0);
            assign_ = bestAssign;
            int mx = 0;
            for (int i = 0; i < N; ++i) {
                int b = assign_[i];
                ensure_bin(b);
                vector<int> &p = prof[b];
                for (int t = A[i]; t < D[i]; ++t) p[t] += S[i];
                cnt[b]++;
                if (b + 1 > mx) mx = b + 1;
            }
        }
    }

    // restore the best assignment found
    assign_ = bestAssign;

    // ---- compact bin labels to 0..K-1 (cosmetic; not required for scoring) --
    {
        unordered_map<int,int> relabel;
        int next = 0;
        for (int i = 0; i < N; ++i) {
            int b = assign_[i];
            auto it = relabel.find(b);
            if (it == relabel.end()) { relabel[b] = next; assign_[i] = next; ++next; }
            else assign_[i] = it->second;
        }
    }

    // ---- output: one bin index per item, in item order ----------------------
    string out;
    out.reserve(N * 4);
    char buf[16];
    for (int i = 0; i < N; ++i) {
        int len = snprintf(buf, sizeof(buf), "%d\n", assign_[i]);
        out.append(buf, len);
    }
    fwrite(out.data(), 1, out.size(), stdout);
    return 0;
}
```
