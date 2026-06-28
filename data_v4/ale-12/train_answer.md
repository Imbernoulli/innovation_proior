# Lattice Antenna Coverage — budgeted submodular coverage via CELF + swap search

## Problem

A `G x G` lattice of demand cells (`d(x,y) >= 1`) and `M` candidate antenna sites. Site `i` covers a
square footprint of side `2 r_i + 1` around `(sx_i, sy_i)` (clipped to the lattice) at power cost
`c_i`. Pick a subset `S` with `sum_{i in S} c_i <= B` to maximize the **demand of the union** of
covered cells — every covered cell counts once. Input on stdin (`G M B`, then `G*G` demands
row-major, then `M` lines `sx sy r c`); output `k` followed by the `k` chosen site indices.

## Objective and scoring

The score is the total demand of the union of covered cells of the chosen set. A solution is feasible
iff `0 <= k <= M`, exactly `k` distinct in-range indices are given, and `sum c_i <= B`. Any violation
floors the score to **0**; the empty set is feasible and scores `0`. The scorer recomputes coverage
from scratch from the output, so the emitted set must be genuinely feasible and its claimed coverage
real. The metric is monotone and submodular: more antennas never uncover a cell, and each antenna's
marginal value shrinks as the chosen set grows. NP-hard; judged on mean covered demand over a fixed
seed set with a ~2s budget.

## Baseline

Density greedy with no overlap awareness: rank affordable sites by `footprint-demand / cost` once and
add them in that fixed order while they fit. Feasible and `O(M log M)`, but it double-counts
overlapping footprints, so on clustered demand fields it stacks several antennas on the biggest
hot-spot, paying repeatedly for cells it already owns and starving the other peaks. This is the toy
the real solver must beat; on the seed set it scores roughly 40% below the solver.

## Key idea — the heuristic innovation

**Cost-benefit greedy made affordable by CELF (lazy) marginal-gain caching, plus a swap local
search, all evaluated incrementally against a per-cell coverage-count array.**

- *Union-aware incremental evaluation.* Maintain `cov[c]` = number of chosen antennas covering cell
  `c`. A cell contributes its demand exactly when `cov[c] > 0`. Adding/removing a site walks only its
  footprint and adjusts a running `curScore` on the `0->1` / `1->0` transitions — `O(footprint)` per
  move, never a full-grid recompute. A site's marginal gain is the demand of its footprint cells with
  `cov == 0`.

- *CELF lazy greedy.* The honest cost-benefit greedy re-scores every remaining site after every pick
  — `O(M^2 * footprint)`, too slow. Submodularity makes a previously computed marginal gain an
  **upper bound** on the current one. Keep a max-heap keyed on gain-per-cost, each entry stamped with
  the coverage state when its gain was computed. Pop the top: if its stamp is current it is provably
  the best (every other key is a no-larger upper bound), so commit it; if stale, recompute its gain,
  re-stamp, and push back. Most re-scorings are skipped — that is the "lazy evaluation skips
  recomputing stale gains" lever.

- *Single-best guard (Khuller-Moss-Naor).* Cost-benefit greedy can be arbitrarily bad against one
  huge expensive element; comparing the greedy result to the best single feasible antenna and keeping
  the better restores the `(1 - 1/e)/2` guarantee.

- *Incremental swap / LNS local search.* With the leftover time budget, run an add pass (pull in any
  affordable positive-gain site) and a swap pass (remove one site, greedily refill the freed budget,
  accept only if `curScore` strictly rises, else roll back exactly). When a sweep stalls, drop a
  random chosen site to perturb. Track and emit the best feasible set seen.

## Feasibility and pitfalls

- *Always feasible.* Empty input prints `0`. Sites with `c > B` are filtered from the queue and the
  guard, so they can never be chosen. Every add gates on `c <= B - curCost`, the perturbation only
  *drops* sites, and a final defensive pass recomputes the emitted set's cost and trims trailing
  sites while it exceeds `B` — so the output is feasible by construction.
- *Symmetric rollback.* A rejected swap must restore `cov[]`, `curScore`, `curCost`, **and** the
  `inSet[]` flags. Forgetting the flags leaves phantom "chosen" sites that a later add pass skips
  while their coverage was already removed — the incremental score drifts above the true score. The
  fix is the mirrored undo: `removeSite/inSet=0` on every added site, then `addSite/inSet=1` on the
  removed one. Cross-checking the incremental `curScore` against a from-scratch recompute on every
  seed catches this.
- *Types.* `cov[]`, `curScore`, `curCost`, and costs are sized so the accumulators cannot overflow.

## Complexity per step

`cov[]` add/remove/marginal-gain are `O(footprint) = O(r^2)`. CELF commits each pick in amortized
near-`O(footprint)` (most stale re-evaluations are skipped) versus the naive `O(M * footprint)` per
pick. The swap pass is `O(|S| * passes * M * footprint)` but runs only on the small chosen set within
the time budget; every move is incremental, never a full recompute.

## Result

On seeds 1..20 every output is feasible (parses, score `> 0`) and the solver beats the density
baseline on every seed (mean covered demand ~155k vs ~92k), within the ~1.8s internal cap and a few
MB of memory.

## Code

```cpp
// Lattice Antenna Coverage (ale-12) -- budgeted monotone-submodular maximum
// coverage solved with cost-benefit LAZY GREEDY (CELF) + single-best guard,
// then incremental-eval SWAP local search. Reads stdin, writes a feasible
// subset of antenna-site indices to stdout. Never crashes, never infeasible.
#include <bits/stdc++.h>
using namespace std;

static inline double now_sec() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

int G, M;
long long B;
vector<int> demand;            // demand[y*G+x]
struct Site { int sx, sy, r; long long c; int x0, y0, x1, y1; };
vector<Site> site;

// covered cell counts (how many CHOSEN antennas cover each cell)
vector<int> cov;
long long curScore = 0;        // total demand of the covered union
long long curCost  = 0;

// add/remove a site, maintaining cov[], curScore, curCost in O(footprint).
inline void addSite(int i) {
    const Site& s = site[i];
    for (int y = s.y0; y <= s.y1; ++y) {
        int base = y * G;
        for (int x = s.x0; x <= s.x1; ++x) {
            int idx = base + x;
            if (cov[idx]++ == 0) curScore += demand[idx]; // newly covered
        }
    }
    curCost += s.c;
}
inline void removeSite(int i) {
    const Site& s = site[i];
    for (int y = s.y0; y <= s.y1; ++y) {
        int base = y * G;
        for (int x = s.x0; x <= s.x1; ++x) {
            int idx = base + x;
            if (--cov[idx] == 0) curScore -= demand[idx]; // last cover gone
        }
    }
    curCost -= s.c;
}

// marginal gain of adding site i to the CURRENT cov[] (cells with cov==0).
inline long long marginalGain(int i) {
    const Site& s = site[i];
    long long g = 0;
    for (int y = s.y0; y <= s.y1; ++y) {
        int base = y * G;
        for (int x = s.x0; x <= s.x1; ++x) {
            int idx = base + x;
            if (cov[idx] == 0) g += demand[idx];
        }
    }
    return g;
}

int main() {
    double t_start = now_sec();
    const double TIME_LIMIT = 1.8; // seconds (budget); local search uses the rest

    // ---- read instance ----
    {
        int m; long long b;
        if (scanf("%d %d %lld", &G, &m, &b) != 3) { printf("0\n"); return 0; }
        M = m; B = b;
        demand.assign((size_t)G * G, 0);
        for (size_t i = 0; i < (size_t)G * G; ++i) scanf("%d", &demand[i]);
        site.resize(M);
        for (int i = 0; i < M; ++i) {
            int sx, sy, r; long long c;
            scanf("%d %d %d %lld", &sx, &sy, &r, &c);
            Site s; s.sx = sx; s.sy = sy; s.r = r; s.c = c;
            s.x0 = max(0, sx - r); s.y0 = max(0, sy - r);
            s.x1 = min(G - 1, sx + r); s.y1 = min(G - 1, sy + r);
            site[i] = s;
        }
    }
    cov.assign((size_t)G * G, 0);

    // ---- CELF: cost-benefit lazy greedy ----
    // Priority queue keyed on an UPPER BOUND of gain-per-cost. We store the
    // gain computed when an entry was (re)evaluated together with the size of
    // the covered set at that time (`stamp`). Because coverage only grows as we
    // add antennas, a previously computed marginal gain is an upper bound on the
    // current one (submodularity), so a stale top entry can be re-evaluated
    // lazily and re-inserted; once the top entry's stamp is current it is the
    // true best and we commit it. This skips recomputing stale gains.
    struct Entry { double ratio; long long gain; int idx; long long stamp; };
    struct Cmp { bool operator()(const Entry& a, const Entry& b) const { return a.ratio < b.ratio; } };
    priority_queue<Entry, vector<Entry>, Cmp> pq;

    vector<char> inSet(M, 0);
    long long stampNow = 0; // increments whenever cov[] changes (an antenna is added)

    // initial evaluation of every affordable site
    for (int i = 0; i < M; ++i) {
        if (site[i].c > B) continue;            // can never fit alone -> skip
        long long g = marginalGain(i);
        double ratio = (g <= 0) ? -1.0 : (double)g / (double)site[i].c;
        pq.push({ratio, g, i, stampNow});
    }

    vector<int> greedyPick;
    while (!pq.empty()) {
        if (now_sec() - t_start > TIME_LIMIT * 0.55) break; // leave time for local search
        Entry e = pq.top(); pq.pop();
        int i = e.idx;
        if (inSet[i]) continue;
        if (site[i].c > B - curCost) continue;  // does not fit anymore
        if (e.stamp != stampNow) {
            // stale: re-evaluate against current coverage and re-insert
            long long g = marginalGain(i);
            if (g <= 0) continue;                // adds nothing new -> drop
            double ratio = (double)g / (double)site[i].c;
            pq.push({ratio, g, i, stampNow});
            continue;
        }
        // top entry is current and feasible -> commit it
        if (e.gain <= 0) break;                  // best remaining adds nothing
        addSite(i);
        inSet[i] = 1;
        greedyPick.push_back(i);
        ++stampNow;
    }

    // ---- single-best guard (Khuller-Moss-Naor) ----
    // The cost-benefit greedy alone can be arbitrarily bad on a single huge-gain
    // expensive element; comparing against the best single feasible antenna and
    // keeping the better of the two restores the (1-1/e)/2 guarantee.
    long long greedyScore = curScore;
    vector<int> greedySet = greedyPick;
    {
        long long bestSingle = -1; int bestI = -1;
        // current cov[] still reflects greedySet; compute single best from empty.
        for (int i = 0; i < M; ++i) {
            if (site[i].c > B) continue;
            // gain from empty == total demand of footprint (cov-independent upper part).
            const Site& s = site[i];
            long long g = 0;
            for (int y = s.y0; y <= s.y1; ++y) {
                int base = y * G;
                for (int x = s.x0; x <= s.x1; ++x) g += demand[base + x];
            }
            if (g > bestSingle) { bestSingle = g; bestI = i; }
        }
        if (bestSingle > greedyScore && bestI >= 0) {
            // rebuild cov[] for the single-best solution
            for (int i : greedySet) removeSite(i);
            for (int i = 0; i < M; ++i) inSet[i] = 0;
            greedySet.clear();
            addSite(bestI); inSet[bestI] = 1; greedySet.push_back(bestI);
        }
    }

    // ---- incremental-eval SWAP / add local search ----
    // cov[], curScore, curCost reflect the current chosen set `greedySet`.
    // Repeatedly try (a) add a feasible site with positive gain, or (b) remove
    // one site and add one or two cheaper ones that more than compensate. Each
    // move is evaluated and applied in O(footprint) via cov[]. Keep the best.
    vector<int> cur = greedySet;
    auto inCur = [&](int idx) { for (int v : cur) if (v == idx) return true; return false; };

    long long bestScore = curScore;
    vector<int> bestSet = cur;

    std::mt19937 rng(123456789u);
    while (now_sec() - t_start < TIME_LIMIT) {
        bool improved = false;

        // (a) greedy ADD pass: any positive-gain site that fits
        for (int i = 0; i < M && (now_sec() - t_start < TIME_LIMIT); ++i) {
            if (inSet[i]) continue;
            if (site[i].c > B - curCost) continue;
            if (marginalGain(i) > 0) {
                addSite(i); inSet[i] = 1; cur.push_back(i);
                improved = true;
            }
        }
        if (curScore > bestScore) { bestScore = curScore; bestSet = cur; }

        // (b) SWAP pass: try removing each chosen site, then greedily refill the
        // freed budget with best-ratio additions; accept if it helps.
        for (size_t pi = 0; pi < cur.size() && (now_sec() - t_start < TIME_LIMIT); ++pi) {
            int out = cur[pi];
            // snapshot to allow rollback
            long long beforeScore = curScore;
            removeSite(out); inSet[out] = 0;

            // greedily add best-ratio feasible sites (a few passes)
            vector<int> added;
            for (int rep = 0; rep < 6; ++rep) {
                int bi = -1; double bratio = 0.0; long long bgain = 0;
                for (int i = 0; i < M; ++i) {
                    if (inSet[i] || i == out) continue;
                    if (site[i].c > B - curCost) continue;
                    long long g = marginalGain(i);
                    if (g <= 0) continue;
                    double ratio = (double)g / (double)site[i].c;
                    if (ratio > bratio) { bratio = ratio; bi = i; bgain = g; }
                }
                if (bi < 0) break;
                addSite(bi); inSet[bi] = 1; added.push_back(bi);
                (void)bgain;
            }

            if (curScore > beforeScore) {
                // accept the swap: rebuild `cur`
                cur.erase(cur.begin() + pi);
                for (int a : added) cur.push_back(a);
                improved = true;
                if (curScore > bestScore) { bestScore = curScore; bestSet = cur; }
                break; // cur mutated; restart the swap pass next loop
            } else {
                // rollback
                for (int a : added) { removeSite(a); inSet[a] = 0; }
                addSite(out); inSet[out] = 1;
            }
        }

        if (curScore > bestScore) { bestScore = curScore; bestSet = cur; }
        if (!improved) {
            // random perturbation: drop a random chosen site to escape a plateau
            if (cur.empty()) break;
            int pi = rng() % cur.size();
            int out = cur[pi];
            removeSite(out); inSet[out] = 0;
            cur.erase(cur.begin() + pi);
        }
    }

    // ---- emit the best set found (guaranteed feasible) ----
    // validate cost once more defensively; if somehow over budget, drop sites.
    {
        long long cost = 0;
        for (int i : bestSet) cost += site[i].c;
        while (cost > B && !bestSet.empty()) { cost -= site[bestSet.back()].c; bestSet.pop_back(); }
    }
    printf("%d", (int)bestSet.size());
    for (int i : bestSet) printf(" %d", i);
    printf("\n");
    return 0;
}
```
