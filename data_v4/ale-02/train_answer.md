**Problem.** An `H x W` floor must be tiled with stock parts drawn from a catalogue
of `K` polyomino shapes. Shape `k` has area `A_k` and a copy bound `cnt_k`; each
placed copy may be rotated `0/90/180/270` degrees and dropped anywhere inside the
floor, with no two placed cells overlapping. Read `H W K` then the `K` shape blocks
from stdin; print `P` and then `P` lines `k rot ar ac`.

**Objective & scoring.** Maximise the number of covered cells. Since feasible
placements never overlap, that equals the summed area of the placed parts. The
local scorer replays the placements and reports the covered-cell count, with a
hard **feasibility -> 0 floor**: any malformed/truncated output, bad shape index,
rotation outside `0..3`, out-of-grid placement, overlap, or use of more than
`cnt_k` copies of a shape scores `0`. An empty solution (`P = 0`) is feasible and
covers `0` cells. The reported coverage ratio is against a first-fit greedy
baseline. Time budget ~2 s.

**Baseline.** First-fit greedy: sort shapes largest-area-first; for each shape and
each distinct rotation, sweep every anchor and place a copy wherever it currently
fits (respecting `cnt_k`). Always feasible and already covers most of the floor,
but it is one-shot — a cell is committed forever the moment a part covers it, so a
few misaligned big parts strand pockets of empty cells that no remaining catalogue
piece fits, leaving a few percent of the floor permanently uncovered. The solver
uses this greedy as both its warm start and the reference baseline.

**Key idea (the heuristic innovation).** Simulated annealing over the *multiset of
placements*, made viable by an O(1)-per-move state. Two ingredients:

- *Rows as bitsets.* A floor row has `<= 30 <= 64` columns, so occupancy is one
  64-bit word per row, `occ[r]`. Each `(shape, rotation)` is precompiled once into
  a list of `(row-offset, column-bitmask)` pairs at anchor column 0. Testing a
  placement at `(ar, ac)` is, per occupied piece-row, a single shifted-AND
  `occ[ar+row] & (mask << ac)`; placing or removing it is the identical XOR. The
  collision test and the edit are a handful of cache-resident word ops, not a
  per-cell scatter over a 2-D array.
- *Incremental coverage.* The score moves by `+area` on add and `-area` on remove;
  the grid is never rescanned.

This buys millions of moves in the budget. The move set is ADD (random feasible
placement, uphill, always accepted), REMOVE (random current placement, downhill,
Metropolis-accepted via `exp(-area/T)`), and REPLACE (remove one, add a random
feasible one, accept net `delta = newArea - oldArea` by Metropolis). REMOVE and
REPLACE are exactly what greedy lacks: they vacate a wasteful part and re-tile the
freed region, climbing out of greedy's stranded-pocket dead ends. Geometric
cooling `T0 = 4.0 -> T1 = 0.02` over wall-clock; the best feasible state is kept and
printed.

**Feasibility & pitfalls.**
- Every add is gated by `usedCnt[k] < cntOf[k]` and an in-grid anchor range, and
  guards a rotation larger than the floor (`s.hh > H || s.ww > W`), so no placement
  is ever out-of-bounds or over-count by construction.
- The subtle bug is the REPLACE reject path: it must restore *exactly the removed
  part* and add nothing else. Re-adding the freshly drawn candidate instead is both
  a score regression and a feasibility hazard (the candidate, drawn against the
  freed floor, may overlap the part you meant to restore). Each branch must
  reconstruct a named legal state: accept = candidate present / old gone; reject =
  old present / candidate gone.
- Print the best feasible state, never the live one, so a run of accepted downhill
  moves at the end cannot make the output worse than the warm start. `P = 0` is the
  ultimate safe fallback. The shift `mask << ac` never drops bits because
  `ac + ww <= W <= 30 < 64`.

**Complexity per step.** Collision test and add/remove: O(rows of the piece) ≈ O(1)
word operations (`<= 6` rows, each one shifted-AND/XOR). Coverage update: O(1).
Greedy warm start: O(K · 4 · H · W · rows) once. Memory: `H` 64-bit words plus the
placement list — a few KB.

**Code.**

```cpp
// ALE-02  Grid Polyomino Packing  --  heuristic solver.
//
// Objective: place copies of given polyomino types (each in one of 4 rotations)
// onto an H x W grid, non-overlapping and inside the grid, using at most cnt[k]
// copies of type k, so as to MAXIMISE the number of covered cells (= sum of the
// areas of the placed pieces). The board is the binding constraint, so this is a
// packing / maximum-coverage problem with no closed form.
//
// INNOVATION (why this file is fast):
//   * Grid occupancy is kept as one 64-bit bitset PER ROW (occ[r], W <= 64), so a
//     candidate placement's collision test is, for each of its rows, a single
//     (occ[r] & mask[r]) != 0 -- O(#rows of the piece), branch-free, cache-tiny.
//   * Add / remove just XORs those row-masks into occ and adds/subtracts the
//     piece area, so the coverage score is maintained INCREMENTALLY in O(1)
//     (plus the O(rows) mask edits) -- we never rescan the grid.
//   * Each (type,rotation) placement at an anchor is precompiled into a list of
//     (row-offset, column-bitmask) pairs once; placing it at column c is a shift.
//
// SEARCH: simulated annealing over the multiset of placements. Moves:
//   ADD     a random feasible placement                       (+area, uphill)
//   REMOVE  a random current placement                        (-area, downhill)
//   REPLACE remove one placement, then ADD a random feasible  (delta = a2 - a1)
// Downhill moves are accepted by the Metropolis rule, which lets the search
// vacate a badly-placed small piece and re-tile the freed region with something
// that packs better -- exactly what the first-fit greedy baseline cannot do.
// The best feasible state seen is remembered and printed, so the output is always
// feasible and never worse than the greedy warm start.
//
// I/O:
//   stdin : "H W K", then K blocks "A_k cnt_k" followed by A_k lines "r c".
//   stdout: "P", then P lines "k rot ar ac" (type, rotation 0..3, anchor row/col).
// Compile: g++ -O2 -std=c++17 sol.cpp
#include <bits/stdc++.h>
using namespace std;

static uint64_t rng_state = 0x9e3779b97f4a7c15ULL;
static inline uint64_t xr() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}
static inline double urand() { return (xr() >> 11) * (1.0 / 9007199254740992.0); }
static inline int randint(int n) { return (int)(xr() % (uint64_t)n); }

int H, W, K;
vector<int> areaOf;                 // area of each type
vector<int> cntOf;                  // available copies of each type

// For each type k and rotation rot (0..3): the cell offsets after normalisation
// (min row = min col = 0). Stored both as offset pairs and as a row-mask list
// anchored at column 0 (mask must be shiftable left by ac without dropping bits).
struct Shape {
    int hh, ww, area;               // bounding box and area of this rotation
    vector<pair<int,int>> cells;    // (dr, dc) offsets, normalised
    vector<pair<int,uint64_t>> rows;// (dr, column-bitmask at anchor col 0)
};
vector<array<Shape,4>> shapes;      // shapes[k][rot]

static vector<pair<int,int>> rotate_norm(const vector<pair<int,int>>& cells, int rot) {
    vector<pair<int,int>> p = cells;
    for (int t = 0; t < (rot & 3); t++)
        for (auto& q : p) q = { q.second, -q.first };   // 90 deg CW
    int mr = INT_MAX, mc = INT_MAX;
    for (auto& q : p) { mr = min(mr, q.first); mc = min(mc, q.second); }
    for (auto& q : p) { q.first -= mr; q.second -= mc; }
    sort(p.begin(), p.end());
    p.erase(unique(p.begin(), p.end()), p.end());
    return p;
}

// A concrete placement currently on the board.
struct Place { int k, rot, ar, ac; int area; };

vector<uint64_t> occ;               // occ[r] = bitset of occupied columns in row r
long long covered = 0;              // current covered-cell count (incremental)
vector<int> usedCnt;                // copies of each type currently placed

// Test whether shape (k,rot) anchored at (ar,ac) fits: in-grid and no overlap.
static inline bool fits(const Shape& s, int ar, int ac) {
    if (ar < 0 || ac < 0 || ar + s.hh > H || ac + s.ww > W) return false;
    for (auto& pr : s.rows) {
        uint64_t m = pr.second << ac;
        if (occ[ar + pr.first] & m) return false;
    }
    return true;
}
static inline void put(const Shape& s, int ar, int ac) {       // assumes fits()
    for (auto& pr : s.rows) occ[ar + pr.first] ^= (pr.second << ac);
    covered += s.area;
}
static inline void take(const Shape& s, int ar, int ac) {      // assumes placed
    for (auto& pr : s.rows) occ[ar + pr.first] ^= (pr.second << ac);
    covered -= s.area;
}

int main() {
    if (scanf("%d %d %d", &H, &W, &K) != 3) return 0;
    areaOf.assign(K, 0);
    cntOf.assign(K, 0);
    shapes.assign(K, {});
    vector<vector<pair<int,int>>> base(K);
    for (int k = 0; k < K; k++) {
        int A, cnt; scanf("%d %d", &A, &cnt);
        areaOf[k] = A; cntOf[k] = cnt;
        base[k].resize(A);
        for (int i = 0; i < A; i++) scanf("%d %d", &base[k][i].first, &base[k][i].second);
    }
    // Precompile the 4 rotations of every type into shiftable row-masks.
    for (int k = 0; k < K; k++) {
        for (int rot = 0; rot < 4; rot++) {
            auto cells = rotate_norm(base[k], rot);
            Shape s; s.cells = cells; s.area = (int)cells.size();
            int hh = 0, ww = 0;
            for (auto& c : cells) { hh = max(hh, c.first + 1); ww = max(ww, c.second + 1); }
            s.hh = hh; s.ww = ww;
            // build row -> column bitmask
            map<int,uint64_t> mp;
            for (auto& c : cells) mp[c.first] |= (uint64_t)1 << c.second;
            for (auto& pr : mp) s.rows.push_back(pr);
            shapes[k][rot] = move(s);
        }
    }

    occ.assign(H, 0);
    usedCnt.assign(K, 0);
    covered = 0;
    vector<Place> placements;

    auto add_place = [&](int k, int rot, int ar, int ac) {
        const Shape& s = shapes[k][rot];
        put(s, ar, ac);
        usedCnt[k]++;
        placements.push_back({k, rot, ar, ac, s.area});
    };
    auto remove_idx = [&](int idx) {
        Place pl = placements[idx];
        take(shapes[pl.k][pl.rot], pl.ar, pl.ac);
        usedCnt[pl.k]--;
        placements[idx] = placements.back();
        placements.pop_back();
    };

    // ---- WARM START: first-fit greedy, largest area first (the baseline). ----
    // This already gives a feasible, non-trivial cover and seeds the SA basin.
    {
        vector<int> order(K);
        iota(order.begin(), order.end(), 0);
        sort(order.begin(), order.end(),
             [&](int a, int b){ return areaOf[a] > areaOf[b]; });
        for (int k : order) {
            for (int rot = 0; rot < 4; rot++) {
                const Shape& s = shapes[k][rot];
                bool dup = false;                       // skip identical rotations
                for (int r2 = 0; r2 < rot; r2++)
                    if (shapes[k][r2].cells == s.cells) { dup = true; break; }
                if (dup) continue;
                for (int ar = 0; ar + s.hh <= H && usedCnt[k] < cntOf[k]; ar++)
                    for (int ac = 0; ac + s.ww <= W && usedCnt[k] < cntOf[k]; ac++)
                        if (fits(s, ar, ac)) add_place(k, rot, ar, ac);
            }
        }
    }

    // remember the best feasible state (start = greedy warm start)
    long long bestCovered = covered;
    vector<Place> bestPlace = placements;

    // helper: draw a uniformly random feasible ADD candidate, or return false.
    auto random_add = [&](int& ok_k, int& ok_rot, int& ok_ar, int& ok_ac) -> bool {
        for (int tries = 0; tries < 24; tries++) {
            int k = randint(K);
            if (usedCnt[k] >= cntOf[k]) continue;
            int rot = randint(4);
            const Shape& s = shapes[k][rot];
            if (s.hh > H || s.ww > W) continue;
            int ar = randint(H - s.hh + 1);
            int ac = randint(W - s.ww + 1);
            if (fits(s, ar, ac)) { ok_k = k; ok_rot = rot; ok_ar = ar; ok_ac = ac; return true; }
        }
        return false;
    };

    // ---- SIMULATED ANNEALING over the placement multiset ----
    const double TIME = 1.8;                 // seconds
    auto t0 = chrono::steady_clock::now();
    double T0 = 4.0, T1 = 0.02;
    long long iter = 0;
    while (true) {
        if ((iter & 1023) == 0) {
            double el = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
            if (el > TIME) break;
        }
        iter++;
        double frac = chrono::duration<double>(chrono::steady_clock::now() - t0).count() / TIME;
        if (frac > 1) frac = 1;
        double T = T0 * pow(T1 / T0, frac);

        int move = randint(3);
        if (move == 0) {
            // ADD: pure uphill, accept whenever a feasible candidate exists.
            int k, rot, ar, ac;
            if (random_add(k, rot, ar, ac)) {
                add_place(k, rot, ar, ac);
                if (covered > bestCovered) { bestCovered = covered; bestPlace = placements; }
            }
        } else if (move == 1) {
            // REMOVE: downhill, Metropolis-accepted to escape local maxima.
            if (placements.empty()) continue;
            int idx = randint((int)placements.size());
            int da = placements[idx].area;             // coverage drops by da
            if (urand() < exp((double)(-da) / T)) {
                remove_idx(idx);
            }
        } else {
            // REPLACE: remove a random placement, try to add a random one.
            if (placements.empty()) continue;
            int idx = randint((int)placements.size());
            Place old = placements[idx];
            remove_idx(idx);
            int k, rot, ar, ac;
            bool got = random_add(k, rot, ar, ac);
            int newArea = got ? shapes[k][rot].area : 0;
            int delta = newArea - old.area;            // change in covered cells
            if (delta >= 0 || urand() < exp((double)delta / T)) {
                if (got) add_place(k, rot, ar, ac);
                if (covered > bestCovered) { bestCovered = covered; bestPlace = placements; }
            } else {
                // reject: restore the removed placement exactly
                if (got) { /* nothing added */ }
                add_place(old.k, old.rot, old.ar, old.ac);
            }
        }
    }

    // ---- emit the best feasible state ----
    printf("%d\n", (int)bestPlace.size());
    for (auto& p : bestPlace) printf("%d %d %d %d\n", p.k, p.rot, p.ar, p.ac);
    fprintf(stderr, "iters=%lld bestCovered=%lld board=%d\n", iter, bestCovered, H * W);
    return 0;
}
```
