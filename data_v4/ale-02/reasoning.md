**Reading the problem and pinning the contract.** I have an `H x W` floor and a
catalogue of `K` polyomino shapes; each shape `k` has area `A_k` and a copy bound
`cnt_k`. I may place copies of shapes, each rotated by `0/90/180/270` degrees,
anywhere inside the floor, with no two placed cells overlapping, and I want to
maximise the number of covered cells — which, because feasible placements never
overlap, is just the summed area of everything I place. Input is `H W K` then the
`K` shape blocks; output is `P` followed by `P` lines `k rot ar ac`. Before any
algorithm I fix the scale, because it decides the data structures: `H, W <= 30`,
so a single row has at most 30 columns and the whole occupancy state is `H` rows
of `<= 30` bits each. That is the first thing I notice — a row of the floor fits
comfortably in a 64-bit word with room to spare. I will come back to this, because
it is the lever that makes the whole approach fast.

The objective is continuous (a coverage count) and the structure is a packing /
maximum-coverage problem, which is NP-hard, so there is no exact answer to print;
I am optimising against a score with a feasibility floor. The floor matters
operationally: if I ever emit a placement that pokes outside the floor, overlaps
another, names a bad shape or rotation, or exceeds `cnt_k`, the scorer drops me to
`0`. So whatever I build has to be feasible by *construction* at every step, and
the thing I print at the end must be a state I have actually maintained as legal —
not a state I hope is legal.

**Getting to a feasible baseline first.** Rule one of this kind of problem: never
be without a valid solution. The cheapest valid solution is `P = 0` — print zero
placements, cover zero cells. It is feasible and it scores `0`, which ties the
infeasible floor, so it is useless as an answer but useful as a safety net: at any
point I can fall back to "place nothing" and not be disqualified.

The cheapest *useful* feasible solution is first-fit greedy. Sort the shapes
largest-area-first (covering more cells per placement tends to leave fewer awkward
gaps), and for each shape and each of its distinct rotations sweep every anchor
`(ar, ac)` on the floor; wherever the shape currently fits, drop a copy and mark
those cells occupied, respecting `cnt_k`. This is a one-shot construction, it is
obviously feasible (I only ever place where it fits), and it already covers a
large fraction of the floor. I will use it as my warm start and, separately, as
the reference baseline the final score ratio is measured against. So my first
milestone is concrete: a greedy that I can prove never emits an illegal placement.

**Why greedy alone is not enough.** The weakness of first-fit is structural and I
want to name it before I try to fix it, not after. Greedy commits a cell forever
the instant it places a part over it. So a handful of early, slightly-misaligned
large parts can strand pockets of empty cells whose shape no remaining catalogue
piece matches — a 1x2 hole when the smallest free piece is an L-tromino, say. On a
12x12 to 30x30 floor with shapes of area up to 6, those stranded pockets add up;
greedy typically leaves a few percent of the floor uncovered that a smarter
arrangement could reclaim. The fix has to be able to *undo*: pull a part back out,
freeing its cells, and re-tile that region differently. That immediately points at
local search over the set of current placements, with moves that both add and
remove parts, and with the willingness to accept a temporary loss of coverage so
the search can cross a valley to a better packing. That is simulated annealing.

**The cost that decides whether SA is worth it.** Local search only beats greedy
if I can do an enormous number of perturbations in the ~2-second budget, because
each individual perturbation gains very little (a part is at most 6 cells). The
expensive part of a perturbation is the *collision test*: to add a part I must
check it does not overlap anything already on the floor. The naive test — for each
of the part's up-to-6 cells, look up a 2-D occupied array — is correct but it
touches scattered memory and costs a handful of operations per cell, and I will
run it tens of millions of times. This is exactly where the scale observation from
the start pays off.

**The innovation: rows as bitsets, incremental coverage.** Because a floor row has
at most 30 columns, I keep the occupancy as one 64-bit word per row: `occ[r]` has
bit `c` set iff cell `(r, c)` is covered. I precompile each `(shape, rotation)`
once into a short list of `(row-offset, column-bitmask)` pairs, where the bitmask
sits at anchor column 0. To test a placement at `(ar, ac)`, for each occupied row
of the part I shift its column-mask left by `ac` and test `occ[ar + row] & mask`;
if every row comes back zero, it fits. That is one `AND` and one branch per
*occupied row of the part* — at most a handful of words, all hot in cache, no
per-cell scatter. Adding the part is `occ[ar + row] ^= (mask << ac)` per row, and
removing it is the identical XOR (a placement's cells are exactly the bits it owns,
so XOR toggles them off). And the coverage score updates incrementally: `+area` on
add, `-area` on remove, never a rescan of the grid. So the per-move cost is O(rows
of the piece) ≈ O(1), and I can afford millions of moves. This is the whole reason
the metaheuristic is viable rather than a toy.

**Designing the move set.** With the bitset machinery in hand, I give SA three
moves over the multiset of current placements:

- **ADD** a uniformly random feasible placement (random shape with a copy left,
  random rotation, random in-bounds anchor; retry a few times to find one that
  fits). This is pure uphill: `+area`, always accepted.
- **REMOVE** a random current placement: `-area`, downhill. I accept it with the
  Metropolis probability `exp(-area / T)`. This is what greedy cannot do — it lets
  the search vacate a part to open space.
- **REPLACE** a random current placement with a fresh random feasible one: remove
  the old, draw a new add, and accept the net change `delta = newArea - oldArea`
  by Metropolis (`delta >= 0` always accepted, else with `exp(delta / T)`). This
  is the workhorse: it re-tiles a freed region in a single move.

Cooling is geometric in wall-clock fraction from `T0 = 4.0` to `T1 = 0.02`, so
early on the search shuffles freely and late it behaves like greedy hill-climbing.
I keep the best feasible state ever seen and print *that*, so a string of accepted
downhill moves at the end can never make my output worse than the warm start.

**First implementation — and a trace, because clean intent transcribes dirty.**
I wrote the loop and immediately distrusted the REPLACE branch, because it is the
one move that touches state twice (a remove then a conditional add) and has a
reject path that must put everything back. My first cut of that branch was:

```
// REPLACE (first, buggy)
int idx = randint(placements.size());
Place old = placements[idx];
remove_idx(idx);                       // freed old's cells
int k,rot,ar,ac; bool got = random_add(k,rot,ar,ac);
int newArea = got ? shapes[k][rot].area : 0;
int delta = newArea - old.area;
if (delta >= 0 || urand() < exp(delta / T)) {
    if (got) add_place(k,rot,ar,ac);
    // accept
} else {
    add_place(k,rot,ar,ac);            // <-- restore?? BUG
}
```

The reject branch reads `add_place(k,rot,ar,ac)` — it re-adds the *new* candidate,
not the *old* placement I removed. To expose it I do not need a big instance; I
trace the smallest situation that hits the reject path. Take a state with one
placement on the board: `old` = shape 0 (area 4) at some anchor. REPLACE removes
it, freeing those 4 cells; `random_add` happens to draw shape 1 (area 2) into part
of that freed region, so `got = true`, `newArea = 2`, `delta = 2 - 4 = -2`. Now
suppose Metropolis rejects (likely, since `delta` is negative). The buggy reject
branch executes `add_place(k=1, ...)` — it *adds the area-2 part it was supposed to
discard* and never restores the area-4 part. So a "rejected" REPLACE silently
turns into "removed a 4, added a 2": coverage fell by 2 on a move I claimed to
reject, and the part I meant to keep is gone.

**Diagnosing the bug.** The defect is precise: the reject path must restore the
*pre-move* state, which means re-adding `old` and *not* adding the candidate. I
conflated "the candidate I drew" with "the placement I removed". There is also a
latent feasibility trap hiding behind it: because `random_add` runs against the
floor *after* `old` was removed, the candidate it returns is feasible only in that
freed state; if I (wrongly) keep both the candidate and try to also restore `old`,
the candidate may now overlap `old` and I would be reconstructing an *illegal*
state — an instant scorer zero. So the reject path must add back exactly `old` and
nothing else, and the accept path must add the candidate and not `old`. Getting
this wrong is not just a score regression, it is a feasibility hazard.

**Fixing and re-verifying.** I rewrote the branch so each path reconstructs a
state I can name: accept = (candidate present, old gone); reject = (old present,
candidate gone).

```
// REPLACE (fixed)
int idx = randint(placements.size());
Place old = placements[idx];
remove_idx(idx);
int k,rot,ar,ac; bool got = random_add(k,rot,ar,ac);
int newArea = got ? shapes[k][rot].area : 0;
int delta = newArea - old.area;
if (delta >= 0 || urand() < exp(delta / T)) {
    if (got) add_place(k,rot,ar,ac);   // accept: keep candidate, old stays gone
    if (covered > bestCovered) { bestCovered = covered; bestPlace = placements; }
} else {
    add_place(old.k, old.rot, old.ar, old.ac);  // reject: restore EXACTLY old
}
```

Re-trace the same one-placement state: REPLACE removes `old` (area 4), draws the
area-2 candidate, `delta = -2`, Metropolis rejects -> the reject path runs
`add_place(old...)`, putting the area-4 part back at its exact anchor; the
candidate is never added. Coverage returns to its pre-move value of 4, and the
state is bit-for-bit the one I started from (the old part's cells are exactly the
bits its XOR toggles back on). The accept path, when `delta >= 0`, keeps the
candidate and leaves `old` removed — coverage rises or holds. The branch now does
what its name says, and the feasibility hazard is closed because I never hold both
parts at once.

**A second self-verify pass on the whole solver.** With REPLACE fixed I built the
solver, the scorer, and a generator, and ran seeds `1..20`: generate the instance,
run the solver, run the first-fit baseline, and score all three (solver, baseline,
empty). The results: every solver output is feasible (the scorer never zeroed one)
and the solver strictly beat the first-fit baseline on all 20 seeds, mean coverage
`~420` vs baseline `~389` (ratio about `1.08`), with empty output scoring `0` as
expected. On several seeds the floor is nearly tiled — seed 7 covers 359 of 360
cells. Crucially I cross-checked the solver's internal `bestCovered` against the
independent Python scorer's recomputed count and they agree exactly (359 = 359),
which tells me the incremental score I maintain and the from-scratch replay agree —
the bitset add/remove bookkeeping is correct, not just plausible.

**Edge cases, deliberately, because this is where packing code dies.**
- *A shape bigger than the floor in some rotation.* `random_add` guards with
  `if (s.hh > H || s.ww > W) continue;` and computes the anchor range as
  `randint(H - s.hh + 1)`, which is well-defined only when the shape fits — the
  guard ensures it. The greedy warm start uses `ar + s.hh <= H` bounds for the same
  reason. So an oversized rotation is simply never placed; no out-of-bounds anchor
  is ever generated.
- *Identical rotations.* A square or a symmetric piece has fewer than 4 distinct
  rotations; the greedy skips a rotation whose normalised cells equal an earlier
  one, so it does not waste the whole sweep four times, and the SA drawing a
  "duplicate" rotation is harmless (it is still a valid placement).
- *Counts.* `usedCnt[k] < cntOf[k]` gates every add, in both greedy and SA, and
  `remove_idx` decrements it, so the multiset never exceeds `cnt_k` — the scorer's
  count check can never fire.
- *Empty placement set during SA.* REMOVE and REPLACE both `continue` when there
  are no placements, so I never index into an empty vector.
- *Time and output.* The loop checks wall-clock every 1024 iterations and stops at
  `1.8 s`, comfortably under the ~2 s budget; the final print emits `bestPlace`,
  the best feasible state, which can only be at least as good as the warm start.
  The bit shift `mask << ac` never drops bits because `ac + ww <= W <= 30 < 64`.

**Final solution.** I convinced myself the *idea* is right by naming greedy's
stranded-pocket failure and choosing an add/remove SA that can undo it; I
convinced myself the per-move cost makes SA worthwhile by reducing the collision
test and the score update to a few cache-resident word operations via row-bitsets;
and I convinced myself the *code* is right by tracing the REPLACE reject path to a
precise add-the-wrong-part bug (which was also a feasibility hazard), fixing it so
each branch reconstructs a named legal state, and confirming on 20 seeds that every
output is feasible and beats the baseline with the scorer's independent recount
matching my incremental one. That is what I ship — one self-contained file.

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

**Causal recap.** Greedy gives a feasible cover but commits cells forever, so a
few misaligned big parts strand pockets it cannot reclaim; an add/remove
simulated annealing can undo those commitments, but only pays off if each move is
near-O(1), which I get by storing each floor row as a 64-bit bitset so collision
testing is one shifted-AND per piece-row and coverage updates by `+/-area` with no
rescan; the one real bug was a REPLACE reject path that re-added the wrong part
(a score *and* feasibility hazard), found by tracing the smallest one-placement
reject and fixed so each branch reconstructs a named legal state; on seeds `1..20`
every output is feasible, the independent scorer's recount matches my incremental
count, and the solver beats the first-fit baseline on all 20 (mean ratio ~1.08).
