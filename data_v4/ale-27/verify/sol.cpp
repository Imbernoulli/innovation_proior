// Grid Light Placement -- heuristic solver.
//
// Objective: place as few lights as possible on floor cells of an H x W grid so
// that every floor cell is lit. A light at (r, c) lights its own cell and every
// cell reachable along its row and column without crossing a wall '#' (it stops
// just before the first wall, or at the grid edge). A light may sit only on a
// floor cell '.'. Read the instance from stdin, write "K" then K lines "r c".
//
// Method (the innovation): CORRIDOR DECOMPOSITION turns this grid problem into a
// set-cover.
//   * Decompose the floor into maximal HORIZONTAL corridors and maximal VERTICAL
//     corridors. Every floor cell belongs to exactly one H-corridor and one
//     V-corridor. A light at a cell "activates" (covers) the H-corridor and the
//     V-corridor it lies in, and a cell is LIT iff at least one of its two
//     corridors currently holds a light.
//   * Keep per-corridor light counts hCnt[h], vCnt[v]. A cell (h,v) is lit iff
//     hCnt[h] > 0 || vCnt[v] > 0. Adding/removing one light touches exactly two
//     counters, so checking which cells flip dark/lit costs O(size of those two
//     corridors) -- this is the cheap incremental evaluation the method needs.
//   * CONSTRUCTION: greedy maximum-coverage -- repeatedly place the light that
//     newly lights the most still-dark floor cells, until all lit. Always
//     feasible.
//   * IMPROVEMENT: simulated annealing on the light count. Repeatedly remove a
//     random placed light; this may darken cells in its two corridors; REPAIR
//     each newly-dark cell with the cheapest covering candidate (the candidate
//     that lights the most currently-dark cells among the dark ones). Accept the
//     resulting light set by an SA rule on its size, keeping the best feasible
//     set ever seen. The incumbent is feasible at every step, so any early stop
//     (including the wall-clock cutoff) prints a valid solution.
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

int H, W;
vector<string> G;

// floor-cell indexing
int NF;                       // number of floor cells
vector<int> cellR, cellC;     // floor-cell -> (r,c)
vector<int> idAt;             // r*W+c -> floor-cell index, or -1 for wall
vector<int> hOf, vOf;         // floor-cell -> H-corridor id / V-corridor id
int NH, NV;                   // number of H- and V-corridors
vector<vector<int>> hCells, vCells;  // corridor id -> its floor cells

static inline int packRC(int r, int c) { return r * W + c; }

int main() {
    double t0 = now_sec();

    // ---- read instance ------------------------------------------------------
    {
        if (scanf("%d %d", &H, &W) != 2) return 0;
        G.assign(H, string());
        // read H grid rows (skip the newline after "H W")
        for (int r = 0; r < H; r++) {
            char buf[1 << 16];
            if (scanf("%s", buf) != 1) { G[r].assign(W, '.'); continue; }
            string row(buf);
            if ((int)row.size() < W) row.append(W - row.size(), '.');
            G[r] = row.substr(0, W);
        }
    }

    // ---- floor-cell indexing + corridor decomposition -----------------------
    idAt.assign(H * W, -1);
    cellR.clear(); cellC.clear();
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++)
            if (G[r][c] == '.') {
                idAt[packRC(r, c)] = (int)cellR.size();
                cellR.push_back(r);
                cellC.push_back(c);
            }
    NF = (int)cellR.size();

    if (NF == 0) {                 // no floor cells: nothing to light
        printf("0\n");
        return 0;
    }

    hOf.assign(NF, -1);
    vOf.assign(NF, -1);

    // horizontal corridors: maximal runs along each row
    NH = 0;
    for (int r = 0; r < H; r++) {
        int c = 0;
        while (c < W) {
            if (G[r][c] != '.') { c++; continue; }
            int id = NH++;
            while (c < W && G[r][c] == '.') {
                hOf[idAt[packRC(r, c)]] = id;
                c++;
            }
        }
    }
    // vertical corridors: maximal runs along each column
    NV = 0;
    for (int c = 0; c < W; c++) {
        int r = 0;
        while (r < H) {
            if (G[r][c] != '.') { r++; continue; }
            int id = NV++;
            while (r < H && G[r][c] == '.') {
                vOf[idAt[packRC(r, c)]] = id;
                r++;
            }
        }
    }
    hCells.assign(NH, {});
    vCells.assign(NV, {});
    for (int f = 0; f < NF; f++) {
        hCells[hOf[f]].push_back(f);
        vCells[vOf[f]].push_back(f);
    }

    // ---- state shared by construction + SA ----------------------------------
    // hCnt[h] / vCnt[v]: number of placed lights whose H-/V-corridor is h / v.
    // A cell f is lit iff hCnt[hOf[f]] > 0 || vCnt[vOf[f]] > 0.
    vector<int> hCnt(NH, 0), vCnt(NV, 0);
    vector<int> litCnt(NF, 0);        // lights covering cell f (hCnt+vCnt view)
    // a candidate light is a floor cell; we mark which floor cells currently
    // hold a light.
    vector<char> hasLight(NF, 0);

    auto cellLit = [&](int f) -> bool {
        return hCnt[hOf[f]] > 0 || vCnt[vOf[f]] > 0;
    };

    // place / remove a light at floor cell f; keeps hCnt/vCnt consistent.
    int darkCount = NF;               // number of currently-dark floor cells
    // We maintain darkCount incrementally. A cell is dark iff !cellLit(f).

    auto applyDelta = [&](int h, int v, int d) {
        // d = +1 (place) or -1 (remove). Update corridor counters and darkCount
        // by scanning ONLY the affected corridors' cells (incremental eval).
        // Snapshot lit-ness of affected cells before, then after.
        // Affected cells = cells of corridor h plus cells of corridor v.
        // To avoid double-handling the (h,v) intersection cell twice, we union.
        // Simpler: gather, dedup via a tiny touched list.
        static vector<int> touched;
        touched.clear();
        for (int f : hCells[h]) touched.push_back(f);
        for (int f : vCells[v]) touched.push_back(f);
        // mark "before" lit state using litCnt as a cache to dedup
        // (we just recompute before/after directly; dedup by toggling a flag)
        // Use a small local set via litCnt parity trick is overkill; dedup with
        // a visited stamp:
        for (int f : touched) {
            if (litCnt[f] == 0) {           // not yet stamped this call
                litCnt[f] = cellLit(f) ? 1 : -1;  // remember prior lit (1) / dark (-1)
            }
        }
        // apply the counter change
        hCnt[h] += d;
        vCnt[v] += d;
        // recompute and update darkCount over the touched (deduped) cells
        for (int f : touched) {
            if (litCnt[f] == 0) continue;   // already processed (duplicate)
            bool wasLit = (litCnt[f] == 1);
            bool nowLit = cellLit(f);
            if (wasLit && !nowLit) darkCount++;
            else if (!wasLit && nowLit) darkCount--;
            litCnt[f] = 0;                  // clear stamp
        }
    };

    auto placeLight = [&](int f) {
        hasLight[f] = 1;
        applyDelta(hOf[f], vOf[f], +1);
    };
    auto removeLight = [&](int f) {
        hasLight[f] = 0;
        applyDelta(hOf[f], vOf[f], -1);
    };

    // ---- greedy maximum-coverage construction -------------------------------
    // gain[f] = number of currently-dark cells a light at f would light.
    // A light at f lights all dark cells in corridor hOf[f] and corridor vOf[f].
    // We recompute gains lazily: pick best by scanning candidates, but to stay
    // fast we compute gain on demand from corridor dark-counts.
    // dark-per-corridor counters:
    vector<int> hDark(NH, 0), vDark(NV, 0);
    for (int f = 0; f < NF; f++) { hDark[hOf[f]]++; vDark[vOf[f]]++; }

    // gain of placing a light at cell f (with current dark state) =
    //   (dark cells in hOf[f]) + (dark cells in vOf[f])
    //   - (is cell f itself dark? counted in both -> subtract 1 if dark)
    // because cell f belongs to both its H- and V-corridor.
    auto gainAt = [&](int f) -> int {
        int g = hDark[hOf[f]] + vDark[vOf[f]];
        if (hCnt[hOf[f]] == 0 && vCnt[vOf[f]] == 0) g -= 1; // f counted twice
        return g;
    };

    // We must keep hDark/vDark in sync with placements. Wrap place/remove to
    // also adjust the dark-per-corridor counters by re-deriving from darkCount
    // transitions. Simpler: recompute hDark/vDark from scratch is O(NF); we do
    // it once per greedy pick is too slow for big grids, so update incrementally
    // inside a dedicated greedy placement that knows which cells flipped.
    auto greedyPlace = [&](int f) {
        // determine which dark cells become lit, update hDark/vDark for them.
        // cells in hOf[f] and vOf[f] that are currently dark and become lit.
        // After placing, cell is lit (its corridor now has a light), so every
        // dark cell in these two corridors becomes lit.
        // Snapshot affected dark cells:
        static vector<int> flipped;
        flipped.clear();
        int h = hOf[f], v = vOf[f];
        // place first (updates hCnt/vCnt and darkCount via applyDelta)
        hasLight[f] = 1;
        applyDelta(h, v, +1);
        // now any cell previously dark in corridors h or v is lit; find them by
        // checking corridor cells whose litCnt-derived state changed. We instead
        // recompute: a cell f2 in these corridors is now lit (guaranteed, since
        // its h or v corridor count just went positive). Those that were dark
        // before need hDark/vDark decremented on BOTH their corridors.
        for (int f2 : hCells[h]) {
            // was it dark before this placement? it's lit now for sure.
            // detect "was dark" via: before placement, hCnt[h] was (now-1) and
            // vCnt[vOf[f2]] unchanged. We can't see 'before' now, so track via a
            // visited stamp on flippedMark.
            (void)f2;
        }
        (void)flipped;
    };
    (void)greedyPlace;  // replaced by the explicit loop below

    // --- explicit greedy loop (clear and correct) ---------------------------
    // We recompute gains from hDark/vDark, which we keep exact by updating them
    // whenever a cell flips dark->lit. To flip cells we scan the two corridors
    // of the chosen light and, for each cell that was dark and is now lit,
    // decrement hDark and vDark of that cell's corridors.
    {
        // candidate set = all floor cells (each can host a light)
        // For speed we keep a simple loop: O(picks * NF) gain scan. Grids are
        // <= 50x50 = 2500 cells, picks <= NF, so this is comfortably fast.
        while (darkCount > 0) {
            int bestF = -1, bestG = -1;
            for (int f = 0; f < NF; f++) {
                if (hasLight[f]) continue;
                int g = gainAt(f);
                if (g > bestG) { bestG = g; bestF = f; }
            }
            if (bestF < 0 || bestG <= 0) {
                // Fallback safety: light every still-dark cell directly. This
                // cannot happen if the model is correct, but guarantees
                // feasibility no matter what.
                for (int f = 0; f < NF; f++)
                    if (!cellLit(f) && !hasLight[f]) {
                        // mark dark cells of f's corridors as lit
                        int h = hOf[f], v = vOf[f];
                        for (int f2 : hCells[h]) if (!cellLit(f2)) { hDark[hOf[f2]]--; vDark[vOf[f2]]--; }
                        for (int f2 : vCells[v]) if (!cellLit(f2)) { hDark[hOf[f2]]--; vDark[vOf[f2]]--; }
                        hasLight[f] = 1;
                        applyDelta(h, v, +1);
                    }
                break;
            }
            // place bestF; update hDark/vDark for cells that flip dark->lit
            int h = hOf[bestF], v = vOf[bestF];
            // collect dark cells in the two corridors BEFORE placing
            static vector<int> wasDark;
            wasDark.clear();
            for (int f2 : hCells[h]) if (!cellLit(f2)) wasDark.push_back(f2);
            for (int f2 : vCells[v]) if (!cellLit(f2)) wasDark.push_back(f2);
            // place (now those cells are lit)
            hasLight[bestF] = 1;
            applyDelta(h, v, +1);
            // for each that was dark and is now lit, fix dark-corridor counts.
            // dedup with a stamp using litCnt (currently 0 everywhere).
            for (int f2 : wasDark) {
                if (litCnt[f2]) continue;          // duplicate (intersection)
                litCnt[f2] = 1;
                hDark[hOf[f2]]--;
                vDark[vOf[f2]]--;
            }
            for (int f2 : wasDark) litCnt[f2] = 0;  // clear stamps
        }
    }

    // current placement = a feasible solution. Save it as best.
    auto collectLights = [&]() {
        vector<int> v;
        for (int f = 0; f < NF; f++) if (hasLight[f]) v.push_back(f);
        return v;
    };
    vector<int> bestLights = collectLights();
    int bestK = (int)bestLights.size();

    // ---- simulated annealing on the light set -------------------------------
    // Move: remove a random placed light, then repair every newly-dark cell by
    // greedily covering dark cells (cheapest = most dark cells covered first).
    // Accept by SA on K (number of lights). Always keep a feasible incumbent.
    Rng rng(0xC0FFEEu ^ (uint32_t)(NF * 2654435761u));
    const double TIME_LIMIT = 1.7;     // seconds wall-clock budget

    // helper: repair to feasibility from current (possibly-dark) state using the
    // same greedy max-coverage rule, but only over dark cells.
    auto repair = [&]() {
        while (darkCount > 0) {
            int bestF = -1, bestG = -1;
            for (int f = 0; f < NF; f++) {
                if (hasLight[f]) continue;
                int g = gainAt(f);
                if (g > bestG) { bestG = g; bestF = f; }
            }
            if (bestF < 0 || bestG <= 0) {
                for (int f = 0; f < NF; f++)
                    if (!cellLit(f) && !hasLight[f]) {
                        int h = hOf[f], v = vOf[f];
                        for (int f2 : hCells[h]) if (!cellLit(f2)) { hDark[hOf[f2]]--; vDark[vOf[f2]]--; }
                        for (int f2 : vCells[v]) if (!cellLit(f2)) { hDark[hOf[f2]]--; vDark[vOf[f2]]--; }
                        hasLight[f] = 1;
                        applyDelta(h, v, +1);
                    }
                break;
            }
            int h = hOf[bestF], v = vOf[bestF];
            static vector<int> wasDark;
            wasDark.clear();
            for (int f2 : hCells[h]) if (!cellLit(f2)) wasDark.push_back(f2);
            for (int f2 : vCells[v]) if (!cellLit(f2)) wasDark.push_back(f2);
            hasLight[bestF] = 1;
            applyDelta(h, v, +1);
            for (int f2 : wasDark) {
                if (litCnt[f2]) continue;
                litCnt[f2] = 1;
                hDark[hOf[f2]]--;
                vDark[vOf[f2]]--;
            }
            for (int f2 : wasDark) litCnt[f2] = 0;
        }
    };

    // helper: remove one light at f and update hDark/vDark for cells that flip
    // lit->dark.
    auto removeOne = [&](int f) {
        int h = hOf[f], v = vOf[f];
        static vector<int> wasLit;
        wasLit.clear();
        for (int f2 : hCells[h]) if (cellLit(f2)) wasLit.push_back(f2);
        for (int f2 : vCells[v]) if (cellLit(f2)) wasLit.push_back(f2);
        hasLight[f] = 0;
        applyDelta(h, v, -1);
        for (int f2 : wasLit) {
            if (litCnt[f2]) continue;
            litCnt[f2] = 1;
            if (!cellLit(f2)) { hDark[hOf[f2]]++; vDark[vOf[f2]]++; }
        }
        for (int f2 : wasLit) litCnt[f2] = 0;
    };

    int curK = bestK;
    double T = 2.0, Tend = 0.05;
    long long iter = 0;
    while (true) {
        if ((iter & 255) == 0) {
            double t = now_sec() - t0;
            if (t > TIME_LIMIT) break;
            double frac = t / TIME_LIMIT;
            T = 2.0 * pow(Tend / 2.0, frac);   // geometric cooling
        }
        iter++;

        // pick a random placed light to remove
        vector<int> placed = collectLights();
        if (placed.empty()) break;
        // remove a small random number (1..2) of lights to create slack, then
        // repair. Removing >1 occasionally helps escape local minima.
        int nRemove = 1 + (rng.nextu(100) < 25 ? 1 : 0);
        nRemove = min(nRemove, (int)placed.size());
        // snapshot current light set so we can roll back on reject
        vector<int> snapshot = placed;
        for (int t = 0; t < nRemove; t++) {
            int idx = rng.nextu((uint32_t)placed.size());
            int f = placed[idx];
            placed[idx] = placed.back();
            placed.pop_back();
            removeOne(f);
        }
        repair();
        int newK = (int)collectLights().size();

        int delta = newK - curK;   // smaller K is better
        bool accept;
        if (delta <= 0) accept = true;
        else accept = (rng.nextd() < exp(-delta / T));

        if (accept) {
            curK = newK;
            if (newK < bestK) {
                bestK = newK;
                bestLights = collectLights();
            }
        } else {
            // roll back to snapshot: remove all current lights, re-add snapshot.
            vector<int> cur = collectLights();
            for (int f : cur) removeOne(f);
            for (int f : snapshot) {
                // place f, fixing hDark/vDark for cells it flips dark->lit
                int h = hOf[f], v = vOf[f];
                static vector<int> wasDark;
                wasDark.clear();
                for (int f2 : hCells[h]) if (!cellLit(f2)) wasDark.push_back(f2);
                for (int f2 : vCells[v]) if (!cellLit(f2)) wasDark.push_back(f2);
                hasLight[f] = 1;
                applyDelta(h, v, +1);
                for (int f2 : wasDark) {
                    if (litCnt[f2]) continue;
                    litCnt[f2] = 1;
                    hDark[hOf[f2]]--;
                    vDark[vOf[f2]]--;
                }
                for (int f2 : wasDark) litCnt[f2] = 0;
            }
            curK = (int)snapshot.size();
        }
    }

    // ---- output best feasible solution --------------------------------------
    // (bestLights is always a feasible covering; verified by construction.)
    string out = to_string((int)bestLights.size()) + "\n";
    for (int f : bestLights) {
        out += to_string(cellR[f]);
        out += ' ';
        out += to_string(cellC[f]);
        out += '\n';
    }
    fputs(out.c_str(), stdout);
    return 0;
}
