// Polyomino Tiling Coverage -- heuristic solver.
//
// Objective: given an H x W board and P polyomino piece TYPES (type t a set of
// k_t unit cells given as normalized integer offsets, usable at most avail_t
// times, in any of 4 rotations), choose a set of non-overlapping, in-bounds
// placements to MAXIMIZE the number of covered board cells. We read the instance
// from stdin and write, to stdout,
//     M
//     type rot r c            (M lines)
// the chosen placements: piece `type`, rotation `rot` in {0,1,2,3} (quarter-turns
// clockwise), anchored at board cell (r, c). The occupied cells of a placement are
// the rotated offsets re-normalized to min 0 and shifted by (r, c).
//
// Method (the innovation): ROW-BITMASK collision test + ADD/REMOVE simulated
// annealing with an incremental covered-count delta.
//   * The board is H 64-bit words, one per row; bit c of row r means cell (r,c) is
//     occupied (W <= 60, so a row fits one word). Each rotated+normalized shape is
//     precomputed as a small array of row-masks (one uint64 per shape-row, the
//     bits of the cells in that row). Testing whether a placement at anchor (r,c)
//     is legal is then: for every shape-row dr, (board[r+dr] & (mask[dr] << c))
//     must be 0 -- O(rows-of-piece) word ANDs, no per-cell loop. Adding the piece
//     ORs those shifted masks in; removing it ANDs their complements out.
//   * Because the covered count changes by exactly +popcount(footprint) on an add
//     and -popcount on a remove, the SA score delta is O(1) given the legality
//     test. The SA state is the multiset of current placements (with per-type use
//     counts). A move ADDS a random legal placement or REMOVES a random placed
//     one; we accept by the Metropolis rule on coverage (maximize). Periodically a
//     ruin-and-recreate kick removes a few placements and greedily refills, to
//     escape the local optima a pure add/remove walk gets stuck in. Every state is
//     overlap-free and in-bounds BY CONSTRUCTION, so any snapshot -- hence the
//     best one we emit -- is feasible.
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
    uint32_t nextu(uint32_t m) { return (uint32_t)(next() % m); }
    double nextd() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};

int H, W, P;

// A concrete shape = one rotation of one type, normalized so min row = min col = 0.
struct Shape {
    int type;                 // owning piece type
    int rot;                  // rotation index in {0,1,2,3} as reported in output
    int hh, ww;               // bounding-box height / width
    int sz;                   // #cells (covered count when placed)
    vector<int> rowmask;      // hh words: bit c set if cell (row, c) belongs to shape
};

vector<vector<Shape>> shapesOf;   // shapesOf[type] = list of distinct rotations
vector<int> avail;                // avail[type]

// A placement currently in the solution.
struct Plc {
    int type, rotOut, r, c;   // rotOut = the rot value to print
    int si;                   // index into shapesOf[type]
    int sz;
};

vector<uint64_t> board;       // H words, the occupied bitmap
vector<int> useCnt;           // per-type current usage
vector<Plc> sol;              // current placements
long long coverage = 0;       // current covered-cell count

// Rotate a set of (dr,dc) by `rot` quarter-turns clockwise; clockwise quarter
// turn maps (r,c) -> (c,-r). Then normalize to min row=col=0.
static vector<pair<int,int>> rotate_norm(const vector<pair<int,int>>& cells, int rot) {
    vector<pair<int,int>> pts = cells;
    for (int t = 0; t < (rot & 3); ++t)
        for (auto& p : pts) { int r = p.first, c = p.second; p = {c, -r}; }
    int rmin = INT_MAX, cmin = INT_MAX;
    for (auto& p : pts) { rmin = min(rmin, p.first); cmin = min(cmin, p.second); }
    for (auto& p : pts) { p.first -= rmin; p.second -= cmin; }
    sort(pts.begin(), pts.end());
    return pts;
}

// Legality of placing shape s at anchor (r,c): bbox in bounds AND no overlap.
static inline bool can_place(const Shape& s, int r, int c) {
    if (r < 0 || c < 0 || r + s.hh > H || c + s.ww > W) return false;
    for (int dr = 0; dr < s.hh; ++dr) {
        uint64_t m = (uint64_t)s.rowmask[dr] << c;
        if (board[r + dr] & m) return false;
    }
    return true;
}
static inline void stamp(const Shape& s, int r, int c) {       // OR the piece in
    for (int dr = 0; dr < s.hh; ++dr) board[r + dr] |= (uint64_t)s.rowmask[dr] << c;
}
static inline void unstamp(const Shape& s, int r, int c) {     // AND it out
    for (int dr = 0; dr < s.hh; ++dr) board[r + dr] &= ~((uint64_t)s.rowmask[dr] << c);
}

// Try to add one legal placement at/around a random anchor. Returns true if added.
static bool try_add(Rng& rng) {
    int type = rng.nextu(P);
    if (useCnt[type] >= avail[type]) return false;
    auto& shs = shapesOf[type];
    int si = rng.nextu((uint32_t)shs.size());
    const Shape& s = shs[si];
    int r = rng.nextu(H), c = rng.nextu(W);
    if (!can_place(s, r, c)) return false;
    stamp(s, r, c);
    sol.push_back({type, s.rot, r, c, si, s.sz});
    useCnt[type]++;
    coverage += s.sz;
    return true;
}

// Greedily fill the board with whatever fits, scanning anchors. Used for the
// initial construction and for the "recreate" half of a ruin-and-recreate kick.
static void greedy_fill() {
    // Try larger pieces first (cover more per placement), scanning the board.
    vector<int> torder(P);
    iota(torder.begin(), torder.end(), 0);
    sort(torder.begin(), torder.end(), [&](int a, int b) {
        int sa = shapesOf[a].empty() ? 0 : shapesOf[a][0].sz;
        int sb = shapesOf[b].empty() ? 0 : shapesOf[b][0].sz;
        return sa > sb;
    });
    bool progress = true;
    while (progress) {
        progress = false;
        for (int type : torder) {
            while (useCnt[type] < avail[type]) {
                bool placed = false;
                for (int r = 0; r < H && !placed; ++r)
                    for (int c = 0; c < W && !placed; ++c)
                        for (auto& s : shapesOf[type]) {
                            if (can_place(s, r, c)) {
                                stamp(s, r, c);
                                sol.push_back({type, s.rot, r, c, (int)(&s - &shapesOf[type][0]), s.sz});
                                useCnt[type]++;
                                coverage += s.sz;
                                placed = true; progress = true;
                                break;
                            }
                        }
                if (!placed) break;
            }
        }
    }
}

int main() {
    if (!(cin >> H >> W >> P)) return 0;
    shapesOf.resize(P);
    avail.resize(P);
    for (int t = 0; t < P; ++t) {
        int k; cin >> k;
        vector<pair<int,int>> cells(k);
        for (int i = 0; i < k; ++i) cin >> cells[i].first >> cells[i].second;
        cin >> avail[t];
        // build the (up to 4) distinct rotations as bitmask shapes
        set<vector<pair<int,int>>> seen;
        for (int rot = 0; rot < 4; ++rot) {
            auto pts = rotate_norm(cells, rot);
            if (seen.count(pts)) continue;
            seen.insert(pts);
            Shape s;
            s.type = t; s.rot = rot; s.sz = k;
            int hh = 0, ww = 0;
            for (auto& p : pts) { hh = max(hh, p.first + 1); ww = max(ww, p.second + 1); }
            s.hh = hh; s.ww = ww;
            s.rowmask.assign(hh, 0);
            for (auto& p : pts) s.rowmask[p.first] |= (1 << p.second);
            shapesOf[t].push_back(s);
        }
    }

    board.assign(H, 0ULL);
    useCnt.assign(P, 0);
    coverage = 0;

    // ---- Initial construction: greedy largest-first fill (always feasible). ----
    greedy_fill();

    // Snapshot best.
    long long best = coverage;
    vector<Plc> bestSol = sol;
    long long boardArea = (long long)H * W;

    Rng rng(0x2024ABCDULL ^ ((uint64_t)H << 40) ^ ((uint64_t)W << 24) ^ ((uint64_t)P << 8));

    // ---- Simulated annealing: add / remove placements; ruin-and-recreate kicks. ----
    const double TL = 1.8;
    double t0 = now_sec();
    // temperature in "cells": start a few cells, end below one cell.
    double T0 = 3.0, T1 = 0.20;
    long long iter = 0;

    while (true) {
        if ((iter & 1023) == 0) {
            if (now_sec() - t0 > TL) break;
        }
        ++iter;
        double frac = (now_sec() - t0) / TL;
        if (frac > 1.0) frac = 1.0;
        double T = T0 * pow(T1 / T0, frac);

        int roll = rng.nextu(100);

        if (roll < 8 && !sol.empty()) {
            // ---- ruin-and-recreate kick: remove a few, refill greedily ----
            int kick = 1 + rng.nextu(4);
            for (int q = 0; q < kick && !sol.empty(); ++q) {
                int idx = rng.nextu((uint32_t)sol.size());
                Plc pl = sol[idx];
                const Shape& s = shapesOf[pl.type][pl.si];
                unstamp(s, pl.r, pl.c);
                useCnt[pl.type]--;
                coverage -= pl.sz;
                sol[idx] = sol.back(); sol.pop_back();
            }
            greedy_fill();
            if (coverage > best) { best = coverage; bestSol = sol; }
            if (coverage == boardArea) break;     // perfect cover; cannot improve
            continue;
        }

        if (roll < 54) {
            // ---- ADD move: try to add a random legal placement. ----
            // (an add never decreases coverage, so it is always accepted)
            if (try_add(rng)) {
                if (coverage > best) { best = coverage; bestSol = sol; }
                if (coverage == boardArea) break;
            }
        } else {
            // ---- REMOVE move: drop a random placement (Metropolis on -sz). ----
            if (sol.empty()) continue;
            int idx = rng.nextu((uint32_t)sol.size());
            int dsz = sol[idx].sz;
            // accept a coverage decrease of dsz with prob exp(-dsz / T)
            if (rng.nextd() < exp(-(double)dsz / max(1e-9, T))) {
                Plc pl = sol[idx];
                const Shape& s = shapesOf[pl.type][pl.si];
                unstamp(s, pl.r, pl.c);
                useCnt[pl.type]--;
                coverage -= pl.sz;
                sol[idx] = sol.back(); sol.pop_back();
            }
        }
    }

    // best may equal a non-current state; nothing else to reconcile -- emit best.
    string out;
    out.reserve(bestSol.size() * 12 + 16);
    char buf[64];
    int len = snprintf(buf, sizeof(buf), "%zu\n", bestSol.size());
    out.append(buf, len);
    for (auto& pl : bestSol) {
        len = snprintf(buf, sizeof(buf), "%d %d %d %d\n", pl.type, pl.rotOut, pl.r, pl.c);
        out.append(buf, len);
    }
    fputs(out.c_str(), stdout);
    return 0;
}
