#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// gen.cpp -- "Rush-Hour Rerouting on the Convex-Congestion Metro".
// Builds a subway network as a grid (guaranteed parallel routes = the whole point of the
// convex-congestion twist) plus a few long chord segments, then a set of O-D demands.
// testId is a difficulty/structure ladder: tiny grid at 1 growing to ~2000 stations at 10.
// Adversarial structure:
//   * PLANTED: the grid always hides many short alternate routes; spreading load across them
//     halves the a_e*x^2 crowding, so a good plan is present but non-obvious.
//   * TRAP (even testIds): origins clustered in one corner, destinations in the opposite one,
//     so every naive shortest route funnels through the same central bottleneck segments
//     (small capacity + high a) -- single-path routing gets crushed by x^2.
//   * NEEDLE (testId % 3 == 0): one huge high-volume demand hidden among many tiny ones; it
//     must be spread and prioritised.
// Fills the constraint envelope on the largest tests (N ~ 2000, M ~ 6000, D ~ 1200).

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    int idx = min(max(testId, 1), 10);

    // grid dimensions per test (rows R, cols C), N = R*C
    int Rs[10] = {2, 4, 6, 10, 14, 18, 24, 30, 40, 44};
    int Cs[10] = {3, 5, 8, 10, 14, 20, 25, 33, 40, 45};
    int R = Rs[idx - 1], C = Cs[idx - 1];
    int N = R * C;

    bool trap   = (idx % 2 == 0);
    bool needle = (idx % 3 == 0);

    auto sid = [&](int r, int c) { return r * C + c + 1; };  // 1-indexed station id

    // ---- build segments (grid) ----
    struct Seg { int u, v, cap, a; };
    vector<Seg> segs;
    set<pair<int,int>> have;
    auto addSeg = [&](int u, int v, int cp, int a) {
        int x = min(u, v), y = max(u, v);
        if (x == y) return;
        if (!have.insert({x, y}).second) return;
        segs.push_back({x, y, cp, a});
    };

    // central bottleneck band: segments near the middle column get small caps + high a
    int midC = C / 2;
    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C; c++) {
            // horizontal segment (r,c)-(r,c+1)
            if (c + 1 < C) {
                bool bottleneck = trap && (c == midC - 1);   // the funnel column boundary
                int cp = bottleneck ? rnd.next(4, 14) : rnd.next(30, 200);
                int a  = bottleneck ? rnd.next(3, 4) : rnd.next(1, 3);
                addSeg(sid(r, c), sid(r, c + 1), cp, a);
            }
            // vertical segment (r,c)-(r+1,c)
            if (r + 1 < R) {
                int cp = rnd.next(30, 200);
                int a  = rnd.next(1, 3);
                addSeg(sid(r, c), sid(r + 1, c), cp, a);
            }
        }
    }

    // a few long chord segments (extra express links) to enrich alternate routes,
    // capped so M stays within the envelope
    int chords = min((int)(N / 8), 6000 - (int)segs.size());
    for (int k = 0; k < chords; k++) {
        int u = rnd.next(1, N), v = rnd.next(1, N);
        if (u == v) continue;
        int cp = rnd.next(20, 120);
        int a  = rnd.next(2, 4);
        addSeg(u, v, cp, a);
        if ((int)segs.size() >= 6000) break;
    }

    int M = (int)segs.size();

    // ---- penalty P per test ----
    // vary so some tests reward serving everyone (large P) and some reward triage (small P)
    int Pvals[10] = {1000, 600, 2500, 800, 1500, 400, 2000, 900, 3000, 1200};
    int P = Pvals[idx - 1];

    // ---- demands ----
    int D = min(1200, max(1, N * 3 / 4));
    struct Dem { int s, t, vol; };
    vector<Dem> dems;

    auto pickCorner = [&](bool leftTop) -> int {
        // pick a station in one corner region of the grid
        int rr, cc;
        if (leftTop) { rr = rnd.next(0, max(0, R / 3)); cc = rnd.next(0, max(0, C / 3)); }
        else         { rr = rnd.next(R - 1 - max(0, R / 3), R - 1);
                       cc = rnd.next(C - 1 - max(0, C / 3), C - 1); }
        rr = min(R - 1, max(0, rr));
        cc = min(C - 1, max(0, cc));
        return sid(rr, cc);
    };

    for (int d = 0; d < D; d++) {
        int s, t;
        if (trap) {
            s = pickCorner(true);
            t = pickCorner(false);
            int guard = 0;
            while (t == s && guard++ < 20) t = pickCorner(false);
        } else {
            s = rnd.next(1, N);
            t = rnd.next(1, N);
            int guard = 0;
            while (t == s && guard++ < 20) t = rnd.next(1, N);
        }
        if (s == t) { t = (s % N) + 1; }
        int vol = rnd.next(1, 200);
        dems.push_back({s, t, vol});
    }
    if (needle && !dems.empty()) {
        // one hidden high-volume commodity among many small ones
        for (auto& dm : dems) dm.vol = rnd.next(1, 20);
        dems[rnd.next(0, (int)dems.size() - 1)].vol = 200;
    }
    D = (int)dems.size();

    // ---- emit ----
    printf("%d %d %d %d\n", N, M, D, P);
    for (auto& s : segs) printf("%d %d %d %d\n", s.u, s.v, s.cap, s.a);
    for (auto& dm : dems) printf("%d %d %d\n", dm.s, dm.t, dm.vol);
    return 0;
}
