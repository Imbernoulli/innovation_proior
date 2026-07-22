#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Carving the Bakery Franchise Map"  (generator)  family: franchise-majority-carve
//
// R x C grid of households. Each household has a favorite recipe (1..6) and a
// spend. The map is built from a small number of large "region" blocks (each a
// contiguous column band, one dominant recipe each) plus planted "thread"
// components: short, thin, straight runs (vertical or horizontal) of a DIFFERENT
// recipe embedded strictly inside a region's interior (always leaving a >=1-cell
// margin to the region's own border and to the grid's own border along the run's
// axis), so removing a thread from its region can never disconnect the region
// (there is always a free bypass row/column past either end of the run).
//
// This directly instantiates the trap: a spatial/geometric partition (row bands,
// Voronoi-by-distance, ...) recovers the big regions but is blind to the thin
// embedded threads -- it never isolates them, so their spend is never captured.
// An insight-driven solver that reasons about SAME-RECIPE CONNECTIVITY (not
// geometry) finds each thread as its own tiny 100%-pure component and can carve
// it out, subject to the fixed territory budget K forcing it to prioritize the
// highest-value threads (packing / plurality engineering, not compact carving).
// -----------------------------------------------------------------------------

int R, C, K;
vector<vector<int>> recipe;
vector<vector<ll>> spend;
vector<vector<char>> occ;

int regionOf(int c, const vector<int>& cuts) {
    // cuts holds the exclusive upper bound of each region's column range.
    for (int i = 0; i < (int)cuts.size(); i++) if (c < cuts[i]) return i;
    return (int)cuts.size() - 1;
}

// Try to place one thread of the given recipe/length/orientation fully inside
// region columns [lo,hi], with margin so removing it cannot disconnect the
// region. Returns true on success.
bool tryPlaceThread(int lo, int hi, bool vertical, int len, int rec, ll spLo, ll spHi) {
    int width = hi - lo + 1;
    for (int attempt = 0; attempt < 60; attempt++) {
        if (vertical) {
            if (width < 3 || R < len + 2) return false;
            int col = lo + 1 + rnd.next(0, width - 3);
            if (R - 2 - len < 0) return false;
            int r0 = 1 + rnd.next(0, R - 2 - len);
            bool ok = true;
            for (int rr = r0 - 1; rr <= r0 + len && ok; rr++)
                for (int cc = col - 1; cc <= col + 1; cc++) {
                    if (rr < 0 || rr >= R || cc < lo || cc > hi) continue;
                    if (occ[rr][cc]) { ok = false; break; }
                }
            if (!ok) continue;
            for (int rr = r0; rr < r0 + len; rr++) {
                recipe[rr][col] = rec;
                spend[rr][col] = rnd.next(spLo, spHi);
                occ[rr][col] = 1;
            }
            return true;
        } else {
            if (R < 3 || width < len + 2) return false;
            int row = 1 + rnd.next(0, R - 3);
            int c0 = lo + 1 + rnd.next(0, width - 2 - len);
            bool ok = true;
            for (int rr = row - 1; rr <= row + 1 && ok; rr++)
                for (int cc = c0 - 1; cc <= c0 + len; cc++) {
                    if (rr < 0 || rr >= R || cc < lo || cc > hi) continue;
                    if (occ[rr][cc]) { ok = false; break; }
                }
            if (!ok) continue;
            for (int cc = c0; cc < c0 + len; cc++) {
                recipe[row][cc] = rec;
                spend[row][cc] = rnd.next(spLo, spHi);
                occ[row][cc] = 1;
            }
            return true;
        }
    }
    return false;
}

void buildRegions(int numRegions, int bgSpLo, int bgSpHi) {
    vector<int> cuts;
    for (int i = 1; i <= numRegions; i++) cuts.push_back((int)((ll)C * i / numRegions));
    vector<int> regRecipe(numRegions);
    for (int i = 0; i < numRegions; i++) {
        int rc;
        do { rc = rnd.next(1, 6); } while (i > 0 && rc == regRecipe[i - 1]);
        regRecipe[i] = rc;
    }
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++) {
            int reg = regionOf(c, cuts);
            recipe[r][c] = regRecipe[reg];
            spend[r][c] = rnd.next(bgSpLo, bgSpHi);
        }
    // stash for thread placement
    // (region column ranges recomputed by caller via cuts / regRecipe as needed)
}

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);

    if (t == 1) {
        // Hardcoded tiny example -- matches statement.txt exactly.
        printf("3 4 3\n");
        printf("1 2 1 1\n1 2 1 1\n1 1 1 1\n");
        printf("1 10 1 1\n1 10 1 1\n1 1 1 1\n");
        return 0;
    }

    struct Plan { int R, C, K, numRegions, nThreads; ll spLo, spHi; bool needle; };
    Plan P;
    switch (t) {
        case 2:  P = {6, 8, 4, 0, 0, 0, 0, false}; break;                 // pure random, sanity
        case 3:  P = {10, 14, 4, 2, 0, 0, 0, false}; break;               // regions only, no threads
        case 4:  P = {16, 20, 4, 2, 3, 250, 600, false}; break;           // trap: 2 regions, 3 threads, budget 2
        case 5:  P = {20, 26, 5, 2, 6, 250, 700, false}; break;           // trap: 2 regions, 6 threads, budget 3
        case 6:  P = {24, 30, 5, 2, 25, 80, 220, true}; break;            // needle: 1 huge + 24 decoys, budget 3
        case 7:  P = {28, 34, 8, 3, 8, 300, 750, false}; break;           // trap: 3 regions, 8 threads, budget 5
        case 8:  P = {32, 40, 9, 3, 15, 300, 750, false}; break;          // trap: 3 regions, 15 threads, budget 6
        case 9:  P = {40, 50, 8, 3, 20, 100, 280, true}; break;           // needle: large, budget 5
        case 10: P = {50, 60, 11, 3, 25, 300, 850, false}; break;         // largest, budget 8
        default: P = {6, 8, 4, 0, 0, 0, 0, false}; break;
    }
    R = P.R; C = P.C; K = P.K;
    recipe.assign(R, vector<int>(C, 1));
    spend.assign(R, vector<ll>(C, 1));
    occ.assign(R, vector<char>(C, 0));

    if (P.numRegions == 0) {
        // pure random sanity case: no planted structure at all
        for (int r = 0; r < R; r++)
            for (int c = 0; c < C; c++) {
                recipe[r][c] = rnd.next(1, 6);
                spend[r][c] = rnd.next(1, 50);
            }
    } else {
        int numRegions = P.numRegions;
        int bgSpLo = 5, bgSpHi = 55;
        vector<int> cuts;
        for (int i = 1; i <= numRegions; i++) cuts.push_back((int)((ll)C * i / numRegions));
        vector<int> regRecipe(numRegions);
        for (int i = 0; i < numRegions; i++) {
            int rc;
            do { rc = rnd.next(1, 6); } while (i > 0 && rc == regRecipe[i - 1]);
            regRecipe[i] = rc;
        }
        for (int r = 0; r < R; r++)
            for (int c = 0; c < C; c++) {
                int reg = regionOf(c, cuts);
                recipe[r][c] = regRecipe[reg];
                spend[r][c] = rnd.next(bgSpLo, bgSpHi);
            }

        // region column ranges [lo,hi]
        vector<pair<int,int>> range(numRegions);
        int prev = 0;
        for (int i = 0; i < numRegions; i++) { range[i] = {prev, cuts[i] - 1}; prev = cuts[i]; }

        int placed = 0;
        int needleIdx = P.needle ? rnd.next(0, P.nThreads - 1) : -1;
        for (int i = 0; i < P.nThreads; i++) {
            int reg = rnd.next(0, numRegions - 1);
            int lo = range[reg].first, hi = range[reg].second;
            bool vertical = rnd.next(0, 1) == 0;
            int len = rnd.next(3, 9);
            int rec;
            do { rec = rnd.next(1, 6); } while (rec == regRecipe[reg]);
            ll spLo = P.spLo, spHi = P.spHi;
            if (i == needleIdx) { spLo = 1600; spHi = 2000; len = rnd.next(6, 9); }
            bool ok = tryPlaceThread(lo, hi, vertical, len, rec, spLo, spHi);
            if (!ok) ok = tryPlaceThread(lo, hi, !vertical, len, rec, spLo, spHi);
            if (!ok && len > 3) ok = tryPlaceThread(lo, hi, vertical, 3, rec, spLo, spHi);
            if (!ok && len > 3) ok = tryPlaceThread(lo, hi, !vertical, 3, rec, spLo, spHi);
            if (ok) placed++;
        }
        (void)placed;
    }

    printf("%d %d %d\n", R, C, K);
    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C; c++) printf("%d%c", recipe[r][c], c + 1 == C ? '\n' : ' ');
    }
    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C; c++) printf("%lld%c", spend[r][c], c + 1 == C ? '\n' : ' ');
    }
    return 0;
}
