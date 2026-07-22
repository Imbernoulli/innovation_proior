#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// -----------------------------------------------------------------------------
// "Punching Holes Without Dropping the Sheet"  (generator)
// family: web-integrity-punching
//
// Grid R x C. '#' = frame (permanent, always present, forms the full outer ring
// so (0,0) is always a valid connectivity anchor). 'x' = void (never material,
// just shapes the blank's silhouette). '.' = punch job.
//
// PLANTED STRUCTURE (one per test): a small ROOM (a solid block of job cells) is
// walled off from the rest of the interior by a 1-cell void margin on every side
// except a single 1-cell-wide CORRIDOR (a straight chain of job cells, itself
// flanked by void on both perpendicular sides its whole length) that is the
// room's ONLY path back to the open field / frame. The corridor's tool is fixed
// to 1 (the lowest tool id); the room's tool is drawn from [2,T]. Consequence:
// sorting all jobs by ascending tool id (the "obvious" turret-minimizing recipe)
// punches every corridor before any room cell it guards -- severing the room
// immediately, since removing a corridor cell while its room is still unpunched
// is exactly what disconnects it (open-field job cells never do this: they are
// never the unique cut-cell for anything). Punching a room's own cells first
// (before its corridor) never disconnects anything.
//
// testId is a size/difficulty ladder: R in [6,12], C in [8,16], T in [2,4],
// growing with testId. 4 of the 10 tests (>= 3 required by spec) carry a
// planted room+corridor trap; the rest are pure open field, where tool-sort
// is genuinely fine (so the "obvious" recipe's mean reflects real wins on
// safe tests, not just smaller losses on every test).
// -----------------------------------------------------------------------------

int R, C, T;
vector<string> grid;

bool allDot(int r0, int r1, int c0, int c1) {
    if (r0 < 0 || c0 < 0 || r1 >= R || c1 >= C || r0 > r1 || c0 > c1) return false;
    for (int r = r0; r <= r1; r++)
        for (int c = c0; c <= c1; c++)
            if (grid[r][c] != '.') return false;
    return true;
}

// side: 0=L (corridor runs left from room, cline=row), 1=R (right, cline=row),
//       2=T (up from room, cline=col),               3=B (down, cline=col)
bool tryPlaceChamber(int rr0, int rr1, int rc0, int rc1, int cline, int clen, int side,
                      vector<pair<int,int>>& roomCells, vector<pair<int,int>>& corridorCells) {
    if (rr0 < 2 || rc0 < 2 || rr1 > R - 3 || rc1 > C - 3) return false;
    int mr0 = rr0 - 1, mr1 = rr1 + 1, mc0 = rc0 - 1, mc1 = rc1 + 1;
    if (!allDot(mr0, mr1, mc0, mc1)) return false;

    int cstart, cend;
    if (side == 0) { cstart = rc0 - 1; cend = cstart - clen + 1; if (cend < 1) return false; if (!allDot(cline - 1, cline + 1, cend, cstart)) return false; }
    else if (side == 1) { cstart = rc1 + 1; cend = cstart + clen - 1; if (cend > C - 2) return false; if (!allDot(cline - 1, cline + 1, cstart, cend)) return false; }
    else if (side == 2) { cstart = rr0 - 1; cend = cstart - clen + 1; if (cend < 1) return false; if (!allDot(cend, cstart, cline - 1, cline + 1)) return false; }
    else { cstart = rr1 + 1; cend = cstart + clen - 1; if (cend > R - 2) return false; if (!allDot(cstart, cend, cline - 1, cline + 1)) return false; }

    // commit: void margin ring
    for (int c = mc0; c <= mc1; c++) { grid[mr0][c] = 'x'; grid[mr1][c] = 'x'; }
    for (int r = mr0; r <= mr1; r++) { grid[r][mc0] = 'x'; grid[r][mc1] = 'x'; }
    // room interior
    for (int r = rr0; r <= rr1; r++)
        for (int c = rc0; c <= rc1; c++) { grid[r][c] = '.'; roomCells.push_back({r, c}); }
    // corridor (punches a gap through the ring + extends out, void-flanked)
    if (side == 0) for (int c = cend; c <= cstart; c++) { grid[cline][c] = '.'; corridorCells.push_back({cline, c}); grid[cline - 1][c] = 'x'; grid[cline + 1][c] = 'x'; }
    else if (side == 1) for (int c = cstart; c <= cend; c++) { grid[cline][c] = '.'; corridorCells.push_back({cline, c}); grid[cline - 1][c] = 'x'; grid[cline + 1][c] = 'x'; }
    else if (side == 2) for (int r = cend; r <= cstart; r++) { grid[r][cline] = '.'; corridorCells.push_back({r, cline}); grid[r][cline - 1] = 'x'; grid[r][cline + 1] = 'x'; }
    else for (int r = cstart; r <= cend; r++) { grid[r][cline] = '.'; corridorCells.push_back({r, cline}); grid[r][cline - 1] = 'x'; grid[r][cline + 1] = 'x'; }
    return true;
}

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    double f = (testId - 1) / 9.0;

    R = 6 + (int)llround(f * 6.0);   // 6..12
    C = 8 + (int)llround(f * 8.0);   // 8..16
    T = 2 + (testId - 1) % 3;        // 2,3,4 cycling

    grid.assign(R, string(C, '.'));
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++)
            if (r == 0 || r == R - 1 || c == 0 || c == C - 1) grid[r][c] = '#';

    // Not every test carries a planted trap: a handful are pure open field,
    // where tool-sort's incidental violations (if any -- an open field can
    // still accidentally pinch off a cell under a bad order) are typically far
    // smaller than a designed chamber's, so greedy's mean is pulled up by real
    // wins there rather than being uniformly bad. >= 3 of 10 tests below DO
    // carry a planted room+corridor (the spec's required floor).
    static const bool wantsChamber[11] = {
        false, // unused (1-indexed)
        false, false, false, true, false, true, false, false, true, true
    };
    vector<pair<int,int>> roomCells, corridorCells;
    bool placed = false;
    int chamberAttempts = wantsChamber[testId] ? 400 : 0;
    for (int attempt = 0; attempt < chamberAttempts && !placed; attempt++) {
        int rh = rnd.next(2, 4), rw = rnd.next(2, 4);
        if (R - 2 - rh < 2 || C - 2 - rw < 2) continue;
        int rr0 = rnd.next(2, R - 2 - rh);
        int rc0 = rnd.next(2, C - 2 - rw);
        int rr1 = rr0 + rh - 1, rc1 = rc0 + rw - 1;
        int side = rnd.next(0, 3);
        int clen = rnd.next(2, 4);
        int cline = (side == 0 || side == 1) ? rnd.next(rr0, rr1) : rnd.next(rc0, rc1);
        vector<pair<int,int>> rc_, cc_;
        if (tryPlaceChamber(rr0, rr1, rc0, rc1, cline, clen, side, rc_, cc_)) {
            roomCells = rc_; corridorCells = cc_; placed = true;
        }
    }

    set<pair<int,int>> roomSet(roomCells.begin(), roomCells.end());
    set<pair<int,int>> corrSet(corridorCells.begin(), corridorCells.end());

    struct Job { int r, c, tool; };
    vector<Job> jobs;
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++)
            if (grid[r][c] == '.') {
                int tool;
                pair<int,int> p = {r, c};
                if (corrSet.count(p)) tool = 1;
                else if (roomSet.count(p)) tool = rnd.next(2, T);
                else tool = rnd.next(1, T);
                jobs.push_back({r, c, tool});
            }

    int N = (int)jobs.size();
    printf("%d %d %d %d\n", R, C, N, T);
    for (int r = 0; r < R; r++) printf("%s\n", grid[r].c_str());
    for (auto& j : jobs) printf("%d %d %d\n", j.r, j.c, j.tool);
    return 0;
}
