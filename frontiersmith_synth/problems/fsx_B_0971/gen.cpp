#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Derandomized Dust: Sculpting a Rotor-Router Aggregate"   family: rotor-router-odometer-shape
//
// A single source cell on an L x L torus emits N particles, one at a time. Each
// particle performs a deterministic ROTOR-ROUTER walk: at the current cell it
// consults that cell's rotor (a pointer that cycles E->N->W->S->E... on every
// visit, i.e. the classic derandomized/"Eulerian walker" exit rule), steps onto
// the neighbor it now points to, and the cell's rotor advances one notch for
// next time. The particle keeps stepping until it lands on a cell that is not
// yet occupied; that cell joins the aggregate and the particle is done. The
// ODOMETER of a cell is simply how many times its rotor has fired.
//
// With EVERY rotor starting at phase 0 (the textbook default), the aggregate is
// known to round itself into a disk centered on the source -- this is the
// well-known rotor-router "shape theorem" trap. The only lever the solver is
// given is the INITIAL phase (0..3) of every cell's rotor before emission
// starts.
//
// PLANTED / TRAP STRUCTURE: the target is FOUR independent "clearance" self-
// avoiding walks (1-cell-wide corridors: no two non-consecutive cells of any
// arm are ever grid-adjacent, including across arms), one launched from each
// of the source's four neighbors -- so, unlike a single corridor, EVERY one of
// the source's four round-robin exits is directly useful (a real solver's
// first-order gain), and the remaining difficulty is threading each arm's
// own winding/spiraling continuation correctly. Small testId -> short, mostly
// straight arms close to the source (near-disk-friendly, sanity cases). Large
// testId -> long, tightly winding / spiraling arms that a disk (or any
// Euclidean-nearest-target gradient) cannot follow, because a point on the far
// side of a spiral arm can be nearer in straight-line terms than the true walk
// order along that arm. That is the trap for both the uniform-rotor default
// AND a naive "always steer toward nearest target cell" greedy.
//
// Input format emitted here:
//   L N
//   sx sy
//   N lines: x y      (the target cell coordinates, 0<=x,y<L, all distinct,
//                       none equal to the source; each of the source's four
//                       neighbors is a target cell, i.e. the start of an arm)
// -----------------------------------------------------------------------------

static int L;
static vector<vector<char>> inpath;   // occupancy while carving all arms

int DX[4] = {1, 0, -1, 0};
int DY[4] = {0, 1, 0, -1};

inline int wrapc(int v) { v %= L; if (v < 0) v += L; return v; }

// true iff, having just stepped into (x,y) from (px,py), the only path-neighbor
// of (x,y) is (px,py) itself (guarantees the induced adjacency graph of the
// eventual target set is a union of simple paths -- no accidental shortcuts,
// even between different arms).
bool clearanceOk(int x, int y, int px, int py) {
    for (int d = 0; d < 4; d++) {
        int nx = wrapc(x + DX[d]), ny = wrapc(y + DY[d]);
        if (inpath[nx][ny]) {
            if (!(nx == px && ny == py)) return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    double f = (testId - 1) / 9.0;   // 0..1 difficulty/windiness ramp

    L = 10 + (int)llround(f * 18.0);              // 10 .. 28
    int sx = L / 2, sy = L / 2;

    int targetLen = max(8, (18 * L * L) / 100);    // ~18% fill -> plenty of room
    int armTarget = max(2, targetLen / 4);

    // windiness: probability weight of continuing straight vs. forced turning.
    // low testId: mostly straight (near-disk friendly). high testId: forced
    // winding / spiral arms that break any Euclidean / disk intuition.
    int straightWeight = (testId <= 2) ? 7 : (testId <= 5) ? 3 : 1;
    int turnWeight      = (testId <= 2) ? 1 : (testId <= 5) ? 3 : 6;
    bool spiralBias = (testId >= 6);   // bias turns to a single rotational sense

    inpath.assign(L, vector<char>(L, 0));
    vector<pair<int,int>> allCells;

    for (int arm = 0; arm < 4; arm++) {
        vector<pair<int,int>> path;
        int startx = wrapc(sx + DX[arm]), starty = wrapc(sy + DY[arm]);
        if (inpath[startx][starty]) continue;   // another arm already claimed it (tiny L)
        path.push_back({startx, starty});
        inpath[startx][starty] = 1;
        int prevDir = arm;
        int spiralTurnSense = (arm % 2 == 0) ? +1 : -1;   // vary sense per arm

        int guard = 0;
        const int GUARD_CAP = 100000;
        while ((int)path.size() < armTarget && guard < GUARD_CAP) {
            guard++;
            auto [cx, cy] = path.back();
            vector<int> cand;
            for (int d = 0; d < 4; d++) {
                int nx = wrapc(cx + DX[d]), ny = wrapc(cy + DY[d]);
                if (nx == sx && ny == sy) continue;              // never step on source
                if (inpath[nx][ny]) continue;                    // no revisits, no other-arm touch
                if (!clearanceOk(nx, ny, cx, cy)) continue;       // no shortcuts
                cand.push_back(d);
            }
            if (cand.empty()) {
                if (path.size() <= 1) break;
                inpath[cx][cy] = 0;
                path.pop_back();
                continue;
            }
            vector<int> bag;
            for (int d : cand) {
                int w;
                if (spiralBias) {
                    int turnedDir = (prevDir + spiralTurnSense + 4) % 4;
                    w = (d == prevDir) ? straightWeight : (d == turnedDir ? turnWeight * 3 : turnWeight);
                } else {
                    w = (d == prevDir) ? straightWeight : turnWeight;
                }
                for (int i = 0; i < w; i++) bag.push_back(d);
            }
            int pick = bag[rnd.next(0, (int)bag.size() - 1)];
            int nx = wrapc(cx + DX[pick]), ny = wrapc(cy + DY[pick]);
            inpath[nx][ny] = 1;
            path.push_back({nx, ny});
            prevDir = pick;
        }
        for (auto &c : path) allCells.push_back(c);
    }

    int N = (int)allCells.size();
    printf("%d %d\n", L, N);
    printf("%d %d\n", sx, sy);
    for (auto &c : allCells) printf("%d %d\n", c.first, c.second);
    return 0;
}
