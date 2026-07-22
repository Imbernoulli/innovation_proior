// TIER: greedy
#include <bits/stdc++.h>
using namespace std;

// "Obvious first pass": try the 4 candidate directions with zero mirrors and
// keep the best one. Then repeatedly find the single highest-weight cell the
// current trajectory has NOT reached (anywhere on the board), and drop a
// mirror on whichever on-trajectory cell is geometrically nearest to that
// target, choosing H vs V by a simple "which axis is farther off" rule --
// WITHOUT ever re-simulating to check whether the placement actually helps.
// This is a plausible, purely proximity-driven heuristic: it has no notion
// of the ray's orbit/period structure, so on a grid where the default orbit
// is locked onto a small gcd-governed sub-lattice, "aim near the target"
// mirrors routinely fail to splice the ray onto a genuinely different orbit.

static int R, C, Bbudget, sr, sc;
static vector<string> grid; // 'S' replaced by '.' after read
static vector<vector<int>> W;

static inline int enc(int dr, int dc) { return ((dr == 1) ? 1 : 0) * 2 + ((dc == 1) ? 1 : 0); }

// Full simulate: returns total value and (optionally) visited-cell mask / path.
static long long simulate(const vector<string> &wg, int dr0, int dc0,
                           vector<char> *visitedCellOut, vector<array<int,2>> *pathOut) {
    int dr = dr0, dc = dc0, r = sr, c = sc;
    vector<char> visitedCell(R * C, 0);
    vector<char> visitedState(R * C * 4, 0);
    long long total = W[r][c];
    visitedCell[r * C + c] = 1;
    visitedState[(r * C + c) * 4 + enc(dr, dc)] = 1;
    if (pathOut) pathOut->push_back({r, c});
    long long maxSteps = 4LL * R * C + 10;
    for (long long s = 0; s < maxSteps; s++) {
        if (r + dr < 0 || r + dr >= R) dr = -dr;
        if (c + dc < 0 || c + dc >= C) dc = -dc;
        int nr = r + dr, nc = c + dc;
        if (wg[nr][nc] == '#') break;
        if (wg[nr][nc] == 'H') dr = -dr;
        else if (wg[nr][nc] == 'V') dc = -dc;
        r = nr; c = nc;
        int idx = (r * C + c) * 4 + enc(dr, dc);
        if (visitedState[idx]) break;
        visitedState[idx] = 1;
        if (pathOut) pathOut->push_back({r, c});
        if (!visitedCell[r * C + c]) { visitedCell[r * C + c] = 1; total += W[r][c]; }
    }
    if (visitedCellOut) *visitedCellOut = visitedCell;
    return total;
}

int main() {
    scanf("%d %d %d", &R, &C, &Bbudget);
    scanf("%d %d", &sr, &sc);
    grid.assign(R, "");
    for (int i = 0; i < R; i++) { char buf[210]; scanf("%s", buf); grid[i] = buf; }
    W.assign(R, vector<int>(C, 0));
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++) scanf("%d", &W[i][j]);
    grid[sr][sc] = '.';

    int dirs[4][2] = {{-1,-1},{-1,1},{1,-1},{1,1}};
    const char* names[4] = {"UL","UR","DL","DR"};
    int best = 3; long long bestVal = -1;
    for (int d = 0; d < 4; d++) {
        long long v = simulate(grid, dirs[d][0], dirs[d][1], nullptr, nullptr);
        if (v > bestVal) { bestVal = v; best = d; }
    }
    int dr = dirs[best][0], dc = dirs[best][1];

    vector<string> work = grid;
    vector<tuple<int,int,char>> mirrors;
    set<pair<int,int>> used;

    for (int step = 0; step < Bbudget; step++) {
        vector<char> vis;
        vector<array<int,2>> path;
        simulate(work, dr, dc, &vis, &path);

        // global highest-weight cell not yet visited
        int ti = -1, tj = -1, tv = -1;
        for (int i = 0; i < R; i++)
            for (int j = 0; j < C; j++)
                if (grid[i][j] != '#' && !vis[i * C + j] && W[i][j] > tv) { tv = W[i][j]; ti = i; tj = j; }
        if (ti < 0) break; // ray already sees everything reachable

        // nearest on-path cell (Chebyshev) that isn't S and isn't already mirrored
        long long bestD = -1; int bi = -1, bj = -1;
        for (auto &pr : path) {
            int r = pr[0], c = pr[1];
            if (r == sr && c == sc) continue;
            if (used.count({r, c})) continue;
            long long d = max(abs(r - ti), abs(c - tj));
            if (bestD < 0 || d < bestD) { bestD = d; bi = r; bj = c; }
        }
        if (bi < 0) break;

        char type = (abs(ti - bi) >= abs(tj - bj)) ? 'H' : 'V';
        work[bi][bj] = type;
        used.insert({bi, bj});
        mirrors.push_back({bi, bj, type});
    }

    printf("%s\n", names[best]);
    printf("%d\n", (int)mirrors.size());
    for (auto &m : mirrors) printf("%d %d %c\n", get<0>(m), get<1>(m), get<2>(m));
    return 0;
}
