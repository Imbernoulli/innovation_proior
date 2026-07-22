// TIER: strong
#include <bits/stdc++.h>
using namespace std;

// Insight: the mirror-free ray's coverage is a fixed billiard orbit -- an
// unfolding argument turns it into a straight line on a lattice, and its
// period (hence which fraction of the board it ever reaches) is a
// gcd(R-1,C-1)-governed invariant, not something a purely local/reactive
// rule can talk its way around. Rather than reacting to individual repeats
// as they occur, we explicitly find WHERE the trajectory's state first
// repeats (the exact point its current orbit "closes"), and place a mirror
// exactly there -- the only point where a change provably alters which
// orbit class the ray continues on. We try BOTH mirror types at that point
// and keep whichever a full re-simulation shows is better (or, on the rare
// case the closing cell is unusable, fall back to the last usable cell on
// the path). Repeating this "detect closure, then splice" step, instead of
// reacting to proximity of high-value cells, is what reliably unlocks new
// territory on a gcd-trapped grid.

static int R, C, Bbudget, sr, sc;
static vector<string> grid;
static vector<vector<int>> W;

static inline int enc(int dr, int dc) { return ((dr == 1) ? 1 : 0) * 2 + ((dc == 1) ? 1 : 0); }

// Full simulate -> total value.
static long long simulateTotal(const vector<string> &wg, int dr0, int dc0) {
    int dr = dr0, dc = dc0, r = sr, c = sc;
    vector<char> visitedCell(R * C, 0);
    vector<char> visitedState(R * C * 4, 0);
    long long total = W[r][c];
    visitedCell[r * C + c] = 1;
    visitedState[(r * C + c) * 4 + enc(dr, dc)] = 1;
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
        if (!visitedCell[r * C + c]) { visitedCell[r * C + c] = 1; total += W[r][c]; }
    }
    return total;
}

// Simulate and report the path (state list) AND whether/where it closes
// (repeats a state) vs escapes by absorption. closedCell = {-1,-1} if it
// terminated by absorption instead of a genuine cycle.
static void simulatePath(const vector<string> &wg, int dr0, int dc0,
                          vector<array<int,2>> &path, array<int,2> &closedCell) {
    int dr = dr0, dc = dc0, r = sr, c = sc;
    vector<char> visitedState(R * C * 4, 0);
    path.clear();
    path.push_back({r, c});
    visitedState[(r * C + c) * 4 + enc(dr, dc)] = 1;
    closedCell = {-1, -1};
    long long maxSteps = 4LL * R * C + 10;
    for (long long s = 0; s < maxSteps; s++) {
        if (r + dr < 0 || r + dr >= R) dr = -dr;
        if (c + dc < 0 || c + dc >= C) dc = -dc;
        int nr = r + dr, nc = c + dc;
        if (wg[nr][nc] == '#') return; // absorbed, no closure
        if (wg[nr][nc] == 'H') dr = -dr;
        else if (wg[nr][nc] == 'V') dc = -dc;
        r = nr; c = nc;
        int idx = (r * C + c) * 4 + enc(dr, dc);
        if (visitedState[idx]) { closedCell = {r, c}; return; }
        visitedState[idx] = 1;
        path.push_back({r, c});
    }
}

static mt19937 rngLocal(12345);

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
        long long v = simulateTotal(grid, dirs[d][0], dirs[d][1]);
        if (v > bestVal) { bestVal = v; best = d; }
    }
    int dr = dirs[best][0], dc = dirs[best][1];

    vector<string> work = grid;
    vector<tuple<int,int,char>> mirrors;
    set<pair<int,int>> used;

    const int SAMPLE_CAP = 220;
    for (int step = 0; step < Bbudget; step++) {
        vector<array<int,2>> path;
        array<int,2> closed;
        simulatePath(work, dr, dc, path, closed);

        // Candidate set: the exact cycle-closing cell (the mathematically
        // principled "orbit splice" point) PLUS an evenly-spaced sample of
        // the rest of the current path (covers cases -- e.g. a near-corner
        // start -- where the first closure is a poor local choice and a
        // splice further along the orbit reaches the planted value faster).
        vector<pair<int,int>> cand;
        if (closed[0] >= 0) cand.push_back({closed[0], closed[1]});
        int psz = (int)path.size();
        int take = min(psz, SAMPLE_CAP);
        for (int t = 0; t < take; t++) {
            int idx = (take == 1) ? 0 : (int)((long long)t * (psz - 1) / (take - 1));
            cand.push_back({path[idx][0], path[idx][1]});
        }

        long long bestF = -1, curF = simulateTotal(work, dr, dc);
        int bi = -1, bj = -1; char bestType = 0;
        for (auto &pr : cand) {
            int r = pr.first, c = pr.second;
            if (r == sr && c == sc) continue;
            if (grid[r][c] != '.') continue;
            if (used.count({r, c})) continue;
            for (char type : {'H', 'V'}) {
                char save = work[r][c];
                work[r][c] = type;
                long long F = simulateTotal(work, dr, dc);
                if (F > bestF) { bestF = F; bi = r; bj = c; bestType = type; }
                work[r][c] = save;
            }
        }
        if (bi < 0 || bestF <= curF) break; // no candidate helps any more
        work[bi][bj] = bestType;
        used.insert({bi, bj});
        mirrors.push_back({bi, bj, bestType});
    }

    printf("%s\n", names[best]);
    printf("%d\n", (int)mirrors.size());
    for (auto &m : mirrors) printf("%d %d %c\n", get<0>(m), get<1>(m), get<2>(m));
    return 0;
}
