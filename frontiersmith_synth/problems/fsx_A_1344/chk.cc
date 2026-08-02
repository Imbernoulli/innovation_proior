// testlib checker/scorer for fsx_A_1344 (Sparse Gates containment).
#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static int N, T, S, K;
static ll TotalBudget;
static int rx0, ry0;
static vector<string> baseGrid;
static vector<pair<int,int>> perim;
static const int dr[8] = {-1,-1,-1, 0,0, 1,1,1};
static const int dc[8] = {-1, 0, 1,-1,1,-1,0,1};

static void buildPerim() {
    perim.clear();
    for (int c = 0; c < N; c++) perim.push_back({0, c});
    for (int r = 1; r < N - 1; r++) perim.push_back({r, N - 1});
    for (int c = N - 1; c >= 0; c--) perim.push_back({N - 1, c});
    for (int r = N - 2; r >= 1; r--) perim.push_back({r, 0});
}

struct FloodResult {
    vector<vector<int>> dist;
    vector<vector<pair<int,int>>> parent;
    int targetR = -1, targetC = -1;
    bool found = false;
    int E = 0;
};

static FloodResult flood(const vector<vector<char>>& blocked, int sr, int sc) {
    FloodResult res;
    res.dist.assign(N, vector<int>(N, -1));
    res.parent.assign(N, vector<pair<int,int>>(N, {-1, -1}));
    queue<pair<int,int>> q;
    res.dist[sr][sc] = 0;
    q.push({sr, sc});
    while (!q.empty()) {
        auto [r, c] = q.front(); q.pop();
        for (int k = 0; k < 8; k++) {
            int nr = r + dr[k], nc = c + dc[k];
            if (nr < 0 || nr >= N || nc < 0 || nc >= N) continue;
            if (baseGrid[nr][nc] != '.') continue;
            if (blocked[nr][nc]) continue;
            if (res.dist[nr][nc] != -1) continue;
            res.dist[nr][nc] = res.dist[r][c] + 1;
            res.parent[nr][nc] = {r, c};
            q.push({nr, nc});
        }
    }
    int bestD = INT_MAX, br = -1, bc = -1;
    for (auto& p : perim) {
        int r = p.first, c = p.second;
        if (res.dist[r][c] == -1) continue;
        int d = res.dist[r][c];
        if (d < bestD || (d == bestD && (r < br || (r == br && c < bc)))) {
            bestD = d; br = r; bc = c;
        }
    }
    if (bestD != INT_MAX) { res.found = true; res.targetR = br; res.targetC = bc; }
    // E: number of maximal contiguous runs of reachable perimeter cells, cyclic
    int P = (int)perim.size();
    vector<char> reach(P);
    bool any = false;
    for (int i = 0; i < P; i++) {
        auto& p = perim[i];
        reach[i] = (res.dist[p.first][p.second] != -1);
        if (reach[i]) any = true;
    }
    if (!any) res.E = 0;
    else {
        int start = -1;
        for (int i = 0; i < P; i++) if (!reach[i]) { start = i; break; }
        if (start == -1) res.E = 1; // entire perimeter open & reachable (degenerate)
        else {
            int cnt = 0;
            for (int i = 0; i < P; i++) {
                int j = (start + i) % P;
                int jp = (start + i - 1 + P) % P;
                if (reach[j] && !reach[jp]) cnt++;
            }
            res.E = cnt;
        }
    }
    return res;
}

// Reconstruct path from (sr,sc) to (tr,tc) using parent[] from a flood rooted at (sr,sc),
// return the cell reached after moving up to `steps` steps along that shortest path.
static pair<int,int> advanceAlong(const FloodResult& fr, int tr, int tc, int steps) {
    vector<pair<int,int>> path;
    int r = tr, c = tc;
    while (true) {
        path.push_back({r, c});
        if (fr.dist[r][c] == 0) break; // reached source
        auto pr = fr.parent[r][c];
        r = pr.first; c = pr.second;
    }
    reverse(path.begin(), path.end()); // path[0] == source ... path.back() == target
    int idx = min((int)path.size() - 1, steps);
    return path[idx];
}

struct SimOut { int turnsContained; int Efinal; };

// Generic simulator: policyFn(turn t (1-indexed), current rx, current ry, blocked-so-far)
// returns the list of cells to block this turn (already assumed feasible -- used only
// for the checker's own internal baseline construction, never for the untrusted output).
template <typename PolicyFn>
static SimOut runPolicy(PolicyFn policyFn, int E0) {
    vector<vector<char>> blocked(N, vector<char>(N, 0));
    int rx = rx0, ry = ry0;
    bool escaped = false, contained = false;
    int tEsc = -1, lastE = E0;
    for (int t = 1; t <= T; t++) {
        if (!escaped && !contained) {
            auto toBlock = policyFn(t, rx, ry, blocked);
            for (auto& pr : toBlock) blocked[pr.first][pr.second] = 1;
            FloodResult fr = flood(blocked, rx, ry);
            lastE = fr.E;
            if (!fr.found) { contained = true; }
            else if (fr.dist[fr.targetR][fr.targetC] <= S) { escaped = true; tEsc = t; }
            else {
                auto np = advanceAlong(fr, fr.targetR, fr.targetC, S);
                rx = np.first; ry = np.second;
            }
        }
    }
    int turnsContained = escaped ? (tEsc - 1) : T;
    return {turnsContained, lastE};
}

static ll scoreOf(int turnsContained, int Efinal, int E0) {
    double raw = 0.6 * ((double)turnsContained / (double)T) +
                 0.4 * ((double)(E0 - Efinal) / (double)max(1, E0));
    return (ll)llround(raw * 1000000.0);
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    N = inf.readInt(); T = inf.readInt(); S = inf.readInt(); K = inf.readInt();
    TotalBudget = inf.readLong();
    rx0 = inf.readInt(); ry0 = inf.readInt();
    baseGrid.assign(N, "");
    for (int r = 0; r < N; r++) baseGrid[r] = inf.readToken();
    buildPerim();

    FloodResult fr0 = flood(vector<vector<char>>(N, vector<char>(N, 0)), rx0, ry0);
    int E0 = fr0.E;
    if (E0 <= 0) quitf(_fail, "internal generator error: no gate reachable at t=0");

    // ---- internal baseline B: block a single gate, front-loaded ("best single unit") ----
    // Enumerate every gate (contiguous open perimeter run) and its distance from the
    // runner's start. Prefer the nearest gate that is actually completable before the
    // runner could arrive (buildTurns = ceil(width/K) <= ETA = ceil(dist0/S)); fall back
    // to the globally nearest gate if none are feasible in time.
    int P = (int)perim.size();
    vector<vector<pair<int,int>>> allGates;
    {
        vector<char> isGateCell(P, 0);
        for (int i = 0; i < P; i++) isGateCell[i] = (baseGrid[perim[i].first][perim[i].second] == '.');
        int start = -1;
        for (int i = 0; i < P; i++) if (!isGateCell[i]) { start = i; break; }
        if (start != -1) {
            int i = 0;
            while (i < P) {
                int j = (start + i) % P;
                if (isGateCell[j]) {
                    vector<pair<int,int>> g;
                    while (i < P && isGateCell[(start + i) % P]) { g.push_back(perim[(start + i) % P]); i++; }
                    allGates.push_back(g);
                } else i++;
            }
        }
    }
    vector<pair<int,int>> nearestGateCells;
    {
        int bestFeasibleDist = INT_MAX, bestAnyDist = INT_MAX;
        int feasibleIdx = -1, anyIdx = -1;
        for (size_t g = 0; g < allGates.size(); g++) {
            int d0 = INT_MAX;
            for (auto& cell : allGates[g]) if (fr0.dist[cell.first][cell.second] != -1) d0 = min(d0, fr0.dist[cell.first][cell.second]);
            if (d0 == INT_MAX) continue;
            if (d0 < bestAnyDist) { bestAnyDist = d0; anyIdx = (int)g; }
            int w = (int)allGates[g].size();
            int buildTurns = (w + K - 1) / K;
            int eta = (d0 + S - 1) / S;
            if (buildTurns <= eta && d0 < bestFeasibleDist) { bestFeasibleDist = d0; feasibleIdx = (int)g; }
        }
        int chosen = (feasibleIdx != -1) ? feasibleIdx : anyIdx;
        if (chosen != -1) nearestGateCells = allGates[chosen];
    }
    ll baselineCells = min((ll)nearestGateCells.size(), TotalBudget);
    vector<vector<pair<int,int>>> baselineSchedule(T + 1);
    {
        int idx = 0;
        for (int t = 1; t <= T && idx < (int)baselineCells; t++) {
            for (int k = 0; k < K && idx < (int)baselineCells; k++) {
                baselineSchedule[t].push_back(nearestGateCells[idx++]);
            }
        }
    }
    auto baselinePolicy = [&](int t, int, int, const vector<vector<char>>&) {
        return (t <= T) ? baselineSchedule[t] : vector<pair<int,int>>();
    };
    SimOut baseOut = runPolicy(baselinePolicy, E0);
    ll B = scoreOf(baseOut.turnsContained, baseOut.Efinal, E0);
    if (B <= 0) B = 1;

    // ---- read + validate + simulate the participant's policy in lockstep ----
    vector<vector<char>> blocked(N, vector<char>(N, 0));
    int rx = rx0, ry = ry0;
    bool escaped = false, contained = false;
    int tEsc = -1, lastE = E0;
    ll budgetUsed = 0;

    for (int t = 1; t <= T; t++) {
        int c = ouf.readInt(0, K, "block_count");
        for (int i = 0; i < c; i++) {
            int r = ouf.readInt(0, N - 1, "r");
            int cc = ouf.readInt(0, N - 1, "c");
            if (baseGrid[r][cc] != '.') quitf(_wa, "turn %d: cell (%d,%d) is not open", t, r, cc);
            if (blocked[r][cc]) quitf(_wa, "turn %d: cell (%d,%d) already blocked", t, r, cc);
            if (r == rx && cc == ry) quitf(_wa, "turn %d: cannot block the runner's current cell (%d,%d)", t, r, cc);
            blocked[r][cc] = 1;
            budgetUsed++;
            if (budgetUsed > TotalBudget) quitf(_wa, "turn %d: total budget %lld exceeded (limit %lld)", t, budgetUsed, TotalBudget);
        }
        if (!escaped && !contained) {
            FloodResult fr = flood(blocked, rx, ry);
            lastE = fr.E;
            if (!fr.found) { contained = true; }
            else if (fr.dist[fr.targetR][fr.targetC] <= S) { escaped = true; tEsc = t; }
            else {
                auto np = advanceAlong(fr, fr.targetR, fr.targetC, S);
                rx = np.first; ry = np.second;
            }
        }
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens in output");

    int turnsContained = escaped ? (tEsc - 1) : T;
    ll F = scoreOf(turnsContained, lastE, E0);

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld E0=%d Ef=%d turns=%d/%d Ratio: %.6f",
          F, B, E0, lastE, turnsContained, T, sc / 1000.0);
    return 0;
}
