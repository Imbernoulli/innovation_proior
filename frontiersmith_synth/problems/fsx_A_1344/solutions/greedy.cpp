// TIER: greedy
// Reactive chase: the ONE approach an average strong coder writes first.
// Every turn, re-evaluate the runner's CURRENT position, find whichever
// gate looks nearest *right now*, and spend this turn's budget directly on
// that gate's still-open cells. There is no global plan: if the "current
// nearest gate" verdict drifts between turns (a plausible symmetric gate
// on a multi-gate map, or a gate that only becomes reachable-fastest once
// the runner has already moved), budget gets split across gates and never
// finishes sealing any of them in time -- always one step behind a
// runner whose whole future path was already computable on turn 1.
#include <bits/stdc++.h>
using namespace std;

static int N, T, S, K;
static long long B;
static vector<string> grid;
static vector<pair<int,int>> perim;
static const int dr[8] = {-1,-1,-1, 0,0, 1,1,1};
static const int dc[8] = {-1, 0, 1,-1,1,-1,0,1};

struct Flood {
    vector<vector<int>> dist;
    vector<vector<pair<int,int>>> parent;
    int tr = -1, tc = -1;
    bool found = false;
};

static Flood flood(vector<vector<char>>& blocked, int sr, int sc) {
    Flood res;
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
            if (grid[nr][nc] != '.') continue;
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
        if (d < bestD || (d == bestD && (r < br || (r == br && c < bc)))) { bestD = d; br = r; bc = c; }
    }
    if (bestD != INT_MAX) { res.found = true; res.tr = br; res.tc = bc; }
    return res;
}

static vector<pair<int,int>> movePathTo(const Flood& fr, int tr, int tc) {
    vector<pair<int,int>> path;
    int r = tr, c = tc;
    while (true) {
        path.push_back({r, c});
        if (fr.dist[r][c] == 0) break;
        auto pr = fr.parent[r][c];
        r = pr.first; c = pr.second;
    }
    reverse(path.begin(), path.end());
    return path;
}

// expand a perimeter index into its full contiguous open ('.') gate run
static vector<pair<int,int>> gateRunAt(int idx) {
    int P = (int)perim.size();
    int lo = idx, hi = idx;
    while (grid[perim[(lo - 1 + P) % P].first][perim[(lo - 1 + P) % P].second] == '.') lo = (lo - 1 + P) % P;
    while (grid[perim[(hi + 1) % P].first][perim[(hi + 1) % P].second] == '.') hi = (hi + 1) % P;
    vector<pair<int,int>> cells;
    for (int i = lo;; i = (i + 1) % P) { cells.push_back(perim[i]); if (i == hi) break; }
    return cells;
}

int main() {
    cin >> N >> T >> S >> K >> B;
    int rx, ry; cin >> rx >> ry;
    grid.assign(N, "");
    for (int r = 0; r < N; r++) cin >> grid[r];
    perim.clear();
    for (int c = 0; c < N; c++) perim.push_back({0, c});
    for (int r = 1; r < N - 1; r++) perim.push_back({r, N - 1});
    for (int c = N - 1; c >= 0; c--) perim.push_back({N - 1, c});
    for (int r = N - 2; r >= 1; r--) perim.push_back({r, 0});
    int P = (int)perim.size();
    vector<int> idxAt(N * N, -1);
    for (int i = 0; i < P; i++) idxAt[perim[i].first * N + perim[i].second] = i;

    vector<vector<char>> blocked(N, vector<char>(N, 0));
    long long used = 0;
    bool escaped = false, contained = false;

    ostringstream out;
    for (int t = 1; t <= T; t++) {
        vector<pair<int,int>> toBlock;
        if (!escaped && !contained) {
            Flood fr = flood(blocked, rx, ry);
            if (!fr.found) {
                contained = true;
            } else {
                int pIdx = idxAt[fr.tr * N + fr.tc];
                vector<pair<int,int>> gate = gateRunAt(pIdx);
                for (auto& cell : gate) {
                    if ((int)toBlock.size() >= K) break;
                    if (used + (int)toBlock.size() >= B) break;
                    if (blocked[cell.first][cell.second]) continue;
                    toBlock.push_back(cell);
                }
                for (auto& pr : toBlock) blocked[pr.first][pr.second] = 1;
                used += (int)toBlock.size();
                Flood fr2 = flood(blocked, rx, ry);
                if (!fr2.found) {
                    contained = true;
                } else if (fr2.dist[fr2.tr][fr2.tc] <= S) {
                    escaped = true;
                } else {
                    vector<pair<int,int>> path2 = movePathTo(fr2, fr2.tr, fr2.tc);
                    int idx = min((int)path2.size() - 1, S);
                    rx = path2[idx].first; ry = path2[idx].second;
                }
            }
        }
        out << (int)toBlock.size();
        for (auto& pr : toBlock) out << " " << pr.first << " " << pr.second;
        out << "\n";
    }
    cout << out.str();
    return 0;
}
