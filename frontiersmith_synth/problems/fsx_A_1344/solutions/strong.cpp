// TIER: strong
// Predictive gate scheduler. Because the map and the runner's rule are both
// known from turn 1, every gate's arrival time (its distance / speed) is
// computable in advance -- there is no need to wait until the runner is
// nearby. Rank gates by how soon they could be reached, then dedicate the
// budget to fully sealing the most urgent still-open gate before moving on
// to the next one, instead of reacting to wherever the runner currently is.
// This is the "sparse barrier built far ahead" insight: it wins because it
// exploits that the runner's reachable set grows by at most S cells/turn,
// so a gate can be closed off long before it is ever under direct threat.
#include <bits/stdc++.h>
using namespace std;

static int N, T, S, K;
static long long B;
static vector<string> grid;
static vector<pair<int,int>> perim;
static const int dr[8] = {-1,-1,-1, 0,0, 1,1,1};
static const int dc[8] = {-1, 0, 1,-1,1,-1,0,1};

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

    // one-time BFS from the runner's start over the (pre-block) open grid
    vector<vector<int>> dist(N, vector<int>(N, -1));
    queue<pair<int,int>> q;
    dist[rx][ry] = 0; q.push({rx, ry});
    while (!q.empty()) {
        auto [r, c] = q.front(); q.pop();
        for (int k = 0; k < 8; k++) {
            int nr = r + dr[k], nc = c + dc[k];
            if (nr < 0 || nr >= N || nc < 0 || nc >= N) continue;
            if (grid[nr][nc] != '.') continue;
            if (dist[nr][nc] != -1) continue;
            dist[nr][nc] = dist[r][c] + 1;
            q.push({nr, nc});
        }
    }

    // identify gates: maximal contiguous runs of open ('.') perimeter cells
    int P = (int)perim.size();
    vector<char> isGateCell(P, 0);
    for (int i = 0; i < P; i++) isGateCell[i] = (grid[perim[i].first][perim[i].second] == '.');
    vector<vector<pair<int,int>>> gates;
    {
        int start = -1;
        for (int i = 0; i < P; i++) if (!isGateCell[i]) { start = i; break; }
        if (start != -1) {
            int i = 0;
            while (i < P) {
                int j = (start + i) % P;
                if (isGateCell[j]) {
                    vector<pair<int,int>> g;
                    while (i < P && isGateCell[(start + i) % P]) { g.push_back(perim[(start + i) % P]); i++; }
                    gates.push_back(g);
                } else i++;
            }
        }
    }

    // rank gates by earliest reachable cell distance from the runner start
    vector<pair<int,int>> order; // (dist0, gate index)
    vector<int> gd0(gates.size(), INT_MAX);
    for (size_t g = 0; g < gates.size(); g++) {
        int best = INT_MAX;
        for (auto& cell : gates[g]) if (dist[cell.first][cell.second] != -1) best = min(best, dist[cell.first][cell.second]);
        gd0[g] = best;
    }
    vector<int> idxs;
    for (size_t g = 0; g < gates.size(); g++) if (gd0[g] != INT_MAX) idxs.push_back((int)g);
    sort(idxs.begin(), idxs.end(), [&](int a, int b) { return gd0[a] < gd0[b]; });

    // schedule: for each gate in urgency (ETA) order, check FEASIBILITY first
    // -- a gate takes buildTurns = ceil(width/K) turns to fully seal, and
    // must finish before the runner's own arrival time ETA = ceil(dist0/S).
    // A gate that cannot be finished in time is skipped entirely (spending
    // even one cell on a doomed gate buys zero credit, since only a FULLY
    // sealed gate stops counting as an open escape route) -- the freed
    // budget goes to the next-nearest gate that CAN be finished in time.
    vector<vector<pair<int,int>>> schedule(T + 1);
    int turnCursor = 1, slotUsed = 0;
    long long budgetUsed = 0;
    for (int gi : idxs) {
        int w = (int)gates[gi].size();
        int buildTurns = (w + K - 1) / K;
        int eta = (gd0[gi] + S - 1) / S;
        // account for a turn already partially spent (slotUsed>0) needing
        // one more turn to finish this gate's own tail
        int turnsIfStartNow = buildTurns;
        // blocks placed during turn `eta` are applied BEFORE that turn's movement
        // check, so finishing exactly on turn `eta` (not just eta-1) still seals
        // the gate in time.
        if (turnCursor + turnsIfStartNow - 1 > eta) continue; // infeasible: skip, keep cursor
        if (budgetUsed >= B || turnCursor > T) break;
        for (auto& cell : gates[gi]) {
            if (budgetUsed >= B) break;
            if (turnCursor > T) break;
            schedule[turnCursor].push_back(cell);
            slotUsed++; budgetUsed++;
            if (slotUsed == K) { slotUsed = 0; turnCursor++; }
        }
        if (slotUsed > 0) { /* leave partial turn open for the next gate's cells too */ }
        if (budgetUsed >= B || turnCursor > T) break;
    }

    ostringstream out;
    for (int t = 1; t <= T; t++) {
        auto& v = schedule[t];
        out << (int)v.size();
        for (auto& pr : v) out << " " << pr.first << " " << pr.second;
        out << "\n";
    }
    cout << out.str();
    return 0;
}
