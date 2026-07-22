// TIER: greedy
#include <bits/stdc++.h>
using namespace std;

struct Basin { int r, c, d, node; };
struct Pump { int r, c, cost, reach, node; };

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int H, W, N, P;
    if (!(cin >> H >> W >> N >> P)) return 0;
    vector<string> grid(H + 1);
    for (int r = 1; r <= H; r++) cin >> grid[r], grid[r] = " " + grid[r];
    auto node = [&](int r, int c) { return (r - 1) * W + c; };
    auto rowOf = [&](int v) { return (v - 1) / W + 1; };
    auto colOf = [&](int v) { return (v - 1) % W + 1; };
    int cells = H * W;

    vector<char> canal(cells + 1, 0);
    for (int r = 1; r <= H; r++) for (int c = 1; c <= W; c++) canal[node(r, c)] = (grid[r][c] != '#');

    vector<Basin> basins(N + 1);
    for (int i = 1; i <= N; i++) {
        cin >> basins[i].r >> basins[i].c >> basins[i].d;
        basins[i].node = node(basins[i].r, basins[i].c);
    }
    vector<Pump> pumps(P + 1);
    for (int j = 1; j <= P; j++) {
        cin >> pumps[j].r >> pumps[j].c >> pumps[j].cost >> pumps[j].reach;
        pumps[j].node = node(pumps[j].r, pumps[j].c);
    }

    auto manh = [&](int i, int p) {
        return abs(basins[i].r - pumps[p].r) + abs(basins[i].c - pumps[p].c);
    };

    vector<char> open(P + 1, 0), covered(N + 1, 0);
    int left = N;
    while (left > 0) {
        int best = -1;
        long long bestNum = -1, bestDen = 1, bestAdd = -1;
        for (int p = 1; p <= P; p++) if (!open[p]) {
            long long add = 0, geom = 0;
            for (int i = 1; i <= N; i++) if (!covered[i]) {
                int md = manh(i, p);
                if (md <= pumps[p].reach) {
                    add += basins[i].d;
                    geom += md;
                }
            }
            if (add == 0) continue;
            long long den = pumps[p].cost + 20 * geom + 1;
            if (best < 0 || add * bestDen > bestNum * den ||
                (add * bestDen == bestNum * den && add > bestAdd)) {
                best = p;
                bestNum = add;
                bestDen = den;
                bestAdd = add;
            }
        }
        if (best < 0) {
            for (int i = 1; i <= N; i++) if (!covered[i]) {
                for (int p = 1; p <= P; p++) {
                    if (pumps[p].node == basins[i].node) {
                        best = p;
                        break;
                    }
                }
                break;
            }
        }
        open[best] = 1;
        for (int i = 1; i <= N; i++) if (!covered[i] && manh(i, best) <= pumps[best].reach) {
            covered[i] = 1;
            left--;
        }
    }

    vector<vector<int>> adj(cells + 1);
    const int dr[4] = {-1, 1, 0, 0};
    const int dc[4] = {0, 0, -1, 1};
    for (int r = 1; r <= H; r++) for (int c = 1; c <= W; c++) {
        int v = node(r, c);
        if (!canal[v]) continue;
        for (int k = 0; k < 4; k++) {
            int nr = r + dr[k], nc = c + dc[k];
            if (nr < 1 || nr > H || nc < 1 || nc > W) continue;
            int u = node(nr, nc);
            if (canal[u]) adj[v].push_back(u);
        }
    }

    auto bfsPath = [&](int s, int t) {
        vector<int> par(cells + 1, -1);
        queue<int> q;
        par[s] = 0;
        q.push(s);
        while (!q.empty()) {
            int v = q.front();
            q.pop();
            if (v == t) break;
            for (int u : adj[v]) if (par[u] < 0) {
                par[u] = v;
                q.push(u);
            }
        }
        vector<int> path;
        if (par[t] < 0) return path;
        for (int v = t; v != 0; v = par[v]) {
            path.push_back(v);
            if (v == s) break;
        }
        reverse(path.begin(), path.end());
        return path;
    };

    vector<int> assign(N + 1, -1);
    vector<vector<int>> paths(N + 1);
    for (int i = 1; i <= N; i++) {
        long long bestVal = LLONG_MAX;
        int best = -1;
        for (int p = 1; p <= P; p++) if (open[p]) {
            int md = manh(i, p);
            if (md > pumps[p].reach) continue;
            long long val = 8LL * basins[i].d * md + 3LL * basins[i].d * md * md;
            if (val < bestVal) {
                bestVal = val;
                best = p;
            }
        }
        if (best < 0) {
            for (int p = 1; p <= P; p++) if (pumps[p].node == basins[i].node) {
                best = p;
                open[p] = 1;
                break;
            }
        }
        assign[i] = best;
        paths[i] = bfsPath(basins[i].node, pumps[best].node);
        if (paths[i].empty()) paths[i] = vector<int>{basins[i].node};
    }

    vector<int> opened;
    for (int p = 1; p <= P; p++) if (open[p]) opened.push_back(p);
    cout << opened.size();
    for (int p : opened) cout << ' ' << p;
    cout << '\n';
    for (int i = 1; i <= N; i++) {
        cout << assign[i] << ' ' << (int)paths[i].size() - 1;
        for (int v : paths[i]) cout << ' ' << v;
        cout << '\n';
    }
    return 0;
}
