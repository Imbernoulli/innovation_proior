// TIER: strong
#include <bits/stdc++.h>
using namespace std;
using ll = long long;

struct Basin { int r, c, d, node; };
struct Pump { int r, c, cost, reach, node; };
struct Adj { int to, edge; };

static long long edgeKey(int a, int b) {
    if (a > b) swap(a, b);
    return (static_cast<long long>(a) << 32) ^ static_cast<unsigned int>(b);
}

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

    vector<int> width(cells + 1, 0);
    vector<char> canal(cells + 1, 0);
    for (int r = 1; r <= H; r++) {
        for (int c = 1; c <= W; c++) {
            int v = node(r, c);
            if (grid[r][c] != '#') {
                canal[v] = 1;
                width[v] = grid[r][c] - '0';
            }
        }
    }

    vector<Basin> basins(N + 1);
    for (int i = 1; i <= N; i++) {
        cin >> basins[i].r >> basins[i].c >> basins[i].d;
        basins[i].node = node(basins[i].r, basins[i].c);
    }
    vector<Pump> pumps(P + 1);
    ll privateSum = 0;
    int privateCnt = 0;
    for (int j = 1; j <= P; j++) {
        cin >> pumps[j].r >> pumps[j].c >> pumps[j].cost >> pumps[j].reach;
        pumps[j].node = node(pumps[j].r, pumps[j].c);
        if (pumps[j].reach == 0) {
            privateSum += pumps[j].cost;
            privateCnt++;
        }
    }

    unordered_map<ll, int> id;
    id.reserve((size_t)cells * 3);
    vector<int> cap;
    vector<vector<Adj>> adj(cells + 1);
    auto addEdge = [&](int a, int b) {
        if (!canal[a] || !canal[b]) return;
        ll k = edgeKey(a, b);
        if (id.find(k) != id.end()) return;
        int e = (int)cap.size();
        id[k] = e;
        cap.push_back(4 * min(width[a], width[b]) + 2);
        adj[a].push_back({b, e});
        adj[b].push_back({a, e});
    };
    for (int r = 1; r <= H; r++) for (int c = 1; c <= W; c++) {
        int v = node(r, c);
        if (!canal[v]) continue;
        if (r < H) addEdge(v, node(r + 1, c));
        if (c < W) addEdge(v, node(r, c + 1));
    }

    auto manh = [&](int i, int p) {
        return abs(basins[i].r - pumps[p].r) + abs(basins[i].c - pumps[p].c);
    };

    ll avgPrivate = privateCnt ? privateSum / privateCnt : 6000;
    int threshold = (int)max(1600LL, avgPrivate / 3);
    vector<char> open(P + 1, 0);
    for (int p = 1; p <= P; p++) {
        if (pumps[p].reach > 0 && pumps[p].cost <= threshold) open[p] = 1;
    }

    for (int i = 1; i <= N; i++) {
        bool has = false;
        int cheapest = -1;
        for (int p = 1; p <= P; p++) {
            if (manh(i, p) <= pumps[p].reach) {
                if (open[p]) has = true;
                if (cheapest < 0 || pumps[p].cost < pumps[cheapest].cost) cheapest = p;
            }
        }
        if (!has && cheapest > 0) open[cheapest] = 1;
    }

    vector<int> order(N);
    iota(order.begin(), order.end(), 1);
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (basins[a].d != basins[b].d) return basins[a].d > basins[b].d;
        return a < b;
    });

    vector<ll> load(cap.size(), 0);
    vector<int> assign(N + 1, -1);
    vector<vector<int>> paths(N + 1);
    const ll INF = (1LL << 62);

    for (int idx : order) {
        int s = basins[idx].node;
        int d = basins[idx].d;
        vector<ll> dist(cells + 1, INF);
        vector<int> par(cells + 1, -1);
        priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<pair<ll,int>>> pq;
        dist[s] = 0;
        pq.push({0, s});
        while (!pq.empty()) {
            auto [du, v] = pq.top();
            pq.pop();
            if (du != dist[v]) continue;
            for (const Adj& a : adj[v]) {
                ll before = (load[a.edge] * load[a.edge] + cap[a.edge] - 1) / cap[a.edge];
                ll nl = load[a.edge] + d;
                ll after = (nl * nl + cap[a.edge] - 1) / cap[a.edge];
                ll w = 8LL * d + (after - before);
                if (du + w < dist[a.to]) {
                    dist[a.to] = du + w;
                    par[a.to] = v;
                    pq.push({dist[a.to], a.to});
                }
            }
        }

        ll bestVal = INF;
        int bestPump = -1;
        for (int p = 1; p <= P; p++) if (open[p]) {
            int md = manh(idx, p);
            if (md > pumps[p].reach) continue;
            if (dist[pumps[p].node] >= INF / 2) continue;
            ll val = dist[pumps[p].node] + 3LL * d * md * md;
            if (val < bestVal) {
                bestVal = val;
                bestPump = p;
            }
        }
        if (bestPump < 0) {
            for (int p = 1; p <= P; p++) if (pumps[p].node == s) {
                bestPump = p;
                open[p] = 1;
                break;
            }
        }

        vector<int> path;
        int target = pumps[bestPump].node;
        if (target == s) {
            path.push_back(s);
        } else if (par[target] >= 0) {
            for (int v = target; v != -1; v = par[v]) {
                path.push_back(v);
                if (v == s) break;
            }
            reverse(path.begin(), path.end());
        }
        if (path.empty() || path.front() != s || path.back() != target) {
            path.clear();
            path.push_back(s);
            bestPump = -1;
            for (int p = 1; p <= P; p++) if (pumps[p].node == s) {
                bestPump = p;
                open[p] = 1;
                break;
            }
        }

        for (size_t k = 1; k < path.size(); k++) {
            auto it = id.find(edgeKey(path[k - 1], path[k]));
            if (it != id.end()) load[it->second] += d;
        }
        assign[idx] = bestPump;
        paths[idx] = path;
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
