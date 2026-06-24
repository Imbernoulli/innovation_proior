#include <bits/stdc++.h>
using namespace std;

struct DSU {
    vector<int> p;
    void init(int n) { p.resize(n); iota(p.begin(), p.end(), 0); }
    int f(int x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }
    void u(int a, int b) { a = f(a); b = f(b); if (a != b) p[a] = b; }
};

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    vector<int> eu(m), ev(m), ec(m);
    {
        vector<long long> raw(m);
        for (int i = 0; i < m; i++) {
            long long u, v, c;
            cin >> u >> v >> c;
            eu[i] = (int)u; ev[i] = (int)v; raw[i] = c;
        }
        // Compress line labels to 0..L-1 (labels may be huge / sparse).
        vector<long long> tmp(raw.begin(), raw.end());
        sort(tmp.begin(), tmp.end());
        tmp.erase(unique(tmp.begin(), tmp.end()), tmp.end());
        for (int i = 0; i < m; i++)
            ec[i] = (int)(lower_bound(tmp.begin(), tmp.end(), raw[i]) - tmp.begin());
    }

    if (n - 1 == 0) { cout << 0 << "\n"; return 0; }   // start == destination

    // ---- Build "line-component" super-nodes. ----------------------------------------------
    // Within one line c, two stations linked by a chain of line-c edges are mutually reachable
    // with ZERO transfers.  So the unit you may roam for free is a *connected component of the
    // subgraph induced by line c* -- NOT the line as a whole (two disjoint segments that merely
    // share a label are not connected).  Each (line, component) becomes one super-node.
    //
    // Layered 0-1 BFS:
    //   station ---- weight 1 ----> line-component node   (boarding / switching onto a line)
    //   line-component node -- weight 0 --> station       (riding for free to any of its stations)
    // dist[station] then counts boardings; with the first board free, transfers = boardings - 1,
    // so the answer for station n-1 is dist[n-1] - 1.

    // Group edge indices by compressed line label.
    int L = 0;
    for (int i = 0; i < m; i++) L = max(L, ec[i] + 1);
    vector<vector<int>> byLine(L);
    for (int i = 0; i < m; i++) byLine[ec[i]].push_back(i);

    DSU dsu; dsu.init(n);
    vector<vector<pair<int,int>>> g(n); // ids 0..n-1 are stations; component nodes appended after
    int nextComp = n;

    for (int c = 0; c < L; c++) {
        auto &es = byLine[c];
        if (es.empty()) continue;
        // Local union-find over only the stations this line touches: reset them to singletons,
        // then union along this line's edges, so find() never escapes the touched set.
        for (int idx : es) { dsu.p[eu[idx]] = eu[idx]; dsu.p[ev[idx]] = ev[idx]; }
        for (int idx : es) dsu.u(eu[idx], ev[idx]);

        unordered_map<int,int> rootToNode;
        rootToNode.reserve(es.size() * 2 + 1);
        auto getNode = [&](int station) -> int {
            int r = dsu.f(station);
            auto it = rootToNode.find(r);
            if (it != rootToNode.end()) return it->second;
            int id = nextComp++;
            rootToNode[r] = id;
            g.emplace_back();        // adjacency row for the new component node
            return id;
        };
        for (int idx : es) {
            for (int s : {eu[idx], ev[idx]}) {
                int node = getNode(s);
                g[s].push_back({node, 1});  // board / switch onto this line-component
                g[node].push_back({s, 0});  // ride for free to this station
            }
        }
    }

    int V = (int)g.size();
    const int INF = INT_MAX;
    vector<int> dist(V, INF);
    deque<int> dq;
    dist[0] = 0;
    dq.push_back(0);
    while (!dq.empty()) {
        int x = dq.front(); dq.pop_front();
        int dx = dist[x];
        for (auto &pr : g[x]) {
            int y = pr.first, w = pr.second;
            if (dx + w < dist[y]) {
                dist[y] = dx + w;
                if (w == 0) dq.push_front(y);
                else dq.push_back(y);
            }
        }
    }

    int dst = n - 1;
    if (dist[dst] >= INF) cout << -1 << "\n";          // unreachable
    else cout << (dist[dst] - 1) << "\n";              // boardings - 1 (first board is free)
    return 0;
}
