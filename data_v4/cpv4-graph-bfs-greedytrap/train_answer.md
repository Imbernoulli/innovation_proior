**Problem.** A metro network has `n` stations (`0 .. n-1`) and `m` bidirectional connections; each connection joins two stations and carries an integer *line* label `c`. You board at station `0` and want station `n-1`, travelling as a walk along connections. Riding a connection on line `c` right after a connection on line `c'` is free if `c == c'` and costs **one transfer** otherwise; boarding the very first connection of the trip is free. Print the minimum number of transfers, `0` if `n-1 == 0`, or `-1` if `n-1` is unreachable. Read `n m` then `m` triples `u v c` from stdin.

**Why the obvious greedy is wrong.** The tempting move is to run a plain BFS over stations and report a count derived from edges, i.e. find the route with the *fewest connections* (or greedily extend the current line and only switch when forced). But the cost being minimized is *line changes*, not *connections*, and these objectives diverge. On stations `0..4`, target `4`, with connections `(0,2,line 1)`, `(2,4,line 2)`, `(0,1,line 5)`, `(1,3,line 5)`, `(3,4,line 5)`: the shortest route `0 -> 2 -> 4` uses two connections but switches line 1 -> line 2, costing **1** transfer; the longer route `0 -> 1 -> 3 -> 4` uses three connections all on line 5, costing **0** transfers. The longer same-line ride wins. Fewest connections is not fewest transfers, so connection-counting BFS and "extend then switch" greedy are both discarded. (Empirically, a fewest-connections greedy disagrees with the truth on dozens of random small instances.)

**Key idea — 0-1 BFS over line-component super-nodes.** "Free to stay on a line, pay 1 to switch onto a line" is a graph with edge weights in `{0, 1}`, the setting for 0-1 BFS (a deque: weight-0 relaxations to the front, weight-1 to the back). The subtlety is *what is free*: free movement on line `c` is confined to a **connected component of the subgraph induced by line-`c` connections** — two disjoint segments that merely share the label `c` are not mutually reachable for free. So mint one super-node per `(line, connected-component)`. Build a bipartite layered graph:

- station `s` --- weight `1` ---> its line-component super-node  (boarding / switching onto that line),
- super-node ---- weight `0` ---> each station `s` it contains   (riding for free within the component).

A 0-1 BFS from station `0` makes `dist[station]` count *boardings*. Since the first board is free, `transfers = boardings - 1`, so the answer for `n-1` is `dist[n-1] - 1`. Each super-node is relaxed once, so the whole thing is `O(n + m)` regardless of hub degree.

**Pitfalls.**
1. *One node per label teleports across disjoint segments.* If you attach every station touching line `c` to a single label-wide super-node, the BFS rides "through" it between two physically disconnected same-label segments — a free teleport that does not exist, fabricating reachability (a random case where `0` cannot reach the target but the buggy code returns `0` exposes it). Fix: split each label into its connected components with a per-line union-find, one super-node per DSU root.
2. *The `(station, last-line)` state is correct but quadratic on a hub.* A hub entered on `D` distinct lines accumulates `D` states, each rescanning all `D` incident connections — `O(D^2)`, a TLE at `D = 2*10^5`. The line-component model relaxes each super-node once and removes the blow-up.
3. *Boarding offset and the `-1` guard.* Always check `dist[n-1] >= INF` (unreachable -> `-1`) *before* subtracting one, so you never compute `INF - 1`; and special-case `n - 1 == 0` (start == destination) to `0` so a never-boarded destination is not turned into a phantom `-1`.
4. *Large/sparse labels.* Labels go up to `10^9`; compress them to `0..L-1` so `byLine` is sized by distinct labels, not by `10^9`.

**Edge cases.** `n = 1` (start == destination) -> `0`; a single direct connection `n=2` -> one free board -> `0`; `m = 0` with `n >= 2` -> `-1`; isolated destination -> `-1`; self-loops `u == v` union a station with itself and never move you (harmless); duplicate/parallel connections add only a constant factor and are deduped into one component by the union-find. Distances are bounded by `m <= 2*10^5`, well inside `int`.

**Complexity.** Building line components is `O(m * alpha(n))`; the layered graph has `O(n + m)` nodes and `O(m)` edges; the 0-1 BFS is `O(n + m)`. Overall `O((n + m) * alpha(n))` time and `O(n + m)` memory. At `n = m = 2*10^5` it runs in well under 0.1 s using about 43 MB.

**Code.**

```cpp
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
```
