// TIER: strong
// Weighted matroid intersection (graphic matroid = forest, partition matroid = per-batch
// cap) via successive shortest augmenting paths in the exchange graph (Lawler's algorithm).
// For k = 0,1,2,... the algorithm produces the MAX-WEIGHT common independent set of size
// exactly k; we track the best total weight seen across all k and output that set.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int V, E, K;
vector<ll> cap_;
vector<int> eu, ev, eb;
vector<ll> ew;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cin >> V >> E >> K;
    cap_.assign(K + 1, 0);
    for (int b = 1; b <= K; b++) cin >> cap_[b];
    eu.assign(E, 0); ev.assign(E, 0); ew.assign(E, 0); eb.assign(E, 0);
    for (int i = 0; i < E; i++) cin >> eu[i] >> ev[i] >> ew[i] >> eb[i];

    vector<char> inS(E, 0);
    vector<ll> batchCnt(K + 1, 0);

    // adjacency of S restricted to each node, for forest BFS (parent/depth reconstruction)
    vector<vector<pair<int,int>>> adj(V + 1); // node -> (neighbor, elemId)

    ll curWeight = 0;
    ll bestWeight = 0;
    vector<char> bestS(E, 0);

    vector<int> parent(V + 1), parentEdge(V + 1), depth_(V + 1), root(V + 1);

    int maxPhases = V + 5;
    for (int phase = 0; phase < maxPhases; phase++) {
        // ---- rebuild forest structure (BFS per component) ----
        adj.assign(V + 1, {});
        for (int i = 0; i < E; i++) if (inS[i]) {
            adj[eu[i]].push_back({ev[i], i});
            adj[ev[i]].push_back({eu[i], i});
        }
        fill(parent.begin(), parent.end(), 0);
        fill(parentEdge.begin(), parentEdge.end(), -1);
        fill(depth_.begin(), depth_.end(), -1);
        fill(root.begin(), root.end(), 0);
        for (int s = 1; s <= V; s++) {
            if (depth_[s] != -1) continue;
            depth_[s] = 0; parent[s] = 0; parentEdge[s] = -1; root[s] = s;
            queue<int> q; q.push(s);
            while (!q.empty()) {
                int u = q.front(); q.pop();
                for (auto &pr : adj[u]) {
                    int v = pr.first, id = pr.second;
                    if (depth_[v] == -1) {
                        depth_[v] = depth_[u] + 1;
                        parent[v] = u;
                        parentEdge[v] = id;
                        root[v] = s;
                        q.push(v);
                    }
                }
            }
        }

        // ---- node costs for this phase ----
        vector<ll> nodecost(E);
        for (int i = 0; i < E; i++) nodecost[i] = inS[i] ? ew[i] : -ew[i];

        // ---- X1 (addable in M1 directly) / X2 (addable in M2 directly) ----
        vector<char> inX1(E, 0), inX2(E, 0);
        for (int i = 0; i < E; i++) if (!inS[i]) {
            if (root[eu[i]] != root[ev[i]]) inX1[i] = 1;
            if (batchCnt[eb[i]] < cap_[eb[i]]) inX2[i] = 1;
        }

        // ---- precompute M1-exchange adjacency: for y in S, adjY1[y] = list of x not in S
        //      whose fundamental cycle (w.r.t. current forest) passes through y ----
        vector<vector<int>> adjY1(E);
        for (int i = 0; i < E; i++) {
            if (inS[i]) continue;
            int u = eu[i], v = ev[i];
            if (root[u] != root[v]) continue; // x already free in M1 (no removal needed)
            int a = u, b = v;
            while (depth_[a] > depth_[b]) { adjY1[parentEdge[a]].push_back(i); a = parent[a]; }
            while (depth_[b] > depth_[a]) { adjY1[parentEdge[b]].push_back(i); b = parent[b]; }
            while (a != b) {
                adjY1[parentEdge[a]].push_back(i); a = parent[a];
                adjY1[parentEdge[b]].push_back(i); b = parent[b];
            }
        }
        // precompute, for each batch, list of S-elements in that batch (for M2 arcs)
        vector<vector<int>> sByBatch(K + 1);
        for (int i = 0; i < E; i++) if (inS[i]) sByBatch[eb[i]].push_back(i);

        // ---- SPFA over exchange graph, multi-source from X1 ----
        const ll INF = (ll)4e18;
        vector<ll> dist(E, INF);
        vector<int> prevElem(E, -1);
        vector<char> inQueue(E, 0);
        deque<int> dq;
        for (int i = 0; i < E; i++) if (!inS[i] && inX1[i]) {
            dist[i] = nodecost[i];
            dq.push_back(i);
            inQueue[i] = 1;
        }

        while (!dq.empty()) {
            int cur = dq.front(); dq.pop_front();
            inQueue[cur] = 0;
            ll dcur = dist[cur];
            if (!inS[cur]) {
                // x = cur, not in S. M2-arc: x -> y for y in S with same batch.
                for (int y : sByBatch[eb[cur]]) {
                    ll nd = dcur + nodecost[y];
                    if (nd < dist[y]) {
                        dist[y] = nd; prevElem[y] = cur;
                        if (!inQueue[y]) { dq.push_back(y); inQueue[y] = 1; }
                    }
                }
            } else {
                // y = cur, in S. M1-arc: y -> x for x not in S whose fundamental cycle
                // (w.r.t. current forest) passes through y.
                for (int x : adjY1[cur]) {
                    ll nd = dcur + nodecost[x];
                    if (nd < dist[x]) {
                        dist[x] = nd; prevElem[x] = cur;
                        if (!inQueue[x]) { dq.push_back(x); inQueue[x] = 1; }
                    }
                }
            }
        }

        // ---- find best sink in X2 ----
        int bestSink = -1; ll bestDist = INF;
        for (int i = 0; i < E; i++) if (!inS[i] && inX2[i] && dist[i] < bestDist) {
            bestDist = dist[i]; bestSink = i;
        }
        if (bestSink == -1) break; // no augmenting path -> S is now maximum cardinality

        // ---- reconstruct path and toggle membership ----
        vector<int> path;
        int cur = bestSink;
        while (cur != -1) { path.push_back(cur); cur = prevElem[cur]; }
        for (int e : path) {
            if (inS[e]) { inS[e] = 0; curWeight -= ew[e]; batchCnt[eb[e]]--; }
            else        { inS[e] = 1; curWeight += ew[e]; batchCnt[eb[e]]++; }
        }
        if (curWeight > bestWeight) { bestWeight = curWeight; bestS = inS; }
    }

    int m = 0;
    for (int i = 0; i < E; i++) if (bestS[i]) m++;
    cout << m << "\n";
    for (int i = 0; i < E; i++) if (bestS[i]) cout << (i + 1) << " ";
    cout << "\n";
    return 0;
}
