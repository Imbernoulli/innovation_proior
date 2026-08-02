// TIER: greedy
// Recursive weighted-centroid edge splitting: K-1 rounds, each round cuts the edge that
// most evenly bisects the CURRENT largest block by yield. Never clears a plot (ignores the
// vertex/hub-clear operation entirely), and never looks past the current round.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, K, LAMBDA;
vector<ll> w;
vector<vector<pair<int,int>>> adj; // node -> (neighbor, edgeIdx)
vector<int> eu, ev, ec;
vector<int> comp;

int main() {
    scanf("%d %d %d", &n, &K, &LAMBDA);
    w.assign(n + 1, 0);
    for (int v = 1; v <= n; v++) { int t; scanf("%d", &t); w[v] = t; }
    for (int v = 1; v <= n; v++) { int t; scanf("%d", &t); } // clear costs, unused by greedy
    eu.assign(n, 0); ev.assign(n, 0); ec.assign(n, 0);
    adj.assign(n + 1, {});
    for (int i = 1; i <= n - 1; i++) {
        int a, b, c; scanf("%d %d %d", &a, &b, &c);
        eu[i] = a; ev[i] = b; ec[i] = c;
        adj[a].push_back({b, i});
        adj[b].push_back({a, i});
    }

    comp.assign(n + 1, 0);
    int nextId = 1;
    vector<int> cutList;

    for (int round = 0; round < K - 1; round++) {
        unordered_map<int, ll> compW;
        for (int v = 1; v <= n; v++) compW[comp[v]] += w[v];
        vector<int> ids;
        for (auto &kv : compW) ids.push_back(kv.first);
        sort(ids.begin(), ids.end());
        int big = ids[0]; ll bestW = -1;
        for (int id : ids) if (compW[id] > bestW) { bestW = compW[id]; big = id; }

        int start = -1;
        for (int v = 1; v <= n; v++) if (comp[v] == big) { start = v; break; }

        vector<int> parent(n + 1, 0), parentEdge(n + 1, 0), order;
        vector<char> vis(n + 1, 0);
        vector<int> stk; stk.push_back(start); vis[start] = 1;
        while (!stk.empty()) {
            int u = stk.back(); stk.pop_back();
            order.push_back(u);
            for (auto &pr : adj[u]) {
                int to = pr.first, eidx = pr.second;
                if (comp[to] != big || vis[to]) continue;
                vis[to] = 1; parent[to] = u; parentEdge[to] = eidx;
                stk.push_back(to);
            }
        }
        vector<ll> subW(n + 1, 0);
        for (int v = 1; v <= n; v++) if (comp[v] == big) subW[v] = w[v];
        for (int i = (int)order.size() - 1; i >= 0; i--) {
            int v = order[i];
            if (v != start) subW[parent[v]] += subW[v];
        }

        int bestEdgeIdx = -1, bestChild = -1;
        ll bestDiff = -1, bestCost = -1;
        for (int v : order) {
            if (v == start) continue;
            ll diff = llabs(2 * subW[v] - bestW);
            int eidx = parentEdge[v];
            if (bestEdgeIdx == -1 || diff < bestDiff ||
                (diff == bestDiff && (ec[eidx] < bestCost ||
                 (ec[eidx] == bestCost && eidx < bestEdgeIdx)))) {
                bestDiff = diff; bestEdgeIdx = eidx; bestChild = v; bestCost = ec[eidx];
            }
        }

        // reassign the child side (not crossing bestEdgeIdx) to a fresh component id
        vector<char> vis2(n + 1, 0);
        vector<int> stk2; stk2.push_back(bestChild); vis2[bestChild] = 1;
        comp[bestChild] = nextId;
        while (!stk2.empty()) {
            int u = stk2.back(); stk2.pop_back();
            for (auto &pr : adj[u]) {
                int to = pr.first, eidx = pr.second;
                if (eidx == bestEdgeIdx) continue;
                if (comp[to] != big || vis2[to]) continue;
                vis2[to] = 1; comp[to] = nextId;
                stk2.push_back(to);
            }
        }
        cutList.push_back(bestEdgeIdx);
        nextId++;
    }

    printf("0\n");
    printf("%d\n", (int)cutList.size());
    for (int e : cutList) printf("%d ", e);
    printf("\n");
    return 0;
}
