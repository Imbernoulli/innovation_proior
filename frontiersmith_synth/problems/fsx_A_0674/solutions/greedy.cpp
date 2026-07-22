// TIER: greedy
// The obvious approach: cross the one shared trunk street, then hand the R
// regions to the K crews round-robin in nearest-spoke order. This ignores
// how much LOCAL work each region actually holds -- a region that happens to
// be huge lands on whichever crew's turn it is, and the fleet ends up badly
// imbalanced whenever region sizes are skewed.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N, M, K;
vector<ll> LEN;
vector<vector<pair<int, int>>> adj; // node -> (neighbor, edge_id)
ll SLOW, FAST;

// Iterative DFS "double cover" of everything reachable from `start` that is
// not yet visited (edges already marked visited, e.g. the spoke back to the
// hub, are skipped, so this stays inside one region). Emits (edge, mode,
// start_time) triples and advances cum in place.
vector<tuple<int, char, ll>> dfsDouble(int start, vector<char> &visitedEdge, ll &cum) {
    vector<tuple<int, char, ll>> out;
    vector<int> idx(N + 1, 0);
    vector<pair<int, int>> stk; // (node, incoming edge id, -1 for start)
    stk.push_back({start, -1});
    while (!stk.empty()) {
        int u = stk.back().first;
        if (idx[u] < (int)adj[u].size()) {
            auto pr = adj[u][idx[u]];
            int v = pr.first, eid = pr.second;
            idx[u]++;
            if (!visitedEdge[eid]) {
                visitedEdge[eid] = 1;
                out.push_back(make_tuple(eid, 'P', cum));
                cum += LEN[eid] * SLOW;
                stk.push_back({v, eid});
            }
        } else {
            int eidIn = stk.back().second;
            stk.pop_back();
            if (!stk.empty()) {
                out.push_back(make_tuple(eidIn, 'F', cum));
                cum += LEN[eidIn] * FAST;
            }
        }
    }
    return out;
}

struct Region {
    int entry;
    int spokeEid;
    ll spokeLen;
    ll localSum;
};

int main() {
    scanf("%d %d %d", &N, &M, &K);
    vector<int> U(M + 1), V(M + 1);
    LEN.assign(M + 1, 0);
    adj.assign(N + 1, {});
    for (int i = 1; i <= M; i++) {
        scanf("%d %d %lld", &U[i], &V[i], &LEN[i]);
        adj[U[i]].push_back({V[i], i});
        adj[V[i]].push_back({U[i], i});
    }
    scanf("%lld %lld", &SLOW, &FAST);
    int depot;
    scanf("%d", &depot);

    int trunkEid = adj[depot][0].second;
    int hub = adj[depot][0].first;
    ll trunkLen = LEN[trunkEid];

    vector<Region> regions;
    for (auto &pr : adj[hub]) {
        int v = pr.first, eid = pr.second;
        if (eid == trunkEid) continue;
        ll localSum = 0;
        vector<char> vis(N + 1, 0), visE(M + 1, 0);
        queue<int> q;
        vis[v] = 1;
        q.push(v);
        while (!q.empty()) {
            int x = q.front();
            q.pop();
            for (auto &pr2 : adj[x]) {
                int y = pr2.first, e2 = pr2.second;
                if (e2 == eid) continue;
                if (!visE[e2]) { visE[e2] = 1; localSum += LEN[e2]; }
                if (!vis[y]) { vis[y] = 1; q.push(y); }
            }
        }
        regions.push_back({v, eid, LEN[eid], localSum});
    }

    // Naive: order regions by nearest spoke first, hand out round-robin.
    sort(regions.begin(), regions.end(), [](const Region &a, const Region &b) {
        return a.spokeLen < b.spokeLen;
    });
    vector<vector<Region>> perPlow(K + 1);
    for (int j = 0; j < (int)regions.size(); j++) {
        int p = (j % K) + 1;
        perPlow[p].push_back(regions[j]);
    }

    // Whoever leads across the trunk pays only trunkLen*SLOW; every follower
    // pays trunkLen*(SLOW+FAST) (wait, then cross fast). Send the busiest
    // crew first so it isn't the one stuck footing the extra wait.
    int firstActive = -1;
    ll bestRaw = -1;
    for (int p = 1; p <= K; p++) {
        if (perPlow[p].empty()) continue;
        ll raw = 0;
        for (auto &reg : perPlow[p]) raw += reg.spokeLen + reg.localSum;
        if (raw > bestRaw) { bestRaw = raw; firstActive = p; }
    }
    ll trunkFinish = trunkLen * SLOW;

    vector<char> visitedEdge(M + 1, 0);
    vector<vector<tuple<int, char, ll>>> outMoves(K + 1);

    for (int p = 1; p <= K; p++) {
        if (perPlow[p].empty()) continue;
        ll cum;
        if (p == firstActive) {
            visitedEdge[trunkEid] = 1;
            outMoves[p].push_back(make_tuple(trunkEid, 'P', 0LL));
            cum = trunkLen * SLOW;
        } else {
            outMoves[p].push_back(make_tuple(trunkEid, 'F', trunkFinish));
            cum = trunkFinish + trunkLen * FAST;
        }
        for (size_t r = 0; r < perPlow[p].size(); r++) {
            Region &reg = perPlow[p][r];
            visitedEdge[reg.spokeEid] = 1;
            outMoves[p].push_back(make_tuple(reg.spokeEid, 'P', cum));
            cum += reg.spokeLen * SLOW;
            auto local = dfsDouble(reg.entry, visitedEdge, cum);
            for (auto &mv : local) outMoves[p].push_back(mv);
            if (r + 1 < perPlow[p].size()) {
                outMoves[p].push_back(make_tuple(reg.spokeEid, 'F', cum));
                cum += reg.spokeLen * FAST;
            }
        }
    }

    for (int p = 1; p <= K; p++) {
        printf("%d\n", (int)outMoves[p].size());
        for (auto &mv : outMoves[p])
            printf("%d %c %lld\n", get<0>(mv), get<1>(mv), get<2>(mv));
    }
    return 0;
}
