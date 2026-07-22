// TIER: strong
// Insight: re-derive the backbone generically via bridge edges (the trunk +
// spokes are exactly the bridges of the graph; everything else is local
// region work reachable only through one spoke). Clear the shared trunk
// once (whoever goes first benefits everyone else, who cross it fast), then
// assign whole regions to crews with longest-processing-time bin packing
// (biggest region first, always to the currently least-loaded crew) so the
// fleet's finish times stay balanced -- this is what round-robin-by-index
// ignores.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N, M, K;
vector<ll> LEN;
vector<vector<pair<int, int>>> adj;
ll SLOW, FAST;

vector<tuple<int, char, ll>> dfsDouble(int start, vector<char> &visitedEdge, ll &cum) {
    vector<tuple<int, char, ll>> out;
    vector<int> idx(N + 1, 0);
    vector<pair<int, int>> stk;
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
    ll weight;
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
        ll weight = (SLOW + FAST) * (LEN[eid] + localSum);
        regions.push_back({v, eid, LEN[eid], localSum, weight});
    }

    // Insight: longest-processing-time bin packing by estimated workload.
    sort(regions.begin(), regions.end(), [](const Region &a, const Region &b) {
        return a.weight > b.weight;
    });
    vector<vector<Region>> perPlow(K + 1);
    vector<ll> load(K + 1, 0);
    for (auto &reg : regions) {
        int best = 1;
        for (int p = 2; p <= K; p++) if (load[p] < load[best]) best = p;
        perPlow[best].push_back(reg);
        load[best] += reg.weight;
    }

    // Second-order refinement: within a crew's own region list, the LAST
    // region visited never needs its spoke re-crossed on the way out -- so
    // save the region with the longest spoke for last.
    for (int p = 1; p <= K; p++) {
        if (perPlow[p].empty()) continue;
        size_t bestIdx = 0;
        for (size_t r = 1; r < perPlow[p].size(); r++)
            if (perPlow[p][r].spokeLen > perPlow[p][bestIdx].spokeLen) bestIdx = r;
        swap(perPlow[p][bestIdx], perPlow[p].back());
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
