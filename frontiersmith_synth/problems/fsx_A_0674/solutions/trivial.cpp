// TIER: trivial
// Single crew, naive: clear the depot's one trunk street, then handle every
// other neighborhood (everything reachable through one more incident street
// at the trunk's far end) completely separately -- walking all the way back
// to the depot and back out again before starting the next one, rather than
// chaining them together. All other crews stay idle. This reproduces the
// checker's internal baseline B exactly.
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

    vector<char> visitedEdge(M + 1, 0);
    vector<tuple<int, char, ll>> moves;
    ll cum = 0;

    if (!adj[depot].empty()) {
        int trunkEid = adj[depot][0].second;
        int hub = adj[depot][0].first;
        ll trunkLen = LEN[trunkEid];

        visitedEdge[trunkEid] = 1;
        moves.push_back(make_tuple(trunkEid, 'P', cum));
        cum += trunkLen * SLOW;

        vector<pair<int, int>> nbhd; // (entry node, spoke edge id)
        for (auto &pr : adj[hub]) {
            if (pr.second == trunkEid) continue;
            nbhd.push_back(pr);
        }
        int R = (int)nbhd.size();
        for (int r = 0; r < R; r++) {
            int entry = nbhd[r].first, spokeEid = nbhd[r].second;
            visitedEdge[spokeEid] = 1;
            moves.push_back(make_tuple(spokeEid, 'P', cum));
            cum += LEN[spokeEid] * SLOW;
            auto local = dfsDouble(entry, visitedEdge, cum);
            for (auto &mv : local) moves.push_back(mv);
            bool isLast = (r == R - 1);
            if (!isLast) {
                moves.push_back(make_tuple(spokeEid, 'F', cum));
                cum += LEN[spokeEid] * FAST;
                moves.push_back(make_tuple(trunkEid, 'F', cum));
                cum += trunkLen * FAST;
                moves.push_back(make_tuple(trunkEid, 'F', cum));
                cum += trunkLen * FAST;
            }
        }
    }
    // Safety net (not expected to trigger on the family's own generator):
    // sweep any leftover unvisited edges from wherever the walk currently is.
    {
        int here = depot;
        // Figure out current position from the last move, if any.
        if (!moves.empty()) {
            // Re-simulate position quickly.
            int pos = depot;
            for (auto &mv : moves) {
                int e = get<0>(mv);
                int u = U[e], v = V[e];
                pos = (u == pos) ? v : ((v == pos) ? u : pos);
            }
            here = pos;
        }
        bool any = false;
        for (int e = 1; e <= M; e++) if (!visitedEdge[e]) { any = true; break; }
        if (any) {
            // Not reachable without crossing back through visited edges in
            // general, but for this family's connected graphs a plain DFS
            // from `here` over the residual graph still covers everything.
            auto extra = dfsDouble(here, visitedEdge, cum);
            for (auto &mv : extra) moves.push_back(mv);
        }
    }

    printf("%d\n", (int)moves.size());
    for (auto &mv : moves)
        printf("%d %c %lld\n", get<0>(mv), get<1>(mv), get<2>(mv));
    for (int p = 2; p <= K; p++) printf("0\n");
    return 0;
}
