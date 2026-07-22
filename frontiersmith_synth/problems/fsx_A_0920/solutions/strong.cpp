// TIER: strong
// The insight: a packet that has NO detour must be given first claim on a
// contended slot; a packet that DOES have a detour should be steered off
// the contended edge even though the contended edge looks locally shorter,
// so its slots stay free for whoever truly needs them.
//
// 1. Compute each packet's durationwise-shortest path A (ignoring
//    contention) and tally, per edge, how many packets' path A use it.
//    An edge used by many packets (>= CONTEND_THRESH) is "contended" --
//    this is discovered from the input, not hard-coded to any topology.
// 2. Recompute each packet's path B in a graph where contended edges carry
//    a huge weight penalty. If B avoids every contended edge AND still
//    meets the packet's own deadline on ACTUAL durations, the packet has a
//    genuine detour ("flexible"); otherwise it structurally needs a
//    contended edge ("captive", possibly because it has no detour at all,
//    possibly because it is one of many packets funnelled through a single
//    non-detourable shared edge).
// 3. Schedule every captive packet first (value-desc, so if a contended
//    edge's window is still oversubscribed the higher-value captives win),
//    reserving their slots on path A. Only THEN schedule the flexible
//    packets on their path B (value-desc) -- by construction this never
//    touches a slot a captive needed.
#include "../routing_lib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static const int CONTEND_THRESH = 3;
static const ll PENALTY = 1000000000LL;

int main(){
    int N, M, K; ll T;
    cin >> N >> M >> K >> T;
    vector<Edge> edges(M);
    for (int i = 0; i < M; i++) cin >> edges[i].u >> edges[i].v >> edges[i].dur;
    vector<int> S(K), Tt(K);
    vector<ll> R(K), D(K), V(K);
    for (int k = 0; k < K; k++) cin >> S[k] >> Tt[k] >> R[k] >> D[k] >> V[k];

    vector<vector<int>> adj = buildAdj(N, edges);

    vector<vector<int>> pathA(K);
    vector<int> usage(M, 0);
    for (int k = 0; k < K; k++){
        pathA[k] = shortestPath(N, edges, adj, S[k], Tt[k]);
        for (int e : pathA[k]) usage[e]++;
    }

    vector<ll> weight(M);
    vector<char> contended(M, 0);
    for (int e = 0; e < M; e++){
        contended[e] = (usage[e] >= CONTEND_THRESH);
        weight[e] = edges[e].dur + (contended[e] ? PENALTY : 0);
    }

    vector<vector<int>> pathB(K);
    vector<char> flexible(K, 0);
    for (int k = 0; k < K; k++){
        if (pathA[k].empty()) continue;
        vector<int> pb = shortestPath(N, edges, adj, S[k], Tt[k], &weight);
        pathB[k] = pb;
        bool avoids = !pb.empty();
        for (int e : pb) if (contended[e]) avoids = false;
        if (avoids && R[k] + realDuration(pb, edges) <= D[k]) flexible[k] = 1;
    }

    vector<int> captiveOrder, flexOrder;
    for (int k = 0; k < K; k++){
        if (pathA[k].empty()) continue;
        if (flexible[k]) flexOrder.push_back(k); else captiveOrder.push_back(k);
    }
    auto byValueDesc = [&](int a, int b){
        if (V[a] != V[b]) return V[a] > V[b];
        return a < b;
    };
    sort(captiveOrder.begin(), captiveOrder.end(), byValueDesc);
    sort(flexOrder.begin(), flexOrder.end(), byValueDesc);

    unordered_set<ll> used;
    vector<int> resultLen(K, -1);
    vector<vector<int>> resultPath(K);
    vector<vector<ll>> resultDepart(K);

    auto commitIfOk = [&](int k, const vector<int>& path){
        vector<ll> depart;
        if (!path.empty() && tryPath(path, edges, used, R[k], D[k], true, &depart)){
            resultLen[k] = (int)path.size();
            resultPath[k] = path;
            resultDepart[k] = depart;
            return true;
        }
        return false;
    };

    for (int k : captiveOrder) commitIfOk(k, pathA[k]);
    for (int k : flexOrder){
        if (!commitIfOk(k, pathB[k])) commitIfOk(k, pathA[k]); // defensive fallback
    }

    for (int k = 0; k < K; k++){
        if (resultLen[k] < 0){ cout << -1 << '\n'; continue; }
        cout << resultLen[k];
        for (int j = 0; j < resultLen[k]; j++) cout << ' ' << resultDepart[k][j] << ' ' << resultPath[k][j];
        cout << '\n';
    }
    return 0;
}
