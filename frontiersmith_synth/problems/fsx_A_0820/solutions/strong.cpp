// TIER: strong
// The insight: for each AND-merge, don't cost its two branches by their
// cheapest (shortest-distance) edges -- reformulate as "find the MINIMUM-cost
// pair of branch edges whose arrival ticks are EXACTLY equal", searching all
// candidate edges into each relay (cheap + delay-line + decoys) and checking
// tick arithmetic directly. This is the only way a merge's value is ever
// really reachable; a merge with no synchronizable combination is dropped
// from consideration entirely (never spend on a doomed package). Then rank
// direct targets and TRUE-cost merge packages together by value/cost and
// greedily fill the budget.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

struct Edge { int u, v; ll w, c; };

int main() {
    int N, M; ll K;
    scanf("%d %d %lld", &N, &M, &K);
    vector<int> reqSync(N + 1, 0);
    vector<ll> val(N + 1, 0);
    for (int i = 2; i <= N; i++) {
        int rs; ll v;
        scanf("%d %lld", &rs, &v);
        reqSync[i] = rs; val[i] = v;
    }
    vector<Edge> e(M + 1);
    for (int i = 1; i <= M; i++) scanf("%d %d %lld %lld", &e[i].u, &e[i].v, &e[i].w, &e[i].c);

    // ALL direct (1 -> x) edges, and cheapest one per x
    vector<vector<int>> directEdges(N + 1);
    vector<ll> cheapCost(N + 1, -1);
    vector<int> cheapEdge(N + 1, -1);
    for (int i = 1; i <= M; i++) {
        if (e[i].u == 1) {
            int v = e[i].v;
            directEdges[v].push_back(i);
            if (cheapCost[v] < 0 || e[i].c < cheapCost[v]) { cheapCost[v] = e[i].c; cheapEdge[v] = i; }
        }
    }
    // ALL (r -> m) edges per (r,m), and the relay set feeding each merge
    map<pair<int,int>, vector<int>> relayToMergeEdges;
    map<int, set<int>> mergeRelays;
    for (int i = 1; i <= M; i++) {
        int m = e[i].v;
        if (reqSync[m] && e[i].u != 1) {
            int r = e[i].u;
            mergeRelays[m].insert(r);
            relayToMergeEdges[{r, m}].push_back(i);
        }
    }

    struct Cand { ll val, cost; vector<int> edges; int key; };
    vector<Cand> cands;

    for (int v = 2; v <= N; v++) {
        if (val[v] > 0 && !reqSync[v] && cheapCost[v] >= 0) {
            cands.push_back({val[v], cheapCost[v], {cheapEdge[v]}, v});
        }
    }

    for (auto& kv : mergeRelays) {
        int m = kv.first;
        if (val[m] <= 0) continue;
        vector<int> relays(kv.second.begin(), kv.second.end());
        sort(relays.begin(), relays.end());
        if (relays.size() < 2) continue;
        int r1 = relays[0], r2 = relays[1];
        if (directEdges[r1].empty() || directEdges[r2].empty()) continue;
        auto& e1s = relayToMergeEdges[{r1, m}];
        auto& e2s = relayToMergeEdges[{r2, m}];
        if (e1s.empty() || e2s.empty()) continue;

        ll bestCost = -1;
        vector<int> bestEdges;
        for (int se1 : directEdges[r1]) {
            for (int sa : e1s) {
                ll tick1 = e[se1].w + e[sa].w;
                for (int se2 : directEdges[r2]) {
                    for (int sb : e2s) {
                        ll tick2 = e[se2].w + e[sb].w;
                        if (tick1 != tick2) continue;
                        ll cost = e[se1].c + e[sa].c + e[se2].c + e[sb].c;
                        if (bestCost < 0 || cost < bestCost) {
                            bestCost = cost;
                            bestEdges = {se1, sa, se2, sb};
                        }
                    }
                }
            }
        }
        if (bestCost >= 0) cands.push_back({val[m], bestCost, bestEdges, m});
    }

    sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
        __int128 lhs = (__int128)a.val * b.cost;
        __int128 rhs = (__int128)b.val * a.cost;
        if (lhs != rhs) return lhs > rhs;
        if (a.cost != b.cost) return a.cost < b.cost;
        return a.key < b.key;
    });

    ll remaining = K;
    vector<int> chosen;
    for (auto& cd : cands) {
        if (cd.cost <= remaining) {
            remaining -= cd.cost;
            for (int idx : cd.edges) chosen.push_back(idx);
        }
    }
    sort(chosen.begin(), chosen.end());
    chosen.erase(unique(chosen.begin(), chosen.end()), chosen.end());

    printf("%d\n", (int)chosen.size());
    for (size_t i = 0; i < chosen.size(); i++) printf("%d ", chosen[i]);
    printf("\n");
    return 0;
}
