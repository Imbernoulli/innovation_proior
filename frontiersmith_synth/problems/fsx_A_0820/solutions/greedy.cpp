// TIER: greedy
// The "obvious" single-pass recipe: build one combined candidate list of
// (a) direct one-hop targets and (b) AND-merge nodes, costed by routing EACH
// relay branch via its CHEAPEST edge (i.e. minimizing distance/cost per
// branch, exactly the "route for shortest distance" reflex) -- WITHOUT ever
// checking whether the two branches' arrival ticks actually coincide. Rank
// everything by value/cost and greedily fill the budget. This wastes budget
// on merges whose cheap branches desynchronize (the trap).
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

    // cheapest direct (1 -> x) edge for every node x
    vector<ll> cheapCost(N + 1, -1);
    vector<int> cheapEdge(N + 1, -1);
    for (int i = 1; i <= M; i++) {
        if (e[i].u == 1) {
            int v = e[i].v;
            if (cheapCost[v] < 0 || e[i].c < cheapCost[v]) { cheapCost[v] = e[i].c; cheapEdge[v] = i; }
        }
    }
    // cheapest (r -> m) edge for every (r, m) pair with m an AND-merge
    map<pair<int,int>, pair<ll,int>> relayToMerge; // (r,m) -> (cost, edgeIdx), pick cheapest w too for reference
    map<int, set<int>> mergeRelays; // m -> set of relay ids (r != 1)
    for (int i = 1; i <= M; i++) {
        int m = e[i].v;
        if (reqSync[m] && e[i].u != 1) {
            int r = e[i].u;
            mergeRelays[m].insert(r);
            auto key = make_pair(r, m);
            auto it = relayToMerge.find(key);
            if (it == relayToMerge.end() || e[i].c < it->second.first) relayToMerge[key] = {e[i].c, i};
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
        if (cheapCost[r1] < 0 || cheapCost[r2] < 0) continue;
        auto e1 = relayToMerge[{r1, m}], e2 = relayToMerge[{r2, m}];
        ll naiveCost = cheapCost[r1] + e1.first + cheapCost[r2] + e2.first;
        vector<int> pkg = {cheapEdge[r1], e1.second, cheapEdge[r2], e2.second};
        cands.push_back({val[m], naiveCost, pkg, m});
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
