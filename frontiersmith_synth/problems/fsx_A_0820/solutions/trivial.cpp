// TIER: trivial
// One-hop-only knapsack: greedily fill the budget, by value-per-cost, using
// only DIRECT edges out of node 1 into positive-value nodes. This is exactly
// the checker's internal baseline B, so this solution always reproduces
// F=B -> ratio 0.1. It never looks at AND-merges at all.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

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
    vector<int> eu(M + 1), ev(M + 1);
    vector<ll> ew(M + 1), ec(M + 1);
    for (int i = 1; i <= M; i++) {
        scanf("%d %d %lld %lld", &eu[i], &ev[i], &ew[i], &ec[i]);
    }

    vector<ll> bestCost(N + 1, -1);
    vector<int> bestEdge(N + 1, -1);
    for (int i = 1; i <= M; i++) {
        if (eu[i] == 1 && val[ev[i]] > 0 && !reqSync[ev[i]]) {
            int v = ev[i];
            if (bestCost[v] < 0 || ec[i] < bestCost[v]) { bestCost[v] = ec[i]; bestEdge[v] = i; }
        }
    }
    struct Cand { int v; ll val, cost, edge; };
    vector<Cand> cands;
    for (int v = 2; v <= N; v++) if (bestCost[v] >= 0) cands.push_back({v, val[v], bestCost[v], bestEdge[v]});
    sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
        __int128 lhs = (__int128)a.val * b.cost;
        __int128 rhs = (__int128)b.val * a.cost;
        if (lhs != rhs) return lhs > rhs;
        if (a.cost != b.cost) return a.cost < b.cost;
        return a.v < b.v;
    });
    ll remaining = K;
    vector<int> chosen;
    for (auto& cd : cands) if (cd.cost <= remaining) { remaining -= cd.cost; chosen.push_back((int)cd.edge); }

    printf("%d\n", (int)chosen.size());
    for (size_t i = 0; i < chosen.size(); i++) printf("%d ", chosen[i]);
    printf("\n");
    return 0;
}
