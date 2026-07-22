// TIER: greedy
// Obvious "coaccess-affinity" recipe: mine pairwise co-occurrence COUNT
// (how often two components appear together in an order, ignoring batch
// size / row weight) and agglomeratively merge the strongest edges into
// cards, capped at a fixed round-number cluster size. Never touches the
// replication budget -- pure "put friends together" partitioning.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int par[221], csize[221];
int find_(int x){ while (par[x] != x){ par[x] = par[par[x]]; x = par[x]; } return x; }

int main(){
    int F, Q, R; ll P, K;
    if (scanf("%d %d %lld %lld %d", &F, &Q, &P, &K, &R) != 5) return 0;
    vector<ll> w(F + 1);
    for (int i = 1; i <= F; i++) scanf("%lld", &w[i]);

    vector<vector<int>> qfields(Q);
    for (int i = 0; i < Q; i++){
        ll rows; int m;
        scanf("%lld %d", &rows, &m);
        qfields[i].resize(m);
        for (int j = 0; j < m; j++) scanf("%d", &qfields[i][j]);
    }

    // raw co-occurrence COUNT (unweighted by traffic)
    map<pair<int,int>, int> cnt;
    for (auto &qf : qfields){
        int m = (int)qf.size();
        for (int a = 0; a < m; a++)
            for (int b = a + 1; b < m; b++){
                int u = qf[a], v = qf[b];
                if (u > v) swap(u, v);
                cnt[{u, v}]++;
            }
    }

    vector<pair<int, pair<int,int>>> edges;   // (count, (u,v))
    edges.reserve(cnt.size());
    for (auto &kv : cnt) edges.push_back({kv.second, kv.first});
    sort(edges.begin(), edges.end(), [](const pair<int,pair<int,int>>&a, const pair<int,pair<int,int>>&b){
        if (a.first != b.first) return a.first > b.first;      // strongest edge first
        return a.second < b.second;                             // deterministic tie-break
    });

    for (int i = 1; i <= F; i++){ par[i] = i; csize[i] = 1; }
    const int SIZE_CAP = 7;   // a round-number "components per card" cap
    for (auto &e : edges){
        int u = find_(e.second.first), v = find_(e.second.second);
        if (u == v) continue;
        if (csize[u] + csize[v] <= SIZE_CAP){
            par[u] = v; csize[v] += csize[u];
        }
    }

    for (int i = 1; i <= F; i++) printf("%d 0\n", find_(i));
    return 0;
}
