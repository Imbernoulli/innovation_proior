// TIER: greedy
// Degree-capped Kruskal: add candidate pipes cheapest-first, skipping any that would
// close a cycle or push a junction past fan-out D. If it fails to span (stuck on the
// degree cap), fall back to the always-feasible backbone path.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, D;
vector<int> eu, ev; vector<ll> ew;

int par[1600006];
int find(int x){ while(par[x]!=x){ par[x]=par[par[x]]; x=par[x];} return x; }

void emitBackbone(){
    vector<int> bestIdx(n + 1, -1);
    vector<ll> bestW(n + 1, LLONG_MAX);
    for (int i = 1; i <= m; i++) if (abs(eu[i]-ev[i])==1){
        int lo = min(eu[i], ev[i]);
        if (ew[i] < bestW[lo]){ bestW[lo] = ew[i]; bestIdx[lo] = i; }
    }
    printf("%d\n", n - 1);
    for (int i = 1; i < n; i++) printf("%d\n", bestIdx[i]);
}

int main(){
    scanf("%d %d %d", &n, &m, &D);
    eu.resize(m+1); ev.resize(m+1); ew.resize(m+1);
    for (int i = 1; i <= m; i++) scanf("%d %d %lld", &eu[i], &ev[i], &ew[i]);

    vector<int> ord(m);
    iota(ord.begin(), ord.end(), 1);
    sort(ord.begin(), ord.end(), [&](int a, int b){ return ew[a] < ew[b]; });

    for (int i = 1; i <= n; i++) par[i] = i;
    vector<int> deg(n + 1, 0);
    vector<int> chosen;
    for (int idx : ord){
        int u = eu[idx], v = ev[idx];
        if (deg[u] >= D || deg[v] >= D) continue;
        if (find(u) == find(v)) continue;
        par[find(u)] = find(v);
        deg[u]++; deg[v]++;
        chosen.push_back(idx);
        if ((int)chosen.size() == n - 1) break;
    }

    if ((int)chosen.size() != n - 1){ emitBackbone(); return 0; }
    printf("%d\n", (int)chosen.size());
    for (int idx : chosen) printf("%d\n", idx);
    return 0;
}
