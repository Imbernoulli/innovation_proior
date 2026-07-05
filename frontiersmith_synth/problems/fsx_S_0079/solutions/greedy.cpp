// TIER: greedy
// Degree-aware Prim growth from station 1: repeatedly attach the cheapest candidate
// cable that links the current fabric to a not-yet-connected station while respecting
// both stations' port limits. A one-pass local heuristic; if it stalls, fall back to
// the always-feasible daisy chain.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m;
vector<int> eu, ev; vector<ll> ew; vector<int> cap;

int main() {
    if (scanf("%d %d", &n, &m) != 2) return 0;
    cap.assign(n + 1, 0);
    for (int v = 1; v <= n; v++) scanf("%d", &cap[v]);
    eu.assign(m + 1, 0); ev.assign(m + 1, 0); ew.assign(m + 1, 0);
    vector<ll> chainW(n + 1, LLONG_MAX); vector<int> chainIdx(n + 1, -1);
    for (int i = 1; i <= m; i++) {
        int u, v; ll w; scanf("%d %d %lld", &u, &v, &w);
        eu[i] = u; ev[i] = v; ew[i] = w;
        int lo = min(u, v), hi = max(u, v);
        if (hi == lo + 1 && w < chainW[lo]) { chainW[lo] = w; chainIdx[lo] = i; }
    }

    vector<char> inTree(n + 1, 0);
    vector<int> deg(n + 1, 0), chosen;
    inTree[1] = 1;
    int cnt = 1;
    bool stuck = false;
    while (cnt < n) {
        ll bestW = LLONG_MAX; int bestE = -1;
        for (int e = 1; e <= m; e++) {
            int a = eu[e], b = ev[e];
            // exactly one endpoint already in the fabric
            if (inTree[a] == inTree[b]) continue;
            if (deg[a] >= cap[a] || deg[b] >= cap[b]) continue;
            if (ew[e] < bestW) { bestW = ew[e]; bestE = e; }
        }
        if (bestE == -1) { stuck = true; break; }
        int a = eu[bestE], b = ev[bestE];
        deg[a]++; deg[b]++;
        inTree[a] = inTree[b] = 1;
        chosen.push_back(bestE); cnt++;
    }

    if (stuck || cnt != n) {
        chosen.clear();
        for (int i = 1; i < n; i++) chosen.push_back(chainIdx[i]);
    }

    printf("%d\n", (int)chosen.size());
    for (int e : chosen) printf("%d\n", e);
    return 0;
}
