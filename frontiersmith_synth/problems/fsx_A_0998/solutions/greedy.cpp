// TIER: greedy
// Obvious "influence = connectivity" recipe: rank every household ONCE by
// raw weighted trust-degree (sum of tie weights), most-connected first --
// the everyday instinct that the best-connected people are the best people
// to convert. Take the smallest degree-ranked prefix (binary search) that
// clears the acreage target, simulating the SAME synchronous majority
// dynamics as the checker to know when to stop. Never looks at cost, and
// never asks whether a tie is actually a big share of the OTHER endpoint's
// decision -- a cheap, high-degree hub looks irresistible by this metric
// even when it buys almost no cascade.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, R, tau;
vector<ll> p, c;
vector<int> s;
vector<vector<pair<int,int>>> adj;
vector<ll> degW;
ll totalAcreage;

ll simulateAcreage(const vector<char> &isZ){
    vector<int> cur(n + 1), nxt(n + 1);
    for (int i = 1; i <= n; i++) cur[i] = isZ[i] ? 1 : s[i];
    for (int t = 0; t < R; t++){
        for (int i = 1; i <= n; i++){
            if (isZ[i]){ nxt[i] = 1; continue; }
            ll w1 = 0;
            for (auto &pr : adj[i]) w1 += (ll)pr.second * cur[pr.first];
            ll w0 = degW[i] - w1;
            if (w1 > w0) nxt[i] = 1;
            else if (w1 < w0) nxt[i] = 0;
            else nxt[i] = cur[i];
        }
        swap(cur, nxt);
    }
    ll acc = 0;
    for (int i = 1; i <= n; i++) if (cur[i] == 1) acc += p[i];
    return acc;
}

int main(){
    if (scanf("%d %d %d %d", &n, &m, &R, &tau) != 4) return 0;
    p.assign(n + 1, 0); c.assign(n + 1, 0); s.assign(n + 1, 0);
    totalAcreage = 0;
    for (int i = 1; i <= n; i++){
        scanf("%lld %lld %d", &p[i], &c[i], &s[i]);
        totalAcreage += p[i];
    }
    adj.assign(n + 1, {}); degW.assign(n + 1, 0);
    for (int e = 0; e < m; e++){
        int u, v, w; scanf("%d %d %d", &u, &v, &w);
        adj[u].push_back({v, w}); adj[v].push_back({u, w});
        degW[u] += w; degW[v] += w;
    }

    vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i + 1;
    sort(order.begin(), order.end(), [&](int a, int b){
        if (degW[a] != degW[b]) return degW[a] > degW[b];
        return a < b;
    });

    // binary search the smallest prefix length that clears the target
    // (monotone: more zealots can only weakly help the final NEW acreage)
    int lo = 0, hi = n;
    auto feasibleWithPrefix = [&](int k) -> bool {
        vector<char> isZ(n + 1, 0);
        for (int i = 0; i < k; i++) isZ[order[i]] = 1;
        ll acc = simulateAcreage(isZ);
        return acc * 100 >= (ll)tau * totalAcreage;
    };
    if (!feasibleWithPrefix(n)) hi = n;   // shouldn't happen (all-zealot always feasible)
    while (lo < hi){
        int mid = (lo + hi) / 2;
        if (feasibleWithPrefix(mid)) hi = mid; else lo = mid + 1;
    }

    printf("%d\n", lo);
    for (int i = 0; i < lo; i++) printf("%d ", order[i]);
    printf("\n");
    return 0;
}
