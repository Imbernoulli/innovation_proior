// TIER: strong
// Insight: a household's opinion is only as "buyable" as the SHARE of a
// neighbor's own trust-weight that a single tie commands. Rank every
// household ONCE by a cost-normalized leverage score
//     leverage(v) = ( sum over ties (v,u) of weight(v,u)^2 / degW(u) ) / c(v)
// -- for each tie, weight^2/degW(u) is how much of NEIGHBOR u's decision
// this one tie controls (a tie into a low-degree neighbor commands a big
// share; a tie into a high-degree hub-neighbor commands almost none), and
// dividing by c(v) prefers cheap households. This rewards members of small,
// mostly-internal dense pockets (their ties point at other low-degree
// pocket members, so mutual leverage is high and the whole pocket becomes
// a self-sustaining NEW domain after seeding roughly half of it) FAR above
// a hub (whose many ties point at decently-connected background households
// for whom the hub tie is a tiny fraction of their own vote). Take the
// smallest leverage-ranked prefix (binary search) that clears the acreage
// target, using the SAME synchronous majority simulation as the checker.
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

    vector<double> lev(n + 1, 0.0);
    for (int v = 1; v <= n; v++){
        double raw = 0.0;
        for (auto &pr : adj[v]){
            int u = pr.first, w = pr.second;
            ll du = max((ll)1, degW[u]);
            raw += (double)w * (double)w / (double)du;
        }
        lev[v] = raw / (double)max((ll)1, c[v]);
    }

    vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i + 1;
    sort(order.begin(), order.end(), [&](int a, int b){
        if (lev[a] != lev[b]) return lev[a] > lev[b];
        return a < b;
    });

    int lo = 0, hi = n;
    auto feasibleWithPrefix = [&](int k) -> bool {
        vector<char> isZ(n + 1, 0);
        for (int i = 0; i < k; i++) isZ[order[i]] = 1;
        ll acc = simulateAcreage(isZ);
        return acc * 100 >= (ll)tau * totalAcreage;
    };
    while (lo < hi){
        int mid = (lo + hi) / 2;
        if (feasibleWithPrefix(mid)) hi = mid; else lo = mid + 1;
    }

    printf("%d\n", lo);
    for (int i = 0; i < lo; i++) printf("%d ", order[i]);
    printf("\n");
    return 0;
}
