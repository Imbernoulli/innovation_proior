// TIER: strong
// Multi-start greedy (several parcel orderings, seeded shuffles) followed by an
// eviction-swap + repacking local search; keep the best total delivered value.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int m, n;
vector<ll> c, w;
vector<vector<ll>> e; // e[i][j]

ll total(const vector<int>& a) {
    ll s = 0;
    for (int j = 1; j <= n; j++) if (a[j] >= 1) s += w[j];
    return s;
}

void buildLoad(const vector<int>& a, vector<ll>& rem) {
    rem = c;
    for (int j = 1; j <= n; j++) if (a[j] >= 1) rem[a[j]] -= e[a[j]][j];
}

// try to place currently-unassigned parcels, allowing single evictions when net gain > 0
void localSearch(vector<int>& a) {
    vector<ll> rem;
    bool changed = true;
    while (changed) {
        changed = false;
        buildLoad(a, rem);

        // 1) direct add: highest value unassigned parcels first
        vector<int> un;
        for (int j = 1; j <= n; j++) if (a[j] == 0) un.push_back(j);
        sort(un.begin(), un.end(), [&](int x, int y){ return w[x] > w[y]; });
        for (int j : un) {
            int best = 0; ll bc = LLONG_MAX;
            for (int i = 1; i <= m; i++)
                if (rem[i] >= e[i][j] && e[i][j] < bc) { bc = e[i][j]; best = i; }
            if (best) { rem[best] -= e[best][j]; a[j] = best; changed = true; }
        }

        // 2) eviction swap: place an unassigned parcel by evicting one cheaper,
        //    strictly-lower-value assigned parcel; net value strictly increases.
        un.clear();
        for (int j = 1; j <= n; j++) if (a[j] == 0) un.push_back(j);
        sort(un.begin(), un.end(), [&](int x, int y){ return w[x] > w[y]; });
        for (int j : un) {
            int chosenI = 0, chosenL = 0; ll bestVictimW = LLONG_MAX;
            for (int i = 1; i <= m; i++) {
                ll need = e[i][j] - rem[i]; // extra capacity required on drone i
                if (need <= 0) continue;    // would be a direct add, handled above
                for (int l = 1; l <= n; l++) {
                    if (a[l] != i) continue;
                    if (w[l] >= w[j]) continue;
                    if (e[i][l] >= need && w[l] < bestVictimW) {
                        bestVictimW = w[l]; chosenI = i; chosenL = l;
                    }
                }
            }
            if (chosenI) {
                a[chosenL] = 0;
                rem[chosenI] += e[chosenI][chosenL];
                rem[chosenI] -= e[chosenI][j];
                a[j] = chosenI;
                changed = true;
            }
        }
    }
}

vector<int> greedyFrom(const vector<int>& order) {
    vector<int> a(n + 1, 0);
    vector<ll> rem = c;
    for (int j : order) {
        int best = 0; ll bc = LLONG_MAX;
        for (int i = 1; i <= m; i++)
            if (rem[i] >= e[i][j] && e[i][j] < bc) { bc = e[i][j]; best = i; }
        if (best) { rem[best] -= e[best][j]; a[j] = best; }
    }
    localSearch(a);
    return a;
}

int main() {
    if (scanf("%d %d", &m, &n) != 2) return 0;
    c.assign(m + 1, 0); w.assign(n + 1, 0);
    e.assign(m + 1, vector<ll>(n + 1, 0));
    for (int i = 1; i <= m; i++) scanf("%lld", &c[i]);
    for (int j = 1; j <= n; j++) {
        scanf("%lld", &w[j]);
        for (int i = 1; i <= m; i++) scanf("%lld", &e[i][j]);
    }

    auto mincost = [&](int j){ ll mn = LLONG_MAX; for (int i=1;i<=m;i++) mn=min(mn,e[i][j]); return mn; };

    vector<int> base(n);
    for (int j = 0; j < n; j++) base[j] = j + 1;

    vector<int> best;
    ll bestVal = -1;

    auto consider = [&](vector<int> order){
        vector<int> a = greedyFrom(order);
        ll v = total(a);
        if (v > bestVal) { bestVal = v; best = a; }
    };

    // deterministic ordered starts
    {
        vector<int> o = base;
        sort(o.begin(), o.end(), [&](int x,int y){
            double rx=(double)w[x]/(double)mincost(x), ry=(double)w[y]/(double)mincost(y);
            if (rx!=ry) return rx>ry; return w[x]>w[y]; });
        consider(o);
    }
    { vector<int> o = base; sort(o.begin(), o.end(), [&](int x,int y){ return w[x]>w[y]; }); consider(o); }
    { vector<int> o = base; sort(o.begin(), o.end(), [&](int x,int y){ return mincost(x)<mincost(y); }); consider(o); }
    consider(base);

    // seeded random multi-starts
    mt19937 rng(987654321u);
    for (int it = 0; it < 60; it++) {
        vector<int> o = base;
        shuffle(o.begin(), o.end(), rng);
        consider(o);
    }

    for (int j = 1; j <= n; j++) printf("%d%c", best[j], j == n ? '\n' : ' ');
    return 0;
}
