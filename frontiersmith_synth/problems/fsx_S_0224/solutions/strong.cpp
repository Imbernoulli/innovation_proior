// TIER: strong
// Flip local search to convergence + multiple seeded random restarts; keep the
// best feasible cut. Deterministic (fixed RNG seed).
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static int n, m; static ll L, U;
static vector<ll> p;
static vector<vector<pair<int,ll>>> adj;

// run flip local search in place; returns cut value
ll localopt(vector<int>& c) {
    ll popA = 0;
    for (int i = 1; i <= n; i++) if (c[i] == 1) popA += p[i];
    bool improved = true;
    int pass = 0;
    while (improved && pass < 60) {
        improved = false; pass++;
        for (int i = 1; i <= n; i++) {
            ll delta = 0;
            for (auto& e : adj[i]) delta += (c[i] == c[e.first] ? e.second : -e.second);
            if (delta <= 0) continue;
            if (c[i] == 0) { if (popA + p[i] > U) continue; c[i] = 1; popA += p[i]; }
            else           { if (popA - p[i] < L) continue; c[i] = 0; popA -= p[i]; }
            improved = true;
        }
    }
    ll cut = 0;
    for (int i = 1; i <= n; i++)
        for (auto& e : adj[i]) if (e.first > i && c[i] != c[e.first]) cut += e.second;
    // adj stores each edge twice; the (e.first>i) guard counts each once
    return cut;
}

int main() {
    scanf("%d %d %lld %lld", &n, &m, &L, &U);
    p.assign(n + 1, 0);
    for (int i = 1; i <= n; i++) scanf("%lld", &p[i]);
    adj.assign(n + 1, {});
    for (int i = 0; i < m; i++) {
        int a, b; ll w; scanf("%d %d %lld", &a, &b, &w);
        adj[a].push_back({b, w});
        adj[b].push_back({a, w});
    }

    mt19937_64 rng(987654321ULL);

    // start 1: balance-only baseline
    vector<int> best(n + 1, 0);
    {
        vector<int> c(n + 1, 0);
        ll sA = 0, sB = 0;
        for (int i = 1; i <= n; i++) {
            if (sA < sB) { c[i] = 1; sA += p[i]; } else { c[i] = 0; sB += p[i]; }
        }
        localopt(c);
        best = c;
    }
    ll bestCut = 0;
    {
        vector<int> tmp = best; bestCut = localopt(tmp); best = tmp;
    }

    int restarts = 20;
    for (int r = 0; r < restarts; r++) {
        vector<int> c(n + 1, 0);
        ll popA = 0;
        for (int i = 1; i <= n; i++) { c[i] = (int)(rng() & 1); if (c[i]) popA += p[i]; }
        // repair into band by flipping random nodes toward the needed side
        int guard = 0;
        while (popA < L && guard++ < 8 * n) {
            int i = (int)(rng() % n) + 1;
            if (c[i] == 0) { c[i] = 1; popA += p[i]; }
        }
        guard = 0;
        while (popA > U && guard++ < 8 * n) {
            int i = (int)(rng() % n) + 1;
            if (c[i] == 1) { c[i] = 0; popA -= p[i]; }
        }
        if (popA < L || popA > U) continue; // couldn't repair; skip
        ll cut = localopt(c);
        if (cut > bestCut) { bestCut = cut; best = c; }
    }

    for (int i = 1; i <= n; i++) printf("%d%c", best[i], i == n ? '\n' : ' ');
    return 0;
}
