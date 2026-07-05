// TIER: strong
// Swap-based local search (Kernighan-Lin flavour) with randomized multi-restart.
// From a starting balanced labeling, repeatedly apply the balanced 0/1 swap with the
// largest positive cut gain (using move-gains D[v] = external-internal, corrected by
// -2*w(a,b) for a swapped pair), plus balance-window single moves that exploit the
// slack. Restarts: one seeded from the sequential-greedy labeling (so it never does
// worse than greedy) plus several random balanced seeds. Keeps the best cut found.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, slack;
vector<int> eu, ev;
vector<ll> ew;
unordered_map<ll, ll> wmap; // summed weight between a pair (key = a*N+b, a<b)
int NK;

ll pairW(int a, int b) {
    if (a > b) swap(a, b);
    auto it = wmap.find((ll)a * NK + b);
    return it == wmap.end() ? 0 : it->second;
}

ll cutOf(const vector<int>& lab) {
    ll F = 0;
    for (int i = 0; i < m; i++) if (lab[eu[i]] != lab[ev[i]]) F += ew[i];
    return F;
}

// local search in place; returns cut of the (possibly imbalanced within window) result
ll localSearch(vector<int>& lab) {
    int lo = n / 2 - slack, hi = n / 2 + slack;
    const int K = 25;
    for (int pass = 0; pass < 80; pass++) {
        // move-gains for MAX-CUT: gain of flipping v = internal(v) - external(v)
        // (same-side edges become cut: +w; opposite edges become uncut: -w)
        vector<ll> D(n + 1, 0);
        for (int i = 0; i < m; i++) {
            int u = eu[i], v = ev[i]; ll w = ew[i];
            if (lab[u] == lab[v]) { D[u] += w; D[v] += w; }
            else { D[u] -= w; D[v] -= w; }
        }
        vector<int> s0, s1;
        int cnt0 = 0;
        for (int v = 1; v <= n; v++) {
            if (lab[v] == 0) { s0.push_back(v); cnt0++; }
            else s1.push_back(v);
        }
        auto byD = [&](int a, int b){ return D[a] > D[b]; };
        sort(s0.begin(), s0.end(), byD);
        sort(s1.begin(), s1.end(), byD);

        ll bestDelta = 0; int bestA = -1, bestB = -1; bool bestIsMove = false; int moveV = -1;

        // best balanced swap among top-K of each side
        int ka = min((int)s0.size(), K), kb = min((int)s1.size(), K);
        for (int i = 0; i < ka; i++)
            for (int j = 0; j < kb; j++) {
                int a = s0[i], b = s1[j];
                ll delta = D[a] + D[b] + 2 * pairW(a, b);
                if (delta > bestDelta) { bestDelta = delta; bestA = a; bestB = b; bestIsMove = false; }
            }
        // best single move that stays within the balance window
        // moving a side-0 zone -> side1 needs cnt0-1 >= lo ; side-1 -> side0 needs cnt0+1 <= hi
        for (int idx = 0; idx < min((int)s0.size(), K); idx++) {
            int v = s0[idx];
            if (cnt0 - 1 >= lo && D[v] > bestDelta) { bestDelta = D[v]; moveV = v; bestIsMove = true; }
        }
        for (int idx = 0; idx < min((int)s1.size(), K); idx++) {
            int v = s1[idx];
            if (cnt0 + 1 <= hi && D[v] > bestDelta) { bestDelta = D[v]; moveV = v; bestIsMove = true; }
        }

        if (bestDelta <= 0) break;
        if (bestIsMove) lab[moveV] ^= 1;
        else { lab[bestA] = 1; lab[bestB] = 0; }
    }
    return cutOf(lab);
}

int main() {
    if (scanf("%d %d %d", &n, &m, &slack) != 3) return 0;
    eu.resize(m); ev.resize(m); ew.resize(m);
    NK = n + 1;
    vector<vector<pair<int,ll>>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v; ll w; scanf("%d %d %lld", &u, &v, &w);
        eu[i] = u; ev[i] = v; ew[i] = w;
        int a = u, b = v; if (a > b) swap(a, b);
        wmap[(ll)a * NK + b] += w;
        adj[u].push_back({v, w}); adj[v].push_back({u, w});
    }

    vector<int> best; ll bestF = -1;

    auto consider = [&](vector<int> lab) {
        ll F = localSearch(lab);
        if (F > bestF) { bestF = F; best = lab; }
    };

    // seed 1: sequential greedy labeling
    {
        int cap = n / 2 + slack;
        vector<int> lab(n + 1, -1);
        int c0 = 0, c1 = 0;
        for (int v = 1; v <= n; v++) {
            ll g0 = 0, g1 = 0;
            for (auto& e : adj[v]) { int u = e.first; if (lab[u] == -1) continue; if (lab[u] == 1) g0 += e.second; else g1 += e.second; }
            bool can0 = c0 < cap, can1 = c1 < cap; int put;
            if (can0 && !can1) put = 0; else if (can1 && !can0) put = 1;
            else if (g0 > g1) put = 0; else if (g1 > g0) put = 1; else put = (c0 <= c1) ? 0 : 1;
            lab[v] = put; if (put == 0) c0++; else c1++;
        }
        consider(lab);
    }

    // random balanced restarts
    mt19937 rng(0xC0FFEEu);
    int restarts = 10;
    for (int r = 0; r < restarts; r++) {
        vector<int> perm(n);
        for (int i = 0; i < n; i++) perm[i] = i + 1;
        shuffle(perm.begin(), perm.end(), rng);
        vector<int> lab(n + 1, 0);
        for (int i = 0; i < n; i++) lab[perm[i]] = (i < n / 2) ? 0 : 1;
        consider(lab);
    }

    if (best.empty()) { best.assign(n + 1, 0); for (int i = n / 2 + 1; i <= n; i++) best[i] = 1; }
    for (int i = 1; i <= n; i++) printf("%d%c", best[i], i == n ? '\n' : ' ');
    return 0;
}
