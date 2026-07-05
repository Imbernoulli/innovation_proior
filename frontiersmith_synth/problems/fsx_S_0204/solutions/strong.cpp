// TIER: strong
// Two constructions (prefix greedy + global cost-per-sack knapsack) each refined by a
// randomized local search over the TRUE objective (operating cost + K * number of runs),
// using single-slot mode changes and active<->pause swap moves. Because the greedy
// construction is one of the seeds and every accepted move strictly lowers F, the result
// is always <= greedy, and usually strictly better (it clusters runs to save startups and
// shifts work onto cheap high-gain spot slots). Deterministic (fixed RNG seed).
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int T; ll W; int q; ll K;
vector<int> avail, g, sc, dc;

// per-slot sack yield / operating cost for a given mode
static inline ll yieldOf(int t, int a) { return a == 1 ? g[t] : a == 2 ? q : 0; }
static inline ll costOf(int t, int a)  { return a == 1 ? sc[t] : a == 2 ? dc[t] : 0; }

ll computeF(const vector<int>& a, ll& sacksOut) {
    ll cost = 0, sacks = 0, segs = 0; bool prev = false;
    for (int t = 1; t <= T; t++) {
        bool act = a[t] != 0;
        sacks += yieldOf(t, a[t]);
        cost += costOf(t, a[t]);
        if (act && !prev) segs++;
        prev = act;
    }
    sacksOut = sacks;
    return cost + K * segs;
}

mt19937 rng(0xC0FFEEu);

// local search on a feasible start; maintains incremental state.
pair<vector<int>, ll> localSearch(vector<int> a) {
    // active[0..T+1] with sentinels
    vector<char> active(T + 2, 0);
    ll sacks = 0, cost = 0, segs = 0;
    for (int t = 1; t <= T; t++) {
        active[t] = a[t] != 0;
        sacks += yieldOf(t, a[t]);
        cost += costOf(t, a[t]);
    }
    for (int t = 1; t <= T; t++) if (active[t] && !active[t - 1]) segs++;

    auto term = [&](int i) -> ll {
        if (i < 1 || i > T) return 0;
        return (active[i] && !active[i - 1]) ? 1 : 0;
    };
    // set slot t to value v, updating sacks/cost/segs incrementally
    auto setSlot = [&](int t, int v) {
        sacks += yieldOf(t, v) - yieldOf(t, a[t]);
        cost  += costOf(t, v)  - costOf(t, a[t]);
        ll oldLocal = term(t) + term(t + 1);
        a[t] = v; active[t] = (v != 0);
        ll newLocal = term(t) + term(t + 1);
        segs += newLocal - oldLocal;
    };
    auto curF = [&]() { return cost + K * segs; };

    vector<int> best = a; ll bestF = curF();

    long long iters = 200000LL + 300LL * T;
    for (long long it = 0; it < iters; it++) {
        int mv = rng() % 3;
        if (mv == 0) {
            // single-slot: try best of the (up to 3) modes for a random slot
            int t = 1 + rng() % T;
            int oldv = a[t];
            int bestv = oldv; ll bestLocalF = curF();
            for (int v = 0; v <= 2; v++) {
                if (v == 1 && !avail[t]) continue;
                if (v == oldv) continue;
                setSlot(t, v);
                if (sacks >= W) {
                    ll f = curF();
                    if (f < bestLocalF) { bestLocalF = f; bestv = v; }
                }
                setSlot(t, oldv); // revert
            }
            if (bestv != oldv) setSlot(t, bestv);
        } else {
            // swap move: deactivate a random active slot i, activate a random slot j
            int i = 1 + rng() % T;
            int j = 1 + rng() % T;
            if (i == j || a[i] == 0) continue;
            int oi = a[i], oj = a[j];
            // pick j's new mode: prefer spot if available & cheaper per sack, else on-demand
            int vj;
            if (avail[j] && (double)sc[j] / g[j] <= (double)dc[j] / q) vj = 1; else vj = 2;
            if (oj == vj && a[i] == 0) continue;
            ll before = curF();
            setSlot(i, 0);
            setSlot(j, vj);
            bool ok = (sacks >= W) && (curF() < before);
            if (!ok) { setSlot(j, oj); setSlot(i, oi); }
        }
        ll f = curF();
        if (f < bestF && sacks >= W) { bestF = f; best = a; }
    }
    return {best, bestF};
}

int main() {
    if (scanf("%d %lld %d %lld", &T, &W, &q, &K) != 4) return 0;
    avail.assign(T + 1, 0); g.assign(T + 1, 0); sc.assign(T + 1, 0); dc.assign(T + 1, 0);
    for (int t = 1; t <= T; t++)
        scanf("%d %d %d %d", &avail[t], &g[t], &sc[t], &dc[t]);

    // --- construction 1: prefix greedy ---
    vector<int> greedy(T + 1, 0);
    {
        ll sacks = 0;
        for (int t = 1; t <= T && sacks < W; t++) {
            double odps = (double)dc[t] / q;
            if (avail[t] && (double)sc[t] / g[t] <= odps) { greedy[t] = 1; sacks += g[t]; }
            else { greedy[t] = 2; sacks += q; }
        }
    }

    // --- construction 2: global cost-per-sack knapsack (best mode per slot) ---
    vector<int> knap(T + 1, 0);
    {
        vector<pair<double,int>> ord; ord.reserve(T);
        for (int t = 1; t <= T; t++) {
            double best = (double)dc[t] / q; // on-demand always usable
            if (avail[t]) best = min(best, (double)sc[t] / g[t]);
            ord.push_back({best, t});
        }
        sort(ord.begin(), ord.end());
        ll sacks = 0;
        for (auto& pr : ord) {
            if (sacks >= W) break;
            int t = pr.second;
            int v;
            if (avail[t] && (double)sc[t] / g[t] <= (double)dc[t] / q) v = 1; else v = 2;
            knap[t] = v; sacks += yieldOf(t, v);
        }
    }

    auto r1 = localSearch(greedy);
    auto r2 = localSearch(knap);

    vector<int>& best = (r1.second <= r2.second) ? r1.first : r2.first;

    for (int t = 1; t <= T; t++) {
        printf("%d", best[t]);
        putchar(t == T ? '\n' : ' ');
    }
    return 0;
}
