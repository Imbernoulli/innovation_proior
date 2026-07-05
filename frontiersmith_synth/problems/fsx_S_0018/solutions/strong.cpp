// TIER: strong
// Regret-based constructive heuristic + local search (relocate + swap) with a few
// deterministic randomized restarts. Beats value-density greedy.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int m, n;
vector<ll> C;
vector<vector<ll>> v, w;

// total thrill of an assignment
ll totalValue(const vector<int>& a) {
    ll F = 0;
    for (int i = 1; i <= n; i++) if (a[i] > 0) F += v[i][a[i]];
    return F;
}

// value-density greedy construction (same idea as the greedy tier) as a strong start
vector<int> constructDensity() {
    struct P { double d; int i, j; };
    vector<P> pairs;
    pairs.reserve((size_t)n * m);
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++)
            pairs.push_back({(double)v[i][j] / (double)w[i][j], i, j});
    sort(pairs.begin(), pairs.end(), [](const P& x, const P& y) {
        if (x.d != y.d) return x.d > y.d;
        if (x.i != y.i) return x.i < y.i;
        return x.j < y.j;
    });
    vector<int> a(n + 1, 0);
    vector<ll> rem = C;
    for (const auto& p : pairs) {
        if (a[p.i] != 0) continue;
        if (w[p.i][p.j] <= rem[p.j]) { rem[p.j] -= w[p.i][p.j]; a[p.i] = p.j; }
    }
    return a;
}

// build load vector from assignment
vector<ll> loadOf(const vector<int>& a) {
    vector<ll> load(m + 1, 0);
    for (int i = 1; i <= n; i++) if (a[i] > 0) load[a[i]] += w[i][a[i]];
    return load;
}

// regret construction; if shuffleTies, perturb the pick among near-best regrets
vector<int> constructRegret(mt19937& rng, bool perturb) {
    vector<int> a(n + 1, 0);
    vector<ll> rem = C;
    vector<char> done(n + 1, 0);
    for (int iter = 0; iter < n; iter++) {
        int bestAct = -1, bestPlat = -1;
        ll bestRegret = -1, bestVal = -1;
        for (int i = 1; i <= n; i++) {
            if (done[i]) continue;
            ll b1 = -1, b2 = -1; int p1 = -1;
            for (int j = 1; j <= m; j++) {
                if (w[i][j] <= rem[j]) {
                    if (v[i][j] > b1) { b2 = b1; b1 = v[i][j]; p1 = j; }
                    else if (v[i][j] > b2) { b2 = v[i][j]; }
                }
            }
            if (p1 == -1) continue;               // cannot place anywhere now
            ll regret = b1 - (b2 < 0 ? 0 : b2);
            ll key = regret * 1000 + b1;          // tie-break by best value
            if (perturb) key += (ll)(rng() % 50); // small deterministic perturbation
            if (key > bestRegret) { bestRegret = key; bestAct = i; bestPlat = p1; bestVal = b1; }
        }
        if (bestAct == -1) break;                 // nothing placeable
        a[bestAct] = bestPlat;
        rem[bestPlat] -= w[bestAct][bestPlat];
        done[bestAct] = 1;
        (void)bestVal;
    }
    // mark remaining as done=1 so we don't loop forever (they stay unassigned)
    return a;
}

// local search: relocate + swap until no improvement
void localSearch(vector<int>& a) {
    vector<ll> load = loadOf(a);
    bool improved = true;
    int guard = 0;
    while (improved && guard++ < 100000) {
        improved = false;
        // relocate: move act i to platform j (or remove) if it raises total value
        for (int i = 1; i <= n; i++) {
            int cur = a[i];
            ll curContrib = (cur > 0) ? v[i][cur] : 0;
            int bestJ = cur; ll bestGain = 0;
            for (int j = 0; j <= m; j++) {
                if (j == cur) continue;
                ll newContrib, feas = 1;
                if (j == 0) { newContrib = 0; }
                else {
                    ll used = load[j] - (cur == j ? w[i][j] : 0);
                    if (used + w[i][j] > C[j]) feas = 0;
                    newContrib = v[i][j];
                }
                if (!feas) continue;
                ll gain = newContrib - curContrib;
                if (gain > bestGain) { bestGain = gain; bestJ = j; }
            }
            if (bestJ != cur) {
                if (cur > 0) load[cur] -= w[i][cur];
                a[i] = bestJ;
                if (bestJ > 0) load[bestJ] += w[i][bestJ];
                improved = true;
            }
        }
        // swap: exchange platforms of two acts if it raises total value
        for (int i = 1; i <= n; i++) {
            for (int k = i + 1; k <= n; k++) {
                int pi = a[i], pk = a[k];
                if (pi == pk) continue;
                ll oldV = (pi > 0 ? v[i][pi] : 0) + (pk > 0 ? v[k][pk] : 0);
                // try i->pk, k->pi
                ll usedPk = load[pk] - (pk > 0 ? w[k][pk] : 0);
                ll usedPi = load[pi] - (pi > 0 ? w[i][pi] : 0);
                ll addI = (pk > 0 ? w[i][pk] : 0);
                ll addK = (pi > 0 ? w[k][pi] : 0);
                bool feas = true;
                if (pk > 0 && usedPk + addI > C[pk]) feas = false;
                if (pi > 0 && usedPi + addK > C[pi]) feas = false;
                if (!feas) continue;
                ll newV = (pk > 0 ? v[i][pk] : 0) + (pi > 0 ? v[k][pi] : 0);
                if (newV > oldV) {
                    if (pi > 0) load[pi] -= w[i][pi];
                    if (pk > 0) load[pk] -= w[k][pk];
                    a[i] = pk; a[k] = pi;
                    if (pk > 0) load[pk] += w[i][pk];
                    if (pi > 0) load[pi] += w[k][pi];
                    improved = true;
                }
            }
        }
    }
}

int main() {
    if (scanf("%d %d", &m, &n) != 2) return 0;
    C.assign(m + 1, 0);
    for (int j = 1; j <= m; j++) scanf("%lld", &C[j]);
    v.assign(n + 1, vector<ll>(m + 1, 0));
    w.assign(n + 1, vector<ll>(m + 1, 0));
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++) scanf("%lld", &v[i][j]);
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++) scanf("%lld", &w[i][j]);

    mt19937 rng(987654321u);
    vector<int> best;
    ll bestF = -1;

    // start 0: density greedy + local search (guarantees strong >= greedy tier)
    {
        vector<int> a = constructDensity();
        localSearch(a);
        ll F = totalValue(a);
        if (F > bestF) { bestF = F; best = a; }
    }
    // starts 1..: regret construction (+ perturbation) + local search
    int restarts = 6;
    for (int r = 0; r < restarts; r++) {
        vector<int> a = constructRegret(rng, r > 0);
        localSearch(a);
        ll F = totalValue(a);
        if (F > bestF) { bestF = F; best = a; }
    }

    for (int i = 1; i <= n; i++) printf("%d%c", best[i], i == n ? '\n' : ' ');
    return 0;
}
