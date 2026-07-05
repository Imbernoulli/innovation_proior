// TIER: strong
// Seeded randomized multi-restart hill-climbing over sensor modes.
// Start from the weighted-majority assignment plus several shuffled random starts;
// repeatedly flip any sensor that increases total satisfied weight; keep the best.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m;
struct Cl { int w; vector<int> lits; };
vector<Cl> cls;
// per-variable incidence: (clause index, sign) for fast flip-gain evaluation
vector<vector<pair<int,int>>> inc; // inc[v] = {(ci, +1/-1)}

// satCount[c] = number of currently-met literals in clause c
static vector<int> satCount;

ll totalSat() {
    ll t = 0;
    for (int c = 0; c < m; c++) if (satCount[c] > 0) t += cls[c].w;
    return t;
}

void rebuild(const vector<int>& a) {
    for (int c = 0; c < m; c++) {
        int cnt = 0;
        for (int lit : cls[c].lits) {
            int v = abs(lit);
            if ((lit > 0 && a[v] == 1) || (lit < 0 && a[v] == 0)) cnt++;
        }
        satCount[c] = cnt;
    }
}

// gain in satisfied weight from flipping variable v given current assignment a
ll flipGain(const vector<int>& a, int v) {
    ll g = 0;
    for (auto& pr : inc[v]) {
        int c = pr.first, sgn = pr.second;
        // is this literal currently met?
        bool metNow = (sgn > 0 && a[v] == 1) || (sgn < 0 && a[v] == 0);
        if (metNow) {
            // after flip it becomes unmet: if it was the only met literal, clause drops
            if (satCount[c] == 1) g -= cls[c].w;
        } else {
            // after flip it becomes met: if clause had none met, clause gained
            if (satCount[c] == 0) g += cls[c].w;
        }
    }
    return g;
}

void applyFlip(vector<int>& a, int v) {
    for (auto& pr : inc[v]) {
        int c = pr.first, sgn = pr.second;
        bool metNow = (sgn > 0 && a[v] == 1) || (sgn < 0 && a[v] == 0);
        if (metNow) satCount[c]--; else satCount[c]++;
    }
    a[v] ^= 1;
}

void hillClimb(vector<int>& a, vector<int>& order) {
    rebuild(a);
    bool improved = true;
    int guard = 0;
    while (improved && guard < 200) {
        improved = false;
        guard++;
        for (int v : order) {
            ll g = flipGain(a, v);
            if (g > 0) { applyFlip(a, v); improved = true; }
        }
    }
}

int main() {
    if (scanf("%d %d", &n, &m) != 2) return 0;
    cls.resize(m);
    inc.assign(n + 1, {});
    for (int i = 0; i < m; i++) {
        int w, L; scanf("%d %d", &w, &L);
        cls[i].w = w; cls[i].lits.resize(L);
        for (int j = 0; j < L; j++) {
            int lit; scanf("%d", &lit);
            cls[i].lits[j] = lit;
            int v = abs(lit);
            inc[v].push_back({i, lit > 0 ? 1 : -1});
        }
    }
    satCount.assign(m, 0);

    // majority start
    vector<ll> wpos(n + 1, 0), wneg(n + 1, 0);
    for (int c = 0; c < m; c++)
        for (int lit : cls[c].lits) {
            int v = abs(lit);
            if (lit > 0) wpos[v] += cls[c].w; else wneg[v] += cls[c].w;
        }
    vector<int> best(n + 1, 0);
    for (int i = 1; i <= n; i++) best[i] = (wpos[i] > wneg[i]) ? 1 : 0;

    vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i + 1;

    // refine the majority start
    {
        vector<int> a = best;
        hillClimb(a, order);
        rebuild(a);   ll fa = totalSat();
        rebuild(best); ll fb = totalSat();
        if (fa > fb) best = a;
    }

    // deterministic seeded multi-restart
    std::mt19937 rng(987654321u ^ (unsigned)(n * 1000003u + m));
    int restarts = 24;
    if ((ll)n * m > 200000) restarts = 10; // keep it fast on the largest cases
    rebuild(best);
    ll bestVal = totalSat();

    for (int r = 0; r < restarts; r++) {
        vector<int> a(n + 1, 0);
        for (int i = 1; i <= n; i++) a[i] = (int)(rng() & 1u);
        // shuffle scan order
        for (int i = n - 1; i > 0; i--) std::swap(order[i], order[rng() % (i + 1)]);
        hillClimb(a, order);
        rebuild(a);
        ll v = totalSat();
        if (v > bestVal) { bestVal = v; best = a; }
    }

    for (int i = 1; i <= n; i++) printf("%d%c", best[i], i < n ? ' ' : '\n');
    return 0;
}
