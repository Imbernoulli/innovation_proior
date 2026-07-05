// TIER: strong
// Randomized-restart sweep hill climbing on top of a weighted-majority start.
// Deterministic (fixed seed). Maintains per-clause true-literal counts for O(deg)
// incremental flip evaluation.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m;
vector<ll> wt;
vector<vector<int>> cl;          // clause -> signed literals
vector<vector<int>> occ;         // var -> clause ids it appears in
vector<int> trueCnt;             // per clause: number of true literals

static inline bool litTrue(int lit, const vector<int>& x) {
    int v = abs(lit);
    return (lit > 0) ? (x[v] == 1) : (x[v] == 0);
}

ll computeSat(const vector<int>& x) {
    ll F = 0;
    for (int c = 0; c < m; c++) {
        int cnt = 0;
        for (int lit : cl[c]) if (litTrue(lit, x)) cnt++;
        trueCnt[c] = cnt;
        if (cnt > 0) F += wt[c];
    }
    return F;
}

// gain of flipping var v given current x and trueCnt
ll flipGain(int v, const vector<int>& x) {
    ll delta = 0;
    for (int c : occ[v]) {
        // find v's literal in clause c and whether currently true
        // (a var appears once per clause by construction)
        bool cur = false;
        for (int lit : cl[c]) if (abs(lit) == v) { cur = litTrue(lit, x); break; }
        if (cur) {                       // literal true now -> becomes false
            if (trueCnt[c] == 1) delta -= wt[c];
        } else {                         // literal false now -> becomes true
            if (trueCnt[c] == 0) delta += wt[c];
        }
    }
    return delta;
}

void applyFlip(int v, vector<int>& x) {
    for (int c : occ[v]) {
        bool cur = false;
        for (int lit : cl[c]) if (abs(lit) == v) { cur = litTrue(lit, x); break; }
        if (cur) trueCnt[c]--; else trueCnt[c]++;
    }
    x[v] ^= 1;
}

int main() {
    if (scanf("%d %d", &n, &m) != 2) return 0;
    wt.assign(m, 0);
    cl.assign(m, {});
    occ.assign(n + 1, {});
    trueCnt.assign(m, 0);
    for (int c = 0; c < m; c++) {
        ll w; int k;
        scanf("%lld %d", &w, &k);
        wt[c] = w;
        cl[c].reserve(k);
        for (int j = 0; j < k; j++) {
            int lit; scanf("%d", &lit);
            cl[c].push_back(lit);
            occ[abs(lit)].push_back(c);
        }
    }

    // weighted-majority start
    vector<ll> posW(n + 1, 0), negW(n + 1, 0);
    for (int c = 0; c < m; c++)
        for (int lit : cl[c]) { if (lit > 0) posW[abs(lit)] += wt[c]; else negW[abs(lit)] += wt[c]; }

    vector<int> start(n + 1, 0);
    for (int v = 1; v <= n; v++) start[v] = (posW[v] > negW[v]) ? 1 : 0;

    mt19937 rng(12345);
    vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i + 1;

    auto hillClimb = [&](vector<int>& x) {
        ll F = computeSat(x);
        const int maxSweeps = 60;
        for (int sw = 0; sw < maxSweeps; sw++) {
            shuffle(order.begin(), order.end(), rng);
            bool improved = false;
            for (int v : order) {
                ll g = flipGain(v, x);
                if (g > 0) { applyFlip(v, x); F += g; improved = true; }
            }
            if (!improved) break;
        }
        return F;
    };

    vector<int> best = start;
    ll bestF = hillClimb(best);   // best is modified in place -> holds local optimum

    // randomized restarts: perturb from best and re-optimize
    int restarts = 6;
    for (int r = 0; r < restarts; r++) {
        vector<int> x = best;
        int flips = max(1, n / 10);
        for (int f = 0; f < flips; f++) x[rng() % n + 1] ^= 1;
        ll F = hillClimb(x);
        if (F > bestF) { bestF = F; best = x; }
    }

    for (int v = 1; v <= n; v++) printf("%s%d", v > 1 ? " " : "", best[v]);
    printf("\n");
    return 0;
}
