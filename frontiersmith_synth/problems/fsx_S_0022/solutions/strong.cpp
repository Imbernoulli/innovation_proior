// TIER: strong
// Randomized multi-restart steepest-ascent local search for weighted Max-SAT.
// Seeds: (1) the independent weighted-vote assignment, (2..) seeded random
// regimes. Each restart hill-climbs by flipping the gate whose flip most
// increases total passable weight, until no improving flip remains. Keeps the
// best regime found across all restarts. Fully deterministic (fixed seed).
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m;
vector<ll> W;
vector<vector<int>> LIT;           // clause -> signed literals
vector<vector<int>> occ;          // variable -> list of clause indices
vector<int> trueCnt;              // per clause: # currently-true literals

static inline bool litTrue(int l, const vector<int>& val) {
    int v = abs(l); bool want = (l > 0);
    return (val[v] == 1) == want;
}

ll evalFull(const vector<int>& val) {
    ll F = 0;
    for (int j = 0; j < m; j++) {
        int c = 0;
        for (int l : LIT[j]) if (litTrue(l, val)) c++;
        trueCnt[j] = c;
        if (c > 0) F += W[j];
    }
    return F;
}

// gain if we flip variable v (uses current trueCnt)
ll gainFlip(int v, const vector<int>& val) {
    ll g = 0;
    int nv = val[v] ^ 1;
    for (int j : occ[v]) {
        // literal on v in clause j
        // find its sign quickly by scanning (clauses are short, k<=5)
        for (int l : LIT[j]) {
            if (abs(l) != v) continue;
            bool wasTrue = ((val[v] == 1) == (l > 0));
            bool nowTrue = ((nv == 1) == (l > 0));
            if (wasTrue == nowTrue) break;
            int c = trueCnt[j];
            int c2 = c + (nowTrue ? 1 : -1);
            bool satOld = (c > 0), satNew = (c2 > 0);
            if (satOld != satNew) g += (satNew ? W[j] : -W[j]);
            break;
        }
    }
    return g;
}

void applyFlip(int v, vector<int>& val) {
    int nv = val[v] ^ 1;
    for (int j : occ[v]) {
        for (int l : LIT[j]) {
            if (abs(l) != v) continue;
            bool wasTrue = ((val[v] == 1) == (l > 0));
            bool nowTrue = ((nv == 1) == (l > 0));
            if (wasTrue == nowTrue) break;
            trueCnt[j] += (nowTrue ? 1 : -1);
            break;
        }
    }
    val[v] = nv;
}

ll hillClimb(vector<int>& val) {
    ll F = evalFull(val);
    int guard = 0, limit = 40 * (n + 1);
    while (guard++ < limit) {
        ll best = 0; int bestV = -1;
        for (int v = 1; v <= n; v++) {
            ll g = gainFlip(v, val);
            if (g > best) { best = g; bestV = v; }
        }
        if (bestV < 0) break;      // no strictly improving flip
        applyFlip(bestV, val);
        F += best;
    }
    return F;
}

int main() {
    if (scanf("%d %d", &n, &m) != 2) return 0;
    W.assign(m, 0); LIT.assign(m, {});
    occ.assign(n + 1, {}); trueCnt.assign(m, 0);
    vector<ll> wPos(n + 1, 0), wNeg(n + 1, 0);
    for (int j = 0; j < m; j++) {
        int w, k; scanf("%d %d", &w, &k);
        W[j] = w; LIT[j].reserve(k);
        for (int i = 0; i < k; i++) {
            int c; scanf("%d", &c);
            LIT[j].push_back(c);
            occ[abs(c)].push_back(j);
            if (c > 0) wPos[abs(c)] += w; else wNeg[abs(c)] += w;
        }
    }

    std::mt19937 rng(0xC0FFEEu);

    vector<int> best(n + 1, 0);
    ll bestF = -1;

    auto consider = [&](vector<int> val) {
        ll F = hillClimb(val);
        if (F > bestF) { bestF = F; best = val; }
    };

    // seed 1: independent weighted vote
    {
        vector<int> val(n + 1, 0);
        for (int g = 1; g <= n; g++) val[g] = (wPos[g] > wNeg[g]) ? 1 : 0;
        consider(val);
    }
    // seed 2: all-high-flow
    {
        vector<int> val(n + 1, 1); val[0] = 0;
        consider(val);
    }
    // seeds 3..: random restarts (count scales down with size to stay in budget)
    int restarts = max(4, min(30, 4000 / (n + 1)));
    for (int r = 0; r < restarts; r++) {
        vector<int> val(n + 1, 0);
        for (int g = 1; g <= n; g++) val[g] = (int)(rng() & 1u);
        consider(val);
    }

    for (int g = 1; g <= n; g++) printf(g == 1 ? "%d" : " %d", best[g]);
    printf("\n");
    return 0;
}
