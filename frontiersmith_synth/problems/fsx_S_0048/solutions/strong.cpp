// TIER: strong
// Two heuristics, keep whichever is cheaper (both feasible by construction):
//   (A) local-search shift: start from reactive (x = h), repeatedly move cooling from an
//       expensive later step to the cheapest reachable earlier step of the same zone
//       whenever thermal slack (floor room) and plant capacity allow and cost drops;
//   (B) price-threshold bang-bang: pre-cool toward the floor on cheap steps, coast on
//       expensive steps.
// Taking the min over two distinct constructions dominates either alone and diverges per
// test from a single-heuristic submission.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int Z, T, C, s;
vector<int> p, Cap, init_;
vector<vector<int>> h;

static ll costOf(const vector<vector<ll>>& x) {
    ll F = 0;
    for (int t = 0; t < T; t++) {
        ll X = 0;
        for (int j = 0; j < Z; j++) X += x[j][t];
        F += (ll)p[t] * X + (X > 0 ? (ll)s : 0);
    }
    return F;
}

// (B) price-threshold bang-bang schedule.
static vector<vector<ll>> bangBang() {
    vector<int> sp = p;
    sort(sp.begin(), sp.end());
    int med = sp[T / 2];
    vector<vector<ll>> x(Z, vector<ll>(T, 0));
    vector<ll> theta(init_.begin(), init_.end());
    for (int t = 0; t < T; t++) {
        bool cheap = (p[t] <= med);
        vector<ll> xmax(Z);
        ll usedMin = 0;
        for (int j = 0; j < Z; j++) {
            ll lo = theta[j] + h[j][t] - Cap[j];
            ll xmin = lo > 0 ? lo : 0;
            xmax[j] = theta[j] + h[j][t];
            x[j][t] = xmin;
            usedMin += xmin;
        }
        ll rem = (ll)C - usedMin;
        if (rem < 0) rem = 0;
        if (cheap) {
            for (int j = 0; j < Z && rem > 0; j++) {
                ll add = min(rem, xmax[j] - x[j][t]);
                if (add < 0) add = 0;
                x[j][t] += add;
                rem -= add;
            }
        }
        for (int j = 0; j < Z; j++) theta[j] = theta[j] + h[j][t] - x[j][t];
    }
    return x;
}

int main() {
    scanf("%d %d %d %d", &Z, &T, &C, &s);
    p.assign(T, 0);
    for (int t = 0; t < T; t++) scanf("%d", &p[t]);
    Cap.assign(Z, 0);
    init_.assign(Z, 0);
    for (int j = 0; j < Z; j++) scanf("%d", &Cap[j]);
    for (int j = 0; j < Z; j++) scanf("%d", &init_[j]);
    h.assign(Z, vector<int>(T));
    for (int j = 0; j < Z; j++)
        for (int t = 0; t < T; t++) scanf("%d", &h[j][t]);
    vector<int>& init = init_;

    // reactive start
    vector<vector<ll>> x(Z, vector<ll>(T, 0));
    for (int j = 0; j < Z; j++)
        for (int t = 0; t < T; t++) x[j][t] = h[j][t];

    // per-step total cooling U[t]
    vector<ll> U(T, 0);
    for (int t = 0; t < T; t++) {
        ll v = 0;
        for (int j = 0; j < Z; j++) v += x[j][t];
        U[t] = v;
    }
    // theta[j][t] = temperature AFTER step t (t in 0..T-1); reactive keeps it at init.
    vector<vector<ll>> th(Z, vector<ll>(T, 0));
    for (int j = 0; j < Z; j++) {
        ll cur = init[j];
        for (int t = 0; t < T; t++) { cur = cur + h[j][t] - x[j][t]; th[j][t] = cur; }
    }

    auto stepCost = [&](ll Uv, ll price) -> ll {
        return price * Uv + (Uv > 0 ? (ll)s : 0);
    };

    const int MAX_PASS = 30;
    for (int pass = 0; pass < MAX_PASS; pass++) {
        bool improved = false;
        for (int j = 0; j < Z; j++) {
            for (int t2 = 1; t2 < T; t2++) {
                if (x[j][t2] <= 0) continue;
                // scan earlier target steps t1 < t2, maintaining running floor room =
                // min over s in [t1+1..t2] of th[j][s-1]  (temperatures that would drop).
                // Moving delta from t2 to t1 lowers th[j][t] for t in [t1..t2-1] by delta.
                ll runMin = LLONG_MAX;      // min of th[j][t] for t in [t1..t2-1]
                ll bestDelta = 0;           // amount to move
                int bestT1 = -1;
                ll bestGain = 0;            // cost reduction (positive = good)
                for (int t1 = t2 - 1; t1 >= 0; t1--) {
                    // extend range: th[j][t1] is now inside [t1..t2-1]
                    if (th[j][t1] < runMin) runMin = th[j][t1];
                    ll floorRoom = runMin;                 // >=0 required after subtracting delta
                    if (floorRoom <= 0) continue;          // cannot pre-cool through a 0
                    ll capRoom = (ll)C - U[t1];
                    if (capRoom <= 0) continue;
                    ll delta = min(x[j][t2], min(capRoom, floorRoom));
                    if (delta <= 0) continue;
                    // cost change at t1 and t2
                    ll before = stepCost(U[t1], p[t1]) + stepCost(U[t2], p[t2]);
                    ll after  = stepCost(U[t1] + delta, p[t1]) + stepCost(U[t2] - delta, p[t2]);
                    ll gain = before - after;              // positive = improvement
                    if (gain > bestGain) { bestGain = gain; bestDelta = delta; bestT1 = t1; }
                }
                if (bestT1 >= 0 && bestGain > 0) {
                    ll d = bestDelta; int t1 = bestT1;
                    x[j][t1] += d;
                    x[j][t2] -= d;
                    U[t1] += d;
                    U[t2] -= d;
                    for (int t = t1; t < t2; t++) th[j][t] -= d;   // temps in [t1..t2-1] drop
                    improved = true;
                }
            }
        }
        if (!improved) break;
    }

    // pick the cheaper of the two feasible constructions
    vector<vector<ll>> xb = bangBang();
    const vector<vector<ll>>& best = (costOf(xb) < costOf(x)) ? xb : x;

    for (int j = 0; j < Z; j++)
        for (int t = 0; t < T; t++)
            printf("%lld%c", best[j][t], t == T - 1 ? '\n' : ' ');
    return 0;
}
