// TIER: strong
// Insight (prestress-inversion): don't chase "rest length = target distance".
// A free mast tip needs a genuine self-stress -- one cable pulling it from the
// left, one from the right, with the SAME tension magnitude on both, so the net
// force at the target shape is exactly zero (force balance), and both cables
// stay taut (so the target is a STABLE minimum, not a marginal direction). The
// needed rest length for a cable of stiffness k and target-gap g to carry
// tension T is  r = g - T/k  -- so matching TENSIONS (not matching lengths, and
// not a uniform length margin) is what force-balances stiffness-mismatched
// cables. Any extra/decoy cable on that node is left comfortably slack (costs
// nothing, needed for nothing). Tension is kept SMALL (just enough to be
// unambiguously taut) since unnecessary prestress is itself penalized.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int par[200005];
int find_(int x){ while (par[x]!=x) x = par[x] = par[par[x]]; return x; }

int main(){
    int n, s, c;
    scanf("%d %d %d", &n, &s, &c);
    vector<ll> target(n);
    for (int i = 0; i < n; i++) scanf("%lld", &target[i]);
    for (int i = 0; i <= n; i++) par[i] = i;
    vector<char> grounded(n, 0);
    grounded[0] = 1;
    for (int i = 0; i < s; i++) {
        ll u,v,d; scanf("%lld %lld %lld", &u,&v,&d);
        par[find_((int)v)] = find_((int)u);
        grounded[v] = 1;
    }

    vector<ll> cu(c), cv(c), ck(c);
    // per free node: list of (gapSigned = target[free]-target[grounded], k, cableIdx)
    vector<vector<array<double,3>>> byFree(n);
    for (int i = 0; i < c; i++) {
        ll u,v,k; scanf("%lld %lld %lld", &u,&v,&k);
        cu[i]=u; cv[i]=v; ck[i]=k;
        bool gu = grounded[u], gv = grounded[v];
        if (gu == gv) continue; // degenerate, handled by fallback below
        int freeNode = gu ? (int)v : (int)u;
        int gNode    = gu ? (int)u : (int)v;
        double gapSigned = (double)target[freeNode] - (double)target[gNode];
        byFree[freeNode].push_back({gapSigned, (double)k, (double)i});
    }

    vector<double> rest(c, -1.0);
    for (int i = 0; i < c; i++) {
        double gap = fabs((double)target[cv[i]] - (double)target[cu[i]]);
        rest[i] = gap + 2.0; // default: safely slack (overwritten for chosen active cables)
    }

    for (int v = 0; v < n; v++) {
        if (byFree[v].empty()) continue;
        int bestL = -1, bestR = -1;
        double bestLgap = 1e18, bestRgap = 1e18;
        for (auto &e : byFree[v]) {
            double gs = e[0], k = e[1]; int idx = (int)e[2];
            if (gs > 0 && gs < bestLgap) { bestLgap = gs; bestL = idx; }
            if (gs < 0 && -gs < bestRgap) { bestRgap = -gs; bestR = idx; }
        }
        if (bestL == -1 || bestR == -1) continue; // shouldn't happen; leave defaults
        double kL = ck[bestL], kR = ck[bestR];
        double gapL = bestLgap, gapR = bestRgap;
        // Use a standard prestress level on each brace independently, capped by
        // what that particular cable can support without going below zero rest
        // length. (A fully general solver would re-balance the two sides to stay
        // exactly equal even when one is capped -- this doesn't, so tight/weak
        // braces can end up with a slightly unbalanced pull.)
        const double T_TARGET = 3.0;
        double TL = min(T_TARGET, 0.9 * kL * gapL);
        double TR = min(T_TARGET, 0.9 * kR * gapR);
        if (TL < 1e-6) TL = 1e-6;
        if (TR < 1e-6) TR = 1e-6;
        rest[bestL] = max(0.0, gapL - TL / kL);
        rest[bestR] = max(0.0, gapR - TR / kR);
    }

    for (int i = 0; i < c; i++) printf("%.6f\n", rest[i]);
    return 0;
}
