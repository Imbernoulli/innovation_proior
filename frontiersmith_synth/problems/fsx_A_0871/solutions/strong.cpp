// TIER: strong
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Insight: side payments are a transfer, not new welfare, so start from the SAME
// welfare-driven matching a greedy weight-sort finds, then treat stability as a
// priced resource. Compute the "envy boundary" -- the set of pairs that would
// actually block this matching under zero payments -- and spend sweeteners only
// on that boundary: raise whichever side of a threatened matched pair clears the
// most blocking edges, within its cap and participation limits. Pairs whose caps
// are too thin to buy off their threats (e.g. contested "star" masters) are left
// to eat the blocking penalty on purpose, rather than wasting payment room that
// can't fully neutralize them. A second pass mops up any newly-exposed knock-on
// vulnerability with leftover cap.
int main() {
    int n, m, e; ll lambda;
    scanf("%d %d %d %lld", &n, &m, &e, &lambda);
    vector<int> ei(e), ej(e), ea(e), eb(e), es(e);
    for (int k = 0; k < e; k++)
        scanf("%d %d %d %d %d", &ei[k], &ej[k], &ea[k], &eb[k], &es[k]);

    // ---- base matching: same weight-sorted greedy as the "greedy" tier ----
    vector<int> order(e);
    for (int k = 0; k < e; k++) order[k] = k;
    sort(order.begin(), order.end(), [&](int x, int y) {
        int wx = ea[x] + eb[x], wy = ea[y] + eb[y];
        if (wx != wy) return wx > wy;
        return x < y;
    });
    vector<int> matchOfI(n + 1, -1), matchOfJ(m + 1, -1);
    for (int k : order) {
        if (matchOfI[ei[k]] < 0 && matchOfJ[ej[k]] < 0) {
            matchOfI[ei[k]] = k;
            matchOfJ[ej[k]] = k;
        }
    }

    vector<ll> p(e, 0);
    vector<ll> u(n + 1, 0), v(m + 1, 0);
    for (int i = 1; i <= n; i++) if (matchOfI[i] >= 0) u[i] = ea[matchOfI[i]];
    for (int j = 1; j <= m; j++) if (matchOfJ[j] >= 0) v[j] = eb[matchOfJ[j]];

    // Two passes: pass 0 uses the zero-payment "envy boundary"; pass 1 re-derives
    // threats from the post-payment utilities to catch knock-on exposure and use
    // any leftover cap.
    for (int pass = 0; pass < 2; pass++) {
        vector<ll> maxThreatA(n + 1, 0), maxThreatB(m + 1, 0);
        vector<int> cntThreatI(n + 1, 0), cntThreatJ(m + 1, 0);
        for (int k = 0; k < e; k++) {
            if (matchOfI[ei[k]] == k) continue; // this is the matched edge itself
            if (ea[k] > u[ei[k]] && eb[k] > v[ej[k]]) {
                maxThreatA[ei[k]] = max(maxThreatA[ei[k]], (ll)ea[k]);
                cntThreatI[ei[k]]++;
                maxThreatB[ej[k]] = max(maxThreatB[ej[k]], (ll)eb[k]);
                cntThreatJ[ej[k]]++;
            }
        }
        for (int i = 1; i <= n; i++) {
            int idx = matchOfI[i];
            if (idx < 0) continue;
            int j = ej[idx];
            ll a = ea[idx], b = eb[idx], s = es[idx];
            ll lo = max(-s, -a), hi = min(s, b);
            ll needUp = maxThreatA[i] - a;     // >0 means i is threatened
            ll needDown = maxThreatB[j] - b;   // >0 means j is threatened
            ll chosen = p[idx];
            if (needUp <= 0 && needDown <= 0) {
                // already safe on both sides; no change needed
            } else {
                ll reqDownP = -needDown; // p <= reqDownP needed to fix j (meaningful when needDown>0)
                ll reqLo = lo, reqHi = hi;
                if (needUp > 0) reqLo = max(reqLo, needUp);
                if (needDown > 0) reqHi = min(reqHi, reqDownP);
                if (reqLo <= reqHi) {
                    // a single payment can clear BOTH sides' threats at once -- free lunch
                    chosen = reqLo;
                } else {
                    // conflicting needs: only one side is affordable -- defend whichever
                    // side clears more blocking edges (fixing i clears cntThreatI[i] of
                    // them in one shot, and symmetrically for j).
                    bool canFixI = (needUp <= 0) || (needUp <= hi);
                    bool canFixJ = (needDown <= 0) || (reqDownP >= lo);
                    bool favorI = cntThreatI[i] >= cntThreatJ[j];
                    if (favorI && canFixI) chosen = max(lo, min(hi, needUp));
                    else if (!favorI && canFixJ) chosen = max(lo, min(hi, reqDownP));
                    else if (canFixI) chosen = max(lo, min(hi, needUp));
                    else if (canFixJ) chosen = max(lo, min(hi, reqDownP));
                    else chosen = favorI ? hi : lo; // best-effort partial defense
                }
            }
            p[idx] = chosen;
            u[i] = a + chosen;
            v[j] = b - chosen;
        }
    }

    vector<int> matchI, matchJ;
    vector<ll> payOut;
    for (int i = 1; i <= n; i++) {
        if (matchOfI[i] >= 0) {
            int idx = matchOfI[i];
            matchI.push_back(i);
            matchJ.push_back(ej[idx]);
            payOut.push_back(p[idx]);
        }
    }
    printf("%d\n", (int)matchI.size());
    for (size_t t = 0; t < matchI.size(); t++)
        printf("%d %d %lld\n", matchI[t], matchJ[t], payOut[t]);
    return 0;
}
