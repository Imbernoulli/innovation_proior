// TIER: strong
// The insight: the outcome only depends on the RATIO WQ/WP (scaling both
// weights by the same factor scales every guild's efficient score by the
// same factor, leaving the ranking, the winner, and the price identical).
// So fix WP=1 and grid-search WQ against every guild's PRINTED cost table
// (not V's shape) jointly with the kink (q0, slopeHi) and a few price caps,
// evaluating the real best-response + second-price simulation the checker
// uses. This can discover that rewarding quality past the naive kink (or
// moving the kink) compresses the winner/runner-up efficient-score gap on
// exactly the guild whose cost curve is cheap in that region -- something
// a rule that only mirrors V can never see.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int Q, n;
ll Pmax;
vector<ll> V;
vector<vector<ll>> C;

ll fq(int q, int q0, int slopeHi) {
    if (q <= q0) return q;
    return (ll)q0 + (ll)slopeHi * (q - q0);
}

double simulate(ll WQ, ll WP, int q0, int slopeHi, ll Pcap) {
    ll bestE = LLONG_MIN, secondE = LLONG_MIN; int win = -1, winQ = 0;
    vector<ll> E(n); vector<int> qBest(n);
    for (int j = 0; j < n; j++) {
        ll best = LLONG_MIN; int bq = 0;
        for (int q = 0; q <= Q; q++) {
            if (C[j][q] > Pcap) continue;
            ll sc = WQ * fq(q, q0, slopeHi) - WP * C[j][q];
            if (sc > best) { best = sc; bq = q; }
        }
        E[j] = best; qBest[j] = bq;
    }
    int w = 0;
    for (int j = 1; j < n; j++) if (E[j] > E[w]) w = j;
    ll E2 = LLONG_MIN;
    for (int j = 0; j < n; j++) if (j != w && E[j] > E2) E2 = E[j];
    int qw = qBest[w];
    double rawPrice = (double)(WQ * fq(qw, q0, slopeHi) - E2) / (double)WP;
    double pw = min((double)Pcap, max(0.0, rawPrice));
    return (double)V[qw] - pw;
}

int main() {
    if (scanf("%d %d %lld", &Q, &n, &Pmax) != 3) return 0;
    V.assign(Q + 1, 0);
    for (int q = 0; q <= Q; q++) scanf("%lld", &V[q]);
    C.assign(n, vector<ll>(Q + 1, 0));
    for (int j = 0; j < n; j++)
        for (int q = 0; q <= Q; q++) scanf("%lld", &C[j][q]);

    ll bestWQ = 1, bestPcap = Pmax;
    int bestQ0 = Q, bestSlope = 1;
    double bestVal = simulate(1, 1, Q, 1, Pmax);

    // Candidate price caps: no cap, and each distinct cost value present in
    // any guild's table at q=Q or the kink-ish region (cheap breakpoint set).
    vector<ll> caps = {Pmax};
    for (int j = 0; j < n; j++)
        for (int q = 0; q <= Q; q += max(1, Q / 4))
            caps.push_back(C[j][q]);
    sort(caps.begin(), caps.end());
    caps.erase(unique(caps.begin(), caps.end()), caps.end());
    if ((int)caps.size() > 8) {
        // thin to <=8 representative caps (always keep Pmax)
        vector<ll> thin;
        int step = max(1, (int)caps.size() / 7);
        for (size_t i = 0; i < caps.size(); i += step) thin.push_back(caps[i]);
        thin.push_back(Pmax);
        sort(thin.begin(), thin.end());
        thin.erase(unique(thin.begin(), thin.end()), thin.end());
        caps = thin;
    }

    for (ll WQ = 1; WQ <= 300; WQ++) {
        for (int q0 = 0; q0 <= Q; q0++) {
            for (int slopeHi = 0; slopeHi <= 8; slopeHi++) {
                double v = simulate(WQ, 1, q0, slopeHi, Pmax);
                if (v > bestVal) {
                    bestVal = v; bestWQ = WQ; bestQ0 = q0; bestSlope = slopeHi; bestPcap = Pmax;
                }
            }
        }
    }
    // refine Pcap around the best (WQ, q0, slopeHi) found so far
    for (ll cap : caps) {
        double v = simulate(bestWQ, 1, bestQ0, bestSlope, cap);
        if (v > bestVal) { bestVal = v; bestPcap = cap; }
    }

    printf("%lld 1 %d %d %lld\n", bestWQ, bestQ0, bestSlope, bestPcap);
    return 0;
}
