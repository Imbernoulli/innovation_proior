#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int Q, n;
ll Pmax;
vector<ll> V;
vector<vector<ll>> C; // C[j][q], j=0..n-1

// f(q) for a given (q0, slopeHi)
ll fq(int q, int q0, int slopeHi) {
    if (q <= q0) return q;
    return (ll)q0 + (ll)slopeHi * (q - q0);
}

// Run the announced-rule best-response + second-price auction.
// Returns the castle's raw true surplus V[q*] - p* (can be negative; caller clamps).
double simulate(ll WQ, ll WP, int q0, int slopeHi, ll Pcap) {
    vector<ll> E(n);
    vector<int> qBest(n);
    for (int j = 0; j < n; j++) {
        ll best = LLONG_MIN; int bq = 0;
        for (int q = 0; q <= Q; q++) {
            if (C[j][q] > Pcap) continue;
            ll sc = WQ * fq(q, q0, slopeHi) - WP * C[j][q];
            if (sc > best) { best = sc; bq = q; }
        }
        // q=0 always has C[j][0]=0<=Pcap (Pcap>=0), so best is always defined.
        E[j] = best; qBest[j] = bq;
    }
    int win = 0;
    for (int j = 1; j < n; j++) if (E[j] > E[win]) win = j;
    ll E2 = LLONG_MIN;
    for (int j = 0; j < n; j++) if (j != win && E[j] > E2) E2 = E[j];
    // n>=2 guaranteed, so E2 is defined.
    int qw = qBest[win];
    double rawPrice = (double)(WQ * fq(qw, q0, slopeHi) - E2) / (double)WP;
    double pw = min((double)Pcap, max(0.0, rawPrice));
    return (double)V[qw] - pw;
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    Q = inf.readInt();
    n = inf.readInt();
    Pmax = inf.readLong();
    if (Q < 1 || n < 2 || Pmax < 0) quitf(_fail, "bad instance header");

    V.assign(Q + 1, 0);
    for (int q = 0; q <= Q; q++) V[q] = inf.readLong();

    C.assign(n, vector<ll>(Q + 1, 0));
    for (int j = 0; j < n; j++)
        for (int q = 0; q <= Q; q++) C[j][q] = inf.readLong();

    // ---- read & validate the participant's announced scoring rule ----
    ll WQ = ouf.readLong(1LL, 300LL, "WQ");
    ll WP = ouf.readLong(1LL, 300LL, "WP");
    int q0 = ouf.readInt(0, Q, "q0");
    int slopeHi = ouf.readInt(0, 8, "slopeHi");
    ll Pcap = ouf.readLong(0LL, Pmax, "Pcap");
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    double raw = simulate(WQ, WP, q0, slopeHi, Pcap);
    double F = max(0.0, raw);

    // internal baseline: a scale-aware capped rule (unit price weight,
    // quality weight set from the coarse Pmax/Q ratio only, a mid-range
    // hard cap on credited quality, no price cap) run through the exact
    // same procedure. This is the "do nothing clever" reference: it never
    // looks at any individual guild's cost table, and the hard cap at the
    // midpoint keeps it from paying for quality indefinitely.
    ll defWQ = max(1LL, min(300LL, Pmax / (ll)Q));
    int defQ0 = max(1, Q / 3);
    double rawB = simulate(defWQ, 1, defQ0, 0, Pmax);
    double B = max(0.0, rawB);
    if (!(B > 1e-9)) quitf(_fail, "bad instance: baseline B=%.6f is not positive", B);

    double sc = min(1000.0, 100.0 * F / B);
    if (!isfinite(sc)) quitf(_fail, "non-finite score");
    quitp(sc / 1000.0, "OK F=%.6f B=%.6f Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
