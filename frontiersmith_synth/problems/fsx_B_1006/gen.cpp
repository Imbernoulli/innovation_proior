#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Generator for "Cascade Reservoir Chain: Scheduling Irreversible Releases"
// family: cascade-reservoir-release
//
// Emits, per testId, a chain of K reservoirs, a T-period inflow schedule to
// reservoir 1, and a T-period local-demand schedule per reservoir. testId 1 is
// the tiny statement example (hand-verified). testId 2,3,5,7,9 are engineered
// TRAP cases: a reactive/current-period-only release policy lands far from an
// anticipatory one because the checker force-spills BEFORE the period's release
// choice is read, and downstream demand must be met by water released lag
// periods EARLIER. testId 4,6,8,10 are denser mixed/random stress scales that
// also plant spikes+droughts but add noise so no strategy reaches a trivial
// zero-penalty optimum. All core structural numbers are fixed by construction;
// `rnd` (seeded deterministically by testId via registerGen) only adds small
// jitter, so the trap is guaranteed to fire regardless of RNG draws.
// -----------------------------------------------------------------------------

struct Case {
    int K, T;
    vector<ll> Cap, S0, fw, dw, lag, I;
    vector<vector<ll>> D; // D[i][t], 1-indexed
};

static void alloc(Case &c) {
    c.Cap.assign(c.K + 1, 0);
    c.S0.assign(c.K + 1, 0);
    c.fw.assign(c.K + 1, 0);
    c.dw.assign(c.K + 1, 0);
    c.lag.assign(c.K + 1, 0);
    c.I.assign(c.T + 1, 0);
    c.D.assign(c.K + 1, vector<ll>(c.T + 1, 0));
}

// ---- testId 1: tiny hand-verified statement example ----
static Case caseExample() {
    Case c; c.K = 2; c.T = 5; alloc(c);
    c.Cap[1] = 50; c.S0[1] = 5; c.fw[1] = 3; c.dw[1] = 2;
    c.Cap[2] = 30; c.S0[2] = 5; c.fw[2] = 4; c.dw[2] = 5;
    c.lag[1] = 1;
    c.I = {0, 5, 5, 40, 5, 5};
    c.D[2][4] = 20;
    return c;
}

// ---- SPIKE trap: quiet baseline inflow with a few sharp spikes; downstream
//      demand placed exactly lag periods after the ideal spike-release arrival.
//      A reactive plan cannot avoid the spill (spill happens before release is
//      read) and, chasing "today's" demand report, delivers late every time.
static Case caseSpikeTrap(int K, int T, vector<ll> lag, ll cap1, ll spikeVal,
                           vector<int> spikeAt, ll baseInflow, int seedTag) {
    Case c; c.K = K; c.T = T; alloc(c);
    for (int i = 1; i <= K; i++) {
        c.Cap[i] = cap1 - (ll)(i - 1) * (cap1 / (2 * K + 2));
        c.S0[i]  = c.Cap[i] / 6 + rnd.next(0, (int)(c.Cap[i] / 20 + 1));
        c.fw[i]  = 3 + rnd.next(0, 4);
        c.dw[i]  = 4 + rnd.next(0, 4);
    }
    for (int i = 1; i < K; i++) c.lag[i] = lag[i - 1];
    for (int t = 1; t <= T; t++)
        c.I[t] = baseInflow + rnd.next(0, (int)(baseInflow / 3 + 1));
    for (int s : spikeAt) if (s >= 1 && s <= T) c.I[s] += spikeVal;
    // cumulative lag to reservoir i (from reservoir 1)
    vector<ll> cum(K + 1, 0);
    for (int i = 2; i <= K; i++) cum[i] = cum[i - 1] + c.lag[i - 1];
    for (int s : spikeAt) {
        for (int i = 2; i <= K; i++) {
            ll t2 = s + cum[i];
            if (t2 >= 1 && t2 <= T)
                c.D[i][t2] += spikeVal / (2 + (i - 1)) + rnd.next(0, 5);
        }
    }
    // Structural penalty floor: at t=1 no water could possibly have reached
    // reservoir K yet (every path needs >=1 lag hop), so demanding slightly
    // more than S0[K] there is an irreducible deficit under ANY strategy.
    // This keeps even a perfect plan's F bounded away from 0 (no per-test
    // score saturation at the 10x cap) without touching the trap dynamics.
    c.D[K][1] = max(c.D[K][1], c.S0[K] + max(1LL, c.Cap[K] / 8));
    (void)seedTag;
    return c;
}

// ---- DROUGHT trap: high inflow for a prefix, then a long zero-inflow stretch,
//      with sustained/rising demand during the drought at a downstream
//      reservoir. Only pre-filling (well before the drought, offset by lag)
//      avoids deficits; reacting to current storage alone drains too early.
static Case caseDroughtTrap(int K, int T, vector<ll> lag, ll cap1, int wetLen,
                             ll wetInflow, ll droughtDemand) {
    Case c; c.K = K; c.T = T; alloc(c);
    for (int i = 1; i <= K; i++) {
        c.Cap[i] = cap1 - (ll)(i - 1) * (cap1 / (2 * K + 2));
        c.S0[i]  = c.Cap[i] / 8;
        c.fw[i]  = 3 + rnd.next(0, 4);
        c.dw[i]  = 5 + rnd.next(0, 5);
    }
    for (int i = 1; i < K; i++) c.lag[i] = lag[i - 1];
    for (int t = 1; t <= T; t++)
        c.I[t] = (t <= wetLen) ? (wetInflow + rnd.next(0, (int)(wetInflow / 4 + 1))) : 0;
    vector<ll> cum(K + 1, 0);
    for (int i = 2; i <= K; i++) cum[i] = cum[i - 1] + c.lag[i - 1];
    for (int t = wetLen + 2; t <= T; t++) {
        ll dem = droughtDemand + (ll)(t - wetLen) * (droughtDemand / 20 + 1);
        c.D[K][t] += dem + rnd.next(0, 4);
        if (K > 1) c.D[K - 1][t] += dem / 3;
    }
    (void)cum;
    return c;
}

// ---- NEEDLE: near-flat schedule everywhere except ONE huge demand deep in the
//      chain at one period; only a multi-hop anticipatory release (started many
//      periods earlier, cascading through K-1 lags) can meet it.
static Case caseNeedle(int K, int T, vector<ll> lag, ll cap1, ll needleVal) {
    Case c; c.K = K; c.T = T; alloc(c);
    for (int i = 1; i <= K; i++) {
        c.Cap[i] = cap1 - (ll)(i - 1) * (cap1 / (2 * K + 2));
        c.S0[i]  = c.Cap[i] / 10;
        c.fw[i]  = 3 + rnd.next(0, 3);
        c.dw[i]  = 6 + rnd.next(0, 4);
    }
    for (int i = 1; i < K; i++) c.lag[i] = lag[i - 1];
    ll flatIn = cap1 / 30 + 1;
    for (int t = 1; t <= T; t++) c.I[t] = flatIn + rnd.next(0, 3);
    for (int i = 1; i <= K; i++)
        for (int t = 1; t <= T; t++)
            c.D[i][t] = rnd.next(0, 2);
    ll cumLag = 0; for (int i = 1; i < K; i++) cumLag += c.lag[i];
    int needleT = (int)min((ll)T - 2, cumLag + T / 2);
    if (needleT < 2) needleT = min(T, (int)cumLag + 2);
    c.D[K][needleT] += needleVal;
    c.D[K][1] = max(c.D[K][1], c.S0[K] + max(1LL, c.Cap[K] / 8)); // irreducible floor, see caseSpikeTrap
    return c;
}

// ---- dense mixed/random stress scale: many overlapping spikes+droughts+noise
//      so no strategy (even a good heuristic) reaches a trivial zero penalty.
static Case caseDense(int K, int T, vector<ll> lag, ll cap1, int nSpikes, int nDroughts) {
    Case c; c.K = K; c.T = T; alloc(c);
    for (int i = 1; i <= K; i++) {
        c.Cap[i] = cap1 - (ll)(i - 1) * (cap1 / (2 * K + 2));
        c.S0[i]  = c.Cap[i] / 5 + rnd.next(0, (int)(c.Cap[i] / 10 + 1));
        c.fw[i]  = 1 + rnd.next(0, 9);
        c.dw[i]  = 1 + rnd.next(0, 9);
    }
    for (int i = 1; i < K; i++) c.lag[i] = lag[i - 1];
    ll base = cap1 / 40 + 1;
    for (int t = 1; t <= T; t++) c.I[t] = base + rnd.next(0, (int)(base * 2 + 1));
    for (int s = 0; s < nSpikes; s++) {
        int t = 1 + rnd.next(0, T - 1);
        c.I[t] += cap1 / 2 + rnd.next(0, (int)(cap1 / 2));
    }
    for (int i = 1; i <= K; i++)
        for (int t = 1; t <= T; t++)
            c.D[i][t] = rnd.next(0, (int)(c.Cap[i] / 40 + 1));
    for (int d = 0; d < nDroughts; d++) {
        int i = 1 + rnd.next(0, K - 1);
        int t = 1 + rnd.next(0, T - 1);
        c.D[i][t] += c.Cap[i] / 3 + rnd.next(0, (int)(c.Cap[i] / 3 + 1));
    }
    return c;
}

int main(int argc, char *argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    Case c;
    switch (testId) {
        case 1:
            c = caseExample();
            break;
        case 2:
            c = caseSpikeTrap(3, 24, {2, 3}, 1200, 1000, {6, 14, 20}, 40, testId);
            break;
        case 3:
            c = caseDroughtTrap(2, 30, {4}, 2000, 10, 900, 250);
            break;
        case 4:
            c = caseDense(4, 40, {1, 2, 2}, 3000, 4, 3);
            break;
        case 5:
            c = caseNeedle(5, 60, {2, 2, 3, 2}, 4000, 9000);
            break;
        case 6: {
            // alternating spike/drought combined chain
            Case a = caseSpikeTrap(3, 80, {3, 2}, 5000, 3500, {10, 30, 50, 70}, 150, testId);
            Case b = caseDroughtTrap(3, 80, {3, 2}, 5000, 22, 3000, 700);
            c = a; c.K = a.K; c.T = a.T;
            for (int t = 1; t <= c.T; t++) c.I[t] = (a.I[t] + (t > 40 ? 0 : b.I[t])) / 2 + 1;
            for (int i = 1; i <= c.K; i++)
                for (int t = 1; t <= c.T; t++)
                    c.D[i][t] = a.D[i][t] + (t > 40 ? b.D[i][t] : 0);
            break;
        }
        case 7:
            c = caseSpikeTrap(4, 100, {2, 3, 2}, 8000, 6000, {12, 24, 36, 48, 60}, 200, testId);
            break;
        case 8:
            c = caseDense(5, 150, {2, 1, 3, 2}, 20000, 10, 8);
            break;
        case 9:
            c = caseSpikeTrap(6, 140, {3, 4, 3, 5, 3}, 30000, 24000, {8, 40, 80}, 400, testId);
            break;
        case 10:
        default:
            c = caseDense(6, 320, {4, 3, 5, 2, 3}, 200000, 24, 16);
            break;
    }

    // clamp to the stated constraint envelope defensively
    for (int i = 1; i <= c.K; i++) {
        c.Cap[i] = max(1LL, min((ll)200000, c.Cap[i]));
        c.S0[i]  = max(0LL, min(c.Cap[i], c.S0[i]));
        c.fw[i]  = max(1LL, min(10LL, c.fw[i]));
        c.dw[i]  = max(1LL, min(10LL, c.dw[i]));
    }
    for (int i = 1; i < c.K; i++) c.lag[i] = max(1LL, min(8LL, c.lag[i]));
    for (int t = 1; t <= c.T; t++) c.I[t] = max(0LL, min(200000LL, c.I[t]));
    for (int i = 1; i <= c.K; i++)
        for (int t = 1; t <= c.T; t++)
            c.D[i][t] = max(0LL, min(200000LL, c.D[i][t]));

    printf("%d %d\n", c.K, c.T);
    for (int i = 1; i <= c.K; i++)
        printf("%lld %lld %lld %lld\n", c.Cap[i], c.S0[i], c.fw[i], c.dw[i]);
    for (int i = 1; i < c.K; i++) printf("%lld%c", c.lag[i], i + 1 < c.K ? ' ' : '\n');
    if (c.K == 1) printf("\n");
    for (int t = 1; t <= c.T; t++) printf("%lld%c", c.I[t], t < c.T ? ' ' : '\n');
    for (int i = 1; i <= c.K; i++) {
        for (int t = 1; t <= c.T; t++) printf("%lld%c", c.D[i][t], t < c.T ? ' ' : '\n');
    }
    return 0;
}
