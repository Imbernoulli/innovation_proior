#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// gen.cpp -- Duchy Flood Ledger.  testId is a difficulty/structure ladder: tiny sanity
// case at 1, growing to N=2600 at 10.  Every test plants a small number of high-value,
// low-tolerance "capital" segments among many cheap, high-tolerance "pasture" segments,
// and sizes the construction Budget well below the cost of fully armouring every segment
// -- so a solver MUST choose where to spend, not just how much.  Tests 3,4,7,8 (>=3 of
// the 10) are engineered TRAP cases: heavily skewed value ratios (pasture value ~1-6 vs
// capital value in the hundreds/thousands) with a tight budget, positioned so that a
// monotone "protect everyone / protect the cheapest-to-fully-armour first" strategy
// leaves the capital exposed to the (nearly) full peak, while sacrificing the pastures'
// free detention and concentrating budget on the capital is dramatically better.
int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    int idx = min(max(testId, 1), 10);

    int Ns[10]           = {3, 8, 18, 35, 70, 160, 420, 900, 1800, 2600};
    int Ts[10]           = {3, 6, 10, 18, 25, 50, 100, 180, 320, 500};
    long long peakQs[10] = {40, 90, 160, 300, 500, 900, 1500, 2500, 4000, 6000};
    int nCitiesArr[10]   = {1, 1, 2, 2, 3, 3, 4, 5, 7, 9};
    // budget fraction of "cost to fully armour every segment against the peak"
    double bfLo[10] = {0.13, 0.065, 0.028, 0.022, 0.055, 0.11, 0.025, 0.022, 0.06, 0.07};
    double bfHi[10] = {0.22, 0.105, 0.048, 0.038, 0.090, 0.17, 0.043, 0.038, 0.11, 0.12};
    // fraction of the total flood excess volume that the pasture floodplains, SUMMED
    // together, can absorb -- kept well under 1.0 so no single segment (or the pastures
    // as a whole) can passively swallow the entire flood; trap tests get it LOW so pure
    // sacrifice is not enough on its own and the capital must also be directly armoured.
    double sfLo[10] = {0.30, 0.28, 0.12, 0.08, 0.28, 0.32, 0.10, 0.08, 0.18, 0.22};
    double sfHi[10] = {0.50, 0.46, 0.22, 0.16, 0.42, 0.50, 0.18, 0.16, 0.30, 0.34};

    int N = Ns[idx - 1];
    int T = Ts[idx - 1];
    long long peakQ = peakQs[idx - 1];
    int nCities = min(nCitiesArr[idx - 1], max(1, N / 3));
    bool trapCase = (idx == 3 || idx == 4 || idx == 7 || idx == 8);

    // ---- hydrograph: rises to a peak then falls, with noise; peak is exact ----
    vector<long long> q(T + 1, 0);
    int peakPos = 1 + rnd.next(0, max(0, T - 1));
    for (int t = 1; t <= T; t++) {
        double frac = (t <= peakPos) ? (double)t / (double)max(1, peakPos)
                                      : (double)(T - t + 1) / (double)max(1, T - peakPos + 1);
        frac = max(0.08, frac);
        long long base = (long long)llround((double)peakQ * frac);
        int noiseSpan = (int)max(1LL, peakQ / 25);
        long long noise = rnd.next(0, noiseSpan) - noiseSpan / 2;
        long long val = base + noise;
        if (val < 0) val = 0;
        q[t] = val;
    }
    q[peakPos] = peakQ;  // guarantee the true peak is actually reached

    // ---- choose which segments are "capitals" (high value, low tolerance) ----
    vector<int> isCity(N + 1, 0);
    {
        vector<int> pool(N);
        for (int i = 1; i <= N; i++) pool[i - 1] = i;
        for (int i = (int)pool.size() - 1; i > 0; i--) swap(pool[i], pool[rnd.next(0, i)]);
        int placed = 0;
        int skipZone = max(1, N / 6);  // bias away from the very first segments
        for (int k = 0; k < (int)pool.size() && placed < nCities; k++) {
            if (pool[k] <= skipZone && rnd.next(0, 99) < 70) continue;
            isCity[pool[k]] = 1;
            placed++;
        }
        for (int k = 0; k < (int)pool.size() && placed < nCities; k++) {
            if (!isCity[pool[k]]) { isCity[pool[k]] = 1; placed++; }
        }
    }

    double bf = bfLo[idx - 1] + rnd.next(0.0, bfHi[idx - 1] - bfLo[idx - 1]);
    double sf = sfLo[idx - 1] + rnd.next(0.0, sfHi[idx - 1] - sfLo[idx - 1]);

    vector<long long> base_cap(N + 1), Hmax(N + 1), cost(N + 1), value(N + 1), store(N + 1);
    long long totalFullCost = 0;
    int nPastures = 0;
    long long totalExcess = 0;  // excess flood volume over a representative "unarmoured" capacity
    long long repBase = max(1LL, peakQ * 15 / 100);
    for (int t = 1; t <= T; t++) totalExcess += max(0LL, q[t] - repBase);
    for (int i = 1; i <= N; i++) if (!isCity[i]) nPastures++;
    long long storePasturePool = max((long long)nPastures, (long long)llround((double)totalExcess * sf));

    for (int i = 1; i <= N; i++) {
        base_cap[i] = rnd.next(peakQ * 5 / 100, peakQ * 30 / 100 + 1);
        Hmax[i] = peakQ + rnd.next(0, (int)(peakQ / 8) + 1);  // generous: enough to fully armour
        // cheap-to-fully-armour segments interleaved with expensive ones (drives the trap:
        // a "cheapest full protection first" greedy happily maxes out many pastures)
        cost[i] = rnd.next(1, 6) + (rnd.next(0, 99) < 20 ? rnd.next(20, 60) : 0);
        if (isCity[i]) {
            value[i] = trapCase ? rnd.next(600, 4000) : rnd.next(150, 1200);
            // small, T-independent tolerance: the capital's floodplain saturates fast
            store[i] = rnd.next(peakQ / 25 + 1, peakQ / 6 + 2);
        } else {
            value[i] = rnd.next(1, trapCase ? 4 : 8);
            // this pasture's share of the pool of total sacrificial capacity, jittered
            long long share = storePasturePool / max(1, nPastures);
            long long lo = max(0LL, share - share / 3), hi = share + share / 3 + 1;
            store[i] = rnd.next(lo, hi);
        }
        long long needed = max(0LL, peakQ - base_cap[i]);
        totalFullCost += cost[i] * needed;
    }
    long long Budget = max(1LL, (long long)llround((double)totalFullCost * bf));

    printf("%d %d %lld\n", N, T, Budget);
    for (int i = 1; i <= N; i++)
        printf("%lld %lld %lld %lld %lld\n", base_cap[i], Hmax[i], cost[i], value[i], store[i]);
    for (int t = 1; t <= T; t++) printf("%lld%c", q[t], t == T ? '\n' : ' ');
    return 0;
}
