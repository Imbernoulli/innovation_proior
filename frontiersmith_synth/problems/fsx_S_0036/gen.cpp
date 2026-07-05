#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// ---------------------------------------------------------------------------
// Polar Research Base: Generator Commitment Replay  (online-to-offline replay)
//
// A polar base must meet an electrical LOAD every step of a fixed horizon.
// Free but intermittent WIND covers part of it; the rest must come from a fleet
// of diesel generators. Each generator, once switched ON, must stay on for a
// minimum number of steps (warm-up) and pays a start-up cost each time it is
// (re)started, plus a per-step no-load cost while on, plus a per-unit fuel rate.
// We author the fixed trace here; the participant picks the ON/OFF schedule.
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- structure ladder ----
    // testId 1 tiny (example scale), growing to a large fleet + long horizon.
    int G = 3 + (testId - 1);                 // 3 .. 12 generators
    int T = 8 + (testId - 1) * 175;           // 8 .. 1583 time steps

    int numBase = max(1, G / 3);              // a few cheap high-capacity units

    // ---- generator fleet (index 0 is always a baseload unit) ----
    vector<long long> P(G), b(G), rate(G), K(G), U(G);
    long long totalCap = 0;
    for (int i = 0; i < G; i++) {
        if (i < numBase) {
            // baseload: big, cheap fuel, expensive to keep on / start, long warm-up
            P[i]    = rnd.next(120, 220);
            rate[i] = rnd.next(2, 4);
            b[i]    = rnd.next(100, 180);
            K[i]    = rnd.next(300, 800);
            U[i]    = min((long long)rnd.next(4, 8), (long long)T);
        } else {
            // peaker: small, dear fuel, cheap to keep on / start, short warm-up
            P[i]    = rnd.next(25, 60);
            rate[i] = rnd.next(8, 15);
            b[i]    = rnd.next(25, 45);
            K[i]    = rnd.next(40, 120);
            U[i]    = min((long long)rnd.next(1, 2), (long long)T);
        }
        totalCap += P[i];
    }

    // ---- demand / wind trace ----
    // Demand is modest relative to fleet capacity: most of the fleet is idle
    // most of the time, so keeping everything on (the baseline) wastes a large
    // amount of no-load / start-up cost that a smart schedule avoids.
    double factor = rnd.next(0.18, 0.32);
    long long peak = (long long)(totalCap * factor);
    peak = min(peak, totalCap);
    long long lo = (long long)(totalCap * rnd.next(0.01, 0.05));
    if (lo >= peak) lo = peak / 2;
    long long amp = peak - lo;

    long long windCap = (long long)(peak * rnd.next(0.40, 0.70));
    int L = rnd.next(6, 24);                    // demand cycle length
    double phase = rnd.next(0.0, 6.28318);

    vector<long long> D(T), W(T);
    long long maxD = 0; int maxDidx = 0;
    for (int t = 0; t < T; t++) {
        double cyc = 0.5 + 0.5 * sin(2.0 * 3.14159265358979 * t / L + phase);
        long long d = lo + (long long)(amp * cyc);
        d += rnd.next(-(int)(amp / 10 + 1), (int)(amp / 10 + 1)); // noise
        if (d < 0) d = 0;
        if (d > totalCap) d = totalCap;
        D[t] = d;
        if (d > maxD) { maxD = d; maxDidx = t; }

        long long w;
        if (rnd.next(0.0, 1.0) < 0.15) w = 0;              // calm spell
        else w = rnd.next(0, (int)max(1LL, windCap));
        W[t] = w;
    }
    // guarantee at least one step needs the generators (peak has no wind)
    W[maxDidx] = 0;

    // ---- emit ----
    printf("%d %d\n", T, G);
    for (int i = 0; i < G; i++)
        printf("%lld %lld %lld %lld %lld\n", P[i], b[i], rate[i], K[i], U[i]);
    for (int t = 0; t < T; t++)
        printf("%lld %lld\n", D[t], W[t]);
    return 0;
}
