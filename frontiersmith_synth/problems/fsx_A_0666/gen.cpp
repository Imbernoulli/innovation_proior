#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Generator for "Additive Interference Spectrum Packing".
// testId is a difficulty/structure ladder:
//  1 tiny sanity, 2 small random, 3/4 dense single-channel TRAP, 5 sparse spread,
//  6 planted well-separated clique amid a dense background TRAP, 7 needle
//  (one huge-value isolated link amid dense low-value clutter), 8 mixed clusters,
//  9 large dense TRAP, 10 largest stress (fills the declared envelope).
// All randomness comes from testlib's rnd, seeded deterministically by testId
// via registerGen(argc, argv, 1).

static int UX[4] = {1, -1, 0, 0};
static int UY[4] = {0, 0, 1, -1};

pair<long long, long long> placeInCluster(long long cx, long long cy, long long R) {
    long long x = cx + rnd.next(-R, R);
    long long y = cy + rnd.next(-R, R);
    return {x, y};
}

pair<long long, long long> unitOffset(long long x, long long y) {
    int d = rnd.next(0, 3);
    return {x + UX[d], y + UY[d]};
}

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int N, C, K;
    long long alpha;
    string mode;
    long long L; // spread half-range for the "sparse" placement

    switch (testId) {
        case 1:  N = 6;   C = 2; K = 3; alpha = 2; mode = "tiny";      L = 60;      break;
        case 2:  N = 15;  C = 2; K = 3; alpha = 2; mode = "small";     L = 2000;    break;
        case 3:  N = 25;  C = 1; K = 3; alpha = 2; mode = "dense1";    L = 5;       break;
        case 4:  N = 30;  C = 2; K = 4; alpha = 4; mode = "dense1";    L = 6;       break;
        case 5:  N = 35;  C = 3; K = 4; alpha = 2; mode = "sparse";    L = 200000;  break;
        case 6:  N = 45;  C = 2; K = 4; alpha = 2; mode = "planted";   L = 6;       break;
        case 7:  N = 55;  C = 2; K = 4; alpha = 4; mode = "needle";    L = 6;       break;
        case 8:  N = 80;  C = 3; K = 5; alpha = 2; mode = "mixed";     L = 7;       break;
        case 9:  N = 90;  C = 5; K = 5; alpha = 4; mode = "dense1";    L = 8;       break;
        case 10: N = 110; C = 6; K = 6; alpha = 2; mode = "mixed";     L = 8;       break;
        default: N = 20;  C = 2; K = 3; alpha = 2; mode = "small";     L = 2000;    break;
    }

    long long theta_i = rnd.next(2, 6);
    long long N0 = rnd.next(1, 3);
    // Baseline safety margin is 1.5x (not a generous 3x): a lone link on its
    // own channel still always clears theta, but co-channel reuse has far
    // less headroom to absorb cumulative interference before failing.
    long long base = (theta_i * N0 * 3 + 1) / 2;
    vector<long long> P(K + 1);
    for (int j = 1; j <= K; j++) P[j] = base * (1LL << (j - 1));

    vector<long long> tx(N + 1), ty(N + 1), rx(N + 1), ry(N + 1), w(N + 1);

    if (mode == "tiny" || mode == "small") {
        for (int i = 1; i <= N; i++) {
            long long x = rnd.next(-L, L), y = rnd.next(-L, L);
            tx[i] = x; ty[i] = y;
            tie(rx[i], ry[i]) = unitOffset(x, y);
            w[i] = rnd.next(1, 50);
        }
    } else if (mode == "dense1") {
        // one tight cluster forces heavy same-channel reuse: the trap.
        long long cx = rnd.next(-100000, 100000), cy = rnd.next(-100000, 100000);
        for (int i = 1; i <= N; i++) {
            auto [x, y] = placeInCluster(cx, cy, L);
            tx[i] = x; ty[i] = y;
            tie(rx[i], ry[i]) = unitOffset(x, y);
            w[i] = rnd.next(1, 60);
        }
    } else if (mode == "sparse") {
        for (int i = 1; i <= N; i++) {
            long long x = rnd.next(-L, L), y = rnd.next(-L, L);
            tx[i] = x; ty[i] = y;
            tie(rx[i], ry[i]) = unitOffset(x, y);
            w[i] = rnd.next(1, 80);
        }
    } else if (mode == "planted") {
        // M well-separated high-value links (a genuinely feasible clique if all
        // activated together) planted among a dense low-value background trap.
        int M = min(N / 4 + 3, 10);
        long long S = 4000; // separation far beyond any interference reach
        for (int i = 1; i <= M; i++) {
            long long gx = (i % 5) * S, gy = (i / 5) * S;
            tx[i] = gx; ty[i] = gy;
            tie(rx[i], ry[i]) = unitOffset(gx, gy);
            w[i] = rnd.next(60, 100);
        }
        long long cx = rnd.next(-100000, 100000), cy = rnd.next(-100000, 100000);
        for (int i = M + 1; i <= N; i++) {
            auto [x, y] = placeInCluster(cx, cy, L);
            tx[i] = x; ty[i] = y;
            tie(rx[i], ry[i]) = unitOffset(x, y);
            w[i] = rnd.next(1, 20);
        }
    } else if (mode == "needle") {
        // link 1: isolated, huge value. Links 2..N: dense low-value clutter.
        long long ix = 1900000, iy = -1900000;
        tx[1] = ix; ty[1] = iy;
        tie(rx[1], ry[1]) = unitOffset(ix, iy);
        w[1] = 2000;
        long long cx = rnd.next(-100000, 100000), cy = rnd.next(-100000, 100000);
        for (int i = 2; i <= N; i++) {
            auto [x, y] = placeInCluster(cx, cy, L);
            tx[i] = x; ty[i] = y;
            tie(rx[i], ry[i]) = unitOffset(x, y);
            w[i] = rnd.next(1, 5);
        }
    } else { // mixed: several clusters plus a sparse tail
        int nClusters = 3;
        vector<pair<long long, long long>> centers;
        for (int c = 0; c < nClusters; c++)
            centers.push_back({rnd.next(-500000, 500000), rnd.next(-500000, 500000)});
        for (int i = 1; i <= N; i++) {
            if (i % 5 == 0) { // sparse tail, low interference
                long long x = rnd.next(-L * 4000, L * 4000), y = rnd.next(-L * 4000, L * 4000);
                tx[i] = x; ty[i] = y;
                tie(rx[i], ry[i]) = unitOffset(x, y);
                w[i] = rnd.next(1, 80);
            } else {
                auto [cx, cy] = centers[rnd.next(0, nClusters - 1)];
                auto [x, y] = placeInCluster(cx, cy, L);
                tx[i] = x; ty[i] = y;
                tie(rx[i], ry[i]) = unitOffset(x, y);
                w[i] = rnd.next(1, 60);
            }
        }
    }

    printf("%d %d %d\n", N, C, K);
    printf("%lld %.6f %lld\n", alpha, (double)theta_i, N0);
    for (int j = 1; j <= K; j++) printf("%lld%c", P[j], j == K ? '\n' : ' ');
    for (int i = 1; i <= N; i++)
        printf("%lld %lld %lld %lld %lld\n", tx[i], ty[i], rx[i], ry[i], w[i]);

    return 0;
}
