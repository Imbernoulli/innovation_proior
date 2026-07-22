#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Generator for "Rotating Star Vertex: Symmetric Crease Selection"
// (family: crease-star-flatfold). Deterministic given testId (testlib rnd is
// seeded from argv by registerGen). Prints:
//   k S mmax beta1000 gamma1000
//   v_1 ... v_{S-1}

static void emit(int k, int S, int mmax, int beta1000, int gamma1000, vector<int>& v) {
    printf("%d %d %d %d %d\n", k, S, mmax, beta1000, gamma1000);
    for (int i = 1; i <= S - 1; i++) printf("%d%c", v[i], i == S - 1 ? '\n' : ' ');
}

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int k, S, mmax, beta1000, gamma1000;
    vector<int> v; // 1-indexed, size S

    switch (testId) {
    case 1: { // tiny sanity, matches the example scale
        k = 2; S = 6; mmax = 1; beta1000 = 1000; gamma1000 = 1000;
        v.assign(S, 0);
        for (int i = 1; i <= S - 1; i++) v[i] = rnd.next(0, 6);
        break;
    }
    case 2: { // normal, k even + mmax even -> auto-regime reachable, random values
        k = 4; S = 10; mmax = 4; beta1000 = 1500; gamma1000 = 1000;
        v.assign(S, 0);
        for (int i = 1; i <= S - 1; i++) v[i] = rnd.next(0, 20);
        break;
    }
    case 3: { // TRAP: k odd, values clustered on one side of the sector
        k = 3; S = 14; mmax = 6; beta1000 = 1200; gamma1000 = 800;
        v.assign(S, 0);
        for (int i = 1; i <= S - 1; i++)
            v[i] = (i <= 5) ? rnd.next(20, 30) : rnd.next(0, 3);
        break;
    }
    case 4: { // TRAP: k odd, a single needle far from the mirror midpoint
        k = 5; S = 20; mmax = 8; beta1000 = 2000; gamma1000 = 1500;
        v.assign(S, 0);
        for (int i = 1; i <= S - 1; i++) v[i] = rnd.next(0, 5);
        int needle = 3; // far from S/2=10, deliberately off-center
        v[needle] = 30;
        break;
    }
    case 5: { // easy/auto: k even, mmax even, moderate random values
        k = 6; S = 16; mmax = 6; beta1000 = 1000; gamma1000 = 1200;
        v.assign(S, 0);
        for (int i = 1; i <= S - 1; i++) v[i] = rnd.next(0, 25);
        break;
    }
    case 6: { // TRAP-ish: k even but mmax ODD -> full-budget pick is NOT auto
        k = 8; S = 24; mmax = 9; beta1000 = 1100; gamma1000 = 900;
        v.assign(S, 0);
        for (int i = 1; i <= S - 1; i++)
            v[i] = (i % 3 == 0) ? rnd.next(22, 30) : rnd.next(0, 4);
        break;
    }
    case 7: { // TRAP: k odd, decreasing-ramp value profile
        k = 7; S = 18; mmax = 7; beta1000 = 1300; gamma1000 = 1000;
        v.assign(S, 0);
        for (int i = 1; i <= S - 1; i++) {
            int base = max(0, 28 - 2 * i);
            v[i] = base + rnd.next(0, 3);
        }
        break;
    }
    case 8: { // larger normal scale, k even, mmax even
        k = 10; S = 30; mmax = 12; beta1000 = 1400; gamma1000 = 1100;
        v.assign(S, 0);
        for (int i = 1; i <= S - 1; i++) v[i] = rnd.next(0, 30);
        break;
    }
    case 9: { // TRAP: k odd, two asymmetric needles
        k = 9; S = 24; mmax = 10; beta1000 = 1600; gamma1000 = 1400;
        v.assign(S, 0);
        for (int i = 1; i <= S - 1; i++) v[i] = rnd.next(0, 4);
        v[2] = 29;
        v[17] = 27;
        break;
    }
    case 10: { // fill the constraint envelope: max k, max S, large mmax
        k = 24; S = 60; mmax = 20; beta1000 = 1800; gamma1000 = 1600;
        v.assign(S, 0);
        for (int i = 1; i <= S - 1; i++) v[i] = rnd.next(0, 30);
        break;
    }
    default: {
        k = 2; S = 6; mmax = 1; beta1000 = 1000; gamma1000 = 1000;
        v.assign(S, 0);
        for (int i = 1; i <= S - 1; i++) v[i] = rnd.next(0, 6);
    }
    }

    emit(k, S, mmax, beta1000, gamma1000, v);
    return 0;
}
